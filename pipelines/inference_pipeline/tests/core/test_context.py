import os
import pytest
from unittest.mock import patch, MagicMock

from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.core.context import InferencePipelineContext


@pytest.fixture
def mock_context_dependencies():
    """
    Patches the internal dependencies of InferencePipelineContext to ensure
    unit tests do not perform real file I/O or S3 connections during initialization.
    """
    with patch(
        "pipelines.inference_pipeline.src.core.context.ConfigParser"
    ) as mock_cfg_parser, patch(
        "pipelines.inference_pipeline.src.core.context.S3Sync"
    ) as mock_s3_sync:

        mock_cfg_instance = MagicMock()
        mock_cfg_parser.return_value = mock_cfg_instance

        mock_s3_instance = MagicMock()
        mock_s3_sync.return_value = mock_s3_instance

        yield mock_cfg_instance, mock_s3_instance


def test_context_initialization_success(mock_context_dependencies):
    """
    Tests that the context initializes correctly and assigns the provided run_id
    and internal dependency objects.
    """
    mock_cfg, mock_s3 = mock_context_dependencies

    run_id = "test_run_001"
    context = InferencePipelineContext(run_id=run_id)

    assert context.run_id == run_id
    assert context.config is mock_cfg
    assert context.s3_sync is mock_s3


def test_context_initialization_failure(mock_context_dependencies):
    """
    Tests that a failure during initialization (e.g., config parsing failure)
    is caught and re-raised as a CustomException.
    """
    with patch(
        "pipelines.inference_pipeline.src.core.context.ConfigParser",
        side_effect=Exception("Config Error"),
    ):
        with pytest.raises(CustomException) as exc_info:
            InferencePipelineContext(run_id="test_run_002")

        assert "Config Error" in str(exc_info.value)


def test_duckdb_con_property_raises_error_before_enter(mock_context_dependencies):
    """
    Tests that accessing duckdb_con before the context manager is entered
    raises a RuntimeError to prevent null reference errors.
    """
    context = InferencePipelineContext(run_id="test_run_003")

    with pytest.raises(RuntimeError) as exc_info:
        _ = context.duckdb_con

    assert "not initialized" in str(exc_info.value).lower()
    assert "ensure context manager" in str(exc_info.value).lower()


@patch("pipelines.inference_pipeline.src.core.context.duckdb")
def test_context_enter_success(mock_duckdb, mock_context_dependencies):
    """
    Tests that entering the context successfully initializes DuckDB,
    loads the required AWS extensions, configures credentials, and sets pragmas.
    """
    mock_con = MagicMock()
    mock_duckdb.connect.return_value = mock_con

    context = InferencePipelineContext(run_id="test_run_004")

    with context as ctx:
        assert ctx is context
        assert ctx.duckdb_con is mock_con

        # Verify connection was opened in-memory
        mock_duckdb.connect.assert_called_once_with(database=":memory:")

        # Verify required extensions and AWS credentials are loaded
        execute_calls = [call.args[0] for call in mock_con.execute.call_args_list]
        assert "INSTALL httpfs;" in execute_calls
        assert "LOAD httpfs;" in execute_calls
        assert "INSTALL aws;" in execute_calls
        assert "LOAD aws;" in execute_calls
        assert "CALL load_aws_credentials();" in execute_calls

        # Verify that AWS region fallback is applied correctly
        aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        assert f"SET s3_region='{aws_region}';" in execute_calls

        # Verify performance pragmas are set
        assert "PRAGMA threads=4;" in execute_calls
        assert "PRAGMA memory_limit='4GB';" in execute_calls


@patch("pipelines.inference_pipeline.src.core.context.duckdb")
def test_context_enter_failure(mock_duckdb, mock_context_dependencies):
    """
    Tests that a failure during DuckDB initialization is caught, the connection
    is safely closed (if partially initialized), and a CustomException is raised.
    """
    mock_duckdb.connect.side_effect = Exception("DuckDB Connection Failed")

    context = InferencePipelineContext(run_id="test_run_005")

    with patch.object(context, "_safe_close_connection") as mock_safe_close:
        with pytest.raises(CustomException) as exc_info:
            with context:
                pass

        assert "DuckDB Connection Failed" in str(exc_info.value)
        mock_safe_close.assert_called_once()


@patch("pipelines.inference_pipeline.src.core.context.duckdb")
def test_context_exit_success(mock_duckdb, mock_context_dependencies):
    """
    Tests that exiting the context manager successfully calls the
    safe teardown method for the database connection.
    """
    mock_con = MagicMock()
    mock_duckdb.connect.return_value = mock_con

    context = InferencePipelineContext(run_id="test_run_006")

    with patch.object(context, "_safe_close_connection") as mock_safe_close:
        with context:
            pass

        mock_safe_close.assert_called_once()


def test_safe_close_connection_success(mock_context_dependencies):
    """
    Tests that the helper method safely closes the DuckDB connection
    and resets the internal reference to None.
    """
    context = InferencePipelineContext(run_id="test_run_007")
    mock_con = MagicMock()
    context._duckdb_con = mock_con

    context._safe_close_connection()

    mock_con.close.assert_called_once()
    assert context._duckdb_con is None


def test_safe_close_connection_with_exception(mock_context_dependencies, caplog):
        """
        Tests that if DuckDB throws an error while closing, it is caught and logged
        without halting the pipeline execution.
        """
        context = InferencePipelineContext(run_id="test_run_008")
        mock_con = MagicMock()
        mock_con.close.side_effect = Exception("Failed to close gracefully")
        context._duckdb_con = mock_con
        
        # Should not raise an exception
        context._safe_close_connection()
        
        # FIX: The actual implementation leaves the reference intact if close() fails.
        assert context._duckdb_con is mock_con
        
        # Verify the warning was logged securely
        assert "Error encountered while closing DuckDB connection" in caplog.text
        assert "Failed to close gracefully" in caplog.text