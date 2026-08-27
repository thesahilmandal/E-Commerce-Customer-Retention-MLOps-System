
import pytest
from unittest.mock import MagicMock, patch

from pipelines.inference_pipeline.src.core.context import InferencePipelineContext
from shared_core.exceptions.custom_exception import CustomException


@patch("pipelines.inference_pipeline.src.core.context.S3Sync")
@patch("pipelines.inference_pipeline.src.core.context.ConfigParser")
def test_context_initialization_success(
    mock_config_parser: MagicMock,
    mock_s3_sync: MagicMock,
) -> None:
    run_id = "test_run_123"

    context = InferencePipelineContext(run_id=run_id)

    assert context.run_id == run_id
    assert context.config is not None
    assert context.s3_sync is not None

    mock_config_parser.assert_called_once()
    mock_s3_sync.assert_called_once()


@patch("pipelines.inference_pipeline.src.core.context.ConfigParser")
def test_context_initialization_failure(
    mock_config_parser: MagicMock,
) -> None:
    mock_config_parser.side_effect = Exception("Config load failed")

    with pytest.raises(CustomException) as exc_info:
        InferencePipelineContext(run_id="test_run_123")

    assert "Config load failed" in str(exc_info.value)


@patch("pipelines.inference_pipeline.src.core.context.S3Sync")
@patch("pipelines.inference_pipeline.src.core.context.ConfigParser")
def test_uninitialized_duckdb_con_raises(
    mock_config_parser: MagicMock,
    mock_s3_sync: MagicMock,
) -> None:
    context = InferencePipelineContext(run_id="test_run_123")

    with pytest.raises(RuntimeError) as exc_info:
        _ = context.duckdb_con

    assert "not initialized" in str(exc_info.value)
    assert "context manager" in str(exc_info.value)


@patch("pipelines.inference_pipeline.src.core.context.duckdb")
@patch("pipelines.inference_pipeline.src.core.context.S3Sync")
@patch("pipelines.inference_pipeline.src.core.context.ConfigParser")
def test_context_manager_lifecycle(
    mock_config_parser: MagicMock,
    mock_s3_sync: MagicMock,
    mock_duckdb: MagicMock,
) -> None:
    mock_con = MagicMock()
    mock_duckdb.connect.return_value = mock_con

    context = InferencePipelineContext(run_id="test_run_123")

    with context as ctx:
        assert ctx.duckdb_con == mock_con

        # Verify extension loading and basic config queries
        mock_con.execute.assert_any_call("INSTALL httpfs;")
        mock_con.execute.assert_any_call("LOAD httpfs;")
        mock_con.execute.assert_any_call("INSTALL aws;")
        mock_con.execute.assert_any_call("LOAD aws;")
        mock_con.execute.assert_any_call("CALL load_aws_credentials();")
        mock_con.execute.assert_any_call("SET s3_region='us-east-1';")
        mock_con.execute.assert_any_call("PRAGMA threads=4;")
        mock_con.execute.assert_any_call(
            "PRAGMA memory_limit='4GB';"
        )

    # Verify cleanup on exit
    mock_con.close.assert_called_once()

    with pytest.raises(RuntimeError):
        _ = context.duckdb_con


@patch("pipelines.inference_pipeline.src.core.context.duckdb")
@patch("pipelines.inference_pipeline.src.core.context.S3Sync")
@patch("pipelines.inference_pipeline.src.core.context.ConfigParser")
def test_context_manager_custom_aws_region(
    mock_config_parser: MagicMock,
    mock_s3_sync: MagicMock,
    mock_duckdb: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")

    mock_con = MagicMock()
    mock_duckdb.connect.return_value = mock_con

    with InferencePipelineContext(run_id="test_run_123"):
        mock_con.execute.assert_any_call(
            "SET s3_region='eu-west-1';"
        )


@patch("pipelines.inference_pipeline.src.core.context.duckdb")
@patch("pipelines.inference_pipeline.src.core.context.S3Sync")
@patch("pipelines.inference_pipeline.src.core.context.ConfigParser")
def test_context_manager_initialization_failure_closes_safely(
    mock_config_parser: MagicMock,
    mock_s3_sync: MagicMock,
    mock_duckdb: MagicMock,
) -> None:
    mock_con = MagicMock()
    mock_duckdb.connect.return_value = mock_con

    # Simulate failure during query execution
    # (e.g., extension fails to install).
    mock_con.execute.side_effect = Exception(
        "DuckDB extension error"
    )

    context = InferencePipelineContext(run_id="test_run_123")

    with patch.object(
        context,
        "_safe_close_connection",
    ) as mock_safe_close:
        with pytest.raises(CustomException) as exc_info:
            with context:
                pass

        assert "DuckDB extension error" in str(exc_info.value)
        mock_safe_close.assert_called_once()


@patch("pipelines.inference_pipeline.src.core.context.S3Sync")
@patch("pipelines.inference_pipeline.src.core.context.ConfigParser")
def test_safe_close_connection_handles_exception(
    mock_config_parser: MagicMock,
    mock_s3_sync: MagicMock,
) -> None:
    context = InferencePipelineContext(run_id="test_run_123")

    mock_con = MagicMock()
    mock_con.close.side_effect = Exception("Forced close exception")
    context._duckdb_con = mock_con

    # Call directly. It should swallow the exception
    # and set the connection to None.
    context._safe_close_connection()

    mock_con.close.assert_called_once()