import os
from unittest.mock import MagicMock, call, patch

import pytest

from pipelines.training_pipeline.src.core.config_parser import ConfigParser
from pipelines.training_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def mock_duckdb() -> MagicMock:
    with patch("pipelines.training_pipeline.src.core.context.duckdb") as mock:
        yield mock


@pytest.fixture
def mock_s3_sync() -> MagicMock:
    with patch("pipelines.training_pipeline.src.core.context.S3Sync") as mock:
        yield mock


@patch("os.makedirs")
@patch("shutil.rmtree")
@patch("os.path.exists", return_value=True)
def test_pipeline_context_lifecycle_and_cleanup(
    mock_exists: MagicMock,
    mock_rmtree: MagicMock,
    mock_makedirs: MagicMock,
    mock_s3_sync: MagicMock,
    mock_duckdb: MagicMock,
    config_parser: ConfigParser,
) -> None:
    run_id = "test_run_123"
    s3_uri = "s3://test/data.parquet"

    with PipelineContext(
        run_id=run_id,
        training_dataset_s3_uri_path=s3_uri,
        config_parser=config_parser,
        cleanup_on_exit=True,
    ) as context:
        assert context.run_id == run_id
        assert context.training_dataset_s3_uri_path == s3_uri
        assert context.config == config_parser
        assert context.cleanup_on_exit is True

        # Verify Directories
        expected_base = os.path.join(
            "artifacts/training_pipeline",
            run_id,
        )

        assert context.run_artifact_dir == expected_base
        assert context.data_processor_dir == os.path.join(
            expected_base,
            "01_data_processor",
        )
        assert context.model_trainer_dir == os.path.join(
            expected_base,
            "02_model_trainer",
        )
        assert context.model_evaluator_dir == os.path.join(
            expected_base,
            "03_model_evaluator",
        )
        assert context.model_registry_dir == os.path.join(
            expected_base,
            "04_model_registry",
        )

        assert mock_makedirs.call_count == 5
        mock_s3_sync.assert_called_once()
        mock_duckdb.connect.assert_called_once_with(database=":memory:")

    # Verify Teardown via Context Manager
    mock_duckdb.connect.return_value.close.assert_called_once()
    mock_rmtree.assert_called_once_with(expected_base)


@patch("os.makedirs")
@patch("shutil.rmtree")
@patch("os.path.exists", return_value=True)
def test_cleanup_on_exit_false(
    mock_exists: MagicMock,
    mock_rmtree: MagicMock,
    mock_makedirs: MagicMock,
    mock_s3_sync: MagicMock,
    mock_duckdb: MagicMock,
    config_parser: ConfigParser,
) -> None:
    context = PipelineContext(
        run_id="test_run_123",
        training_dataset_s3_uri_path="s3://test/data.parquet",
        config_parser=config_parser,
        cleanup_on_exit=False,
    )

    context.close()

    mock_duckdb.connect.return_value.close.assert_called_once()
    mock_rmtree.assert_not_called()


@patch("os.makedirs")
def test_duckdb_initialization_sequence(
    mock_makedirs: MagicMock,
    mock_s3_sync: MagicMock,
    mock_duckdb: MagicMock,
    config_parser: ConfigParser,
) -> None:
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn

    PipelineContext(
        run_id="test_run_123",
        training_dataset_s3_uri_path="s3://test/data.parquet",
        config_parser=config_parser,
    )

    expected_calls = [
        call("INSTALL httpfs;"),
        call("LOAD httpfs;"),
        call("INSTALL aws;"),
        call("LOAD aws;"),
        call("CALL load_aws_credentials();"),
    ]

    mock_conn.execute.assert_has_calls(
        expected_calls,
        any_order=False,
    )


@patch("os.makedirs")
def test_initialization_failure_raises_custom_exception(
    mock_makedirs: MagicMock,
    mock_duckdb: MagicMock,
    config_parser: ConfigParser,
) -> None:
    mock_duckdb.connect.side_effect = Exception("DuckDB init failed")

    with pytest.raises(CustomException) as exc_info:
        PipelineContext(
            run_id="test_run_123",
            training_dataset_s3_uri_path="s3://test/data.parquet",
            config_parser=config_parser,
        )

    assert "DuckDB init failed" in str(exc_info.value)


@patch("shutil.rmtree")
def test_close_safe_when_partially_initialized(
    mock_rmtree: MagicMock,
) -> None:
    context = object.__new__(PipelineContext)

    context.cleanup_on_exit = True
    context.close()

    mock_rmtree.assert_not_called()