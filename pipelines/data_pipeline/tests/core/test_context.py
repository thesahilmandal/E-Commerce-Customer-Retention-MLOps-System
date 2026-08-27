import pytest

from unittest.mock import MagicMock, patch

from pipelines.data_pipeline.src.core.config_parser import DataPipelineConfig
from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException


@patch("pipelines.data_pipeline.src.core.context.os.makedirs")
@patch("pipelines.data_pipeline.src.core.context.duckdb.connect")
def test_pipeline_context_initialization_success(
    mock_duckdb_connect: MagicMock,
    mock_makedirs: MagicMock,
    dummy_pipeline_config: DataPipelineConfig,
    mock_s3_sync: MagicMock,
) -> None:
    mock_con = MagicMock()
    mock_duckdb_connect.return_value = mock_con

    context = PipelineContext(
        run_id="test_run_01",
        start_date="2023-01-01",
        end_date="2023-06-01",
        config=dummy_pipeline_config,
        s3_sync=mock_s3_sync,
    )

    assert context.run_id == "test_run_01"
    assert context.start_date == "2023-01-01"
    assert context.end_date == "2023-06-01"
    assert context.config == dummy_pipeline_config
    assert context.s3_sync == mock_s3_sync
    assert context.db_con == mock_con

    mock_makedirs.assert_called_once_with(
        "/tmp/test_data_pipeline_spill",
        exist_ok=True,
    )
    mock_duckdb_connect.assert_called_once_with(database=":memory:")

    mock_con.execute.assert_any_call("PRAGMA threads=2;")
    mock_con.execute.assert_any_call("PRAGMA memory_limit='4GB';")
    mock_con.execute.assert_any_call(
        "PRAGMA temp_directory='/tmp/test_data_pipeline_spill';"
    )
    mock_con.execute.assert_any_call("INSTALL httpfs;")
    mock_con.execute.assert_any_call("LOAD httpfs;")
    mock_con.execute.assert_any_call("CALL load_aws_credentials();")


@patch("pipelines.data_pipeline.src.core.context.os.makedirs")
@patch("pipelines.data_pipeline.src.core.context.duckdb.connect")
def test_pipeline_context_initialization_failure(
    mock_duckdb_connect: MagicMock,
    mock_makedirs: MagicMock,
    dummy_pipeline_config: DataPipelineConfig,
    mock_s3_sync: MagicMock,
) -> None:
    mock_duckdb_connect.side_effect = Exception(
        "DuckDB initialization failed"
    )

    with pytest.raises(CustomException):
        PipelineContext(
            run_id="test_run_02",
            start_date="2023-01-01",
            end_date="2023-06-01",
            config=dummy_pipeline_config,
            s3_sync=mock_s3_sync,
        )


@patch("pipelines.data_pipeline.src.core.context.os.makedirs")
@patch("pipelines.data_pipeline.src.core.context.duckdb.connect")
def test_pipeline_context_close_success(
    mock_duckdb_connect: MagicMock,
    mock_makedirs: MagicMock,
    dummy_pipeline_config: DataPipelineConfig,
    mock_s3_sync: MagicMock,
) -> None:
    mock_con = MagicMock()
    mock_duckdb_connect.return_value = mock_con

    context = PipelineContext(
        run_id="test_run_03",
        start_date="2023-01-01",
        end_date="2023-06-01",
        config=dummy_pipeline_config,
        s3_sync=mock_s3_sync,
    )

    context.close()

    mock_con.close.assert_called_once()
    assert context.db_con is None


@patch("pipelines.data_pipeline.src.core.context.os.makedirs")
@patch("pipelines.data_pipeline.src.core.context.duckdb.connect")
def test_pipeline_context_close_with_exception_suppressed(
    mock_duckdb_connect: MagicMock,
    mock_makedirs: MagicMock,
    dummy_pipeline_config: DataPipelineConfig,
    mock_s3_sync: MagicMock,
) -> None:
    mock_con = MagicMock()
    mock_con.close.side_effect = Exception("Failed to close gracefully")
    mock_duckdb_connect.return_value = mock_con

    context = PipelineContext(
        run_id="test_run_04",
        start_date="2023-01-01",
        end_date="2023-06-01",
        config=dummy_pipeline_config,
        s3_sync=mock_s3_sync,
    )

    context.close()

    mock_con.close.assert_called_once()
    assert context.db_con is None


@patch("pipelines.data_pipeline.src.core.context.os.makedirs")
@patch("pipelines.data_pipeline.src.core.context.duckdb.connect")
def test_pipeline_context_manager_lifecycle(
    mock_duckdb_connect: MagicMock,
    mock_makedirs: MagicMock,
    dummy_pipeline_config: DataPipelineConfig,
    mock_s3_sync: MagicMock,
) -> None:
    mock_con = MagicMock()
    mock_duckdb_connect.return_value = mock_con

    with PipelineContext(
        run_id="test_run_05",
        start_date="2023-01-01",
        end_date="2023-06-01",
        config=dummy_pipeline_config,
        s3_sync=mock_s3_sync,
    ) as context:
        assert context.db_con is mock_con

    mock_con.close.assert_called_once()
    assert context.db_con is None