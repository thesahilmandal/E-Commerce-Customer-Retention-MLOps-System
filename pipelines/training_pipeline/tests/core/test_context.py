import os
from unittest.mock import MagicMock, patch

import pytest

from pipelines.training_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException


@patch("pipelines.training_pipeline.src.core.context.duckdb.connect")
@patch("pipelines.training_pipeline.src.core.context.S3Sync")
def test_context_initialization_and_directory_creation(
    mock_s3_sync, mock_duckdb, mock_config_parser
):
    context = PipelineContext(
        run_id="test_run_dirs",
        training_dataset_s3_uri_path="s3://dummy-bucket/master_panel.parquet",
        config_parser=mock_config_parser,
    )

    assert context.run_id == "test_run_dirs"
    assert (
        context.training_dataset_s3_uri_path
        == "s3://dummy-bucket/master_panel.parquet"
    )
    assert context.config == mock_config_parser

    assert os.path.exists(context.run_artifact_dir)
    assert os.path.exists(context.data_processor_dir)
    assert os.path.exists(context.model_trainer_dir)
    assert os.path.exists(context.model_evaluator_dir)
    assert os.path.exists(context.model_registry_dir)

    mock_duckdb.assert_called_once_with(database=":memory:")

    executed_queries = [
        call.args[0] for call in context.db_conn.execute.call_args_list
    ]

    assert "INSTALL httpfs;" in executed_queries
    assert "LOAD httpfs;" in executed_queries
    assert "INSTALL aws;" in executed_queries
    assert "LOAD aws;" in executed_queries
    assert "CALL load_aws_credentials();" in executed_queries


@patch("pipelines.training_pipeline.src.core.context.duckdb.connect")
@patch("pipelines.training_pipeline.src.core.context.S3Sync")
def test_context_cleanup_on_exit_true(
    mock_s3_sync, mock_duckdb, mock_config_parser
):
    with PipelineContext(
        run_id="test_cleanup_true",
        training_dataset_s3_uri_path="s3://dummy-bucket/master_panel.parquet",
        config_parser=mock_config_parser,
        cleanup_on_exit=True,
    ) as context:
        run_dir = context.run_artifact_dir
        assert os.path.exists(run_dir)

    assert not os.path.exists(
        run_dir
    ), "Workspace should be deleted when cleanup_on_exit is True"
    context.db_conn.close.assert_called_once()


@patch("pipelines.training_pipeline.src.core.context.duckdb.connect")
@patch("pipelines.training_pipeline.src.core.context.S3Sync")
def test_context_cleanup_on_exit_false(
    mock_s3_sync, mock_duckdb, mock_config_parser
):
    with PipelineContext(
        run_id="test_cleanup_false",
        training_dataset_s3_uri_path="s3://dummy-bucket/master_panel.parquet",
        config_parser=mock_config_parser,
        cleanup_on_exit=False,
    ) as context:
        run_dir = context.run_artifact_dir
        assert os.path.exists(run_dir)

    assert os.path.exists(
        run_dir
    ), "Workspace should remain intact when cleanup_on_exit is False"
    context.db_conn.close.assert_called_once()


@patch("pipelines.training_pipeline.src.core.context.duckdb.connect")
@patch("pipelines.training_pipeline.src.core.context.S3Sync")
def test_context_initialization_duckdb_failure(
    mock_s3_sync, mock_duckdb, mock_config_parser
):
    mock_duckdb.side_effect = Exception(
        "Simulated DuckDB initialization failure"
    )

    with pytest.raises(CustomException) as exc_info:
        PipelineContext(
            run_id="test_db_fail",
            training_dataset_s3_uri_path="s3://dummy-bucket/master_panel.parquet",
            config_parser=mock_config_parser,
        )

    assert "Simulated DuckDB initialization failure" in str(exc_info.value)


@patch("pipelines.training_pipeline.src.core.context.ConfigParser")
@patch("pipelines.training_pipeline.src.core.context.duckdb.connect")
@patch("pipelines.training_pipeline.src.core.context.S3Sync")
def test_context_auto_initializes_config_parser(
    mock_s3_sync, mock_duckdb, mock_config_class, tmp_path
):
    mock_config_instance = MagicMock()
    mock_config_instance.get_global_config.return_value = {
        "local_artifact_dir": str(tmp_path)
    }
    mock_config_class.return_value = mock_config_instance

    context = PipelineContext(
        run_id="test_auto_config",
        training_dataset_s3_uri_path="s3://dummy-bucket/master_panel.parquet",
    )

    mock_config_class.assert_called_once()
    assert context.config == mock_config_instance

    context.close()


@patch("pipelines.training_pipeline.src.core.context.duckdb.connect")
@patch("pipelines.training_pipeline.src.core.context.S3Sync")
def test_close_handles_exceptions_gracefully(
    mock_s3_sync, mock_duckdb, mock_config_parser
):
    context = PipelineContext(
        run_id="test_close_fail",
        training_dataset_s3_uri_path="s3://dummy-bucket/master_panel.parquet",
        config_parser=mock_config_parser,
        cleanup_on_exit=True,
    )

    run_dir = context.run_artifact_dir

    mock_conn = MagicMock()
    mock_conn.close.side_effect = RuntimeError(
        "Simulated DuckDB close failure"
    )
    context.db_conn = mock_conn

    context.close()

    mock_conn.close.assert_called_once()
    assert not os.path.exists(
        run_dir
    ), "_cleanup_workspace should still execute even if close fails"