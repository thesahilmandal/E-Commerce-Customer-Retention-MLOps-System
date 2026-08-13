import sys
import pytest
from unittest.mock import MagicMock, patch

from pipelines.training_pipeline.src.runner import TrainingPipeline, main
from shared_core.exceptions.custom_exception import CustomException


@patch("pipelines.training_pipeline.src.runner.ModelRegistry")
@patch("pipelines.training_pipeline.src.runner.ModelEvaluator")
@patch("pipelines.training_pipeline.src.runner.ModelTrainer")
@patch("pipelines.training_pipeline.src.runner.DataProcessor")
@patch("pipelines.training_pipeline.src.runner.PipelineContext")
@patch("pipelines.training_pipeline.src.runner.ConfigParser")
def test_training_pipeline_run_success(
    mock_config_parser,
    mock_pipeline_context,
    mock_data_processor,
    mock_model_trainer,
    mock_model_evaluator,
    mock_model_registry,
):
    mock_context_instance = MagicMock()
    mock_pipeline_context.return_value.__enter__.return_value = mock_context_instance

    mock_registry_artifact = MagicMock()
    mock_registry_artifact.deployment_status = True
    mock_registry_artifact.s3_model_uri = "s3://test-bucket/models/run123"
    mock_model_registry.return_value.run.return_value = mock_registry_artifact

    TrainingPipeline.run(
        run_id="run123",
        training_dataset_s3_uri_path="s3://test-bucket/data.parquet",
    )

    mock_config_parser.assert_called_once()
    mock_pipeline_context.assert_called_once()

    mock_data_processor.return_value.run.assert_called_once()
    mock_model_trainer.return_value.run.assert_called_once()
    mock_model_evaluator.return_value.run.assert_called_once()
    mock_model_registry.return_value.run.assert_called_once()


@patch("pipelines.training_pipeline.src.runner.ModelRegistry")
@patch("pipelines.training_pipeline.src.runner.ModelEvaluator")
@patch("pipelines.training_pipeline.src.runner.ModelTrainer")
@patch("pipelines.training_pipeline.src.runner.DataProcessor")
@patch("pipelines.training_pipeline.src.runner.PipelineContext")
@patch("pipelines.training_pipeline.src.runner.ConfigParser")
def test_training_pipeline_run_challenger_rejected(
    mock_config_parser,
    mock_pipeline_context,
    mock_data_processor,
    mock_model_trainer,
    mock_model_evaluator,
    mock_model_registry,
):
    mock_context_instance = MagicMock()
    mock_pipeline_context.return_value.__enter__.return_value = mock_context_instance

    mock_registry_artifact = MagicMock()
    mock_registry_artifact.deployment_status = False
    mock_model_registry.return_value.run.return_value = mock_registry_artifact

    TrainingPipeline.run(
        run_id="run123",
        training_dataset_s3_uri_path="s3://test-bucket/data.parquet",
    )

    mock_model_registry.return_value.run.assert_called_once()


@patch("pipelines.training_pipeline.src.runner.DataProcessor")
@patch("pipelines.training_pipeline.src.runner.PipelineContext")
@patch("pipelines.training_pipeline.src.runner.ConfigParser")
def test_training_pipeline_run_exception_propagation(
    mock_config_parser,
    mock_pipeline_context,
    mock_data_processor,
):
    mock_context_instance = MagicMock()
    mock_pipeline_context.return_value.__enter__.return_value = mock_context_instance

    mock_data_processor.return_value.run.side_effect = Exception(
        "Simulated component failure"
    )

    with pytest.raises(CustomException) as exc_info:
        TrainingPipeline.run(
            run_id="run123",
            training_dataset_s3_uri_path="s3://test-bucket/data.parquet",
        )

    assert "Simulated component failure" in str(exc_info.value)


@patch(
    "sys.argv",
    [
        "runner.py",
        "--run-id",
        "test_run_001",
        "--dataset-uri",
        "s3://bucket/test.parquet",
    ],
)
@patch("pipelines.training_pipeline.src.runner.TrainingPipeline.run")
def test_main_cli_success(mock_pipeline_run):
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    mock_pipeline_run.assert_called_once_with(
        run_id="test_run_001",
        training_dataset_s3_uri_path="s3://bucket/test.parquet",
    )


@patch(
    "sys.argv",
    ["runner.py", "--run-id", "test_run_001"],
)
def test_main_cli_missing_dataset_uri():
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


@patch(
    "sys.argv",
    ["runner.py", "--dataset-uri", "s3://bucket/test.parquet"],
)
def test_main_cli_missing_run_id():
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


@patch(
    "sys.argv",
    [
        "runner.py",
        "--run-id",
        "   ",
        "--dataset-uri",
        "s3://bucket/test.parquet",
    ],
)
def test_main_cli_empty_run_id():
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


@patch(
    "sys.argv",
    [
        "runner.py",
        "--run-id",
        "test_run_001",
        "--dataset-uri",
        "local_file.parquet",
    ],
)
def test_main_cli_invalid_dataset_uri_protocol():
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


@patch(
    "sys.argv",
    [
        "runner.py",
        "--run-id",
        "test_run_001",
        "--dataset-uri",
        "s3://bucket/test.csv",
    ],
)
def test_main_cli_invalid_dataset_uri_extension():
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


@patch(
    "sys.argv",
    [
        "runner.py",
        "--run-id",
        "test_run_001",
        "--dataset-uri",
        "s3://bucket/test.parquet",
    ],
)
@patch("pipelines.training_pipeline.src.runner.TrainingPipeline.run")
def test_main_cli_pipeline_failure(mock_pipeline_run):
    mock_pipeline_run.side_effect = CustomException(
        Exception("Fatal error"), sys
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1