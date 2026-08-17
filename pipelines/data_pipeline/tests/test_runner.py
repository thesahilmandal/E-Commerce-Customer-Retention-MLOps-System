import sys
import runpy
from unittest.mock import MagicMock, patch

import pytest

from pipelines.data_pipeline.src.runner import DataPipeline, DEFAULT_CONFIG_PATH
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def mock_dependencies():
    with patch("pipelines.data_pipeline.src.runner.PipelineConfigParser") as parser, \
        patch("pipelines.data_pipeline.src.runner.S3Sync") as s3, \
        patch("pipelines.data_pipeline.src.runner.PipelineContext") as ctx, \
        patch("pipelines.data_pipeline.src.runner.DataDiscovery") as disc, \
        patch("pipelines.data_pipeline.src.runner.DataValidation") as val, \
        patch("pipelines.data_pipeline.src.runner.FeatureMaterializer") as mat, \
        patch("pipelines.data_pipeline.src.runner.MetadataRegistry") as reg:

        ctx.return_value.__enter__.return_value = MagicMock()
        yield {
            "parser": parser, "s3": s3, "ctx": ctx,
            "disc": disc, "val": val, "mat": mat, "reg": reg
        }


@pytest.fixture
def mock_origins():
    with patch("pipelines.data_pipeline.src.core.config_parser.PipelineConfigParser") as parser, \
        patch("shared_core.cloud.s3_operations.S3Sync") as s3, \
        patch("pipelines.data_pipeline.src.core.context.PipelineContext") as ctx, \
        patch("pipelines.data_pipeline.src.components.data_discovery.DataDiscovery") as disc, \
        patch("pipelines.data_pipeline.src.components.data_validation.DataValidation") as val, \
        patch("pipelines.data_pipeline.src.components.feature_materializer.FeatureMaterializer") as mat, \
        patch("pipelines.data_pipeline.src.components.metadata_registry.MetadataRegistry") as reg:

        ctx.return_value.__enter__.return_value = MagicMock()
        yield {
            "parser": parser, "s3": s3, "ctx": ctx,
            "disc": disc, "val": val, "mat": mat, "reg": reg
        }


def test_datapipeline_run_success(mock_dependencies):
    DataPipeline.run(
        run_id="test_run_1",
        start_date="2016-09-01",
        end_date="2018-03-01",
        config_path="custom/path/config.yaml"
    )

    mock_dependencies["parser"].assert_called_once_with(config_file_path="custom/path/config.yaml")
    mock_dependencies["s3"].assert_called_once()
    mock_dependencies["ctx"].assert_called_once()

    ctx_args = mock_dependencies["ctx"].call_args.kwargs
    assert ctx_args["run_id"] == "test_run_1"
    assert ctx_args["start_date"] == "2016-09-01"
    assert ctx_args["end_date"] == "2018-03-01"

    mock_dependencies["disc"].return_value.run.assert_called_once()
    mock_dependencies["val"].return_value.run.assert_called_once()
    mock_dependencies["mat"].return_value.run.assert_called_once()
    mock_dependencies["reg"].return_value.run.assert_called_once()


def test_datapipeline_run_generates_run_id_if_none(mock_dependencies):
    DataPipeline.run(
        start_date="2016-09-01",
        end_date="2018-03-01"
    )

    ctx_args = mock_dependencies["ctx"].call_args.kwargs
    assert ctx_args["run_id"].startswith("run_")
    assert len(ctx_args["run_id"]) > 4


def test_datapipeline_run_missing_dates():
    with pytest.raises(ValueError, match="Both 'start_date' and 'end_date' must be provided"):
        DataPipeline.run(start_date=None, end_date="2018-03-01")

    with pytest.raises(ValueError, match="Both 'start_date' and 'end_date' must be provided"):
        DataPipeline.run(start_date="2016-09-01", end_date="")


def test_datapipeline_run_component_failure(mock_dependencies):
    mock_dependencies["val"].return_value.run.side_effect = Exception("Simulated Validation Error")

    with pytest.raises(CustomException) as exc_info:
        DataPipeline.run(
            run_id="test_run_fail",
            start_date="2016-09-01",
            end_date="2018-03-01"
        )

    assert "Simulated Validation Error" in str(exc_info.value)


def test_cli_execution_success(monkeypatch, mock_origins):
    test_args = [
        "runner.py",
        "--run-id", "cli_run_01",
        "--start-date", "2016-09-01",
        "--end-date", "2018-03-01",
        "--config-path", "test/config.yaml"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    runpy.run_module("pipelines.data_pipeline.src.runner", run_name="__main__")

    mock_origins["ctx"].assert_called_once()
    ctx_args = mock_origins["ctx"].call_args.kwargs
    assert ctx_args["run_id"] == "cli_run_01"
    assert ctx_args["start_date"] == "2016-09-01"
    assert ctx_args["end_date"] == "2018-03-01"

    mock_origins["parser"].assert_called_once_with(config_file_path="test/config.yaml")


def test_cli_execution_default_config_path(monkeypatch, mock_origins):
    test_args = [
        "runner.py",
        "--start-date", "2016-09-01",
        "--end-date", "2018-03-01"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    runpy.run_module("pipelines.data_pipeline.src.runner", run_name="__main__")

    mock_origins["parser"].assert_called_once_with(config_file_path=DEFAULT_CONFIG_PATH)


def test_cli_execution_missing_required_args(monkeypatch):
    test_args = ["runner.py", "--run-id", "cli_run_01"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("pipelines.data_pipeline.src.runner", run_name="__main__")

    assert exc_info.value.code == 2


def test_cli_execution_handles_unrecoverable_failure(monkeypatch, mock_origins):
    test_args = [
        "runner.py",
        "--start-date", "2016-09-01",
        "--end-date", "2018-03-01"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    mock_origins["disc"].return_value.run.side_effect = Exception(
        "Simulated unrecoverable pipeline crash"
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("pipelines.data_pipeline.src.runner", run_name="__main__")

    assert exc_info.value.code == 1