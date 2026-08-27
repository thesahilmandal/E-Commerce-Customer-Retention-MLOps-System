import pytest

from typing import Dict, Generator
from unittest.mock import MagicMock, patch

from pipelines.data_pipeline.src.runner import DataPipeline
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def mock_runner_deps() -> Generator[Dict[str, MagicMock], None, None]:
    with (
        patch("pipelines.data_pipeline.src.runner.load_dotenv") as mock_dotenv,
        patch(
            "pipelines.data_pipeline.src.runner.PipelineConfigParser"
        ) as mock_parser,
        patch("pipelines.data_pipeline.src.runner.S3Sync") as mock_s3_sync,
        patch(
            "pipelines.data_pipeline.src.runner.PipelineContext"
        ) as mock_context,
        patch("pipelines.data_pipeline.src.runner.DataDiscovery") as mock_discovery,
        patch(
            "pipelines.data_pipeline.src.runner.DataValidation"
        ) as mock_validation,
        patch(
            "pipelines.data_pipeline.src.runner.FeatureMaterializer"
        ) as mock_materializer,
        patch(
            "pipelines.data_pipeline.src.runner.MetadataRegistry"
        ) as mock_registry,
    ):
        # Setup the context manager mock.
        mock_context_instance = MagicMock()
        mock_context.return_value.__enter__.return_value = mock_context_instance

        yield {
            "dotenv": mock_dotenv,
            "parser": mock_parser,
            "s3_sync": mock_s3_sync,
            "context": mock_context,
            "context_instance": mock_context_instance,
            "discovery": mock_discovery,
            "validation": mock_validation,
            "materializer": mock_materializer,
            "registry": mock_registry,
        }


def test_run_success(
    mock_runner_deps: Dict[str, MagicMock],
) -> None:
    DataPipeline.run(
        run_id="test_runner_id",
        start_date="2023-01-01",
        end_date="2023-06-01",
        config_path="custom_config.yaml",
    )

    mock_runner_deps["dotenv"].assert_called_once()

    mock_runner_deps["parser"].assert_called_once_with(
        config_file_path="custom_config.yaml"
    )
    mock_runner_deps["parser"].return_value.parse.assert_called_once()

    mock_runner_deps["s3_sync"].assert_called_once()

    mock_runner_deps["context"].assert_called_once_with(
        run_id="test_runner_id",
        start_date="2023-01-01",
        end_date="2023-06-01",
        config=mock_runner_deps["parser"].return_value.parse.return_value,
        s3_sync=mock_runner_deps["s3_sync"].return_value,
    )

    mock_runner_deps["discovery"].assert_called_once_with(
        context=mock_runner_deps["context_instance"]
    )
    mock_runner_deps["discovery"].return_value.run.assert_called_once()

    mock_runner_deps["validation"].assert_called_once_with(
        context=mock_runner_deps["context_instance"]
    )
    mock_runner_deps["validation"].return_value.run.assert_called_once()

    mock_runner_deps["materializer"].assert_called_once_with(
        context=mock_runner_deps["context_instance"]
    )
    mock_runner_deps["materializer"].return_value.run.assert_called_once()

    mock_runner_deps["registry"].assert_called_once_with(
        context=mock_runner_deps["context_instance"]
    )
    mock_runner_deps["registry"].return_value.run.assert_called_once()


def test_run_auto_generates_run_id(
    mock_runner_deps: Dict[str, MagicMock],
) -> None:
    DataPipeline.run(
        start_date="2023-01-01",
        end_date="2023-06-01",
    )

    context_call_kwargs = mock_runner_deps["context"].call_args.kwargs

    assert "run_id" in context_call_kwargs
    assert isinstance(context_call_kwargs["run_id"], str)
    assert context_call_kwargs["run_id"].startswith("run_")


@pytest.mark.parametrize(
    "start_date, end_date",
    [
        (None, "2023-06-01"),
        ("2023-01-01", None),
        (None, None),
        ("", "2023-06-01"),
        ("2023-01-01", ""),
    ],
)
def test_run_missing_dates_raises_value_error(
    start_date: str,
    end_date: str,
    mock_runner_deps: Dict[str, MagicMock],
) -> None:
    with pytest.raises(ValueError) as exc_info:
        DataPipeline.run(
            start_date=start_date,
            end_date=end_date,
        )

    assert "must be provided" in str(exc_info.value)
    mock_runner_deps["parser"].assert_not_called()


def test_run_component_failure_halts_execution(
    mock_runner_deps: Dict[str, MagicMock],
) -> None:
    mock_runner_deps["validation"].return_value.run.side_effect = Exception(
        "Validation logic failed"
    )

    with pytest.raises(CustomException) as exc_info:
        DataPipeline.run(
            start_date="2023-01-01",
            end_date="2023-06-01",
        )

    assert "Validation logic failed" in str(exc_info.value)

    mock_runner_deps["discovery"].return_value.run.assert_called_once()
    mock_runner_deps["validation"].return_value.run.assert_called_once()
    mock_runner_deps["materializer"].return_value.run.assert_not_called()
    mock_runner_deps["registry"].return_value.run.assert_not_called()


def test_run_config_parser_failure(
    mock_runner_deps: Dict[str, MagicMock],
) -> None:
    mock_runner_deps["parser"].side_effect = Exception(
        "YAML formatting issue"
    )

    with pytest.raises(CustomException) as exc_info:
        DataPipeline.run(
            start_date="2023-01-01",
            end_date="2023-06-01",
        )

    assert "YAML formatting issue" in str(exc_info.value)
    mock_runner_deps["context"].assert_not_called()