import sys
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.runner import InferencePipeline


@pytest.fixture
def mock_pipeline_components():
    """
    Mocks all downstream components and the execution context to isolate the
    runner's orchestration logic. Ensures no actual database connections, S3
    calls, or heavy computations occur during orchestration tests.
    """
    with patch(
        "pipelines.inference_pipeline.src.runner.InferencePipelineContext"
    ) as mock_ctx, patch(
        "pipelines.inference_pipeline.src.runner.ModelLoader"
    ) as mock_ml, patch(
        "pipelines.inference_pipeline.src.runner.FeatureMatrixBuilder"
    ) as mock_fmb, patch(
        "pipelines.inference_pipeline.src.runner.InferenceValidator"
    ) as mock_iv, patch(
        "pipelines.inference_pipeline.src.runner.ReportGenerator"
    ) as mock_rg, patch(
        "pipelines.inference_pipeline.src.runner.ReportPublisher"
    ) as mock_rp:

        # Mock Context Manager behavior
        mock_ctx_instance = MagicMock()
        mock_ctx.return_value.__enter__.return_value = mock_ctx_instance

        # Mock Component Artifacts
        mock_ml_art = MagicMock()
        mock_ml.return_value.run.return_value = mock_ml_art

        mock_fmb_art = MagicMock()
        mock_fmb.return_value.run.return_value = mock_fmb_art

        # By default, validator passes
        mock_iv_art = MagicMock(is_valid=True)
        mock_iv.return_value.run.return_value = mock_iv_art

        mock_rg_art = MagicMock()
        mock_rg.return_value.run.return_value = mock_rg_art

        mock_rp_art = MagicMock()
        mock_rp.return_value.run.return_value = mock_rp_art

        yield {
            "context": mock_ctx,
            "context_instance": mock_ctx_instance,
            "model_loader": mock_ml,
            "model_loader_artifact": mock_ml_art,
            "feature_matrix_builder": mock_fmb,
            "feature_matrix_artifact": mock_fmb_art,
            "inference_validator": mock_iv,
            "validator_artifact": mock_iv_art,
            "report_generator": mock_rg,
            "report_generator_artifact": mock_rg_art,
            "report_publisher": mock_rp,
            "publisher_artifact": mock_rp_art
        }


def test_inference_pipeline_run_success(mock_pipeline_components):
    """
    Tests the complete happy path of the runner orchestration.
    Validates correct sequential invocation and deterministic artifact propagation.
    """
    artifact = InferencePipeline.run(run_id="test_run_123")

    components = mock_pipeline_components

    # 0. Context Initialization
    components["context"].assert_called_once_with(run_id="test_run_123")

    # 1. Model Loader
    components["model_loader"].assert_called_once_with(
        components["context_instance"]
    )
    components["model_loader"].return_value.run.assert_called_once()

    # 2. Feature Matrix Builder
    components["feature_matrix_builder"].assert_called_once_with(
        components["context_instance"]
    )
    components["feature_matrix_builder"].return_value.run.assert_called_once()

    # 3. Inference Validator
    components["inference_validator"].assert_called_once_with(
        components["context_instance"]
    )
    components["inference_validator"].return_value.run.assert_called_once_with(
        model_loader_artifact=components["model_loader_artifact"],
        feature_matrix_artifact=components["feature_matrix_artifact"]
    )

    # 4. Report Generator
    components["report_generator"].assert_called_once_with(
        components["context_instance"]
    )
    components["report_generator"].return_value.run.assert_called_once_with(
        model_loader_artifact=components["model_loader_artifact"],
        feature_matrix_artifact=components["feature_matrix_artifact"]
    )

    # 5. Report Publisher
    components["report_publisher"].assert_called_once_with(
        components["context_instance"]
    )
    components["report_publisher"].return_value.run.assert_called_once_with(
        model_loader_artifact=components["model_loader_artifact"],
        feature_matrix_artifact=components["feature_matrix_artifact"],
        validator_artifact=components["validator_artifact"],
        report_generator_artifact=components["report_generator_artifact"]
    )

    # Output Validation
    assert artifact is components["publisher_artifact"]


def test_inference_pipeline_run_auto_generates_run_id(mock_pipeline_components):
    """
    Tests that the runner automatically generates a timestamp-based run_id
    if one is not explicitly provided.
    """
    InferencePipeline.run(run_id=None)

    components = mock_pipeline_components
    call_kwargs = components["context"].call_args.kwargs

    assert "run_id" in call_kwargs
    assert call_kwargs["run_id"].startswith("run_")


def test_inference_pipeline_validation_failure_halts_execution(
    mock_pipeline_components
):
    """
    Tests the fail-fast mechanism. If the data contract validation fails, the
    pipeline must raise a CustomException and halt before batch scoring occurs.
    """
    components = mock_pipeline_components

    # Simulate a structural validation breach
    components["validator_artifact"].is_valid = False
    components["validator_artifact"].report_file_path = "/path/to/report.json"

    with pytest.raises(CustomException) as exc_info:
        InferencePipeline.run(run_id="test_fail_123")

    assert "Structural Data Contract Validation failed" in str(exc_info.value)

    # Verify strict halting: downstream components MUST NOT be called
    components["report_generator"].return_value.run.assert_not_called()
    components["report_publisher"].return_value.run.assert_not_called()


def test_inference_pipeline_component_exception_propagation(
    mock_pipeline_components
):
    """
    Tests that an unhandled exception inside any component is caught, logged,
    and securely wrapped in a CustomException before halting the pipeline.
    """
    components = mock_pipeline_components

    # Inject a failure into the Feature Matrix Builder
    components[
        "feature_matrix_builder"
    ].return_value.run.side_effect = Exception("DuckDB OOM Error")

    with pytest.raises(CustomException) as exc_info:
        InferencePipeline.run(run_id="test_exception")

    assert "DuckDB OOM Error" in str(exc_info.value)

    # Downstream components should not be executed
    components["inference_validator"].return_value.run.assert_not_called()


def test_cli_help_command():
    """
    Validates that the argparse CLI is correctly configured and responds to --help.
    This executes the module via subprocess to mirror actual Docker ENTRYPOINT behavior.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.inference_pipeline.src.runner",
            "--help"
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "Batch Inference Pipeline" in result.stdout
    assert "--run-id" in result.stdout


def test_cli_unrecognized_arguments():
    """
    Validates that argparse enforces argument validation and exits cleanly
    with code 2 when provided with unknown flags.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.inference_pipeline.src.runner",
            "--invalid-flag"
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --invalid-flag" in result.stderr