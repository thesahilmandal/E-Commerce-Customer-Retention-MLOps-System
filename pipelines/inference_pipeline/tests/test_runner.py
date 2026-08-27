import pytest

from unittest.mock import MagicMock, patch

from pipelines.inference_pipeline.src.runner import InferencePipeline
from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    ModelLoaderArtifact,
    FeatureMatrixBuilderArtifact,
    InferenceValidatorArtifact,
    ReportGeneratorArtifact,
    ReportPublisherArtifact,
)


@patch("pipelines.inference_pipeline.src.runner.ReportPublisher")
@patch("pipelines.inference_pipeline.src.runner.ReportGenerator")
@patch("pipelines.inference_pipeline.src.runner.InferenceValidator")
@patch("pipelines.inference_pipeline.src.runner.FeatureMatrixBuilder")
@patch("pipelines.inference_pipeline.src.runner.ModelLoader")
@patch("pipelines.inference_pipeline.src.runner.InferencePipelineContext")
def test_inference_pipeline_run_success(
    mock_context: MagicMock,
    mock_model_loader: MagicMock,
    mock_feature_builder: MagicMock,
    mock_validator: MagicMock,
    mock_generator: MagicMock,
    mock_publisher: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
    dummy_inference_validator_artifact: InferenceValidatorArtifact,
    dummy_report_generator_artifact: ReportGeneratorArtifact,
    dummy_report_publisher_artifact: ReportPublisherArtifact,
) -> None:
    # Setup Context Manager mock.
    mock_context_instance = MagicMock()
    mock_context.return_value.__enter__.return_value = (
        mock_context_instance
    )

    # Setup Component Returns.
    mock_model_loader.return_value.run.return_value = (
        dummy_model_loader_artifact
    )
    mock_feature_builder.return_value.run.return_value = (
        dummy_feature_matrix_artifact
    )
    mock_validator.return_value.run.return_value = (
        dummy_inference_validator_artifact
    )
    mock_generator.return_value.run.return_value = (
        dummy_report_generator_artifact
    )
    mock_publisher.return_value.run.return_value = (
        dummy_report_publisher_artifact
    )

    # Execute.
    run_id = "test_orchestrator_123"
    artifact = InferencePipeline.run(run_id=run_id)

    # Assert Context Initialization.
    mock_context.assert_called_once_with(run_id=run_id)

    # Assert Component Initializations.
    mock_model_loader.assert_called_once_with(
        mock_context_instance
    )
    mock_feature_builder.assert_called_once_with(
        mock_context_instance
    )
    mock_validator.assert_called_once_with(
        mock_context_instance
    )
    mock_generator.assert_called_once_with(
        mock_context_instance
    )
    mock_publisher.assert_called_once_with(
        mock_context_instance
    )

    # Assert Execution Order & Arguments.
    mock_model_loader.return_value.run.assert_called_once()
    mock_feature_builder.return_value.run.assert_called_once()

    mock_validator.return_value.run.assert_called_once_with(
        model_loader_artifact=dummy_model_loader_artifact,
        feature_matrix_artifact=dummy_feature_matrix_artifact,
    )

    mock_generator.return_value.run.assert_called_once_with(
        model_loader_artifact=dummy_model_loader_artifact,
        feature_matrix_artifact=dummy_feature_matrix_artifact,
    )

    mock_publisher.return_value.run.assert_called_once_with(
        model_loader_artifact=dummy_model_loader_artifact,
        feature_matrix_artifact=dummy_feature_matrix_artifact,
        validator_artifact=dummy_inference_validator_artifact,
        report_generator_artifact=dummy_report_generator_artifact,
    )

    assert artifact == dummy_report_publisher_artifact


@patch("pipelines.inference_pipeline.src.runner.ReportPublisher")
@patch("pipelines.inference_pipeline.src.runner.ReportGenerator")
@patch("pipelines.inference_pipeline.src.runner.InferenceValidator")
@patch("pipelines.inference_pipeline.src.runner.FeatureMatrixBuilder")
@patch("pipelines.inference_pipeline.src.runner.ModelLoader")
@patch("pipelines.inference_pipeline.src.runner.InferencePipelineContext")
def test_inference_pipeline_run_auto_run_id(
    mock_context: MagicMock,
    mock_model_loader: MagicMock,
    mock_feature_builder: MagicMock,
    mock_validator: MagicMock,
    mock_generator: MagicMock,
    mock_publisher: MagicMock,
) -> None:
    mock_context.return_value.__enter__.return_value = (
        MagicMock()
    )

    # Ensure validation passes.
    validator_artifact = MagicMock(is_valid=True)
    mock_validator.return_value.run.return_value = (
        validator_artifact
    )

    InferencePipeline.run(run_id=None)

    mock_context.assert_called_once()

    _, kwargs = mock_context.call_args

    assert "run_id" in kwargs
    assert kwargs["run_id"].startswith("run_")


@patch("pipelines.inference_pipeline.src.runner.ReportPublisher")
@patch("pipelines.inference_pipeline.src.runner.ReportGenerator")
@patch("pipelines.inference_pipeline.src.runner.InferenceValidator")
@patch("pipelines.inference_pipeline.src.runner.FeatureMatrixBuilder")
@patch("pipelines.inference_pipeline.src.runner.ModelLoader")
@patch("pipelines.inference_pipeline.src.runner.InferencePipelineContext")
def test_inference_pipeline_run_validation_failure_halts_execution(
    mock_context: MagicMock,
    mock_model_loader: MagicMock,
    mock_feature_builder: MagicMock,
    mock_validator: MagicMock,
    mock_generator: MagicMock,
    mock_publisher: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
) -> None:
    mock_context.return_value.__enter__.return_value = (
        MagicMock()
    )

    mock_model_loader.return_value.run.return_value = (
        dummy_model_loader_artifact
    )
    mock_feature_builder.return_value.run.return_value = (
        dummy_feature_matrix_artifact
    )

    # Mock validator to fail.
    invalid_artifact = InferenceValidatorArtifact(
        is_valid=False,
        report_file_path="/tmp/validation_report.json",
    )

    mock_validator.return_value.run.return_value = (
        invalid_artifact
    )

    with pytest.raises(CustomException) as exc_info:
        InferencePipeline.run(run_id="test_fail")

    assert (
        "Structural Data Contract Validation failed"
        in str(exc_info.value)
    )

    # Generator and Publisher should NOT be executed.
    mock_generator.return_value.run.assert_not_called()
    mock_publisher.return_value.run.assert_not_called()


@patch("pipelines.inference_pipeline.src.runner.InferencePipelineContext")
@patch("pipelines.inference_pipeline.src.runner.ModelLoader")
def test_inference_pipeline_run_exception_propagation(
    mock_model_loader: MagicMock,
    mock_context: MagicMock,
) -> None:
    mock_context.return_value.__enter__.return_value = (
        MagicMock()
    )

    mock_model_loader.return_value.run.side_effect = Exception(
        "Model loading out of memory"
    )

    with pytest.raises(CustomException) as exc_info:
        InferencePipeline.run(run_id="test_exception")

    assert (
        "Model loading out of memory"
        in str(exc_info.value)
    )