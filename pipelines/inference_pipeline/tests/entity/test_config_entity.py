import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from pipelines.inference_pipeline.src.entity.config_entity import (
    ModelLoaderConfig,
    FeatureMatrixBuilderConfig,
    InferenceValidatorConfig,
    ReportGeneratorConfig,
    ReportPublisherConfig
)
from shared_core.exceptions.custom_exception import CustomException


@patch("pipelines.inference_pipeline.src.entity.config_entity.os.makedirs")
def test_model_loader_config_from_context(mock_makedirs: MagicMock, mock_pipeline_context: MagicMock) -> None:
    config = ModelLoaderConfig.from_context(mock_pipeline_context)
    
    assert "01_model_loader" in config.model_loader_root_dir
    assert config.model_file_path.endswith("model.pkl")
    assert config.schema_file_path.endswith("schema.json")
    assert config.baseline_metrics_file_path.endswith("baseline_performance_metrics.json")
    assert config.reference_distributions_file_path.endswith("reference_feature_distributions.json")
    assert config.metadata_file_path.endswith("metadata.json")
    
    expected_s3_uri = "s3://test-pipeline-run-artifacts/model_registry/state/model_state.json"
    assert config.s3_pointer_uri == expected_s3_uri
    
    mock_makedirs.assert_called_once_with(config.model_loader_root_dir, exist_ok=True)


@patch("pipelines.inference_pipeline.src.entity.config_entity.os.makedirs")
def test_feature_matrix_builder_config_from_context(mock_makedirs: MagicMock, mock_pipeline_context: MagicMock) -> None:
    config = FeatureMatrixBuilderConfig.from_context(mock_pipeline_context)
    
    assert "02_input_feature_matrix_builder" in config.builder_root_dir
    assert config.feature_matrix_file_path.endswith("input_feature_matrix.parquet")
    
    expected_s3_uri = "s3://test-data-lake/bronze"
    assert config.s3_data_lake_uri == expected_s3_uri
    
    try:
        datetime.strptime(config.snapshot_date, "%Y-%m-%d")
    except ValueError:
        pytest.fail(f"snapshot_date {config.snapshot_date} is not in YYYY-MM-DD format")
        
    mock_makedirs.assert_called_once_with(config.builder_root_dir, exist_ok=True)


@patch("pipelines.inference_pipeline.src.entity.config_entity.os.makedirs")
def test_inference_validator_config_from_context(mock_makedirs: MagicMock, mock_pipeline_context: MagicMock) -> None:
    config = InferenceValidatorConfig.from_context(mock_pipeline_context)
    
    assert "03_validator" in config.validator_root_dir
    assert config.report_file_path.endswith("report.json")
    assert config.metadata_file_path.endswith("metadata.json")
    
    mock_makedirs.assert_called_once_with(config.validator_root_dir, exist_ok=True)


@patch("pipelines.inference_pipeline.src.entity.config_entity.os.makedirs")
def test_report_generator_config_from_context(mock_makedirs: MagicMock, mock_pipeline_context: MagicMock) -> None:
    config = ReportGeneratorConfig.from_context(mock_pipeline_context)
    
    assert "04_report_generator" in config.generator_root_dir
    assert config.run_id == "test_run_123"
    assert config.probability_threshold == 0.5
    assert "customer_unique_id" in config.system_columns_to_drop
    assert config.csv_report_path.endswith("churn_predictions.csv")
    assert config.telemetry_log_path.endswith("telemetry.parquet")
    
    mock_makedirs.assert_called_once_with(config.generator_root_dir, exist_ok=True)


@pytest.mark.parametrize("invalid_threshold", [-0.1, 1.1])
def test_report_generator_config_threshold_validation(invalid_threshold: float) -> None:
    with pytest.raises(ValueError) as exc_info:
        ReportGeneratorConfig(
            generator_root_dir="/tmp",
            run_id="test_123",
            probability_threshold=invalid_threshold,
            system_columns_to_drop=[],
            csv_report_path="/tmp/report.csv",
            telemetry_log_path="/tmp/telemetry.parquet",
            metadata_file_path="/tmp/metadata.json"
        )
    assert "probability_threshold must be between 0 and 1" in str(exc_info.value)


@patch("pipelines.inference_pipeline.src.entity.config_entity.os.makedirs")
def test_report_publisher_config_from_context(mock_makedirs: MagicMock, mock_pipeline_context: MagicMock) -> None:
    config = ReportPublisherConfig.from_context(mock_pipeline_context)
    
    assert "05_report_publisher" in config.publisher_root_dir
    assert config.run_id == "test_run_123"
    
    assert config.s3_business_reports_base_uri == "s3://test-pipeline-run-artifacts/inference_pipeline_artifacts/business_reports/customer_churn"
    assert config.s3_telemetry_logs_base_uri == "s3://test-pipeline-run-artifacts/inference_pipeline_artifacts/mlops_telemetry/inference_logs"
    assert config.s3_metadata_base_uri == "s3://test-pipeline-run-artifacts/inference_pipeline_artifacts/inference_metadata"
    
    mock_makedirs.assert_called_once_with(config.publisher_root_dir, exist_ok=True)


@pytest.mark.parametrize("config_class", [
    ModelLoaderConfig,
    FeatureMatrixBuilderConfig,
    InferenceValidatorConfig,
    ReportGeneratorConfig,
    ReportPublisherConfig
])
def test_config_initialization_exceptions(config_class: type, mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.config.get_system_config.side_effect = KeyError("artifact_dir")
    
    with pytest.raises(CustomException) as exc_info:
        config_class.from_context(mock_pipeline_context)
        
    assert "artifact_dir" in str(exc_info.value)