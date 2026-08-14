import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import pytest

from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.entity.config_entity import (
    ModelLoaderConfig,
    FeatureMatrixBuilderConfig,
    InferenceValidatorConfig,
    ReportGeneratorConfig,
    ReportPublisherConfig
)


def test_model_loader_config_from_context_success(mock_pipeline_context):
    """
    Tests successful initialization of ModelLoaderConfig from the pipeline context.
    Validates path construction, directory creation side effects, and S3 URI formatting.
    """
    config = ModelLoaderConfig.from_context(mock_pipeline_context)

    expected_root = os.path.join(
        mock_pipeline_context.config.get_system_config()["artifact_dir"],
        "inference_pipeline",
        "test_run_123",
        "01_model_loader"
    )

    assert config.model_loader_root_dir == expected_root
    assert os.path.exists(config.model_loader_root_dir)
    assert config.model_file_path.endswith("model.pkl")
    assert config.s3_pointer_uri == "s3://test-model-registry/model_registry/state/model_state.json"


def test_model_loader_config_from_context_failure(mock_pipeline_context):
    """
    Tests that a malformed context (e.g., missing system configs) correctly
    raises a CustomException during ModelLoaderConfig initialization.
    """
    with patch.object(mock_pipeline_context.config, "get_system_config", return_value={}):
        with pytest.raises(CustomException):
            ModelLoaderConfig.from_context(mock_pipeline_context)


def test_feature_matrix_builder_config_from_context_success(mock_pipeline_context):
    """
    Tests successful initialization of FeatureMatrixBuilderConfig, focusing on
    S3 Data Lake URI construction and the dynamic T-1 UTC snapshot date calculation.
    """
    config = FeatureMatrixBuilderConfig.from_context(mock_pipeline_context)

    expected_root = os.path.join(
        mock_pipeline_context.config.get_system_config()["artifact_dir"],
        "inference_pipeline",
        "test_run_123",
        "02_input_feature_matrix_builder"
    )

    expected_snapshot_date = (
        datetime.now(timezone.utc) - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    assert config.builder_root_dir == expected_root
    assert os.path.exists(config.builder_root_dir)
    assert config.s3_data_lake_uri == "s3://test-data-lake/bronze"
    assert config.snapshot_date == expected_snapshot_date


def test_feature_matrix_builder_config_from_context_failure(mock_pipeline_context):
    """
    Tests CustomException handling for FeatureMatrixBuilderConfig initialization.
    """
    with patch.object(
        mock_pipeline_context.config,
        "get_cloud_storage_config",
        return_value={}
    ):
        with pytest.raises(CustomException):
            FeatureMatrixBuilderConfig.from_context(mock_pipeline_context)


def test_inference_validator_config_from_context_success(mock_pipeline_context):
    """
    Tests successful initialization of InferenceValidatorConfig.
    """
    config = InferenceValidatorConfig.from_context(mock_pipeline_context)

    expected_root = os.path.join(
        mock_pipeline_context.config.get_system_config()["artifact_dir"],
        "inference_pipeline",
        "test_run_123",
        "03_validator"
    )

    assert config.validator_root_dir == expected_root
    assert os.path.exists(config.validator_root_dir)
    assert config.report_file_path.endswith("report.json")


def test_inference_validator_config_from_context_failure(mock_pipeline_context):
    """
    Tests CustomException handling for InferenceValidatorConfig initialization.
    """
    with patch.object(
        mock_pipeline_context.config,
        "get_components_config",
        return_value={}
    ):
        with pytest.raises(CustomException):
            InferenceValidatorConfig.from_context(mock_pipeline_context)


def test_report_generator_config_from_context_success(mock_pipeline_context):
    """
    Tests successful initialization of ReportGeneratorConfig, verifying
    business logic extraction and thresholds.
    """
    config = ReportGeneratorConfig.from_context(mock_pipeline_context)

    expected_root = os.path.join(
        mock_pipeline_context.config.get_system_config()["artifact_dir"],
        "inference_pipeline",
        "test_run_123",
        "04_report_generator"
    )

    assert config.generator_root_dir == expected_root
    assert os.path.exists(config.generator_root_dir)
    assert config.probability_threshold == 0.5
    assert "customer_unique_id" in config.system_columns_to_drop
    assert config.run_id == "test_run_123"


def test_report_generator_config_from_context_failure(mock_pipeline_context):
    """
    Tests CustomException handling for ReportGeneratorConfig initialization.
    """
    with patch.object(
        mock_pipeline_context.config,
        "get_business_logic_config",
        return_value={}
    ):
        with pytest.raises(CustomException):
            ReportGeneratorConfig.from_context(mock_pipeline_context)


@pytest.mark.parametrize("invalid_threshold", [-0.1, 1.1, 5.0, -100.0])
def test_report_generator_config_invalid_thresholds(invalid_threshold):
    """
    Tests that ReportGeneratorConfig enforces constraints on probability_threshold
    through its **post_init** validation.
    """
    with pytest.raises(ValueError) as exc_info:
        ReportGeneratorConfig(
            generator_root_dir="/fake/dir",
            run_id="run123",
            probability_threshold=invalid_threshold,
            system_columns_to_drop=[],
            csv_report_path="/fake/dir/report.csv",
            telemetry_log_path="/fake/dir/log.parquet",
            metadata_file_path="/fake/dir/meta.json"
        )

    assert "probability_threshold must be between 0 and 1" in str(exc_info.value)


@pytest.mark.parametrize("valid_threshold", [0.0, 0.5, 1.0])
def test_report_generator_config_valid_thresholds(valid_threshold):
    """
    Tests that ReportGeneratorConfig accepts valid probability boundary values.
    """
    config = ReportGeneratorConfig(
        generator_root_dir="/fake/dir",
        run_id="run123",
        probability_threshold=valid_threshold,
        system_columns_to_drop=[],
        csv_report_path="/fake/dir/report.csv",
        telemetry_log_path="/fake/dir/log.parquet",
        metadata_file_path="/fake/dir/meta.json"
    )
    assert config.probability_threshold == valid_threshold


def test_report_publisher_config_from_context_success(mock_pipeline_context):
    """
    Tests successful initialization of ReportPublisherConfig, focusing on
    the construction of nested S3 output directories for reports and telemetry.
    """
    config = ReportPublisherConfig.from_context(mock_pipeline_context)

    expected_root = os.path.join(
        mock_pipeline_context.config.get_system_config()["artifact_dir"],
        "inference_pipeline",
        "test_run_123",
        "05_report_publisher"
    )

    assert config.publisher_root_dir == expected_root
    assert os.path.exists(config.publisher_root_dir)
    assert config.run_id == "test_run_123"

    expected_s3_base = "s3://test-model-registry/inference_pipeline_artifacts"
    assert config.s3_business_reports_base_uri == (
        f"{expected_s3_base}/business_reports/customer_churn"
    )
    assert config.s3_telemetry_logs_base_uri == (
        f"{expected_s3_base}/mlops_telemetry/inference_logs"
    )
    assert config.s3_metadata_base_uri == (
        f"{expected_s3_base}/inference_metadata"
    )


def test_report_publisher_config_from_context_failure(mock_pipeline_context):
    """
    Tests CustomException handling for ReportPublisherConfig initialization.
    """
    with patch.object(
        mock_pipeline_context.config,
        "get_cloud_storage_config",
        return_value={}
    ):
        with pytest.raises(CustomException):
            ReportPublisherConfig.from_context(mock_pipeline_context)