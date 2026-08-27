
import pytest
from typing import Any, Dict
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

from pipelines.inference_pipeline.src.entity.artifact_entity import (
    ModelLoaderArtifact,
    FeatureMatrixBuilderArtifact,
    InferenceValidatorArtifact,
    ReportGeneratorArtifact,
    ReportPublisherArtifact,
)


@pytest.fixture
def mock_global_config_dict() -> Dict[str, Any]:
    return {
        "system": {
            "artifact_dir": "test_inference_artifacts",
            "pipeline_name": "inference_pipeline",
        },
        "business_logic": {
            "target_column": "target_is_churn",
            "churn_probability_threshold": 0.5,
            "system_columns_to_drop": [
                "customer_unique_id",
                "snapshot_date",
            ],
        },
        "cloud_storage": {
            "data_lake": {
                "database_name": "test-data-lake",
                "bronze_dir": "bronze",
            },
            "model_registry": {
                "bucket_name": "test-pipeline-run-artifacts",
                "registry_dir": "model_registry",
                "state_dir": "state",
                "pointer_file": "model_state.json",
            },
            "inference_outputs": {
                "base_artifact_dir": "inference_pipeline_artifacts",
                "business_reports_dir": "business_reports/customer_churn",
                "mlops_telemetry_dir": "mlops_telemetry/inference_logs",
                "metadata_dir": "inference_metadata",
            },
        },
        "components": {
            "model_loader": {
                "dir_name": "01_model_loader",
                "model_file": "model.pkl",
                "schema_file": "schema.json",
                "baseline_metrics_file": "baseline_performance_metrics.json",
                "reference_distributions_file": "reference_feature_distributions.json",
                "metadata_file": "metadata.json",
            },
            "feature_matrix_builder": {
                "dir_name": "02_input_feature_matrix_builder",
                "feature_matrix_file": "input_feature_matrix.parquet",
                "schema_file": "schema.json",
                "metadata_file": "metadata.json",
            },
            "inference_validator": {
                "dir_name": "03_validator",
                "report_file": "report.json",
                "metadata_file": "metadata.json",
            },
            "report_generator": {
                "dir_name": "04_report_generator",
                "csv_report_file": "churn_predictions.csv",
                "telemetry_file": "telemetry.parquet",
                "metadata_file": "metadata.json",
            },
            "report_publisher": {
                "dir_name": "05_report_publisher",
                "metadata_file": "metadata.json",
            },
        },
    }


@pytest.fixture
def mock_pipeline_context(
    mock_global_config_dict: Dict[str, Any],
) -> MagicMock:
    context = MagicMock()
    context.run_id = "test_run_123"

    config_parser = MagicMock()
    config_parser.get_system_config.return_value = (
        mock_global_config_dict["system"]
    )
    config_parser.get_business_logic_config.return_value = (
        mock_global_config_dict["business_logic"]
    )
    config_parser.get_cloud_storage_config.return_value = (
        mock_global_config_dict["cloud_storage"]
    )
    config_parser.get_components_config.return_value = (
        mock_global_config_dict["components"]
    )

    context.config = config_parser
    context.s3_sync = MagicMock()
    context.duckdb_con = MagicMock()

    context.__enter__.return_value = context
    context.__exit__.return_value = None

    return context


@pytest.fixture
def dummy_model_loader_artifact() -> ModelLoaderArtifact:
    return ModelLoaderArtifact(
        model_file_path=(
            "/tmp/inference/test_run_123/"
            "01_model_loader/model.pkl"
        ),
        schema_file_path=(
            "/tmp/inference/test_run_123/"
            "01_model_loader/schema.json"
        ),
        champion_run_id="train_run_001",
        metadata_file_path=(
            "/tmp/inference/test_run_123/"
            "01_model_loader/metadata.json"
        ),
    )


@pytest.fixture
def dummy_feature_matrix_artifact() -> FeatureMatrixBuilderArtifact:
    return FeatureMatrixBuilderArtifact(
        feature_matrix_file_path=(
            "/tmp/inference/test_run_123/"
            "02_input_feature_matrix_builder/"
            "input_feature_matrix.parquet"
        ),
        schema_file_path=(
            "/tmp/inference/test_run_123/"
            "02_input_feature_matrix_builder/schema.json"
        ),
        metadata_file_path=(
            "/tmp/inference/test_run_123/"
            "02_input_feature_matrix_builder/metadata.json"
        ),
        snapshot_date="2026-08-26",
    )


@pytest.fixture
def dummy_inference_validator_artifact() -> InferenceValidatorArtifact:
    return InferenceValidatorArtifact(
        is_valid=True,
        report_file_path=(
            "/tmp/inference/test_run_123/"
            "03_validator/report.json"
        ),
    )


@pytest.fixture
def dummy_report_generator_artifact() -> ReportGeneratorArtifact:
    return ReportGeneratorArtifact(
        csv_report_path=(
            "/tmp/inference/test_run_123/"
            "04_report_generator/churn_predictions.csv"
        ),
        telemetry_log_path=(
            "/tmp/inference/test_run_123/"
            "04_report_generator/telemetry.parquet"
        ),
        metadata_file_path=(
            "/tmp/inference/test_run_123/"
            "04_report_generator/metadata.json"
        ),
    )


@pytest.fixture
def dummy_report_publisher_artifact() -> ReportPublisherArtifact:
    return ReportPublisherArtifact(
        published_business_report_uri=(
            "s3://test-bucket/"
            "business_reports/churn_predictions.csv"
        ),
        published_telemetry_log_uri=(
            "s3://test-bucket/"
            "mlops_telemetry/telemetry.parquet"
        ),
        metadata_file_path=(
            "/tmp/inference/test_run_123/"
            "05_report_publisher/metadata.json"
        ),
    )


@pytest.fixture
def sample_feature_matrix_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_unique_id": [
                "CUST-001",
                "CUST-002",
                "CUST-003",
            ],
            "snapshot_date": [
                "2026-08-26",
                "2026-08-26",
                "2026-08-26",
            ],
            "recency_days": [10, 45, 2],
            "frequency": [5, 1, 12],
            "monetary_total": [150.0, 20.0, 1200.0],
            "max_delivery_delay_days": [0, 5, 0],
            "has_undelivered_order": [0, 1, 0],
            "total_canceled_orders": [0, 0, 1],
        }
    )


@pytest.fixture
def mock_predictive_model() -> MagicMock:
    model = MagicMock()

    model.predict_proba.return_value = np.array(
        [
            [0.85, 0.15],
            [0.20, 0.80],
            [0.95, 0.05],
        ]
    )

    return model