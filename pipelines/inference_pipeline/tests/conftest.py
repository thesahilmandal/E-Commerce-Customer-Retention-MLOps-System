import json
import pytest
import yaml
import joblib
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.dummy import DummyClassifier

from pipelines.inference_pipeline.src.core.config_parser import ConfigParser
from pipelines.inference_pipeline.src.core.context import InferencePipelineContext


@pytest.fixture(autouse=True)
def mock_aws_env_vars(monkeypatch):
    """
    Globally mocks AWS environment variables to ensure no test accidentally
    attempts to authenticate with real AWS credentials or reaches out to real services.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing_mock_key_id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing_mock_secret_key")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing_mock_session_token")


@pytest.fixture
def temp_workspace(tmp_path):
    """
    Provides a clean temporary directory for the execution of isolated tests,
    ensuring artifacts don't persist across runs or pollute the local file system.
    """
    workspace = tmp_path / "inference_pipeline_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def mock_global_config_path(temp_workspace):
    """
    Generates a temporary configuration YAML file mirroring the structure of the
    production global_config.yaml, but with paths pointed to the isolated temp_workspace.
    """
    config_data = {
        "system": {
            "artifact_dir": str(temp_workspace / "artifacts"),
            "pipeline_name": "inference_pipeline"
        },
        "business_logic": {
            "target_column": "target_is_churn",
            "churn_probability_threshold": 0.5,
            "system_columns_to_drop": ["customer_unique_id", "snapshot_date"]
        },
        "cloud_storage": {
            "data_lake": {
                "database_name": "test-data-lake",
                "bronze_dir": "bronze"
            },
            "model_registry": {
                "bucket_name": "test-model-registry",
                "registry_dir": "model_registry",
                "state_dir": "state",
                "pointer_file": "model_state.json"
            },
            "inference_outputs": {
                "base_artifact_dir": "inference_pipeline_artifacts",
                "business_reports_dir": "business_reports/customer_churn",
                "mlops_telemetry_dir": "mlops_telemetry/inference_logs",
                "metadata_dir": "inference_metadata"
            }
        },
        "components": {
            "model_loader": {
                "dir_name": "01_model_loader",
                "model_file": "model.pkl",
                "schema_file": "schema.json",
                "baseline_metrics_file": "baseline_performance_metrics.json",
                "reference_distributions_file": "reference_feature_distributions.json",
                "metadata_file": "metadata.json"
            },
            "feature_matrix_builder": {
                "dir_name": "02_input_feature_matrix_builder",
                "feature_matrix_file": "input_feature_matrix.parquet",
                "schema_file": "schema.json",
                "metadata_file": "metadata.json"
            },
            "inference_validator": {
                "dir_name": "03_validator",
                "report_file": "report.json",
                "metadata_file": "metadata.json"
            },
            "report_generator": {
                "dir_name": "04_report_generator",
                "csv_report_file": "churn_predictions.csv",
                "telemetry_file": "telemetry.parquet",
                "metadata_file": "metadata.json"
            },
            "report_publisher": {
                "dir_name": "05_report_publisher",
                "metadata_file": "metadata.json"
            }
        }
    }

    config_path = temp_workspace / "mock_global_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    return str(config_path)


@pytest.fixture
def mock_s3_sync():
    """
    Provides a mocked instance of S3Sync to prevent real network calls.
    """
    mock_sync = MagicMock()
    mock_sync.download_file.return_value = None
    mock_sync.upload_file.return_value = None
    return mock_sync


@pytest.fixture
def mock_duckdb_connection():
    """
    Provides a MagicMock for the DuckDB connection to safely bypass the loading
    of AWS/HTTPFS extensions that require valid credentials during unit testing.
    """
    return MagicMock()


@pytest.fixture
def mock_pipeline_context(mock_global_config_path, mock_s3_sync, mock_duckdb_connection):
    """
    Yields a fully mocked InferencePipelineContext. Bypasses actual S3 and DuckDB
    instantiation while maintaining the correct property interfaces and configuration parsing.
    """
    with patch("pipelines.inference_pipeline.src.core.context.ConfigParser") as mock_parser_class:
        real_parser = ConfigParser(config_path=mock_global_config_path)
        mock_parser_class.return_value = real_parser

        with patch("pipelines.inference_pipeline.src.core.context.S3Sync", return_value=mock_s3_sync):
            context = InferencePipelineContext(run_id="test_run_123")

            # Patch the context manager protocols to bypass the actual extension execution
            with patch.object(InferencePipelineContext, "__enter__", return_value=context):
                with patch.object(InferencePipelineContext, "__exit__", return_value=None):
                    context._duckdb_con = mock_duckdb_connection
                    yield context


@pytest.fixture
def dummy_model_state_path(temp_workspace):
    """
    Creates a dummy model_state.json mimicking the Training Pipeline's registry pointer.
    """
    state_path = temp_workspace / "model_state.json"
    state_data = {
        "run_id": "champion_run_456",
        "s3_model_path": "s3://test-bucket/model.pkl",
        "s3_schema_path": "s3://test-bucket/schema.json",
        "s3_monitoring_baselines_path": "s3://test-bucket/baselines.json",
        "s3_reference_distributions_path": "s3://test-bucket/distributions.json"
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f)
    return str(state_path)


@pytest.fixture
def dummy_model_schema_path(temp_workspace):
    """
    Creates a dummy schema.json representing the expected predictive features.
    """
    path = temp_workspace / "model_schema.json"
    schema_data = {
        "features": {
            "numerical": ["f1", "f2", "monetary_value"],
            "categorical": {"c1": ["A", "B"]}
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema_data, f)
    return str(path)


@pytest.fixture
def dummy_builder_schema_path(temp_workspace):
    """
    Creates a dummy schema.json representing the structural output from DuckDB.
    """
    path = temp_workspace / "builder_schema.json"
    schema_data = [
        {"name": "customer_unique_id", "physical_type": "VARCHAR"},
        {"name": "snapshot_date", "physical_type": "VARCHAR"},
        {"name": "f1", "physical_type": "DOUBLE"},
        {"name": "f2", "physical_type": "DOUBLE"},
        {"name": "c1", "physical_type": "VARCHAR"},
        {"name": "monetary_value", "physical_type": "DOUBLE"}
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema_data, f)
    return str(path)


@pytest.fixture
def dummy_model_artifact_path(temp_workspace):
    """
    Creates a real Scikit-Learn DummyClassifier, fits it to expected data shapes,
    and serializes it via joblib to allow genuine load/predict integration testing.
    """
    model_path = temp_workspace / "dummy_model.pkl"

    X = pd.DataFrame({
        "f1": [1.0, 2.0],
        "f2": [0.5, 0.6],
        "monetary_value": [10.0, 20.0],
        "c1": ["A", "B"]
    })
    y = np.array([0, 1])

    model = DummyClassifier(strategy="uniform", random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_path)
    return str(model_path)


@pytest.fixture
def dummy_feature_matrix_path(temp_workspace):
    """
    Creates a physical Parquet file representing the generated out-of-core feature matrix.
    """
    parquet_path = temp_workspace / "input_feature_matrix.parquet"

    df = pd.DataFrame({
        "customer_unique_id": ["C1", "C2", "C3"],
        "snapshot_date": ["2023-01-01", "2023-01-01", "2023-01-01"],
        "f1": [10.5, 20.1, 15.0],
        "f2": [1.1, 2.2, 3.3],
        "c1": ["A", "B", "A"],
        "monetary_value": [100.0, 200.0, 150.0]
    })

    df.to_parquet(parquet_path, index=False, compression="snappy")
    return str(parquet_path)