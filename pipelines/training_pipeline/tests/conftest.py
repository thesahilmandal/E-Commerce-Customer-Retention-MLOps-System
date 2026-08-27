import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipelines.training_pipeline.src.core.config_parser import ConfigParser
from pipelines.training_pipeline.src.core.context import PipelineContext
from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataProcessorArtifact,
    ModelTrainerArtifact,
    ModelEvaluatorArtifact,
    ModelRegistryArtifact,
)


@pytest.fixture(autouse=True)
def mock_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(autouse=True)
def block_boto3_network_calls() -> None:
    with patch("boto3.client"):
        yield


@pytest.fixture
def env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "S3_PIPELINE_RUN_ARTIFACTS",
        "test-pipeline-artifacts-bucket",
    )


@pytest.fixture
def dummy_yaml_config(tmp_path: Path) -> str:
    yaml_content = """
training_pipeline:
  global:
    s3_bucket_name: "${S3_PIPELINE_RUN_ARTIFACTS}"
    local_artifact_dir: "artifacts/training_pipeline"
    target_column: "target_is_churn"

  data_processor:
    val_size: 0.15
    test_size: 0.15
    random_state: 42
    system_columns_to_drop:
      - "customer_unique_id"
      - "snapshot_date"
      - "ingested_at_utc"
      - "target_180d_ltv"

  model_trainer:
    mlflow_experiment_name: "Customer_Retention_Optimization"
    random_state: 42
    optuna_n_trials: 2
    early_stopping_rounds: 5
    calibration_method: "isotonic"
    calibration_cv_folds: 2
    hyperparameter_search_space:
      n_estimators:
        min: 10
        max: 20
        step: 10
      learning_rate:
        min: 0.01
        max: 0.2
      max_depth:
        min: 3
        max: 9
      min_child_weight:
        min: 1
        max: 10
      subsample:
        min: 0.6
        max: 1.0
      colsample_bytree:
        min: 0.6
        max: 1.0
      gamma:
        min: 0.0
        max: 5.0

  model_evaluator:
    min_eroi_threshold: 0.05
    eroi_hysteresis_margin: 0.02
    business_assumptions:
      campaign_cost: 10.0
      customer_ltv: 500.0
      intervention_save_rate: 0.10

  model_registry:
    s3_registry_base_dir: "model_registry"
    s3_models_dir: "models"
    s3_state_dir: "state"
    s3_pointer_file_name: "model_state.json"
    deployment_environment: "testing"
"""

    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(yaml_content)

    return str(config_path)


@pytest.fixture
def config_parser(
    env_vars: None,
    dummy_yaml_config: str,
) -> ConfigParser:
    return ConfigParser(config_filepath=dummy_yaml_config)


@pytest.fixture
def mock_pipeline_context(
    config_parser: ConfigParser,
) -> MagicMock:
    context = MagicMock(spec=PipelineContext)

    context.run_id = "test_run_12345"
    context.training_dataset_s3_uri_path = (
        "s3://test-bucket/master_panel.parquet"
    )
    context.config = config_parser

    context.run_artifact_dir = "/tmp/artifacts/test_run_12345"
    context.data_processor_dir = (
        "/tmp/artifacts/test_run_12345/01_data_processor"
    )
    context.model_trainer_dir = (
        "/tmp/artifacts/test_run_12345/02_model_trainer"
    )
    context.model_evaluator_dir = (
        "/tmp/artifacts/test_run_12345/03_model_evaluator"
    )
    context.model_registry_dir = (
        "/tmp/artifacts/test_run_12345/04_model_registry"
    )

    context.s3_sync = MagicMock()
    context.db_conn = MagicMock()

    return context


@pytest.fixture
def dummy_data_processor_artifact() -> DataProcessorArtifact:
    return DataProcessorArtifact(
        preprocessor_file_path="/tmp/preprocessor.pkl",
        schema_file_path="/tmp/schema.json",
        metadata_file_path="/tmp/dp_metadata.json",
        x_train_file_path="/tmp/x_train.parquet",
        y_train_file_path="/tmp/y_train.parquet",
        x_val_file_path="/tmp/x_val.parquet",
        y_val_file_path="/tmp/y_val.parquet",
        x_test_file_path="/tmp/x_test.parquet",
        y_test_file_path="/tmp/y_test.parquet",
    )


@pytest.fixture
def dummy_model_trainer_artifact() -> ModelTrainerArtifact:
    return ModelTrainerArtifact(
        model_file_path="/tmp/model.pkl",
        shap_summary_file_path="/tmp/shap_summary.png",
        shap_feature_importance_file_path=(
            "/tmp/shap_feature_importance.json"
        ),
        reference_feature_distributions_file_path=(
            "/tmp/reference_feature_distributions.json"
        ),
        metadata_file_path="/tmp/mt_metadata.json",
    )


@pytest.fixture
def dummy_model_evaluator_artifact() -> ModelEvaluatorArtifact:
    return ModelEvaluatorArtifact(
        approval_status=True,
        report_file_path="/tmp/evaluation_report.json",
        baseline_performance_metrics_file_path=(
            "/tmp/baseline_performance_metrics.json"
        ),
        metadata_file_path="/tmp/me_metadata.json",
    )


@pytest.fixture
def dummy_model_registry_artifact() -> ModelRegistryArtifact:
    return ModelRegistryArtifact(
        deployment_status=True,
        s3_model_uri=(
            "s3://test-pipeline-artifacts-bucket/"
            "model_registry/models/test_run_12345"
        ),
        metadata_file_path="/tmp/mr_metadata.json",
    )