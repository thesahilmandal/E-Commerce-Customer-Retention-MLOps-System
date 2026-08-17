import yaml
import joblib
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from pipelines.training_pipeline.src.core.config_parser import ConfigParser
from pipelines.training_pipeline.src.core.context import PipelineContext
from pipelines.training_pipeline.src.components.data_processor import CategoricalSchemaEnforcer

from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataProcessorArtifact,
    ModelTrainerArtifact,
    ModelEvaluatorArtifact,
    ModelRegistryArtifact,
)

from shared_core.cloud.s3_operations import S3Sync
from shared_core.utils.main_utils import write_json_file


@pytest.fixture
def mock_s3_sync():
    """Provides a mocked S3Sync instance for isolating tests from AWS S3."""
    s3_mock = MagicMock(spec=S3Sync)

    s3_mock.download_file = MagicMock(return_value=True)

    s3_mock.upload_file = MagicMock(return_value=True)

    s3_mock.sync_folder_to_s3 = MagicMock(return_value=True)

    return s3_mock


@pytest.fixture
def sample_config_dict(tmp_path):
    """Provides a valid configuration dictionary targeting temporary test paths."""
    artifact_dir = str(tmp_path / "artifacts")

    return {
        "training_pipeline": {
            "global": {
                "s3_bucket_name": "test-ml-platform-bucket",
                "local_artifact_dir": artifact_dir,
                "target_column": "target_is_churn",
            },
            "data_processor": {
                "val_size": 0.20,
                "test_size": 0.20,
                "random_state": 42,
                "system_columns_to_drop": [
                    "customer_unique_id",
                    "snapshot_date",
                    "ingested_at_utc",
                    "target_180d_ltv",
                ],
            },
            "model_trainer": {
                "mlflow_experiment_name": "Test_Experiment",
                "random_state": 42,
                "optuna_n_trials": 2,
                "early_stopping_rounds": 5,
                "calibration_method": "isotonic",
                "calibration_cv_folds": 2,
                "hyperparameter_search_space": {
                    "n_estimators": {"min": 10, "max": 20, "step": 10},
                    "learning_rate": {"min": 0.01, "max": 0.1},
                    "max_depth": {"min": 2, "max": 4},
                    "min_child_weight": {"min": 1, "max": 3},
                    "subsample": {"min": 0.8, "max": 1.0},
                    "colsample_bytree": {"min": 0.8, "max": 1.0},
                    "gamma": {"min": 0.0, "max": 1.0},
                },
            },
            "model_evaluator": {
                "min_eroi_threshold": 0.05,
                "eroi_hysteresis_margin": 0.02,
                "business_assumptions": {
                    "campaign_cost": 10.0,
                    "customer_ltv": 500.0,
                    "intervention_save_rate": 0.10,
                },
            },
            "model_registry": {
                "s3_registry_base_dir": "model_registry",
                "s3_models_dir": "models",
                "s3_state_dir": "state",
                "s3_pointer_file_name": "model_state.json",
                "deployment_environment": "testing",
            },
        }
    }


@pytest.fixture
def sample_yaml_config_file(tmp_path, sample_config_dict):
    """Creates a temporary YAML configuration file for testing ConfigParser."""
    config_path = tmp_path / "pipeline_config.yaml"

    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)

    return str(config_path)


@pytest.fixture
def mock_config_parser(sample_yaml_config_file):
    """Provides an initialized ConfigParser pointed to a temporary valid YAML file."""
    return ConfigParser(config_filepath=sample_yaml_config_file)


@pytest.fixture
def pipeline_context(tmp_path, sample_yaml_config_file, mock_s3_sync):
    """
    Provides a fully initialized PipelineContext with mocked DuckDB and S3 Sync.
    """
    config_parser = ConfigParser(config_filepath=sample_yaml_config_file)

    mock_db_conn = MagicMock()

    with patch(
        "pipelines.training_pipeline.src.core.context.S3Sync",
        return_value=mock_s3_sync,
    ), patch.object(
        PipelineContext,
        "_initialize_duckdb",
        return_value=mock_db_conn,
    ):
        context = PipelineContext(
            run_id="test_run_001",
            training_dataset_s3_uri_path="s3://test-ml-platform-bucket/feature_store/test_run_001/dataset.parquet",
            config_parser=config_parser,
            cleanup_on_exit=False,
        )

        yield context

        context.close()


@pytest.fixture
def synthetic_dataframe():
    """Generates a deterministic synthetic DataFrame matching the Training Pipeline's expected schema."""
    np.random.seed(42)

    n_rows = 100

    data = {
        "customer_unique_id": [f"cust_{i}" for i in range(n_rows)],
        "snapshot_date": ["2023-01-01"] * n_rows,
        "ingested_at_utc": ["2023-01-01T00:00:00Z"] * n_rows,
        "target_180d_ltv": np.random.uniform(10, 100, n_rows),
        "target_is_churn": np.random.choice(
            [0, 1], size=n_rows, p=[0.7, 0.3]
        ),
        "num_feature_1": np.random.randn(n_rows),
        "num_feature_2": np.random.uniform(0, 100, n_rows),
        "cat_feature_1": np.random.choice(["A", "B", "C"], size=n_rows),
        "cat_feature_2": np.random.choice(["X", "Y"], size=n_rows),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_parquet_file(tmp_path, synthetic_dataframe):
    """Saves synthetic_dataframe as a local Parquet file for ingestion testing."""
    file_path = tmp_path / "master_panel.parquet"

    synthetic_dataframe.to_parquet(file_path, index=False)

    return str(file_path)


@pytest.fixture
def data_processor_artifact(tmp_path, synthetic_dataframe):
    """Creates real Parquet datasets and pickled schema enforcer artifacts on disk for downstream test fixtures."""
    dp_dir = tmp_path / "01_data_processor"

    dp_dir.mkdir(parents=True, exist_ok=True)

    X = synthetic_dataframe.drop(
        columns=[
            "customer_unique_id",
            "snapshot_date",
            "ingested_at_utc",
            "target_180d_ltv",
            "target_is_churn",
        ]
    )

    for col in ["cat_feature_1", "cat_feature_2"]:
        X[col] = X[col].astype("category")

    y = synthetic_dataframe[["target_is_churn"]]

    x_train_path = str(dp_dir / "x_train.parquet")
    y_train_path = str(dp_dir / "y_train.parquet")
    x_val_path = str(dp_dir / "x_val.parquet")
    y_val_path = str(dp_dir / "y_val.parquet")
    x_test_path = str(dp_dir / "x_test.parquet")
    y_test_path = str(dp_dir / "y_test.parquet")
    preprocessor_path = str(dp_dir / "preprocessor.pkl")
    schema_path = str(dp_dir / "schema.json")
    metadata_path = str(dp_dir / "metadata.json")

    X.to_parquet(x_train_path, index=False)
    y.to_parquet(y_train_path, index=False)
    X.to_parquet(x_val_path, index=False)
    y.to_parquet(y_val_path, index=False)
    X.to_parquet(x_test_path, index=False)
    y.to_parquet(y_test_path, index=False)

    enforcer = CategoricalSchemaEnforcer(
        categorical_features=["cat_feature_1", "cat_feature_2"],
        numerical_features=["num_feature_1", "num_feature_2"],
    )

    enforcer.fit(X)

    joblib.dump(enforcer, preprocessor_path)

    schema_blueprint = {
        "features": {
            "numerical": ["num_feature_1", "num_feature_2"],
            "categorical": {
                "cat_feature_1": ["A", "B", "C"],
                "cat_feature_2": ["X", "Y"],
            },
        }
    }

    write_json_file(schema_path, schema_blueprint)
    write_json_file(metadata_path, {"status": "ok"})

    return DataProcessorArtifact(
        preprocessor_file_path=preprocessor_path,
        schema_file_path=schema_path,
        metadata_file_path=metadata_path,
        x_train_file_path=x_train_path,
        y_train_file_path=y_train_path,
        x_val_file_path=x_val_path,
        y_val_file_path=y_val_path,
        x_test_file_path=x_test_path,
        y_test_file_path=y_test_path,
    )


@pytest.fixture
def model_trainer_artifact(tmp_path, data_processor_artifact):
    """Creates trained model and SHAP/distribution artifacts on disk for downstream testing."""
    mt_dir = tmp_path / "02_model_trainer"

    mt_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(mt_dir / "model.pkl")
    shap_summary_path = str(mt_dir / "shap_summary.png")
    shap_importance_path = str(mt_dir / "shap_feature_importance.json")
    ref_dist_path = str(mt_dir / "reference_feature_distributions.json")
    metadata_path = str(mt_dir / "metadata.json")

    preprocessor = joblib.load(data_processor_artifact.preprocessor_file_path)

    X = pd.read_parquet(data_processor_artifact.x_train_file_path)

    y = pd.read_parquet(
        data_processor_artifact.y_train_file_path
    ).values.ravel()

    base_xgb = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
    )

    calibrated = CalibratedClassifierCV(estimator=base_xgb, cv=2)

    calibrated.fit(X, y)

    mega_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", calibrated)]
    )

    joblib.dump(mega_pipeline, model_path)

    with open(shap_summary_path, "wb") as f:
        f.write(b"DUMMY_PNG_CONTENT")

    write_json_file(
        shap_importance_path,
        {"num_feature_1": 0.45, "num_feature_2": 0.20},
    )

    write_json_file(
        ref_dist_path,
        {"num_feature_1": {"type": "numerical", "mean": 0.0}},
    )

    write_json_file(metadata_path, {"status": "ok"})

    return ModelTrainerArtifact(
        model_file_path=model_path,
        shap_summary_file_path=shap_summary_path,
        shap_feature_importance_file_path=shap_importance_path,
        reference_feature_distributions_file_path=ref_dist_path,
        metadata_file_path=metadata_path,
    )


@pytest.fixture
def model_evaluator_artifact(tmp_path):
    """Creates evaluation report and baseline performance artifacts on disk."""
    me_dir = tmp_path / "03_model_evaluator"

    me_dir.mkdir(parents=True, exist_ok=True)

    report_path = str(me_dir / "evaluation_report.json")
    baseline_path = str(me_dir / "baseline_performance_metrics.json")
    metadata_path = str(me_dir / "metadata.json")

    write_json_file(
        report_path,
        {"approval_status": True, "challenger_metrics": {"eroi": 12.5}},
    )

    write_json_file(
        baseline_path,
        {
            "metrics": {"log_loss": 0.25, "roc_auc": 0.85},
            "operational": {"expected_roi_per_user": 12.5},
        },
    )

    write_json_file(metadata_path, {"status": "ok"})

    return ModelEvaluatorArtifact(
        approval_status=True,
        report_file_path=report_path,
        baseline_performance_metrics_file_path=baseline_path,
        metadata_file_path=metadata_path,
    )


@pytest.fixture
def model_registry_artifact(tmp_path):
    """Provides a ModelRegistryArtifact instance for testing."""
    mr_dir = tmp_path / "04_model_registry"

    mr_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = str(mr_dir / "metadata.json")

    write_json_file(metadata_path, {"status": "ok"})

    return ModelRegistryArtifact(
        deployment_status=True,
        s3_model_uri="s3://test-ml-platform-bucket/model_registry/models/test_run_001",
        metadata_file_path=metadata_path,
    )