import os
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from pipelines.training_pipeline.src.entity.config_entity import (
    DataProcessorConfig,
    ModelEvaluatorConfig,
    ModelRegistryConfig,
    ModelTrainerConfig,
)


def test_data_processor_config_from_context(
    mock_pipeline_context: MagicMock,
) -> None:
    config = DataProcessorConfig.from_context(mock_pipeline_context)

    assert (
        config.training_dataset_s3_uri_path
        == "s3://test-bucket/master_panel.parquet"
    )
    assert config.target_column == "target_is_churn"
    assert config.val_size == 0.15
    assert config.test_size == 0.15
    assert config.random_state == 42
    assert config.system_columns_to_drop == [
        "customer_unique_id",
        "snapshot_date",
        "ingested_at_utc",
        "target_180d_ltv",
    ]

    base_dir = "/tmp/artifacts/test_run_12345/01_data_processor"
    assert config.data_processor_dir == base_dir
    assert config.preprocessor_file_path == os.path.join(
        base_dir,
        "preprocessor.pkl",
    )
    assert config.schema_file_path == os.path.join(
        base_dir,
        "schema.json",
    )
    assert config.metadata_file_path == os.path.join(
        base_dir,
        "metadata.json",
    )
    assert config.x_train_file_path == os.path.join(
        base_dir,
        "x_train.parquet",
    )
    assert config.y_train_file_path == os.path.join(
        base_dir,
        "y_train.parquet",
    )
    assert config.x_val_file_path == os.path.join(
        base_dir,
        "x_val.parquet",
    )
    assert config.y_val_file_path == os.path.join(
        base_dir,
        "y_val.parquet",
    )
    assert config.x_test_file_path == os.path.join(
        base_dir,
        "x_test.parquet",
    )
    assert config.y_test_file_path == os.path.join(
        base_dir,
        "y_test.parquet",
    )


def test_model_trainer_config_from_context(
    mock_pipeline_context: MagicMock,
) -> None:
    config = ModelTrainerConfig.from_context(mock_pipeline_context)

    assert config.mlflow_experiment_name == "Customer_Retention_Optimization"
    assert config.random_state == 42
    assert config.optuna_n_trials == 2
    assert config.early_stopping_rounds == 5
    assert config.calibration_method == "isotonic"
    assert config.calibration_cv_folds == 2
    assert config.hyperparameter_search_space["n_estimators"]["min"] == 10

    base_dir = "/tmp/artifacts/test_run_12345/02_model_trainer"
    assert config.model_trainer_dir == base_dir
    assert config.model_file_path == os.path.join(
        base_dir,
        "model.pkl",
    )
    assert config.shap_summary_file_path == os.path.join(
        base_dir,
        "shap_summary.png",
    )
    assert config.shap_feature_importance_file_path == os.path.join(
        base_dir,
        "shap_feature_importance.json",
    )
    assert config.reference_feature_distributions_file_path == os.path.join(
        base_dir,
        "reference_feature_distributions.json",
    )
    assert config.metadata_file_path == os.path.join(
        base_dir,
        "metadata.json",
    )


def test_model_evaluator_config_from_context(
    mock_pipeline_context: MagicMock,
) -> None:
    config = ModelEvaluatorConfig.from_context(mock_pipeline_context)

    assert config.min_eroi_threshold == 0.05
    assert config.eroi_hysteresis_margin == 0.02
    assert config.campaign_cost == 10.0
    assert config.customer_ltv == 500.0
    assert config.intervention_save_rate == 0.10

    expected_s3_pointer_uri = (
        "s3://test-pipeline-artifacts-bucket/"
        "model_registry/state/model_state.json"
    )
    assert config.s3_pointer_uri == expected_s3_pointer_uri

    base_dir = "/tmp/artifacts/test_run_12345/03_model_evaluator"
    assert config.model_evaluator_dir == base_dir
    assert config.report_file_path == os.path.join(
        base_dir,
        "evaluation_report.json",
    )
    assert config.baseline_performance_metrics_file_path == os.path.join(
        base_dir,
        "baseline_performance_metrics.json",
    )
    assert config.metadata_file_path == os.path.join(
        base_dir,
        "metadata.json",
    )


def test_model_registry_config_from_context(
    mock_pipeline_context: MagicMock,
) -> None:
    config = ModelRegistryConfig.from_context(mock_pipeline_context)

    assert config.deployment_environment == "testing"

    expected_models_uri = (
        "s3://test-pipeline-artifacts-bucket/model_registry/models"
    )
    expected_pointer_uri = (
        "s3://test-pipeline-artifacts-bucket/"
        "model_registry/state/model_state.json"
    )

    assert config.s3_models_dir_uri == expected_models_uri
    assert config.s3_pointer_uri == expected_pointer_uri

    base_dir = "/tmp/artifacts/test_run_12345/04_model_registry"
    assert config.model_registry_dir == base_dir
    assert config.staging_dir == os.path.join(
        base_dir,
        "staging",
    )
    assert config.metadata_file_path == os.path.join(
        base_dir,
        "metadata.json",
    )


@pytest.mark.parametrize(
    "config_class",
    [
        DataProcessorConfig,
        ModelTrainerConfig,
        ModelEvaluatorConfig,
        ModelRegistryConfig,
    ],
)
def test_config_entities_are_immutable(
    config_class: type,
    mock_pipeline_context: MagicMock,
) -> None:
    config = config_class.from_context(mock_pipeline_context)

    with pytest.raises(FrozenInstanceError):
        config.some_arbitrary_field = "new_value"