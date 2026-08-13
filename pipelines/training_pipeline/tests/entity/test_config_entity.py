import os
from dataclasses import FrozenInstanceError

import pytest

from pipelines.training_pipeline.src.entity.config_entity import (
    DataProcessorConfig,
    ModelTrainerConfig,
    ModelEvaluatorConfig,
    ModelRegistryConfig,
)


def test_data_processor_config_from_context(pipeline_context):
    """Verifies DataProcessorConfig correctly maps values from PipelineContext and derives paths."""
    config = DataProcessorConfig.from_context(pipeline_context)

    assert config.training_dataset_s3_uri_path == "s3://test-ml-platform-bucket/feature_store/test_run_001/dataset.parquet"
    assert config.target_column == "target_is_churn"
    assert config.val_size == 0.20
    assert config.test_size == 0.20
    assert config.random_state == 42
    assert "target_180d_ltv" in config.system_columns_to_drop

    base_dir = pipeline_context.data_processor_dir
    assert config.data_processor_dir == base_dir
    assert config.preprocessor_file_path == os.path.join(base_dir, "preprocessor.pkl")
    assert config.schema_file_path == os.path.join(base_dir, "schema.json")
    assert config.metadata_file_path == os.path.join(base_dir, "metadata.json")
    assert config.x_train_file_path == os.path.join(base_dir, "x_train.parquet")
    assert config.y_test_file_path == os.path.join(base_dir, "y_test.parquet")


def test_model_trainer_config_from_context(pipeline_context):
    """Verifies ModelTrainerConfig correctly extracts optimization settings and constructs paths."""
    config = ModelTrainerConfig.from_context(pipeline_context)

    assert config.mlflow_experiment_name == "Test_Experiment"
    assert config.random_state == 42
    assert config.optuna_n_trials == 2
    assert config.early_stopping_rounds == 5
    assert config.calibration_method == "isotonic"
    assert config.calibration_cv_folds == 2

    search_space = config.hyperparameter_search_space
    assert "n_estimators" in search_space
    assert search_space["n_estimators"]["min"] == 10

    base_dir = pipeline_context.model_trainer_dir
    assert config.model_trainer_dir == base_dir
    assert config.model_file_path == os.path.join(base_dir, "model.pkl")
    assert config.shap_summary_file_path == os.path.join(base_dir, "shap_summary.png")
    assert config.shap_feature_importance_file_path == os.path.join(
        base_dir, "shap_feature_importance.json"
    )
    assert config.reference_feature_distributions_file_path == os.path.join(
        base_dir, "reference_feature_distributions.json"
    )
    assert config.metadata_file_path == os.path.join(base_dir, "metadata.json")


def test_model_evaluator_config_from_context(pipeline_context):
    """Verifies ModelEvaluatorConfig correctly flattens nested business assumptions and resolves S3 pointer."""
    config = ModelEvaluatorConfig.from_context(pipeline_context)

    assert config.min_eroi_threshold == 0.05
    assert config.eroi_hysteresis_margin == 0.02
    assert config.campaign_cost == 10.0
    assert config.customer_ltv == 500.0
    assert config.intervention_save_rate == 0.10

    expected_s3_pointer = (
        "s3://test-ml-platform-bucket/model_registry/state/model_state.json"
    )
    assert config.s3_pointer_uri == expected_s3_pointer

    base_dir = pipeline_context.model_evaluator_dir
    assert config.model_evaluator_dir == base_dir
    assert config.report_file_path == os.path.join(
        base_dir, "evaluation_report.json"
    )
    assert config.baseline_performance_metrics_file_path == os.path.join(
        base_dir, "baseline_performance_metrics.json"
    )
    assert config.metadata_file_path == os.path.join(base_dir, "metadata.json")


def test_model_registry_config_from_context(pipeline_context):
    """Verifies ModelRegistryConfig correctly pieces together the S3 WORM vault and state directories."""
    config = ModelRegistryConfig.from_context(pipeline_context)

    assert config.deployment_environment == "testing"

    expected_models_uri = (
        "s3://test-ml-platform-bucket/model_registry/models"
    )
    expected_pointer_uri = (
        "s3://test-ml-platform-bucket/model_registry/state/model_state.json"
    )
    assert config.s3_models_dir_uri == expected_models_uri
    assert config.s3_pointer_uri == expected_pointer_uri

    base_dir = pipeline_context.model_registry_dir
    assert config.model_registry_dir == base_dir
    assert config.staging_dir == os.path.join(base_dir, "staging")
    assert config.metadata_file_path == os.path.join(base_dir, "metadata.json")


def test_config_entities_are_strictly_immutable(pipeline_context):
    """Ensures configuration dataclasses are frozen and prevent runtime mutation."""
    dp_config = DataProcessorConfig.from_context(pipeline_context)

    with pytest.raises(FrozenInstanceError):
        dp_config.val_size = 0.5

    mt_config = ModelTrainerConfig.from_context(pipeline_context)

    with pytest.raises(FrozenInstanceError):
        mt_config.optuna_n_trials = 100