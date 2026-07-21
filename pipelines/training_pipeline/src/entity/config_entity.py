import os
from dataclasses import dataclass
from typing import List, Dict, Any

from pipelines.training_pipeline.src.core.context import PipelineContext


@dataclass(frozen=True)
class DataProcessorConfig:
    """
    Configuration entity for the Data Processor component.
    Strictly immutable and holds all parameters and dynamically calculated 
    local paths required for Out-Of-Core dataset splitting and schema enforcement.
    """
    training_dataset_s3_uri_path: str
    target_column: str
    val_size: float
    test_size: float
    random_state: int
    system_columns_to_drop: List[str]
    
    data_processor_dir: str
    preprocessor_file_path: str
    schema_file_path: str
    metadata_file_path: str
    x_train_file_path: str
    y_train_file_path: str
    x_val_file_path: str
    y_val_file_path: str
    x_test_file_path: str
    y_test_file_path: str

    @classmethod
    def from_context(cls, context: PipelineContext) -> "DataProcessorConfig":
        """Factory method to instantiate config dynamically from PipelineContext."""
        global_config = context.config.get_global_config()
        dp_config = context.config.get_data_processor_config()
        base_dir = context.data_processor_dir

        return cls(
            training_dataset_s3_uri_path=context.training_dataset_s3_uri_path,
            target_column=global_config["target_column"],
            val_size=dp_config["val_size"],
            test_size=dp_config["test_size"],
            random_state=dp_config["random_state"],
            system_columns_to_drop=dp_config["system_columns_to_drop"],
            data_processor_dir=base_dir,
            preprocessor_file_path=os.path.join(base_dir, "preprocessor.pkl"),
            schema_file_path=os.path.join(base_dir, "schema.json"),
            metadata_file_path=os.path.join(base_dir, "metadata.json"),
            x_train_file_path=os.path.join(base_dir, "x_train.parquet"),
            y_train_file_path=os.path.join(base_dir, "y_train.parquet"),
            x_val_file_path=os.path.join(base_dir, "x_val.parquet"),
            y_val_file_path=os.path.join(base_dir, "y_val.parquet"),
            x_test_file_path=os.path.join(base_dir, "x_test.parquet"),
            y_test_file_path=os.path.join(base_dir, "y_test.parquet"),
        )


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration entity for the Model Trainer component.
    Encapsulates Optuna optimization settings, XGBoost hyperparameter search space, 
    calibration settings, and output paths for models and explainability artifacts.
    """
    mlflow_experiment_name: str
    random_state: int
    optuna_n_trials: int
    early_stopping_rounds: int
    calibration_method: str
    calibration_cv_folds: int
    hyperparameter_search_space: Dict[str, Any]
    
    model_trainer_dir: str
    model_file_path: str
    shap_summary_file_path: str
    shap_feature_importance_file_path: str
    reference_feature_distributions_file_path: str
    metadata_file_path: str

    @classmethod
    def from_context(cls, context: PipelineContext) -> "ModelTrainerConfig":
        trainer_config = context.config.get_model_trainer_config()
        base_dir = context.model_trainer_dir

        return cls(
            mlflow_experiment_name=trainer_config["mlflow_experiment_name"],
            random_state=trainer_config["random_state"],
            optuna_n_trials=trainer_config["optuna_n_trials"],
            early_stopping_rounds=trainer_config["early_stopping_rounds"],
            calibration_method=trainer_config["calibration_method"],
            calibration_cv_folds=trainer_config["calibration_cv_folds"],
            hyperparameter_search_space=trainer_config["hyperparameter_search_space"],
            model_trainer_dir=base_dir,
            model_file_path=os.path.join(base_dir, "model.pkl"),
            shap_summary_file_path=os.path.join(base_dir, "shap_summary.png"),
            shap_feature_importance_file_path=os.path.join(base_dir, "shap_feature_importance.json"),
            reference_feature_distributions_file_path=os.path.join(base_dir, "reference_feature_distributions.json"),
            metadata_file_path=os.path.join(base_dir, "metadata.json"),
        )


@dataclass(frozen=True)
class ModelEvaluatorConfig:
    """
    Configuration entity for the Model Evaluator component.
    Defines business gating thresholds, expected ROI (EROI) hysteresis margins, 
    business assumption costs, and exact S3 URIs required to fetch the production Champion.
    """
    min_eroi_threshold: float
    eroi_hysteresis_margin: float
    campaign_cost: float
    customer_ltv: float
    intervention_save_rate: float
    
    s3_pointer_uri: str
    model_evaluator_dir: str
    report_file_path: str
    baseline_performance_metrics_file_path: str
    metadata_file_path: str

    @classmethod
    def from_context(cls, context: PipelineContext) -> "ModelEvaluatorConfig":
        global_config = context.config.get_global_config()
        eval_config = context.config.get_model_evaluator_config()
        reg_config = context.config.get_model_registry_config()
        base_dir = context.model_evaluator_dir

        bucket = global_config["s3_bucket_name"]
        s3_pointer_uri = (
            f"s3://{bucket}/{reg_config['s3_registry_base_dir']}/"
            f"{reg_config['s3_state_dir']}/{reg_config['s3_pointer_file_name']}"
        )

        return cls(
            min_eroi_threshold=eval_config["min_eroi_threshold"],
            eroi_hysteresis_margin=eval_config["eroi_hysteresis_margin"],
            campaign_cost=eval_config["business_assumptions"]["campaign_cost"],
            customer_ltv=eval_config["business_assumptions"]["customer_ltv"],
            intervention_save_rate=eval_config["business_assumptions"]["intervention_save_rate"],
            s3_pointer_uri=s3_pointer_uri,
            model_evaluator_dir=base_dir,
            report_file_path=os.path.join(base_dir, "evaluation_report.json"),
            baseline_performance_metrics_file_path=os.path.join(base_dir, "baseline_performance_metrics.json"),
            metadata_file_path=os.path.join(base_dir, "metadata.json"),
        )


@dataclass(frozen=True)
class ModelRegistryConfig:
    """
    Configuration entity for the Model Registry component.
    Defines the Two-Phase Commit endpoints in AWS S3, including the immutable models vault 
    and the mutable state pointer.
    """
    deployment_environment: str
    s3_models_dir_uri: str
    s3_pointer_uri: str
    
    model_registry_dir: str
    staging_dir: str
    metadata_file_path: str

    @classmethod
    def from_context(cls, context: PipelineContext) -> "ModelRegistryConfig":
        global_config = context.config.get_global_config()
        reg_config = context.config.get_model_registry_config()
        base_dir = context.model_registry_dir

        bucket = global_config["s3_bucket_name"]
        base_registry_uri = f"s3://{bucket}/{reg_config['s3_registry_base_dir']}"

        s3_models_dir_uri = f"{base_registry_uri}/{reg_config['s3_models_dir']}"
        s3_pointer_uri = f"{base_registry_uri}/{reg_config['s3_state_dir']}/{reg_config['s3_pointer_file_name']}"

        return cls(
            deployment_environment=reg_config["deployment_environment"],
            s3_models_dir_uri=s3_models_dir_uri,
            s3_pointer_uri=s3_pointer_uri,
            model_registry_dir=base_dir,
            staging_dir=os.path.join(base_dir, "staging"),
            metadata_file_path=os.path.join(base_dir, "metadata.json"),
        )