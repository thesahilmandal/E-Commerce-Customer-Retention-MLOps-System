import os
import sys
from datetime import datetime, timezone

from pipelines.training_pipeline.src import constants
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class PipelineConfig:
    """
    Base configuration for the Training Pipeline.
    Responsible for creating the unique run ID and root artifact directory.
    """

    def __init__(self) -> None:
        try:
            self.run_id: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            self.root_dir: str = os.path.join(
                constants.ARTIFACT_DIR_NAME,
                constants.TRAINING_PIPELINE_ROOT_DIR_NAME,
                self.run_id,
            )
            os.makedirs(self.root_dir, exist_ok=True)
            logging.info("TrainingPipelineConfig initialized. Run ID: %s", self.run_id)

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineConfig.")
            raise CustomException(e, sys) from e


class DataIngestionConfig:
    """
    Configuration for the Training Pipeline Data Ingestion component.
    Defines S3 source URIs, local artifact paths for the splits, and OOT parameters.
    """

    def __init__(self, training_pipeline_config: PipelineConfig) -> None:
        try:
            self.data_ingestion_root_dir: str = os.path.join(
                training_pipeline_config.root_dir,
                constants.DATA_INGESTION_ROOT_DIR_NAME,
            )
            
            # Local artifact paths
            self.train_data_path: str = os.path.join(
                self.data_ingestion_root_dir, constants.DATA_INGESTION_TRAIN_FILE_NAME
            )
            self.val_data_path: str = os.path.join(
                self.data_ingestion_root_dir, constants.DATA_INGESTION_VAL_FILE_NAME
            )
            self.test_data_path: str = os.path.join(
                self.data_ingestion_root_dir, constants.DATA_INGESTION_TEST_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.data_ingestion_root_dir, constants.DATA_INGESTION_METADATA_FILE_NAME
            )

            # S3 remote source for the bitemporal master panel
            self.s3_master_panel_uri: str = (
                f"s3://{constants.S3_BUCKET_NAME}/{constants.S3_FEATURE_STORE_DIR_NAME}/"
                f"{constants.LOADER_MASTER_PANEL_LOCAL_FILE_NAME}"
            )

            # Splitting configuration based on bitemporal snapshots
            self.train_snapshots: list = constants.TRAIN_SNAPSHOT
            self.val_snapshot: str = constants.VAL_SNAPSHOT
            self.test_snapshot: str = constants.TEST_SNAPSHOT

            os.makedirs(self.data_ingestion_root_dir, exist_ok=True)
            logging.info("TrainingPipelineDataIngestionConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineDataIngestionConfig.")
            raise CustomException(e, sys) from e


class FeatureTransformationConfig:
    """
    Configuration for the Training Pipeline Data Transformation component.
    Defines output paths for the preprocessor artifact, transformed datasets, 
    and schema validation rules.
    """

    def __init__(self, training_pipeline_config: PipelineConfig) -> None:
        try:
            self.data_transformation_root_dir: str = os.path.join(
                training_pipeline_config.root_dir,
                constants.DATA_TRANSFORMATION_ROOT_DIR_NAME,
            )
            
            # Local artifact paths for the serialized preprocessor and metadata
            self.preprocessor_file_path: str = os.path.join(
                self.data_transformation_root_dir,
                constants.DATA_TRANSFORMATION_PREPROCESSOR_FILE_NAME,
            )
            self.schema_file_path: str = os.path.join(
                self.data_transformation_root_dir,
                constants.DATA_TRANSFORMATION_SCHEMA_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.data_transformation_root_dir,
                constants.DATA_TRANSFORMATION_METADATA_FILE_NAME,
            )

            # Transformed Feature Matrix (X) and Target Vector (y) paths
            self.x_train_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_X_TRAIN_FILE_NAME
            )
            self.y_train_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_Y_TRAIN_FILE_NAME
            )
            self.x_val_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_X_VAL_FILE_NAME
            )
            self.y_val_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_Y_VAL_FILE_NAME
            )
            self.x_test_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_X_TEST_FILE_NAME
            )
            self.y_test_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_Y_TEST_FILE_NAME
            )

            # Feature schema and processing constants
            self.target_column: str = constants.TARGET_COLUMN
            self.columns_to_drop: list = constants.SYSTEM_COLUMNS_TO_DROP

            os.makedirs(self.data_transformation_root_dir, exist_ok=True)
            logging.info("TrainingPipelineDataTransformationConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineDataTransformationConfig.")
            raise CustomException(e, sys) from e


class ModelTrainingConfig:
    """
    Configuration for the Training Pipeline Model Trainer component.
    Defines output paths for the calibrated model, SHAP summaries, and MLflow metadata.
    """

    def __init__(self, training_pipeline_config: PipelineConfig) -> None:
        try:
            self.model_trainer_root_dir: str = os.path.join(
                training_pipeline_config.root_dir,
                constants.MODEL_TRAINER_ROOT_DIR_NAME,
            )
            
            self.model_file_path: str = os.path.join(
                self.model_trainer_root_dir,
                constants.MODEL_TRAINER_MODEL_FILE_NAME,
            )
            self.shap_summary_file_path: str = os.path.join(
                self.model_trainer_root_dir,
                constants.MODEL_TRAINER_SHAP_SUMMARY_FILE_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.model_trainer_root_dir,
                constants.MODEL_TRAINER_METADATA_FILE_NAME,
            )

            # JSON artifacts required for Downstream Monitoring Pipeline
            self.reference_feature_distributions_file_path: str = os.path.join(
                self.model_trainer_root_dir,
                "reference_feature_distributions.json",
            )
            self.shap_feature_importance_summary_file_path: str = os.path.join(
                self.model_trainer_root_dir,
                "shap_feature_importance_summary.json",
            )

            self.mlflow_experiment_name: str = constants.MODEL_TRAINER_MLFLOW_EXPERIMENT_NAME

            os.makedirs(self.model_trainer_root_dir, exist_ok=True)
            logging.info("TrainingPipelineModelTrainerConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineModelTrainerConfig.")
            raise CustomException(e, sys) from e


class ModelEvaluationConfig:
    """
    Configuration for the Training Pipeline Model Evaluation component.
    Defines output paths for the evaluation report and gating thresholds.
    """

    def __init__(self, training_pipeline_config: PipelineConfig) -> None:
        try:
            self.model_evaluation_root_dir: str = os.path.join(
                training_pipeline_config.root_dir,
                constants.MODEL_EVALUATION_ROOT_DIR_NAME,
            )
            
            self.report_file_path: str = os.path.join(
                self.model_evaluation_root_dir,
                constants.MODEL_EVALUATION_REPORT_FILE_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.model_evaluation_root_dir,
                constants.MODEL_EVALUATION_METADATA_FILE_NAME,
            )

            # JSON artifact required for Downstream Monitoring Pipeline
            self.baseline_performance_metrics_file_path: str = os.path.join(
                self.model_evaluation_root_dir,
                "baseline_performance_metrics.json",
            )

            # Business and Hysteresis Thresholds
            self.min_eroi_threshold: float = constants.MODEL_EVALUATION_MIN_EROI_THRESHOLD
            self.eroi_hysteresis_margin: float = constants.MODEL_EVALUATION_EROI_HYSTERESIS_MARGIN

            os.makedirs(self.model_evaluation_root_dir, exist_ok=True)
            logging.info("TrainingPipelineModelEvaluationConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineModelEvaluationConfig.")
            raise CustomException(e, sys) from e


class ModelRegistrationConfig:
    """
    Configuration for the Training Pipeline Model Registry component.
    Defines S3 URIs for immutable artifact storage and the mutable production pointer.
    """

    def __init__(self, training_pipeline_config: PipelineConfig) -> None:
        try:
            self.model_registry_root_dir: str = os.path.join(
                training_pipeline_config.root_dir,
                constants.MODEL_REGISTRY_ROOT_DIR_NAME,
            )
            
            self.metadata_file_path: str = os.path.join(
                self.model_registry_root_dir,
                constants.MODEL_REGISTRY_METADATA_FILE_NAME,
            )

            # S3 Registry Configurations
            self.s3_bucket_name: str = constants.S3_BUCKET_NAME
            self.s3_registry_base_uri: str = f"s3://{self.s3_bucket_name}/{constants.S3_MODEL_REGISTRY_DIR_NAME}"
            
            self.s3_models_dir_uri: str = f"{self.s3_registry_base_uri}/{constants.S3_MODEL_REGISTRY_MODELS_DIR}"
            self.s3_state_dir_uri: str = f"{self.s3_registry_base_uri}/{constants.S3_MODEL_REGISTRY_STATE_DIR}"
            self.s3_pointer_file_uri: str = f"{self.s3_state_dir_uri}/{constants.S3_MODEL_REGISTRY_POINTER_FILE_NAME}"

            os.makedirs(self.model_registry_root_dir, exist_ok=True)
            logging.info("TrainingPipelineModelRegistryConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineModelRegistryConfig.")
            raise CustomException(e, sys) from e