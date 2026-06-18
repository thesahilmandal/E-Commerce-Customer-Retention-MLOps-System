import os
import sys
from datetime import datetime, timezone

from shared_core import constants
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class InferencePipelineConfig:
    """
    Base configuration for the Inference Pipeline.
    Responsible for creating the unique run ID and root artifact directory.
    """

    def __init__(self) -> None:
        try:
            self.run_id: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            self.root_dir: str = os.path.join(
                constants.ARTIFACT_DIR_NAME,
                constants.INFERENCE_PIPELINE_ROOT_DIR_NAME,
                self.run_id,
            )
            os.makedirs(self.root_dir, exist_ok=True)
            logging.info("InferencePipelineConfig initialized. Run ID: %s", self.run_id)

        except Exception as e:
            logging.exception("Error initializing InferencePipelineConfig.")
            raise CustomException(e, sys) from e


class ModelLoadingConfig:
    """
    Configuration for the Inference Pipeline Current Production Model Loader component.
    Defines S3 URIs for retrieving the active production pointer and local paths 
    to safely stash the downloaded immutable model and schema artifacts.
    """

    def __init__(self, inference_pipeline_config: InferencePipelineConfig) -> None:
        try:
            # Root directory for this specific component
            self.model_loader_root_dir: str = os.path.join(
                inference_pipeline_config.root_dir,
                constants.INFERENCE_MODEL_LOADER_ROOT_DIR_NAME,
            )

            # Local scratchpad paths for the downloaded assets
            self.model_file_path: str = os.path.join(
                self.model_loader_root_dir,
                constants.INFERENCE_MODEL_LOADER_MODEL_FILE_NAME,
            )
            self.schema_file_path: str = os.path.join(
                self.model_loader_root_dir,
                constants.INFERENCE_MODEL_LOADER_SCHEMA_FILE_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.model_loader_root_dir,
                constants.INFERENCE_MODEL_LOADER_METADATA_FILE_NAME,
            )

            # S3 Registry Configurations
            # Aligns precisely with the Phase 2 (Model Registry) atomic state pointer
            self.s3_bucket_name: str = constants.S3_BUCKET_NAME
            self.s3_registry_base_uri: str = f"s3://{self.s3_bucket_name}/{constants.S3_MODEL_REGISTRY_DIR_NAME}"
            self.s3_pointer_uri: str = f"{self.s3_registry_base_uri}/model_state.json"

            # Pre-create the directory structure for safe local I/O
            os.makedirs(self.model_loader_root_dir, exist_ok=True)
            logging.info("InferenceModelLoaderConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing InferenceModelLoaderConfig.")
            raise CustomException(e, sys) from e


class FeatureMatrixGenerationConfig:
    """
    Configuration for the Input Feature Matrix Builder component.
    Defines S3 Data Lake URIs, DuckDB parameters, and local persistence paths
    for the generated inference feature matrix.
    """
    def __init__(self, inference_pipeline_config: InferencePipelineConfig) -> None:
        try:
            # Component Root Directory
            self.feature_matrix_root_dir: str = os.path.join(
                inference_pipeline_config.root_dir,
                constants.INFERENCE_FEATURE_MATRIX_BUILDER_ROOT_DIR_NAME,
            )
            
            # Local Artifact Paths
            self.feature_matrix_file_path: str = os.path.join(
                self.feature_matrix_root_dir, constants.INFERENCE_FEATURE_MATRIX_FILE_NAME
            )
            self.schema_file_path: str = os.path.join(
                self.feature_matrix_root_dir, constants.INFERENCE_FEATURE_MATRIX_SCHEMA_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.feature_matrix_root_dir, constants.INFERENCE_FEATURE_MATRIX_METADATA_FILE_NAME
            )

            # Upstream S3 Data Lake Location
            self.s3_data_lake_uri: str = f"s3://{constants.S3_CUSTOMER_DATABASE_NAME}/{constants.S3_DATA_LAKE_BRONZE_DIR_NAME}"

            # Snapshot Logic (Scoring Population Temporal Bound)
            # The pipeline runs on Day T, scoring data up to T-1 (Yesterday).
            # We set snapshot_date to exactly 00:00:00 of the execution day.
            # The SharedFeatureGenerator uses strictly "< snapshot_date".
            self.snapshot_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            os.makedirs(self.feature_matrix_root_dir, exist_ok=True)
            logging.info(
                "InferenceInputFeatureMatrixBuilderConfig initialized. Snapshot date anchor: %s", 
                self.snapshot_date
            )

        except Exception as e:
            logging.exception("Error initializing InferenceInputFeatureMatrixBuilderConfig.")
            raise CustomException(e, sys) from e


class InferenceValidationConfig:
    def __init__(self, inference_pipeline_config: InferencePipelineConfig) -> None:
        try:
            # Root directory for this specific component
            self.validator_root_dir: str = os.path.join(
                inference_pipeline_config.root_dir,
                constants.INFERENCE_VALIDATOR_ROOT_DIR_NAME,
            )

            self.report_file_path: str = os.path.join(
                self.validator_root_dir,
                constants.INFERENCE_VALIDATOR_REPORT_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.validator_root_dir,
                constants.INFERENCE_VALIDATOR_METADATA_FILE_NAME
            )

            # Pre-create the directory structure for safe local I/O
            os.makedirs(self.validator_root_dir, exist_ok=True)
            logging.info("InferenceValidatorConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing InferenceValidatorConfig.")
            raise CustomException(e, sys) from e


class ReportGenerationConfig:
    """
    Configuration for the Inference Publisher (Report Generator) component.
    Defines the probability threshold for churn classification and output 
    paths for the CSV, Parquet, and JSON artifacts.
    """
    def __init__(self, inference_pipeline_config: InferencePipelineConfig) -> None:
        try:
            self.run_id: str = inference_pipeline_config.run_id
            self.report_generator_root_dir: str = os.path.join(
                inference_pipeline_config.root_dir,
                constants.INFERENCE_REPORT_GENERATOR_ROOT_DIR_NAME,
            )

            self.csv_report_path: str = os.path.join(
                self.report_generator_root_dir,
                constants.INFERENCE_REPORT_GENERATOR_CSV_FILE_NAME,
            )
            self.telemetry_log_path: str = os.path.join(
                self.report_generator_root_dir,
                constants.INFERENCE_REPORT_GENERATOR_TELEMETRY_FILE_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.report_generator_root_dir,
                constants.INFERENCE_REPORT_GENERATOR_METADATA_FILE_NAME,
            )
            
            self.probability_threshold: float = constants.INFERENCE_REPORT_GENERATOR_PROBABILITY_THRESHOLD

            os.makedirs(self.report_generator_root_dir, exist_ok=True)
            logging.info("InferenceReportGeneratorConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing InferenceReportGeneratorConfig.")
            raise CustomException(e, sys) from e


class ReportPublishingConfig:
    """
    Configuration for the Inference Report Publisher component.
    Defines S3 base URIs for partitioned uploads and local paths for metadata tracking.
    """
    def __init__(self, inference_pipeline_config: InferencePipelineConfig) -> None:
        try:
            self.run_id: str = inference_pipeline_config.run_id
            self.report_publisher_root_dir: str = os.path.join(
                inference_pipeline_config.root_dir,
                constants.INFERENCE_REPORT_PUBLISHER_ROOT_DIR_NAME,
            )
            
            self.metadata_file_path: str = os.path.join(
                self.report_publisher_root_dir,
                constants.INFERENCE_REPORT_PUBLISHER_METADATA_FILE_NAME,
            )

            # S3 Base URIs (will be dynamically appended with Hive partitions during execution)
            self.s3_business_reports_base_uri: str = f"s3://{constants.S3_BUCKET_NAME}/{constants.S3_INFERENCE_BUSINESS_REPORTS_DIR}"
            self.s3_telemetry_logs_base_uri: str = f"s3://{constants.S3_BUCKET_NAME}/{constants.S3_INFERENCE_MLOPS_TELEMETRY_DIR}"

            os.makedirs(self.report_publisher_root_dir, exist_ok=True)
            logging.info("InferenceReportPublisherConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing InferenceReportPublisherConfig.")
            raise CustomException(e, sys) from e