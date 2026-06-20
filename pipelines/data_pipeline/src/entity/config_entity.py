import os
from datetime import datetime, timezone
from typing import Optional

from pipelines.data_pipeline.src import constants
from shared_core.logging.custom_logging import logging


class DataPipelineConfig:
    """
    Base configuration for the Data Pipeline.
    Responsible for creating or resuming a unique run ID and root artifact directory.
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id: str = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        self.root_dir: str = os.path.join(
            constants.ARTIFACT_DIR_NAME,
            constants.DATA_PIPELINE_ROOT_DIR_NAME,
            self.run_id,
        )
        os.makedirs(self.root_dir, exist_ok=True)
        
        if run_id:
            logging.info("Resuming DataPipeline execution. Using Run ID: %s", self.run_id)
        else:
            logging.info("DataPipelineConfig initialized. New Run ID: %s", self.run_id)


class DataExtractorConfig:
    """
    Configuration for the Extractor component.
    Defines local directory paths for extracted data and remote S3 paths for ingestion.
    """

    def __init__(self, data_pipeline_config: DataPipelineConfig) -> None:
        self.extractor_root_dir: str = os.path.join(
            data_pipeline_config.root_dir,
            constants.EXTRACTOR_ROOT_DIR_NAME,
        )
        self.raw_data_dir_path: str = os.path.join(
            self.extractor_root_dir,
            constants.EXTRACTOR_RAW_DATA_DIR_NAME,
        )
        self.raw_data_schema_file_path: str = os.path.join(
            self.extractor_root_dir,
            constants.EXTRACTOR_RAW_DATA_SCHEMA_FILE_NAME,
        )
        self.metadata_file_path: str = os.path.join(
            self.extractor_root_dir,
            constants.EXTRACTOR_METADATA_FILE_NAME,
        )

        self.s3_bucket_name: str = constants.S3_BUCKET_NAME
        self.s3_raw_data_dir: str = constants.S3_RAW_DATA_DIR_NAME
        self.s3_raw_data_uri: str = f"s3://{self.s3_bucket_name}/{self.s3_raw_data_dir}"


class DataValidationConfig:
    """
    Configuration for the Validator component.
    Defines paths for validation reports and the reference schema.
    """

    def __init__(self, data_pipeline_config: DataPipelineConfig) -> None:
        self.validator_root_dir: str = os.path.join(
            data_pipeline_config.root_dir,
            constants.VALIDATOR_ROOT_DIR_NAME,
        )
        self.report_file_path: str = os.path.join(
            self.validator_root_dir,
            constants.VALIDATOR_REPORT_FILE_NAME,
        )
        self.is_valid: Optional[bool] = None
        self.reference_schema_file_path: str = str(constants.REFERENCE_SCHEMA_FILE_PATH)


class DataTransformationConfig:
    """
    Configuration for the Transformer component.
    Defines paths, core business logic parameters (target days, snapshots), and thread limits.
    """

    def __init__(self, data_pipeline_config: DataPipelineConfig) -> None:
        self.transformer_root_dir: str = os.path.join(
            data_pipeline_config.root_dir,
            constants.TRANSFORMER_ROOT_DIR_NAME,
        )
        self.metadata_file_path: str = os.path.join(
            self.transformer_root_dir,
            constants.TRANSFORMER_METADATA_FILE_NAME,
        )
        
        # Dedicated temporary directory for DuckDB disk-spilling to avoid concurrent DB overlap
        self.duckdb_temp_dir: str = os.path.join(
            self.transformer_root_dir,
            "tmp",
        )
        
        self.target_days: int = constants.TARGET_DAYS
        self.snapshots: list = constants.SNAPSHOT_DATES
        self.threads: int = constants.COMPUTE_THREADS

        os.makedirs(self.transformer_root_dir, exist_ok=True)
        os.makedirs(self.duckdb_temp_dir, exist_ok=True)


class DataLoadingConfig:
    """
    Configuration for the Loader component.
    Defines local artifact paths for metadata and remote S3 URIs for the feature store.
    """

    def __init__(self, data_pipeline_config: DataPipelineConfig) -> None:
        self.loader_root_dir: str = os.path.join(
            data_pipeline_config.root_dir,
            constants.LOADER_ROOT_DIR_NAME,
        )
        self.metadata_file_path: str = os.path.join(
            self.loader_root_dir,
            constants.LOADER_METADATA_FILE_NAME,
        )
        self.s3_bucket_name: str = constants.S3_BUCKET_NAME
        self.s3_feature_store_dir: str = constants.S3_FEATURE_STORE_DIR_NAME
        self.s3_master_panel_uri: str = (
            f"s3://{self.s3_bucket_name}/{self.s3_feature_store_dir}/"
            f"{constants.LOADER_MASTER_PANEL_LOCAL_FILE_NAME}"
        )

        os.makedirs(self.loader_root_dir, exist_ok=True)