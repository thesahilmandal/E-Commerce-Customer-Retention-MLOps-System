import os
import sys
from typing import Optional

from pipelines.data_pipeline.src import constants
from pipelines.data_pipeline.src.components.data_extractor import DataExtractor
from pipelines.data_pipeline.src.components.data_loading import DataLoading
from pipelines.data_pipeline.src.components.data_transformation import (
    DataTransformation,
)
from pipelines.data_pipeline.src.components.data_validation import DataValidation
from pipelines.data_pipeline.src.entity.artifact_entity import (
    DataExtractorArtifact,
    DataLoadingArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from pipelines.data_pipeline.src.entity.config_entity import (
    DataExtractorConfig,
    DataLoadingConfig,
    DataPipelineConfig,
    DataTransformationConfig,
    DataValidationConfig,
)
from shared_core.cloud.s3_operations import S3Sync
from shared_core.features.shared_feature import SharedFeatureGenerator
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import read_json_file


class DataPipeline:
    """
    Orchestrates the end-to-end Data Pipeline.

    Pipeline Stages:
        1. Data Extraction
        2. Data Validation
        3. Data Transformation
        4. Data Loading
        5. Artifact Synchronization (S3)

    Responsibilities:
        - Execute pipeline stages sequentially.
        - Pass artifacts safely between stages.
        - Enforce data quality gatekeeping through validation.
        - Provide idempotency for graceful resumption on transient failures.
        - Inject dependencies (e.g., S3Sync) to downstream components.
        - Synchronize generated artifacts to AWS S3.
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        """
        Initialize pipeline configuration and inject shared services.
        
        Args:
            run_id (Optional[str]): Existing run_id to resume a failed pipeline, 
                                    or None to generate a new execution context.
        """
        logging.info("Initializing Data Pipeline Orchestrator.")
        self.pipeline_config = DataPipelineConfig(run_id=run_id)
        
        # Dependency Injection: Centralized cloud synchronization utility
        self.s3_sync = S3Sync()

    def _run_data_extraction(self) -> DataExtractorArtifact:
        """Execute data extraction stage with idempotency check."""
        logging.info(">>> Starting Phase 1: Data Extraction")
        config = DataExtractorConfig(self.pipeline_config)

        # Idempotency Check
        if os.path.exists(config.metadata_file_path):
            logging.info("Phase 1 Artifacts found. Skipping execution.")
            return DataExtractorArtifact(
                raw_data_dir_path=config.raw_data_dir_path,
                raw_data_schema_file_path=config.raw_data_schema_file_path,
                metadata_file_path=config.metadata_file_path,
            )

        extractor = DataExtractor(config=config, s3_sync=self.s3_sync)
        artifact = extractor.run()
        logging.info("<<< Phase 1: Data Extraction completed successfully.")
        return artifact

    def _run_data_validation(
        self,
        extractor_artifact: DataExtractorArtifact,
    ) -> DataValidationArtifact:
        """Execute data validation stage with idempotency check."""
        logging.info(">>> Starting Phase 2: Data Validation")
        config = DataValidationConfig(self.pipeline_config)

        # Idempotency Check
        if os.path.exists(config.report_file_path):
            logging.info("Phase 2 Artifacts found. Skipping execution.")
            report = read_json_file(config.report_file_path)
            is_valid = report.get("summary", {}).get("is_valid", False)
            return DataValidationArtifact(
                report_file_path=config.report_file_path,
                is_valid=is_valid,
            )

        validator = DataValidation(
            config=config,
            extractor_artifact=extractor_artifact,
        )
        artifact = validator.run()
        logging.info("<<< Phase 2: Data Validation completed successfully.")
        return artifact

    def _run_data_transformation(
        self,
        extractor_artifact: DataExtractorArtifact,
    ) -> DataTransformationArtifact:
        """Execute data transformation stage with idempotency check."""
        logging.info(">>> Starting Phase 3: Data Transformation")
        config = DataTransformationConfig(self.pipeline_config)
        transformed_path = os.path.join(config.transformer_root_dir, "master_panel.parquet")

        # Idempotency Check
        if os.path.exists(config.metadata_file_path) and os.path.exists(transformed_path):
            logging.info("Phase 3 Artifacts found. Skipping execution.")
            return DataTransformationArtifact(
                transformed_data_file_path=transformed_path,
                metadata_file_path=config.metadata_file_path,
            )

        # Dependency Injection: Centralized feature logic
        feature_generator = SharedFeatureGenerator(
            data_dir=extractor_artifact.raw_data_dir_path,
            is_partitioned=False,
        )

        transformer = DataTransformation(
            config=config,
            extractor_artifact=extractor_artifact,
            feature_generator=feature_generator,
        )
        artifact = transformer.run()
        logging.info("<<< Phase 3: Data Transformation completed successfully.")
        return artifact

    def _run_data_loading(
        self,
        transformation_artifact: DataTransformationArtifact,
    ) -> DataLoadingArtifact:
        """Execute data loading stage with idempotency check."""
        logging.info(">>> Starting Phase 4: Data Loading")
        config = DataLoadingConfig(self.pipeline_config)

        # Idempotency Check
        if os.path.exists(config.metadata_file_path):
            logging.info("Phase 4 Artifacts found. Skipping execution.")
            return DataLoadingArtifact(
                s3_file_uri=config.s3_master_panel_uri,
                metadata_file_path=config.metadata_file_path,
            )

        loader = DataLoading(
            config=config,
            transformer_artifact=transformation_artifact,
            s3_sync=self.s3_sync,
        )
        artifact = loader.run()
        logging.info("<<< Phase 4: Data Loading completed successfully.")
        return artifact
    
    def _sync_artifacts(self) -> None:
        """Synchronize pipeline artifacts to AWS S3."""
        logging.info(">>> Starting Phase 5: Artifact Sync (S3)")
        
        # Append the specific run_id to the S3 bucket URL to prevent directory nesting overlap
        s3_bucket_url = (
            f"s3://{constants.S3_BUCKET_NAME}/"
            f"{constants.ARTIFACT_DIR_NAME}/"
            f"{constants.DATA_PIPELINE_ROOT_DIR_NAME}/"
            f"{self.pipeline_config.run_id}"
        )

        # Sync ONLY the current run's artifacts, not the entire historical folder
        self.s3_sync.sync_folder_to_s3(
            folder=self.pipeline_config.root_dir,
            aws_bucket_url=s3_bucket_url,
        )
        logging.info("<<< Phase 5: Artifact Sync completed successfully.")

    def run(self) -> None:
        """
        Execute the complete Data Pipeline safely.

        Raises:
            CustomException: If an uncaught failure bubbles up from a pipeline stage.
        """
        try:
            logging.info("=" * 60)
            logging.info("STARTING DATA PIPELINE EXECUTION")
            logging.info("=" * 60)

            extractor_artifact = self._run_data_extraction()
            validation_artifact = self._run_data_validation(extractor_artifact)

            if not getattr(validation_artifact, "is_valid", False):
                logging.warning(
                    "Gate Check Failed: Data validation unsuccessful. "
                    "Skipping transformation, loading, and artifact sync."
                )
                return

            logging.info("Gate Check Passed: Data validation successful.")

            transformation_artifact = self._run_data_transformation(extractor_artifact)
            self._run_data_loading(transformation_artifact)

            self._sync_artifacts()

            logging.info("=" * 60)
            logging.info("DATA PIPELINE EXECUTION COMPLETED")
            logging.info("=" * 60)

        except Exception as exc:
            logging.exception("Critical failure during Data Pipeline execution.")
            raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    try:
        # Support for graceful resumption of failed runs via CLI argument
        target_run_id = sys.argv[1] if len(sys.argv) > 1 else None
        pipeline = DataPipeline(run_id=target_run_id)
        pipeline.run()
    except Exception:
        logging.critical(
            "Pipeline execution terminated due to an unrecoverable error.",
            exc_info=True,
        )
        sys.exit(1)