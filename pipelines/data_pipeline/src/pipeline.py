import sys
from typing import Callable, TypeVar

from shared_core import constants
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
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging

T = TypeVar("T")


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
        - Synchronize generated artifacts to AWS S3.
        - Provide centralized logging and exception handling.
    """

    def __init__(self) -> None:
        """
        Initialize pipeline configuration.
        """
        try:
            logging.info("Initializing Data Pipeline.")
            self.pipeline_config = DataPipelineConfig()
        except Exception as exc:
            logging.exception("Failed to initialize Data Pipeline.")
            raise CustomException(exc, sys) from exc

    @staticmethod
    def _execute_phase(
        phase_name: str,
        operation: Callable[[], T],
    ) -> T:
        """
        Execute a pipeline phase with standardized logging and exception handling.

        Args:
            phase_name: Human-readable phase name.
            operation: Callable responsible for executing the phase.

        Returns:
            The artifact or result returned by the phase.

        Raises:
            CustomException: If phase execution fails.
        """
        try:
            logging.info(">>> Starting %s", phase_name)
            result = operation()
            logging.info("<<< %s completed successfully.", phase_name)
            return result
        except Exception as exc:
            logging.exception("%s failed.", phase_name)
            raise CustomException(exc, sys) from exc

    def _run_data_extraction(self) -> DataExtractorArtifact:
        """
        Execute data extraction stage.

        Returns:
            DataExtractorArtifact
        """

        def operation() -> DataExtractorArtifact:
            config = DataExtractorConfig(self.pipeline_config)
            return DataExtractor(config=config).run()

        return self._execute_phase("Phase 1: Data Extraction", operation)

    def _run_data_validation(
        self,
        extractor_artifact: DataExtractorArtifact,
    ) -> DataValidationArtifact:
        """
        Execute data validation stage.

        Args:
            extractor_artifact: Data extraction artifact.

        Returns:
            DataValidationArtifact
        """

        def operation() -> DataValidationArtifact:
            config = DataValidationConfig(self.pipeline_config)

            return DataValidation(
                config=config,
                extractor_artifact=extractor_artifact,
            ).run()

        return self._execute_phase("Phase 2: Data Validation", operation)

    def _run_data_transformation(
        self,
        extractor_artifact: DataExtractorArtifact,
    ) -> DataTransformationArtifact:
        """
        Execute data transformation stage.

        Args:
            extractor_artifact: Data extraction artifact.

        Returns:
            DataTransformationArtifact
        """

        def operation() -> DataTransformationArtifact:
            config = DataTransformationConfig(self.pipeline_config)

            return DataTransformation(
                config=config,
                extractor_artifact=extractor_artifact,
            ).run()

        return self._execute_phase("Phase 3: Data Transformation", operation)

    def _run_data_loading(
        self,
        transformation_artifact: DataTransformationArtifact,
    ) -> DataLoadingArtifact:
        """
        Execute data loading stage.

        Args:
            transformation_artifact: Data transformation artifact.

        Returns:
            DataLoadingArtifact
        """

        def operation() -> DataLoadingArtifact:
            config = DataLoadingConfig(self.pipeline_config)

            return DataLoading(
                config=config,
                transformer_artifact=transformation_artifact,
            ).run()

        return self._execute_phase("Phase 4: Data Loading", operation)

    def _sync_artifacts(self) -> None:
        """
        Synchronize pipeline artifacts to AWS S3.
        """

        def operation() -> None:
            s3_bucket_url = (
                f"s3://{constants.S3_BUCKET_NAME}/"
                f"{constants.ARTIFACT_DIR_NAME}/"
                f"{constants.DATA_PIPELINE_ROOT_DIR_NAME}"
            )

            S3Sync().sync_folder_to_s3(
                folder=constants.ARTIFACT_DIR_NAME,
                aws_bucket_url=s3_bucket_url,
            )

        self._execute_phase("Phase 5: Artifact Sync (S3)", operation)

    @staticmethod
    def _is_data_valid(
        validation_artifact: DataValidationArtifact,
    ) -> bool:
        """
        Determine whether the dataset passed validation.

        Args:
            validation_artifact: Validation artifact.

        Returns:
            True if validation passed, otherwise False.
        """
        return bool(getattr(validation_artifact, "is_valid", False))

    def run(self) -> None:
        """
        Execute the complete Data Pipeline.

        Raises:
            CustomException: If a critical pipeline failure occurs.
        """
        try:
            logging.info("=" * 60)
            logging.info("STARTING DATA PIPELINE EXECUTION")
            logging.info("=" * 60)

            extractor_artifact = self._run_data_extraction()

            validation_artifact = self._run_data_validation(
                extractor_artifact=extractor_artifact
            )

            if self._is_data_valid(validation_artifact):
                logging.info(
                    "Gate Check Passed: Data validation successful."
                )

                transformation_artifact = self._run_data_transformation(
                    extractor_artifact=extractor_artifact
                )

                self._run_data_loading(
                    transformation_artifact=transformation_artifact
                )

                self._sync_artifacts()
            else:
                logging.warning(
                    "Gate Check Failed: Data validation unsuccessful. "
                    "Skipping transformation, loading, and artifact sync."
                )

            logging.info("=" * 60)
            logging.info("DATA PIPELINE EXECUTION COMPLETED")
            logging.info("=" * 60)

        except Exception as exc:
            logging.exception(
                "Critical failure during Data Pipeline execution."
            )
            raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    try:
        DataPipeline().run()
    except Exception:
        logging.critical(
            "Pipeline execution terminated due to an unrecoverable error.",
            exc_info=True,
        )
        sys.exit(1)