import sys
from typing import Any

from dotenv import load_dotenv

from pipelines.inference_pipeline.src import constants
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging

from pipelines.inference_pipeline.src.entity.config_entity import (
    FeatureMatrixGenerationConfig,
    InferencePipelineConfig,
    InferenceValidationConfig,
    ModelLoadingConfig,
    ReportGenerationConfig,
    ReportPublishingConfig,
)
from pipelines.inference_pipeline.src.components.feature_matrix_generation import (
    FeatureMatrixGeneration,
)
from pipelines.inference_pipeline.src.components.inference_validation import (
    InferenceValidation,
)
from pipelines.inference_pipeline.src.components.model_loading import ModelLoading
from pipelines.inference_pipeline.src.components.report_generation import (
    ReportGeneration,
)
from pipelines.inference_pipeline.src.components.report_publishing import (
    ReportPublishing,
)

load_dotenv()


class InferencePipeline:
    """
    Orchestrates the end-to-end inference workflow.

    Pipeline Stages:
        1. Model Loading
        2. Feature Matrix Generation
        3. Feature Validation
        4. Report Generation
        5. Report Publishing
        6. Artifact Synchronization to AWS S3
    """

    def __init__(self) -> None:
        """
        Initialize the inference pipeline configuration.
        """
        try:
            logging.info("Initializing inference pipeline.")
            self.pipeline_config = InferencePipelineConfig()
        except Exception as exc:
            logging.exception(
                "Failed to initialize inference pipeline configuration."
            )
            raise CustomException(exc, sys) from exc

    @staticmethod
    def _log_artifact(artifact: Any) -> None:
        """
        Log pipeline artifact details.

        Args:
            artifact: Pipeline stage artifact.
        """
        logging.debug("Generated artifact: %s", artifact)

    def _run_model_loader(self) -> Any:
        """
        Execute the model loading stage.

        Returns:
            Any: Model loading artifact.
        """
        try:
            logging.info("Phase 1/6 - Model Loading started.")

            config = ModelLoadingConfig(self.pipeline_config)
            artifact = ModelLoading(config).run()

            self._log_artifact(artifact)

            logging.info("Phase 1/6 - Model Loading completed successfully.")
            return artifact

        except Exception as exc:
            logging.exception("Model Loading stage failed.")
            raise CustomException(exc, sys) from exc

    def _run_feature_matrix_generation(self) -> Any:
        """
        Execute the feature matrix generation stage.

        Returns:
            Any: Feature matrix generation artifact.
        """
        try:
            logging.info("Phase 2/6 - Feature Matrix Generation started.")

            config = FeatureMatrixGenerationConfig(self.pipeline_config)
            artifact = FeatureMatrixGeneration(config).run()

            self._log_artifact(artifact)

            logging.info(
                "Phase 2/6 - Feature Matrix Generation completed successfully."
            )
            return artifact

        except Exception as exc:
            logging.exception("Feature Matrix Generation stage failed.")
            raise CustomException(exc, sys) from exc

    def _run_feature_validation(
        self,
        model_loading_artifact: Any,
        feature_matrix_artifact: Any,
    ) -> Any:
        """
        Execute the feature validation stage.

        Args:
            model_loading_artifact: Output from model loading stage.
            feature_matrix_artifact: Output from feature matrix generation stage.

        Returns:
            Any: Feature validation artifact.
        """
        try:
            logging.info("Phase 3/6 - Feature Validation started.")

            config = InferenceValidationConfig(self.pipeline_config)

            artifact = InferenceValidation(config).run(
                model_loading_artifact,
                feature_matrix_artifact,
            )

            self._log_artifact(artifact)

            logging.info(
                "Phase 3/6 - Feature Validation completed successfully."
            )
            return artifact

        except Exception as exc:
            logging.exception("Feature Validation stage failed.")
            raise CustomException(exc, sys) from exc

    def _run_report_generation(
        self,
        model_loading_artifact: Any,
        feature_matrix_artifact: Any,
    ) -> Any:
        """
        Execute the report generation stage.

        Args:
            model_loading_artifact: Output from model loading stage.
            feature_matrix_artifact: Output from feature matrix generation stage.

        Returns:
            Any: Report generation artifact.
        """
        try:
            logging.info("Phase 4/6 - Report Generation started.")

            config = ReportGenerationConfig(self.pipeline_config)

            artifact = ReportGeneration(
                config=config,
                loader_artifact=model_loading_artifact,
                builder_artifact=feature_matrix_artifact,
            ).run()

            self._log_artifact(artifact)

            logging.info(
                "Phase 4/6 - Report Generation completed successfully."
            )
            return artifact

        except Exception as exc:
            logging.exception("Report Generation stage failed.")
            raise CustomException(exc, sys) from exc

    def _run_report_publishing(
        self,
        report_generation_artifact: Any,
    ) -> Any:
        """
        Execute the report publishing stage.

        Args:
            report_generation_artifact: Output from report generation stage.

        Returns:
            Any: Report publishing artifact.
        """
        try:
            logging.info("Phase 5/6 - Report Publishing started.")

            config = ReportPublishingConfig(self.pipeline_config)

            artifact = ReportPublishing(
                config,
                report_generation_artifact,
            ).run()

            self._log_artifact(artifact)

            logging.info(
                "Phase 5/6 - Report Publishing completed successfully."
            )
            return artifact

        except Exception as exc:
            logging.exception("Report Publishing stage failed.")
            raise CustomException(exc, sys) from exc

    def _sync_artifacts_to_s3(self) -> None:
        """
        Synchronize generated artifacts to AWS S3.
        """
        try:
            logging.info("Phase 6/6 - Artifact Synchronization started.")

            s3_path = (
                f"s3://{constants.S3_BUCKET_NAME}/"
                f"{constants.ARTIFACT_DIR_NAME}/"
                f"{constants.INFERENCE_PIPELINE_ROOT_DIR_NAME}/"
                f"{self.pipeline_config.run_id}"
            )

            S3Sync().sync_folder_to_s3(
                folder=self.pipeline_config.root_dir,
                aws_bucket_url=s3_path,
            )

            logging.info(
                "Phase 6/6 - Artifact Synchronization completed successfully."
            )

        except Exception as exc:
            logging.exception("Artifact Synchronization stage failed.")
            raise CustomException(exc, sys) from exc

    def run(self) -> None:
        """
        Execute the complete inference pipeline.
        """
        try:
            logging.info("=" * 80)
            logging.info("INFERENCE PIPELINE EXECUTION STARTED")
            logging.info("=" * 80)

            model_loading_artifact = self._run_model_loader()

            feature_matrix_artifact = (
                self._run_feature_matrix_generation()
            )

            validation_artifact = self._run_feature_validation(
                model_loading_artifact=model_loading_artifact,
                feature_matrix_artifact=feature_matrix_artifact,
            )

            if not validation_artifact.is_valid:
                logging.warning(
                    "Feature validation failed. "
                    "Skipping downstream pipeline stages."
                )
                return

            report_generation_artifact = self._run_report_generation(
                model_loading_artifact=model_loading_artifact,
                feature_matrix_artifact=feature_matrix_artifact,
            )

            self._run_report_publishing(
                report_generation_artifact=report_generation_artifact
            )

            self._sync_artifacts_to_s3()

            logging.info("=" * 80)
            logging.info("INFERENCE PIPELINE EXECUTION COMPLETED")
            logging.info("=" * 80)

        except Exception as exc:
            logging.exception(
                "Critical failure during inference pipeline execution."
            )
            raise CustomException(exc, sys) from exc


def main() -> int:
    """
    Application entry point.

    Returns:
        int: Process exit code.
    """
    try:
        pipeline = InferencePipeline()
        pipeline.run()
        return 0

    except Exception:
        logging.critical(
            "Inference pipeline execution terminated unexpectedly.",
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())