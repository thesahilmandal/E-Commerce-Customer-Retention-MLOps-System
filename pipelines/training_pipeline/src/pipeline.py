import sys
from typing import Callable, TypeVar

from pipelines.training_pipeline.src.components.data_ingestion import DataIngestion
from pipelines.training_pipeline.src.components.feature_transformation import (
    FeatureTransformation,
)
from pipelines.training_pipeline.src.components.model_evaluation import (
    ModelEvaluation,
)
from pipelines.training_pipeline.src.components.model_registration import (
    ModelRegistration,
)
from pipelines.training_pipeline.src.components.model_training import ModelTraining
from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataIngestionArtifact,
    FeatureTransformationArtifact,
    ModelEvaluationArtifact,
    ModelRegistrationArtifact,
    ModelTrainingArtifact,
)
from pipelines.training_pipeline.src.entity.config_entity import (
    DataIngestionConfig,
    FeatureTransformationConfig,
    ModelEvaluationConfig,
    ModelRegistrationConfig,
    ModelTrainingConfig,
    PipelineConfig,
)
from shared_core import constants
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging

T = TypeVar("T")


class TrainingPipeline:
    """
    Orchestrates the end-to-end Continuous Training (CT) pipeline.

    Pipeline Stages:
        1. Data Ingestion
        2. Feature Transformation
        3. Model Training
        4. Model Evaluation
        5. Model Registration (conditional)
        6. Artifact Synchronization (S3)

    Responsibilities:
        - Execute pipeline stages sequentially.
        - Pass artifacts safely between stages.
        - Enforce deployment gatekeeping based on evaluation results.
        - Synchronize generated artifacts to AWS S3.
        - Provide centralized logging and exception handling.
    """

    def __init__(self) -> None:
        """
        Initialize pipeline configuration.
        """
        try:
            logging.info("Initializing Training Pipeline.")
            self.pipeline_config = PipelineConfig()
        except Exception as exc:
            logging.exception("Failed to initialize Training Pipeline.")
            raise CustomException(exc, sys) from exc

    @staticmethod
    def _execute_phase(
        phase_name: str,
        operation: Callable[[], T],
    ) -> T:
        """
        Execute a pipeline phase with standardized logging and exception handling.

        Args:
            phase_name: Name of the pipeline phase.
            operation: Callable responsible for executing the phase.

        Returns:
            Result returned by the phase execution.

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

    def _run_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute data ingestion stage.

        Returns:
            DataIngestionArtifact
        """

        def operation() -> DataIngestionArtifact:
            config = DataIngestionConfig(self.pipeline_config)
            return DataIngestion(config=config).run()

        return self._execute_phase("Phase 1: Data Ingestion", operation)

    def _run_feature_transformation(
        self,
        ingestion_artifact: DataIngestionArtifact,
    ) -> FeatureTransformationArtifact:
        """
        Execute feature transformation stage.

        Args:
            ingestion_artifact: Data ingestion artifact.

        Returns:
            FeatureTransformationArtifact
        """

        def operation() -> FeatureTransformationArtifact:
            config = FeatureTransformationConfig(self.pipeline_config)

            return FeatureTransformation(
                config=config,
                ingestion_artifact=ingestion_artifact,
            ).run()

        return self._execute_phase(
            "Phase 2: Feature Transformation",
            operation,
        )

    def _run_model_training(
        self,
        transformation_artifact: FeatureTransformationArtifact,
    ) -> ModelTrainingArtifact:
        """
        Execute model training stage.

        Args:
            transformation_artifact: Feature transformation artifact.

        Returns:
            ModelTrainingArtifact
        """

        def operation() -> ModelTrainingArtifact:
            config = ModelTrainingConfig(self.pipeline_config)

            return ModelTraining(
                config=config,
                transformation_artifact=transformation_artifact,
            ).run()

        return self._execute_phase("Phase 3: Model Training", operation)

    def _run_model_evaluation(
        self,
        trainer_artifact: ModelTrainingArtifact,
        ingestion_artifact: DataIngestionArtifact,
    ) -> ModelEvaluationArtifact:
        """
        Execute model evaluation stage.

        Args:
            trainer_artifact: Model training artifact.
            ingestion_artifact: Data ingestion artifact.

        Returns:
            ModelEvaluationArtifact
        """

        def operation() -> ModelEvaluationArtifact:
            config = ModelEvaluationConfig(self.pipeline_config)

            return ModelEvaluation(
                config=config,
                trainer_artifact=trainer_artifact,
                ingestion_artifact=ingestion_artifact,
            ).run()

        return self._execute_phase("Phase 4: Model Evaluation", operation)

    def _run_model_registration(
        self,
        trainer_artifact: ModelTrainingArtifact,
        evaluation_artifact: ModelEvaluationArtifact,
        transformation_artifact: FeatureTransformationArtifact,
    ) -> ModelRegistrationArtifact:
        """
        Execute model registration stage.

        Args:
            trainer_artifact: Model training artifact.
            evaluation_artifact: Model evaluation artifact.
            transformation_artifact: Feature transformation artifact.

        Returns:
            ModelRegistrationArtifact
        """

        def operation() -> ModelRegistrationArtifact:
            config = ModelRegistrationConfig(self.pipeline_config)

            return ModelRegistration(
                config=config,
                transformation_artifact=transformation_artifact,
                trainer_artifact=trainer_artifact,
                evaluation_artifact=evaluation_artifact,
            ).run()

        return self._execute_phase("Phase 5: Model Registration", operation)

    def _sync_artifacts(self) -> None:
        """
        Synchronize pipeline artifacts to AWS S3.
        """

        def operation() -> None:
            s3_bucket_url = (
                f"s3://{constants.S3_BUCKET_NAME}/"
                f"{constants.ARTIFACT_DIR_NAME}/"
                f"{constants.TRAINING_PIPELINE_ROOT_DIR_NAME}"
            )

            S3Sync().sync_folder_to_s3(
                folder=constants.ARTIFACT_DIR_NAME,
                aws_bucket_url=s3_bucket_url,
            )

        self._execute_phase("Phase 6: Artifact Sync (S3)", operation)

    @staticmethod
    def _is_model_approved(
        evaluation_artifact: ModelEvaluationArtifact,
    ) -> bool:
        """
        Check whether the model has been approved for registration.

        Args:
            evaluation_artifact: Model evaluation artifact.

        Returns:
            True if approved, otherwise False.
        """
        return bool(
            getattr(evaluation_artifact, "approval_status", False)
        )

    def run(self) -> None:
        """
        Execute the complete Continuous Training pipeline.

        Raises:
            CustomException: If a critical pipeline failure occurs.
        """
        try:
            logging.info("=" * 60)
            logging.info("STARTING CONTINUOUS TRAINING (CT) PIPELINE")
            logging.info("=" * 60)

            ingestion_artifact = self._run_data_ingestion()

            transformation_artifact = self._run_feature_transformation(
                ingestion_artifact=ingestion_artifact
            )

            trainer_artifact = self._run_model_training(
                transformation_artifact=transformation_artifact
            )

            evaluation_artifact = self._run_model_evaluation(
                trainer_artifact=trainer_artifact,
                ingestion_artifact=ingestion_artifact,
            )

            if self._is_model_approved(evaluation_artifact):
                logging.info(
                    "Gate Check Passed: Model approved for registration."
                )

                self._run_model_registration(
                    trainer_artifact=trainer_artifact,
                    evaluation_artifact=evaluation_artifact,
                    transformation_artifact=transformation_artifact,
                )
            else:
                logging.warning(
                    "Gate Check Failed: Challenger model rejected. "
                    "Skipping Model Registration stage."
                )

            self._sync_artifacts()

            logging.info("=" * 60)
            logging.info(
                "CONTINUOUS TRAINING PIPELINE EXECUTION COMPLETED"
            )
            logging.info("=" * 60)

        except Exception as exc:
            logging.exception(
                "Critical failure during Training Pipeline execution."
            )
            raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    try:
        TrainingPipeline().run()
    except Exception:
        logging.critical(
            "Pipeline execution terminated due to an unrecoverable error.",
            exc_info=True,
        )
        sys.exit(1)