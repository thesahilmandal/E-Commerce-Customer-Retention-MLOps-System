"""
Continuous Training (CT) Pipeline Orchestrator.

This module acts as the master execution graph for the Training Pipeline.
It orchestrates Data Ingestion, Feature Transformation, Model Training, 
Model Evaluation, and Model Registration in a strict, acyclic sequence.
It enforces production gatekeeping by halting deployment if the Challenger 
model fails business evaluation criteria, and optimizes cloud storage by 
synchronizing only the current run's artifacts to S3.
"""

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
        1. Data Ingestion (Out-Of-Time Splitting)
        2. Feature Transformation (Stateful Schema Enforcement)
        3. Model Training (Optuna Tuning & Isotonic Calibration)
        4. Model Evaluation (Champion vs. Challenger Hysteresis)
        5. Model Registration (Atomic Deployment - Conditional)
        6. Artifact Synchronization (Run-specific S3 Sync)

    Responsibilities:
        - Execute pipeline stages sequentially via functional wrappers.
        - Pass immutable Dataclass artifacts safely between stages.
        - Enforce deployment gatekeeping based on business evaluation results.
        - Synchronize generated artifacts to AWS S3, restricting payload to the current run.
        - Provide centralized logging and robust exception handling.
    """

    def __init__(self) -> None:
        """
        Initialize the pipeline orchestration context and configuration.
        """
        try:
            logging.info("Initializing Training Pipeline Context.")
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
            phase_name: Standardized name of the pipeline phase.
            operation: Callable responsible for executing the phase component.

        Returns:
            Result returned by the phase execution (typically an Artifact Dataclass).

        Raises:
            CustomException: If the underlying component execution fails.
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
        Execute the data ingestion and bitemporal Out-Of-Time (OOT) splitting stage.
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
        Execute the stateful feature transformation and schema enforcement stage.
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
        Execute the model training, tuning, and probability calibration stage.
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
        Execute the Champion vs. Challenger evaluation and business ROI translation stage.
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
        Execute the atomic model registration and S3 pointer deployment stage.
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
        Synchronize specific pipeline run artifacts to AWS S3.
        Restricts synchronization strictly to the current run to prevent 
        bandwidth and storage bloat from uploading historical local artifacts.
        """
        def operation() -> None:
            s3_bucket_url = (
                f"s3://{constants.S3_BUCKET_NAME}/"
                f"{constants.ARTIFACT_DIR_NAME}/"
                f"{constants.TRAINING_PIPELINE_ROOT_DIR_NAME}/"
                f"{self.pipeline_config.run_id}"
            )

            S3Sync().sync_folder_to_s3(
                folder=self.pipeline_config.root_dir,
                aws_bucket_url=s3_bucket_url,
            )

        self._execute_phase("Phase 6: Artifact Sync (S3)", operation)

    @staticmethod
    def _is_model_approved(
        evaluation_artifact: ModelEvaluationArtifact,
    ) -> bool:
        """
        Inspect the evaluation gatekeeper boolean to determine deployment eligibility.
        """
        return bool(
            getattr(evaluation_artifact, "approval_status", False)
        )

    def run(self) -> None:
        """
        Execute the complete Continuous Training (CT) execution graph.

        Raises:
            CustomException: If a critical pipeline failure occurs at any stage.
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
                    "Skipping Model Registration stage to protect production."
                )

            # Artifacts (logs, metrics, local schema blueprints) are synced regardless 
            # of deployment status to maintain a complete historical audit trail.
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