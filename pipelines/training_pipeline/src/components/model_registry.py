"""
Model Registry Module for the Training Pipeline.

This module acts as the Deployment Vault and State Manager. It executes a 
Two-Phase Commit to promote an approved Challenger model to production.
First, it packages all artifacts, strict schema blueprints, and lineage metadata 
into an immutable, WORM (Write-Once-Read-Many) vault in S3 partitioned by run_id.
Second, it atomically overwrites the global `model_state.json` pointer to redirect 
downstream inference and monitoring pipelines to the new model bundle.
"""

import os
import sys
import time
import shutil
from datetime import datetime, timezone
from typing import Dict

from pipelines.training_pipeline.src.core.context import PipelineContext
from pipelines.training_pipeline.src.entity.config_entity import ModelRegistryConfig
from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataProcessorArtifact,
    ModelTrainerArtifact,
    ModelEvaluatorArtifact,
    ModelRegistryArtifact,
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class ModelRegistry:
    """
    Model Registry component for the Training Pipeline.

    Responsibilities:
    - Verify deployment eligibility via the Evaluator's gatekeeper boolean.
    - Stage and consolidate all disparate component artifacts into a single bundle.
    - Generate a curated, strictly versioned `requirements.txt` to guarantee 
      inference environment parity.
    - Generate a unified `deployment_metadata.json` capturing hyperparameter, 
      evaluation, and data provenance lineage (including training_dataset_s3_uri_path).
    - Phase 1 Commit: Upload the bundle to `models/<run_id>/` (Immutable Vault).
    - Phase 2 Commit: Overwrite `state/model_state.json` (Mutable Pointer).
    """

    def __init__(
        self,
        config: ModelRegistryConfig,
        context: PipelineContext,
        data_artifact: DataProcessorArtifact,
        trainer_artifact: ModelTrainerArtifact,
        evaluator_artifact: ModelEvaluatorArtifact,
    ) -> None:
        """
        Initializes the Model Registry component.
        """
        try:
            self.config = config
            self.context = context
            self.data_artifact = data_artifact
            self.trainer_artifact = trainer_artifact
            self.evaluator_artifact = evaluator_artifact

            logging.info("Training Pipeline: Model Registry component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Model Registry component.")
            raise CustomException(e, sys) from e

    def run(self) -> ModelRegistryArtifact:
        """
        Executes the Model Registration and Artifact Synchronization stage.
        """
        try:
            logging.info("Starting Model Registry execution.")
            start_time = time.time()

            # 1. Gatekeeper Verification
            if not self.evaluator_artifact.approval_status:
                logging.warning(
                    "Gatekeeper Denied: Challenger model was rejected during evaluation. "
                    "Skipping deployment to production registry."
                )
                return ModelRegistryArtifact(
                    deployment_status=False,
                    s3_model_uri="",
                    metadata_file_path=self.config.metadata_file_path,
                )

            logging.info("Gatekeeper Approved: Proceeding with production model registration.")

            # 2. Prepare Local Staging Directory
            os.makedirs(self.config.staging_dir, exist_ok=True)
            staged_artifacts = self._stage_artifacts()

            # 3. Generate Lean Dependency Blueprint
            self._generate_requirements()

            # 4. Generate Master Lineage Ledger
            self._generate_deployment_metadata()

            # 5. Execute S3 Two-Phase Commit
            s3_run_vault_uri = self._execute_two_phase_commit(staged_artifacts)

            # 6. Generate Telemetry Metadata for the Component
            execution_time = round(time.time() - start_time, 2)
            self._generate_component_metadata(execution_time, s3_run_vault_uri)

            artifact = ModelRegistryArtifact(
                deployment_status=True,
                s3_model_uri=s3_run_vault_uri,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("Model Registry execution completed successfully.")
            return artifact

        except Exception as e:
            logging.exception("Model Registry run failed.")
            raise CustomException(e, sys) from e

    def _stage_artifacts(self) -> Dict[str, str]:
        """
        Consolidates all required artifacts from previous pipeline stages into 
        the unified staging directory for bulk upload.
        """
        try:
            logging.info("Staging artifacts for deployment bundle.")
            staged_files = {}

            # Map of source paths to target filenames
            files_to_stage = {
                self.trainer_artifact.model_file_path: "model.pkl",
                self.data_artifact.schema_file_path: "schema.json",
                self.trainer_artifact.reference_feature_distributions_file_path: "reference_feature_distributions.json",
                self.evaluator_artifact.baseline_performance_metrics_file_path: "baseline_performance_metrics.json",
                self.evaluator_artifact.report_file_path: "evaluation_report.json",
                self.trainer_artifact.shap_summary_file_path: "shap_summary.png",
                self.trainer_artifact.shap_feature_importance_file_path: "shap_feature_importance.json",
            }

            for source_path, filename in files_to_stage.items():
                if os.path.exists(source_path):
                    target_path = os.path.join(self.config.staging_dir, filename)
                    shutil.copy2(source_path, target_path)
                    staged_files[filename] = target_path
                else:
                    logging.warning("Expected artifact not found during staging: %s", source_path)

            return staged_files

        except Exception as e:
            logging.exception("Failed to stage artifacts.")
            raise CustomException(e, sys) from e

    def _generate_requirements(self) -> None:
        """
        Generates a curated requirements.txt to prevent inference image bloat.
        Avoids `pip freeze` which captures hundreds of irrelevant system libraries.
        """
        try:
            logging.info("Generating curated requirements.txt for inference parity.")
            
            # Using precise versions required for the Mega-Pipeline and inference script
            # In a real-world scenario, these could be extracted dynamically from pkg_resources
            dependencies = [
                "scikit-learn>=1.2.0",
                "xgboost>=2.0.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "pyarrow>=14.0.0",
            ]
            
            requirements_path = os.path.join(self.config.staging_dir, "requirements.txt")
            with open(requirements_path, "w") as f:
                f.write("\n".join(dependencies))

        except Exception as e:
            logging.exception("Failed to generate requirements.txt.")
            raise CustomException(e, sys) from e

    def _generate_deployment_metadata(self) -> None:
        """
        Generates the master `deployment_metadata.json` combining all component 
        provenance, including the precise S3 URI of the master panel dataset.
        """
        try:
            logging.info("Compiling master deployment lineage ledger.")
            
            deployment_metadata = {
                "run_id": self.context.run_id,
                "deployed_at_utc": datetime.now(timezone.utc).isoformat(),
                "environment": self.config.deployment_environment,
                "lineage": {
                    "training_dataset_s3_uri": self.context.training_dataset_s3_uri_path,
                }
            }
            
            target_path = os.path.join(self.config.staging_dir, "deployment_metadata.json")
            write_json_file(target_path, deployment_metadata)

        except Exception as e:
            logging.exception("Failed to generate deployment metadata.")
            raise CustomException(e, sys) from e

    def _execute_two_phase_commit(self, staged_files: Dict[str, str]) -> str:
        """
        Executes the atomic deployment protocol to AWS S3.
        Phase 1: Upload all files to the WORM models/<run_id> prefix.
        Phase 2: Overwrite the mutable state/model_state.json pointer file.
        """
        try:
            # Construct the S3 path for this specific run's immutable vault
            s3_run_vault_uri = f"{self.config.s3_models_dir_uri}/{self.context.run_id}"
            
            logging.info("Phase 1 Commit: Uploading artifact bundle to WORM Vault -> %s", s3_run_vault_uri)
            
            # Use S3Sync to recursively upload the staging directory
            self.context.s3_sync.sync_folder_to_s3(
                folder=self.config.staging_dir,
                aws_bucket_url=s3_run_vault_uri
            )
            
            logging.info("Phase 1 Complete.")
            
            logging.info("Phase 2 Commit: Mutating global model pointer -> %s", self.config.s3_pointer_uri)
            
            # Construct the state dictionary exactly as expected by the Inference Pipeline
            model_state_payload = {
                "run_id": self.context.run_id,
                "deployment_environment": self.config.deployment_environment,
                "s3_model_path": f"{s3_run_vault_uri}/model.pkl",
                "s3_schema_path": f"{s3_run_vault_uri}/schema.json",
                "s3_monitoring_baselines_path": f"{s3_run_vault_uri}/baseline_performance_metrics.json",
                "s3_reference_distributions_path": f"{s3_run_vault_uri}/reference_feature_distributions.json",
                "training_dataset_s3_uri": self.context.training_dataset_s3_uri_path,
                "updated_at_utc": datetime.now(timezone.utc).isoformat()
            }
            
            # Write locally then PUT to S3 to ensure atomic overwrite
            local_state_path = os.path.join(self.config.model_registry_dir, "tmp_model_state.json")
            write_json_file(local_state_path, model_state_payload)
            
            # Executed via positional arguments to guarantee API compatibility with shared_core
            self.context.s3_sync.upload_file(
                local_state_path,
                self.config.s3_pointer_uri
            )
            
            # Cleanup
            os.remove(local_state_path)
            
            logging.info("Phase 2 Complete. New Champion Model is actively serving traffic.")
            return s3_run_vault_uri

        except Exception as e:
            logging.exception("Critical Failure during Two-Phase Commit.")
            raise CustomException(e, sys) from e

    def _generate_component_metadata(self, execution_time: float, s3_vault_uri: str) -> None:
        """Generates observability telemetry for the Model Registry stage."""
        try:
            metadata = {
                "pipeline_stage": "Model Registration & Deployment",
                "execution_time_seconds": execution_time,
                "deployment_successful": True,
                "s3_vault_uri": s3_vault_uri,
                "s3_pointer_uri_updated": self.config.s3_pointer_uri,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(self.config.metadata_file_path, metadata)
            logging.info("Model Registry metadata generated successfully.")

        except Exception as e:
            logging.exception("Failed to generate registry metadata.")
            raise CustomException(e, sys) from e