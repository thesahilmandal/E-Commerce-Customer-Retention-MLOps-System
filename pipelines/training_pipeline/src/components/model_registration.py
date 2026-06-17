import os
import sys
import json
import time
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any

from shared_core import constants
from pipelines.training_pipeline.src.entity.config_entity import ModelRegistrationConfig
from pipelines.training_pipeline.src.entity.artifact_entity import (
    FeatureTransformationArtifact,
    ModelTrainingArtifact,
    ModelEvaluationArtifact,
    ModelRegistrationArtifact,
)
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class ModelRegistration:
    """
    Model Registry component for the Training Pipeline.

    Responsibilities:
    - Act as the final deployment execution layer (The Vault).
    - Halt execution cleanly if the Challenger model was rejected by the Evaluator.
    - Generate a fully pinned `requirements.txt` for exact reproducibility.
    - Merge Trainer and Evaluation metadata into a single, unified `metadata.json`.
    - Upload the complete immutable bundle (Model, Schema, Requirements, Metadata) to S3.
    - Execute a transactional, atomic update of `model_state.json` for zero-downtime deployment.
    """

    def __init__(
        self,
        config: ModelRegistrationConfig,
        transformation_artifact: FeatureTransformationArtifact,
        trainer_artifact: ModelTrainingArtifact,
        evaluation_artifact: ModelEvaluationArtifact,
    ) -> None:
        """
        Initializes the Model Registry component.
        """
        try:
            self.config = config
            self.transformation_artifact = transformation_artifact
            self.trainer_artifact = trainer_artifact
            self.evaluation_artifact = evaluation_artifact
            self.s3_sync = S3Sync()

            # Dynamically override legacy config to strictly enforce the new S3 structure
            self.s3_registry_base_uri = f"s3://{constants.S3_BUCKET_NAME}/model_registry"
            self.s3_pointer_uri = f"{self.s3_registry_base_uri}/model_state.json"

            # Create a dedicated local staging directory for bundling before upload
            self.staging_dir = os.path.join(self.config.model_registry_root_dir, "staging")
            os.makedirs(self.staging_dir, exist_ok=True)
            
            logging.info("Training Pipeline: Model Registry component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Model Registry component.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> ModelRegistrationArtifact:
        """
        Executes the registry and deployment pipeline based on evaluation gating.
        """
        try:
            logging.info("Starting Model Registry execution phase.")
            start_time = time.time()

            # 1. The Gatekeeper Check
            approval_status = self.evaluation_artifact.approval_status
            previous_run_id = "None"
            
            if not approval_status:
                logging.warning("Challenger model was REJECTED. Halting deployment sequence.")
                s3_run_dir_uri = "N/A"
                deployment_status = False
            else:
                logging.info("Challenger model APPROVED. Initiating production deployment sequence.")
                
                # Fetch rollback pointer before any state mutations occur
                previous_run_id = self._get_previous_champion_run_id()
                
                # Generate unique, timestamped deployment ID to prevent collisions
                run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                s3_run_dir_uri = f"{self.s3_registry_base_uri}/{run_id}"
                
                # 2. Bundle Artifacts Locally (Requirements, Metadata)
                self._generate_requirements()
                self._merge_metadata(run_id, previous_run_id)

                # 3. Two-Phase Commit to S3
                # Phase 1: Upload Immutable Bundle
                self._upload_immutable_bundle(s3_run_dir_uri)
                
                # Phase 2: Atomic State Pointer Update (Zero-Downtime Switch)
                self._update_model_state(run_id, previous_run_id, s3_run_dir_uri)
                
                deployment_status = True

            # 4. Generate local component observability metadata (Not uploaded to registry)
            execution_time = round(time.time() - start_time, 2)
            self._generate_component_metadata(
                deployment_status=deployment_status, 
                s3_model_uri=s3_run_dir_uri, 
                execution_time=execution_time,
                previous_run_id=previous_run_id
            )

            # 5. Package Artifact
            artifact = ModelRegistrationArtifact(
                s3_model_uri=s3_run_dir_uri,
                metadata_file_path=self.config.metadata_file_path,
                deployment_status=deployment_status,
            )

            logging.info("Model Registry execution completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Model Registry run failed critically.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # ARTIFACT BUNDLING & GENERATION
    # ==========================================================
    def _generate_requirements(self) -> None:
        """
        Dynamically freezes the current Python environment to ensure absolute reproducibility.
        """
        try:
            logging.info("Generating fully pinned requirements.txt for the deployment bundle.")
            req_path = os.path.join(self.staging_dir, "requirements.txt")
            
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            with open(req_path, "w") as f:
                f.write(result.stdout)
                
            logging.debug("requirements.txt generated successfully.")
            
        except subprocess.CalledProcessError as e:
            logging.error("Subprocess failed during pip freeze: %s", e.stderr)
            raise CustomException(e, sys) from e
        except Exception as e:
            logging.exception("Failed to generate requirements.txt.")
            raise CustomException(e, sys) from e

    def _merge_metadata(self, run_id: str, previous_run_id: str) -> None:
        """
        Fuses Trainer metrics, Evaluation metrics, and Registry deployment logic into 
        a single cohesive metadata.json file.
        """
        try:
            logging.info("Consolidating pipeline telemetry into unified metadata.json.")
            
            # Extract Trainer Data
            with open(self.trainer_artifact.metadata_file_path, "r") as f:
                trainer_data = json.load(f)
                
            # Extract Evaluator Data
            with open(self.evaluation_artifact.report_file_path, "r") as f:
                eval_data = json.load(f)

            # Construct Unified Payload
            unified_metadata = {
                "registry_context": {
                    "run_id": run_id,
                    "previous_champion_run_id": previous_run_id,
                    "deployment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "environment": "production"
                },
                "training_context": trainer_data,
                "evaluation_context": eval_data
            }

            merged_path = os.path.join(self.staging_dir, "metadata.json")
            write_json_file(file_path=merged_path, content=unified_metadata)
            logging.debug("Unified metadata.json generated successfully.")

        except Exception as e:
            logging.exception("Failed to merge artifact metadata.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # S3 STATE MANAGEMENT
    # ==========================================================
    def _get_previous_champion_run_id(self) -> str:
        """
        Retrieves the run ID of the currently deployed Champion model.
        Forms the 'Linked List' allowing for immediate, stateful rollbacks.
        """
        try:
            local_tmp_path = os.path.join(self.staging_dir, "tmp_model_state.json")
            try:
                self.s3_sync.download_file(self.s3_pointer_uri, local_tmp_path)
                with open(local_tmp_path, "r") as f:
                    pointer_data = json.load(f)
                os.remove(local_tmp_path)
                
                previous_run_id = pointer_data.get("champion_run_id", "None")
                logging.info("Retrieved previous Champion Run ID for rollback state: %s", previous_run_id)
                return previous_run_id
                
            except Exception:
                logging.info("No active model_state.json found. Initializing Cold Start protocol (No rollback available).")
                return "None"
                
        except Exception as e:
            logging.warning("System error while fetching rollback pointer. Defaulting to 'None'. Error: %s", str(e))
            return "None"

    def _upload_immutable_bundle(self, s3_run_dir_uri: str) -> None:
        """
        PHASE 1 COMMIT: Uploads the compiled model, schema contract, dependencies, 
        and metadata to an isolated, immutable S3 directory.
        """
        try:
            logging.info("Initiating Phase 1 Commit: Uploading immutable bundle to %s", s3_run_dir_uri)
            
            # 1. Compiled Mega-Pipeline Model
            self.s3_sync.upload_file(
                local_path=self.trainer_artifact.model_file_path,
                s3_uri=f"{s3_run_dir_uri}/model.pkl"
            )
            
            # 2. Strict Physical Data Contract
            self.s3_sync.upload_file(
                local_path=self.transformation_artifact.schema_file_path,
                s3_uri=f"{s3_run_dir_uri}/schema.json"
            )
            
            # 3. Pinned Dependencies
            self.s3_sync.upload_file(
                local_path=os.path.join(self.staging_dir, "requirements.txt"),
                s3_uri=f"{s3_run_dir_uri}/requirements.txt"
            )
            
            # 4. Merged Telemetry
            self.s3_sync.upload_file(
                local_path=os.path.join(self.staging_dir, "metadata.json"),
                s3_uri=f"{s3_run_dir_uri}/metadata.json"
            )

            logging.info("Phase 1 Commit successful. Immutable bundle securely vaulted.")

        except Exception as e:
            logging.exception("Failed during Phase 1 Commit (Immutable Bundle Upload).")
            raise CustomException(e, sys) from e

    def _update_model_state(self, run_id: str, previous_run_id: str, s3_run_dir_uri: str) -> None:
        """
        PHASE 2 COMMIT: Atomic replacement of the global pointer file. 
        Downstream Inference Services strictly poll this file for the active model.
        """
        try:
            logging.info("Initiating Phase 2 Commit: Overwriting global model_state.json")
            
            # Extract the EROI to log in the pointer for rapid sanity checks
            with open(os.path.join(self.staging_dir, "metadata.json"), "r") as f:
                merged_data = json.load(f)
            
            eroi = merged_data.get("evaluation_context", {}).get("challenger_metrics", {}).get("eroi", 0.0)

            state_payload = {
                "schema_version": 2, 
                "champion_run_id": run_id,
                "previous_champion_run_id": previous_run_id,
                "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
                "eroi_baseline": eroi,
                "s3_bundle_uri": s3_run_dir_uri,
                "s3_model_path": f"{s3_run_dir_uri}/model.pkl",
                "s3_schema_path": f"{s3_run_dir_uri}/schema.json"
            }

            local_state_path = os.path.join(self.staging_dir, "model_state.json")
            write_json_file(file_path=local_state_path, content=state_payload)

            self.s3_sync.upload_file(
                local_path=local_state_path,
                s3_uri=self.s3_pointer_uri
            )
            
            logging.info("Phase 2 Commit successful. Production pointer updated at: %s", self.s3_pointer_uri)

        except Exception as e:
            logging.exception("Failed during Phase 2 Commit (Model State Update).")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY METADATA
    # ==========================================================
    def _generate_component_metadata(
        self, 
        deployment_status: bool, 
        s3_model_uri: str, 
        execution_time: float, 
        previous_run_id: str
    ) -> None:
        """
        Generates standard local telemetry metadata for the pipeline component.
        This is for local tracking/MLflow, separate from the S3 vaulted bundle.
        """
        try:
            metadata: Dict[str, Any] = {
                "pipeline_stage": "Model Registry & Deployment",
                "execution_time_seconds": execution_time,
                "deployment_executed": deployment_status,
                "s3_vault_uri": s3_model_uri,
                "state_management": {
                    "rollback_pointer_preserved": previous_run_id != "None",
                    "previous_champion": previous_run_id,
                    "global_pointer_uri": self.s3_pointer_uri
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(file_path=self.config.metadata_file_path, content=metadata)

        except Exception as e:
            logging.exception("Failed to generate local registry metadata.")
            raise CustomException(e, sys) from e