import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

from pipelines.inference_pipeline.src.entity.config_entity import ModelLoadingConfig
from pipelines.inference_pipeline.src.entity.artifact_entity import InferenceModelLoaderArtifact
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class ModelLoading:
    """
    Model Loader component for the Inference Pipeline.

    Responsibilities:
    - Act as the dynamic bridge between the Model Registry and the Inference Engine.
    - Query the global atomic pointer (model_state.json) to resolve the active Champion model.
    - Validate the integrity of the deployment pointer to prevent corrupted execution.
    - Download the immutable model.pkl and schema.json assets to a local isolated scratchpad.
    - Generate observability telemetry to track exactly which model version scored the batch.
    """

    def __init__(self, config: ModelLoadingConfig) -> None:
        """
        Initializes the Model Loader component.

        Args:
            config (InferenceModelLoaderConfig): Configuration object containing S3 URIs and local paths.
        """
        try:
            self.config = config
            self.s3_sync = S3Sync()
            
            # Temporary local path for downloading and inspecting the global pointer
            self.local_pointer_path = os.path.join(
                self.config.model_loader_root_dir, "temp_model_state.json"
            )

            logging.info("Inference Pipeline: Model Loader component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Model Loader component.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> InferenceModelLoaderArtifact:
        """
        Executes the atomic model resolution and download process.

        Returns:
            InferenceModelLoaderArtifact: Local paths to the downloaded model, schema, and metadata.
        """
        try:
            logging.info("Starting Current Production Model Loader execution.")
            start_time = time.time()

            # 1. Resolve State Pointer
            champion_run_id, s3_model_path, s3_schema_path = self._fetch_and_parse_pointer()

            # 2. Download Immutable Assets
            self._download_artifacts(s3_model_path, s3_schema_path)

            # 3. Generate Component Telemetry
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(champion_run_id, s3_model_path, s3_schema_path, execution_time)

            # 4. Package Artifact
            artifact = InferenceModelLoaderArtifact(
                model_file_path=self.config.model_file_path,
                schema_file_path=self.config.schema_file_path,
                champion_run_id=champion_run_id,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("Model Loader execution completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            # Fail-deadly: If the model cannot be loaded, the pipeline must crash immediately.
            logging.exception("Critical Failure: Model Loader run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # STATE RESOLUTION & VALIDATION
    # ==========================================================
    def _fetch_and_parse_pointer(self) -> Tuple[str, str, str]:
        """
        Downloads the global model_state.json pointer from the S3 registry,
        extracts the explicit asset URIs, and validates structural integrity.

        Returns:
            Tuple[str, str, str]: champion_run_id, s3_model_path, s3_schema_path
        """
        try:
            logging.info("Resolving global active pointer at: %s", self.config.s3_pointer_uri)

            # Download pointer to a temporary local file
            self.s3_sync.download_file(
                s3_uri=self.config.s3_pointer_uri, 
                local_path=self.local_pointer_path
            )

            # Parse and extract
            with open(self.local_pointer_path, "r") as f:
                pointer_data = json.load(f)

            # Clean up the temporary pointer file to keep the scratchpad pristine
            os.remove(self.local_pointer_path)

            # Strict extraction based on the Phase 2 Model Registry schema
            champion_run_id = pointer_data.get("champion_run_id")
            s3_model_path = pointer_data.get("s3_model_path")
            s3_schema_path = pointer_data.get("s3_schema_path")

            # Validation Pass: Ensure the pointer is not corrupted
            if not all([champion_run_id, s3_model_path, s3_schema_path]):
                raise ValueError(
                    f"Corrupted Registry Pointer: Missing required keys. "
                    f"Parsed Data: run_id={champion_run_id}, model={s3_model_path}, schema={s3_schema_path}"
                )

            logging.info("Active Champion resolved -> Run ID: %s", champion_run_id)
            return champion_run_id, s3_model_path, s3_schema_path

        except Exception as e:
            logging.exception("Failed to fetch or parse the production pointer from S3.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # ASSET DOWNLOAD
    # ==========================================================
    def _download_artifacts(self, s3_model_path: str, s3_schema_path: str) -> None:
        """
        Downloads the immutable model and schema artifacts from S3 to the local disk.
        """
        try:
            logging.info("Downloading model bundle from registry to local scratchpad.")

            # Download Model
            self.s3_sync.download_file(
                s3_uri=s3_model_path, 
                local_path=self.config.model_file_path
            )
            logging.debug("Downloaded model.pkl successfully.")

            # Download Schema
            self.s3_sync.download_file(
                s3_uri=s3_schema_path, 
                local_path=self.config.schema_file_path
            )
            logging.debug("Downloaded schema.json successfully.")

        except Exception as e:
            logging.exception("Failed to download deployment artifacts from S3.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY METADATA
    # ==========================================================
    def _generate_metadata(
        self, 
        champion_run_id: str, 
        s3_model_path: str, 
        s3_schema_path: str, 
        execution_time: float
    ) -> None:
        """
        Generates standard telemetry metadata ensuring complete data lineage 
        between the deployed model version and the current inference run.
        """
        try:
            logging.info("Generating Model Loader observability metadata.")

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Inference Current Production Model Loader",
                "execution_time_seconds": execution_time,
                "resolved_state": {
                    "pointer_uri_polled": self.config.s3_pointer_uri,
                    "champion_run_id_loaded": champion_run_id,
                },
                "downloaded_assets": {
                    "s3_model_source": s3_model_path,
                    "s3_schema_source": s3_schema_path,
                    "local_model_destination": self.config.model_file_path,
                    "local_schema_destination": self.config.schema_file_path,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            logging.info(
                "Model Loader metadata securely saved at: %s", 
                self.config.metadata_file_path
            )

        except Exception as e:
            logging.exception("Failed to generate Model Loader metadata.")
            raise CustomException(e, sys) from e