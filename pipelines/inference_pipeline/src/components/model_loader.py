import json
import os
import sys
import time
from typing import Any, Dict

from pipelines.inference_pipeline.src.core.context import InferenceContext
from pipelines.inference_pipeline.src.entity.artifact_entity import ModelLoadingArtifact
from pipelines.inference_pipeline.src.entity.config_entity import ModelLoaderConfig
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class ModelLoader:
    """
    Model Loader Component.

    Responsible for resolving the active Champion model from the Model Registry's
    atomic state pointer and downloading the deployment artifacts (model and schema)
    to the local execution workspace.
    """

    def __init__(self, config: ModelLoaderConfig, context: InferenceContext) -> None:
        """
        Initializes the ModelLoader component.

        Args:
            config (ModelLoaderConfig): Configuration specifying registry URIs and target paths.
            context (InferenceContext): Centralized pipeline execution context.
        """
        try:
            self.config = config
            self.context = context
            self.s3_sync = S3Sync()
        except Exception as e:
            logging.exception("Failed to initialize ModelLoader component.")
            raise CustomException(e, sys) from e

    def run(self) -> ModelLoadingArtifact:
        """
        Executes the model loading phase, resolving the active pointer and fetching artifacts.

        Returns:
            ModelLoadingArtifact: The local paths to the deployment artifacts and registry ID.
        """
        try:
            logging.info("Starting ModelLoader execution.")
            start_time = time.time()

            # 1. Download and parse the atomic pointer file
            local_pointer_path = os.path.join(self.context.run_dir, "model_state.json")
            logging.debug("Downloading registry pointer from: %s", self.config.s3_pointer_uri)
            
            self.s3_sync.download_file(
                s3_uri=self.config.s3_pointer_uri,
                local_path=local_pointer_path
            )
            
            with open(local_pointer_path, "r", encoding="utf-8") as f:
                model_state = json.load(f)

            champion_run_id = model_state.get("run_id")
            s3_model_uri = model_state.get("s3_model_path")
            s3_schema_uri = model_state.get("s3_schema_path")

            if not all([champion_run_id, s3_model_uri, s3_schema_uri]):
                raise ValueError("Model state pointer is missing required routing keys.")

            logging.info("Resolved Champion Model ID: %s", champion_run_id)

            # 2. Download deployment artifacts
            logging.debug("Downloading model artifact from: %s", s3_model_uri)
            self.s3_sync.download_file(
                s3_uri=s3_model_uri,
                local_path=self.config.local_model_file_path
            )
            
            logging.debug("Downloading schema contract from: %s", s3_schema_uri)
            self.s3_sync.download_file(
                s3_uri=s3_schema_uri,
                local_path=self.config.local_schema_file_path
            )

            # 3. Log component telemetry to context
            execution_time = round(time.time() - start_time, 2)
            telemetry: Dict[str, Any] = {
                "champion_run_id": champion_run_id,
                "s3_pointer_resolved": self.config.s3_pointer_uri,
                "s3_model_uri": s3_model_uri,
                "s3_schema_uri": s3_schema_uri,
                "execution_time_seconds": execution_time
            }
            self.context.add_metadata("ModelLoader", telemetry)

            # 4. Construct and return artifact
            artifact = ModelLoadingArtifact(
                model_file_path=self.config.local_model_file_path,
                schema_file_path=self.config.local_schema_file_path,
                champion_run_id=champion_run_id
            )

            logging.info("ModelLoader execution completed successfully.")
            return artifact

        except Exception as e:
            logging.exception("Critical Failure in ModelLoader component.")
            raise CustomException(e, sys) from e