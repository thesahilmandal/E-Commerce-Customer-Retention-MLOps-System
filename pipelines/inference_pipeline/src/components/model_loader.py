import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from pipelines.inference_pipeline.src.core.context import InferencePipelineContext
from pipelines.inference_pipeline.src.entity.config_entity import ModelLoaderConfig
from pipelines.inference_pipeline.src.entity.artifact_entity import ModelLoaderArtifact


class ModelLoader:
    """
    Model Loader Component.

    Responsibilities:
    - Acts as the secure bridge between the Model Registry and the Inference Pipeline.
    - Fetches the active state pointer (`model_state.json`) from the centralized Model Registry.
    - Parses the pointer to identify the current Champion model's specific `run_id` and artifact S3 URIs.
    - Dynamically resolves and downloads the serialized model artifact (`model.pkl`), its data contract 
      schema (`schema.json`), monitoring baselines, and reference distributions to the local environment.
    - Validates the integrity of the state pointer and the existence of all referenced artifacts.
    - Generates strict operational metadata ensuring model provenance and traceability.
    """

    def __init__(self, context: InferencePipelineContext) -> None:
        """
        Initializes the Model Loader component.

        Args:
            context (InferencePipelineContext): The centralized execution context.
        """
        try:
            self.context = context
            self.config = ModelLoaderConfig.from_context(context)
            logging.info("Inference Pipeline: Model Loader component initialized.")
        except Exception as e:
            logging.exception("Failed to initialize Model Loader component.")
            raise CustomException(e, sys) from e

    def run(self) -> ModelLoaderArtifact:
        """
        Executes the secure artifact resolution and downloading workflow.

        Returns:
            ModelLoaderArtifact: Dataclass containing the resolved champion run ID 
                                 and the local paths to the model and schema artifacts.
        """
        try:
            logging.info("Starting Model Loading Sequence.")
            start_time = time.time()

            # 1. Fetch and Parse Registry Pointer (model_state.json)
            pointer_data = self._fetch_and_parse_pointer()
            champion_run_id = pointer_data["run_id"]
            logging.info("Resolved Champion Model Run ID: %s", champion_run_id)

            # 2. Download Production Artifacts Dynamically
            self._download_artifacts(pointer_data=pointer_data)

            # 3. Generate Traceability Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(
                pointer_data=pointer_data,
                execution_time=execution_time
            )

            # 4. Package Artifact
            # Note: We return model and schema paths for immediate downstream compatibility
            # while having securely downloaded monitoring baselines for sidecar/telemetry processes.
            artifact = ModelLoaderArtifact(
                model_file_path=self.config.model_file_path,
                schema_file_path=self.config.schema_file_path,
                champion_run_id=champion_run_id,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("Model Loading completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Critical Failure inside Model Loader execution routine.")
            raise CustomException(e, sys) from e

    def _fetch_and_parse_pointer(self) -> Dict[str, Any]:
        """
        Downloads the active state pointer from the registry vault and parses its JSON content.
        
        Returns:
            Dict[str, Any]: The validated pointer dictionary mapping to the current champion artifacts.
            
        Raises:
            CustomException: If the pointer file is malformed, missing required keys, or cannot be found.
        """
        local_pointer_path = os.path.join(self.config.model_loader_root_dir, "model_state.json")
        try:
            logging.debug("Fetching model registry state pointer from: %s", self.config.s3_pointer_uri)
            self.context.s3_sync.download_file(
                s3_uri=self.config.s3_pointer_uri, 
                local_path=local_pointer_path
            )
        except Exception as e:
            error_msg = (
                f"Failed to fetch model registry state pointer from {self.config.s3_pointer_uri}. "
                "Ensure the Training Pipeline has successfully published the model_state.json file."
            )
            logging.error(error_msg)
            raise CustomException(error_msg, sys) from e

        try:
            with open(local_pointer_path, "r", encoding="utf-8") as f:
                pointer_data = json.load(f)

            # Validate structural contract of the pointer generated by the Training Pipeline
            required_keys = [
                "run_id",
                "s3_model_path",
                "s3_schema_path",
                "s3_monitoring_baselines_path",
                "s3_reference_distributions_path"
            ]
            
            missing_keys = [
                key for key in required_keys 
                if key not in pointer_data or not pointer_data[key]
            ]

            if missing_keys:
                raise ValueError(
                    f"Invalid or incomplete state pointer format. Missing required keys: {missing_keys}"
                )

            return pointer_data

        except Exception as e:
            logging.exception("Failed to parse or validate the registry state pointer.")
            raise CustomException(e, sys) from e

    def _download_artifacts(self, pointer_data: Dict[str, Any]) -> None:
        """
        Downloads all required Champion model artifacts from S3 to local storage
        based dynamically on the URIs provided in the state pointer.
        """
        downloads = [
            (pointer_data["s3_model_path"], self.config.model_file_path, "Model Artifact"),
            (pointer_data["s3_schema_path"], self.config.schema_file_path, "Schema Contract"),
            (pointer_data["s3_monitoring_baselines_path"], self.config.baseline_metrics_file_path, "Monitoring Baselines"),
            (pointer_data["s3_reference_distributions_path"], self.config.reference_distributions_file_path, "Reference Distributions")
        ]

        for s3_uri, local_path, artifact_name in downloads:
            try:
                logging.info("Downloading %s from: %s", artifact_name, s3_uri)
                self.context.s3_sync.download_file(
                    s3_uri=s3_uri, 
                    local_path=local_path
                )
            except Exception as e:
                error_msg = f"Failed to download {artifact_name} from S3 URI: {s3_uri}."
                logging.error(error_msg)
                raise CustomException(error_msg, sys) from e
                
        logging.debug("All required registry artifacts successfully secured locally.")

    def _generate_metadata(self, pointer_data: Dict[str, Any], execution_time: float) -> None:
        """
        Generates comprehensive operational metadata to ensure model provenance,
        embedding the full state pointer payload for exact traceability.
        """
        try:
            metadata = {
                "pipeline_stage": "Model Loader",
                "inference_run_id": self.context.run_id,
                "execution_time_seconds": execution_time,
                "model_provenance": pointer_data,
                "local_artifact_sizes_bytes": {
                    "model": os.path.getsize(self.config.model_file_path),
                    "schema": os.path.getsize(self.config.schema_file_path),
                    "monitoring_baselines": os.path.getsize(self.config.baseline_metrics_file_path),
                    "reference_distributions": os.path.getsize(self.config.reference_distributions_file_path)
                },
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }

            with open(self.config.metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
                
            logging.debug("Model Loader metadata successfully saved to: %s", self.config.metadata_file_path)

        except Exception as e:
            logging.exception("Failed to generate Model Loader metadata.")
            raise CustomException(e, sys) from e