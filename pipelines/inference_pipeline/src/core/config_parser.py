import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class InferenceConfigParser:
    """
    Configuration parser for the Inference Pipeline.

    Responsible for securely loading, validating, and centralizing environment variables
    and static configurations required for batch scoring into a structured dictionary.
    """

    def __init__(self) -> None:
        """
        Initializes the InferenceConfigParser and parses the configuration state.
        """
        try:
            load_dotenv()
            self._config: Dict[str, Any] = self._parse_configuration()
            logging.debug("Inference Pipeline configuration parsed successfully.")
        except Exception as e:
            logging.exception("Failed to parse Inference Pipeline configuration.")
            raise CustomException(e, sys) from e

    def _parse_configuration(self) -> Dict[str, Any]:
        """
        Parses and validates all required configuration parameters, explicitly 
        routing them into logical domains for downstream component injection.
        
        Returns:
            Dict[str, Any]: A strongly typed, validated configuration dictionary.
        """
        return {
            "system": {
                "artifact_dir": self._get_env("ARTIFACT_DIR_NAME", "artifacts"),
                "s3_bucket_name": self._get_env("ML_S3_BUCKET_NAME"),
            },
            "registry": {
                "registry_dir": self._get_env("S3_MODEL_REGISTRY_DIR_NAME", "model_registry"),
                "state_dir": self._get_env("S3_MODEL_REGISTRY_STATE_DIR", "state"),
                "pointer_file": self._get_env("S3_MODEL_REGISTRY_POINTER_FILE_NAME", "model_state.json"),
            },
            "data_lake": {
                "database_name": self._get_env("S3_CUSTOMER_DATABASE_NAME", "company-central-data-lake"),
                "bronze_dir": self._get_env("S3_DATA_LAKE_BRONZE_DIR_NAME", "bronze"),
            },
            "destinations": {
                "business_reports_dir": self._get_env("S3_INFERENCE_BUSINESS_REPORTS_DIR", "business_reports"),
                "mlops_telemetry_dir": self._get_env("S3_INFERENCE_MLOPS_TELEMETRY_DIR", "mlops_telemetry"),
                "metadata_manifest_dir": self._get_env("S3_INFERENCE_METADATA_DIR", "inference_metadata"),
            },
            "business_logic": {
                "probability_threshold": float(self._get_env("INFERENCE_PROBABILITY_THRESHOLD", "0.5")),
                "system_columns_to_drop": [
                    "customer_unique_id",
                    "snapshot_date",
                    "ingested_at_utc",
                    "target_180d_ltv"
                ],
            }
        }

    @staticmethod
    def _get_env(key: str, default: Any = None) -> str:
        """
        Safely retrieves environment variables with validation for required keys.
        
        Args:
            key (str): The environment variable key.
            default (Any, optional): The default value if the key is not found.
            
        Returns:
            str: The resolved environment variable value.
            
        Raises:
            ValueError: If the environment variable is missing and no default is provided.
        """
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(
                f"Configuration Error: Required environment variable '{key}' "
                "is missing or undefined."
            )
        return str(value)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the parsed and validated configuration dictionary.
        """
        return self._config