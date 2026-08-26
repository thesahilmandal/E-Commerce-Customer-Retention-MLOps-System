import os
import sys
import re
import yaml
from typing import Any, Dict, List

from dotenv import load_dotenv

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging

class ConfigParser:
    """
    Parses and validates the global configuration YAML file for the Inference Pipeline.

    ```
    Acts as the centralized configuration provider, replacing hardcoded constants.
    Reads the configuration file, validates top-level structural integrity, 
    resolves environment variable interpolations, and exposes distinct sections 
    for downstream entity instantiation.
    """

    REQUIRED_TOP_LEVEL_KEYS: List[str] = [
        "system",
        "business_logic",
        "cloud_storage",
        "components"
    ]

    ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")

    def __init__(self, config_path: str = "pipelines/inference_pipeline/configs/global_config.yaml") -> None:
        """
        Initializes the ConfigParser and loads the YAML configuration.

        Args:
            config_path (str): Path to the global_config.yaml file.
        """
        try:
            load_dotenv()
            self.config_path = config_path
            self.config = self._read_and_validate_yaml()
        except Exception as e:
            logging.exception("Failed to initialize ConfigParser.")
            raise CustomException(e, sys) from e

    def _resolve_env_vars(self, content: str) -> str:
        """
        Resolves bash-style environment variable interpolations in the configuration string.
        Supports both ${VAR_NAME} and${VAR_NAME:-default_value} syntax.
        """
        def replacer(match: re.Match) -> str:
            env_var = match.group(1)
            default_val = match.group(2)
            
            if env_var in os.environ:
                return os.environ[env_var]
            elif default_val is not None:
                return default_val
            else:
                return ""
                
        return self.ENV_PATTERN.sub(replacer, content)

    def _read_and_validate_yaml(self) -> Dict[str, Any]:
        """
        Reads the YAML file from disk, resolves environment variables, and validates 
        its structural integrity.

        Returns:
            Dict[str, Any]: The parsed configuration dictionary.
        
        Raises:
            CustomException: If the file is missing, malformed, or missing required keys.
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, "r", encoding="utf-8") as file:
                raw_content = file.read()

            resolved_content = self._resolve_env_vars(raw_content)
            parsed_config = yaml.safe_load(resolved_content)

            if not isinstance(parsed_config, dict):
                raise ValueError("Parsed YAML configuration must be a dictionary.")

            # Validate top-level keys to ensure contract integrity
            missing_keys = [
                key for key in self.REQUIRED_TOP_LEVEL_KEYS if key not in parsed_config
            ]
            if missing_keys:
                raise KeyError(f"Configuration is missing required top-level keys: {missing_keys}")

            logging.debug("Configuration successfully loaded and structurally validated from %s", self.config_path)
            return parsed_config

        except Exception as e:
            logging.exception("Error parsing or validating YAML configuration.")
            raise CustomException(e, sys) from e

    def get_system_config(self) -> Dict[str, Any]:
        """Retrieves the system-level configuration."""
        return self.config.get("system", {})

    def get_business_logic_config(self) -> Dict[str, Any]:
        """Retrieves the business logic configuration, including thresholds and target definitions."""
        return self.config.get("business_logic", {})

    def get_cloud_storage_config(self) -> Dict[str, Any]:
        """Retrieves the S3 and Data Lake routing configuration."""
        return self.config.get("cloud_storage", {})

    def get_components_config(self) -> Dict[str, Any]:
        """Retrieves the specific configuration blocks for pipeline components."""
        return self.config.get("components", {})