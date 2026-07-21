import os
import sys
import yaml
from typing import Dict, Any

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class ConfigParser:
    """
    Configuration Parser for the Training Pipeline.

    Responsibilities:
    - Load the centralized pipeline_config.yaml file.
    - Validate that all required configuration sections are present and properly formatted.
    - Expose structured, strictly-typed configuration dictionaries to the PipelineContext.
    - Handle configuration file missing or parsing errors gracefully.
    """

    DEFAULT_CONFIG_PATH = "pipelines/training_pipeline/configs/pipeline_config.yaml"

    def __init__(self, config_filepath: str = DEFAULT_CONFIG_PATH) -> None:
        """
        Initializes the ConfigParser and loads the configuration.

        Args:
            config_filepath (str): Path to the YAML configuration file.
        """
        try:
            self.config_filepath = config_filepath
            self.config_data = self._read_yaml()
            self._validate_config()
            logging.info("Training Pipeline configuration loaded and validated successfully.")
            
        except Exception as e:
            logging.exception("Failed to initialize ConfigParser.")
            raise CustomException(e, sys) from e

    def _read_yaml(self) -> Dict[str, Any]:
        """
        Safely reads the YAML configuration file.

        Returns:
            Dict[str, Any]: The parsed configuration dictionary.
            
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file is empty or invalid.
        """
        try:
            logging.debug("Reading configuration from: %s", self.config_filepath)
            
            if not os.path.exists(self.config_filepath):
                raise FileNotFoundError(f"Configuration file not found at: {self.config_filepath}")

            with open(self.config_filepath, "r") as file:
                config = yaml.safe_load(file)

            # Ensure the file wasn't completely empty and parsed as a valid dictionary
            if not config or not isinstance(config, dict):
                raise ValueError(f"Configuration file is empty or invalid: {self.config_filepath}")

            # Extract the 'training_pipeline' namespace if it exists and is a valid dictionary.
            # If a formatting/indentation error caused it to be None, or if the config 
            # doesn't use the namespace, we gracefully fall back to the root dictionary.
            config_data = config.get("training_pipeline")
            
            if config_data is None or not isinstance(config_data, dict):
                config_data = config

            return config_data
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def _validate_config(self) -> None:
        """
        Validates the presence of all required top-level configuration sections
        and ensures they contain valid configuration dictionaries.
        
        Raises:
            KeyError: If a required section is missing.
            ValueError: If a required section is empty or malformed (None).
        """
        try:
            required_sections = [
                "global",
                "data_processor",
                "model_trainer",
                "model_evaluator",
                "model_registry"
            ]
            
            # Check if the keys exist at all
            missing_sections = [
                section for section in required_sections 
                if section not in self.config_data
            ]
            
            if missing_sections:
                raise KeyError(f"Missing required configuration sections: {missing_sections}")

            # Validate that each section is actually a populated dictionary, not None.
            # This catches YAML indentation loss where a section key is parsed with no children.
            invalid_sections = [
                section for section in required_sections
                if not isinstance(self.config_data.get(section), dict)
            ]

            if invalid_sections:
                raise ValueError(
                    f"Configuration sections {invalid_sections} are empty or invalid. "
                    "This typically occurs due to missing YAML indentation. Please ensure "
                    "the variables beneath these sections are properly indented with spaces."
                )
                
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_global_config(self) -> Dict[str, Any]:
        """Returns the global configuration parameters."""
        return self.config_data["global"]

    def get_data_processor_config(self) -> Dict[str, Any]:
        """Returns the data processor component configuration."""
        return self.config_data["data_processor"]

    def get_model_trainer_config(self) -> Dict[str, Any]:
        """Returns the model trainer component configuration."""
        return self.config_data["model_trainer"]

    def get_model_evaluator_config(self) -> Dict[str, Any]:
        """Returns the model evaluator component configuration."""
        return self.config_data["model_evaluator"]

    def get_model_registry_config(self) -> Dict[str, Any]:
        """Returns the model registry component configuration."""
        return self.config_data["model_registry"]