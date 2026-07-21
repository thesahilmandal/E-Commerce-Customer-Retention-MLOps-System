"""
Configuration Parser Module.

This module is responsible for loading, parsing, and validating the centralized 
YAML configuration file (pipeline_config.yaml) for the Data Pipeline.
It maps the raw YAML structure into strongly-typed, immutable Data Classes 
to ensure type safety and configuration integrity throughout the pipeline's execution.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


# ==========================================================
# CONFIGURATION DATA CLASSES
# ==========================================================

@dataclass(frozen=True)
class PipelineInfoConfig:
    """Metadata about the pipeline."""
    name: str
    version: str


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for an individual dataset in the Bronze Data Lake."""
    name: str
    partition_keys: List[str]


@dataclass(frozen=True)
class BronzeDataLakeConfig:
    """Configuration for the S3 Bronze Data Lake."""
    base_uri: str
    datasets: List[DatasetConfig]


@dataclass(frozen=True)
class FeatureStoreConfig:
    """Configuration for the S3 Feature Store."""
    base_uri: str
    artifact_name: str
    metadata_name: str
    export_format: str
    export_compression: str


@dataclass(frozen=True)
class StorageConfig:
    """Aggregated storage configurations."""
    bronze_data_lake: BronzeDataLakeConfig
    feature_store: FeatureStoreConfig


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware limits for DuckDB out-of-core execution."""
    threads: int
    memory_limit: str


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime environment settings for compute."""
    temp_directory: str
    extensions: List[str]


@dataclass(frozen=True)
class ComputeConfig:
    """Aggregated compute engine configurations."""
    engine: str
    hardware: HardwareConfig
    runtime: RuntimeConfig


@dataclass(frozen=True)
class TargetDefinitionConfig:
    """Business logic configuration for target variables."""
    churn_window_days: int
    churn_column_name: str
    ltv_column_name: str


@dataclass(frozen=True)
class CohortDefinitionConfig:
    """Business logic configuration for cohort selection."""
    minimum_orders: int
    active_status_codes: List[str]


@dataclass(frozen=True)
class BusinessLogicConfig:
    """Aggregated business logic configurations."""
    target_definition: TargetDefinitionConfig
    cohort_definition: CohortDefinitionConfig


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for data quality and schema validation gates."""
    enable_schema_checks: bool
    enable_null_checks: bool
    strict_mode: bool
    fail_fast: bool


@dataclass(frozen=True)
class DataPipelineConfig:
    """Root configuration object encompassing all pipeline settings."""
    pipeline: PipelineInfoConfig
    storage: StorageConfig
    compute: ComputeConfig
    business_logic: BusinessLogicConfig
    validation: ValidationConfig


# ==========================================================
# CONFIGURATION PARSER
# ==========================================================

class PipelineConfigParser:
    """
    Parses the centralized YAML configuration into strongly-typed Data Classes.

    Responsibilities:
    - Safely read the YAML configuration file from the filesystem.
    - Validate the presence of all required configuration blocks.
    - Instantiate and return an immutable DataPipelineConfig object.
    """

    def __init__(self, config_file_path: str) -> None:
        """
        Initializes the parser with the target configuration file path.

        Args:
            config_file_path (str): Absolute or relative path to the YAML config file.
        """
        self.config_file_path = config_file_path
        logging.info("PipelineConfigParser initialized for path: %s", self.config_file_path)

    def _read_yaml(self) -> Dict[str, Any]:
        """
        Reads and parses the YAML file into a Python dictionary.

        Returns:
            Dict[str, Any]: The parsed YAML structure.

        Raises:
            CustomException: If the file does not exist or contains invalid YAML.
        """
        if not os.path.exists(self.config_file_path):
            error_msg = f"Configuration file not found at: {self.config_file_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(self.config_file_path, "r", encoding="utf-8") as file:
                parsed_yaml = yaml.safe_load(file)

            if not parsed_yaml:
                raise ValueError(f"Configuration file {self.config_file_path} is empty.")

            return parsed_yaml

        except Exception as exc:
            logging.exception("Failed to read YAML configuration file.")
            raise CustomException(exc, sys) from exc

    def parse(self) -> DataPipelineConfig:
        """
        Executes the parsing and mapping logic.

        Returns:
            DataPipelineConfig: The fully hydrated, immutable configuration object.

        Raises:
            CustomException: If any required keys are missing or mapping fails.
        """
        logging.debug("Starting configuration parsing and mapping.")
        raw_config = self._read_yaml()

        try:
            # 1. Map Pipeline Info
            pipeline_info = PipelineInfoConfig(**raw_config["pipeline"])

            # 2. Map Storage Config
            bronze_raw = raw_config["storage"]["bronze_data_lake"]
            datasets = [DatasetConfig(**ds) for ds in bronze_raw.get("datasets", [])]
            bronze_data_lake = BronzeDataLakeConfig(
                base_uri=bronze_raw["base_uri"],
                datasets=datasets
            )
            feature_store = FeatureStoreConfig(**raw_config["storage"]["feature_store"])
            storage = StorageConfig(
                bronze_data_lake=bronze_data_lake,
                feature_store=feature_store
            )

            # 3. Map Compute Config
            compute_raw = raw_config["compute"]
            hardware = HardwareConfig(**compute_raw["hardware"])
            runtime = RuntimeConfig(**compute_raw["runtime"])
            compute = ComputeConfig(
                engine=compute_raw["engine"],
                hardware=hardware,
                runtime=runtime
            )

            # 4. Map Business Logic Config
            logic_raw = raw_config["business_logic"]
            target_def = TargetDefinitionConfig(**logic_raw["target_definition"])
            cohort_def = CohortDefinitionConfig(**logic_raw["cohort_definition"])
            business_logic = BusinessLogicConfig(
                target_definition=target_def,
                cohort_definition=cohort_def
            )

            # 5. Map Validation Config
            validation = ValidationConfig(**raw_config["validation"])

            # Combine into root configuration object
            pipeline_config = DataPipelineConfig(
                pipeline=pipeline_info,
                storage=storage,
                compute=compute,
                business_logic=business_logic,
                validation=validation
            )

            logging.info("Pipeline configuration parsed successfully.")
            return pipeline_config

        except KeyError as exc:
            logging.error("Missing required configuration key: %s", exc)
            raise CustomException(exc, sys) from exc
        except Exception as exc:
            logging.exception("Unexpected error during configuration mapping.")
            raise CustomException(exc, sys) from exc