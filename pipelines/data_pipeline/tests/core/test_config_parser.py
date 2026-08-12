import os
from pathlib import Path

import pytest

from pipelines.data_pipeline.src.core.config_parser import DataPipelineConfig, PipelineConfigParser
from shared_core.exceptions.custom_exception import CustomException


def test_parse_successful_configuration(sample_config_yaml_path: str):
    parser = PipelineConfigParser(config_file_path=sample_config_yaml_path)
    config = parser.parse()

    assert isinstance(config, DataPipelineConfig)
    assert config.pipeline.name == "continual_learning_data_pipeline_test"
    assert config.pipeline.version == "1.0.0"

    assert config.storage.bronze_data_lake.base_uri == "s3://test-bucket/bronze"
    assert len(config.storage.bronze_data_lake.datasets) == 3
    assert config.storage.bronze_data_lake.datasets[0].name == "orders"
    assert "year" in config.storage.bronze_data_lake.datasets[0].partition_keys

    assert config.storage.feature_store.export_format == "parquet"
    assert config.compute.engine == "duckdb"
    assert config.compute.hardware.threads == 1
    assert "httpfs" in config.compute.runtime.extensions

    assert config.business_logic.target_definition.churn_window_days == 180
    assert config.business_logic.cohort_definition.minimum_orders == 1

    assert config.validation.enable_schema_checks is True
    assert config.validation.strict_mode is True


def test_parse_file_not_found():
    parser = PipelineConfigParser(config_file_path="non_existent_path.yaml")
    with pytest.raises(FileNotFoundError) as exc_info:
        parser.parse()

    assert "Configuration file not found" in str(exc_info.value)


def test_parse_empty_file(temp_workspace: Path):
    empty_file_path = temp_workspace / "empty_config.yaml"
    empty_file_path.touch()

    parser = PipelineConfigParser(config_file_path=str(empty_file_path))
    with pytest.raises(CustomException) as exc_info:
        parser.parse()

    assert "is empty" in str(exc_info.value)


def test_parse_invalid_yaml_syntax(temp_workspace: Path):
    invalid_yaml_path = temp_workspace / "invalid_config.yaml"
    invalid_yaml_path.write_text(
        "pipeline:\n"
        "  name: test_pipeline\n"
        " invalid_indentation: True\n"
        "- broken_list_item"
    )

    parser = PipelineConfigParser(config_file_path=str(invalid_yaml_path))
    with pytest.raises(CustomException):
        parser.parse()


def test_parse_missing_required_key(temp_workspace: Path):
    missing_key_yaml_path = temp_workspace / "missing_key_config.yaml"
    missing_key_yaml_path.write_text(
        "pipeline:\n"
        "  name: test_pipeline\n"
        "  version: 1.0.0\n"
        "# storage block is intentionally missing\n"
        "compute:\n"
        "  engine: duckdb\n"
    )

    parser = PipelineConfigParser(config_file_path=str(missing_key_yaml_path))
    with pytest.raises(CustomException) as exc_info:
        parser.parse()

    assert "storage" in str(exc_info.value)