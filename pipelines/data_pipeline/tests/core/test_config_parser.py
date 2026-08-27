import pytest

from pathlib import Path

from pipelines.data_pipeline.src.core.config_parser import (
    DataPipelineConfig,
    PipelineConfigParser,
)
from shared_core.exceptions.custom_exception import CustomException


VALID_YAML_CONTENT = """
pipeline:
  name: "continual_learning_data_pipeline"
  version: "2.0.0"

storage:
  bronze_data_lake:
    base_uri: "s3://${S3_CUSTOMER_DATABASE}/bronze"
    datasets:
      - name: "orders"
        partition_keys: ["year", "month", "day"]

  feature_store:
    base_uri: "s3://${S3_PIPELINE_RUN_ARTIFACTS}/feature_store"
    artifact_name: "dataset.parquet"
    metadata_name: "metadata.json"
    export_format: "parquet"
    export_compression: "snappy"

compute:
  engine: "duckdb"
  hardware:
    threads: 4
    memory_limit: "8GB"
  runtime:
    temp_directory: "/tmp/data_pipeline_spill"
    extensions:
      - "httpfs"

business_logic:
  target_definition:
    churn_window_days: 180
    churn_column_name: "target_is_churn"
    ltv_column_name: "target_180d_ltv"

  cohort_definition:
    minimum_orders: 1
    active_status_codes:
      - "delivered"
      - "shipped"

validation:
  enable_schema_checks: true
  enable_null_checks: true
  strict_mode: true
  fail_fast: true
"""


def test_parse_success(
    temp_config_file: Path,
    mock_env_vars: None,
) -> None:
    temp_config_file.write_text(VALID_YAML_CONTENT)

    parser = PipelineConfigParser(str(temp_config_file))
    config = parser.parse()

    assert isinstance(config, DataPipelineConfig)
    assert config.pipeline.name == "continual_learning_data_pipeline"
    assert config.storage.bronze_data_lake.base_uri == "s3://test-customer-db/bronze"
    assert (
        config.storage.feature_store.base_uri
        == "s3://test-pipeline-artifacts/feature_store"
    )
    assert len(config.storage.bronze_data_lake.datasets) == 1
    assert config.compute.engine == "duckdb"
    assert config.business_logic.target_definition.churn_window_days == 180
    assert config.validation.strict_mode is True


def test_parse_file_not_found() -> None:
    parser = PipelineConfigParser("non_existent_config.yaml")

    with pytest.raises(FileNotFoundError):
        parser.parse()


def test_parse_missing_env_var(
    temp_config_file: Path,
    mock_env_vars: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Explicitly clear the required environment variables to trigger validation failure
    monkeypatch.delenv("S3_CUSTOMER_DATABASE", raising=False)
    monkeypatch.delenv("S3_PIPELINE_RUN_ARTIFACTS", raising=False)

    temp_config_file.write_text(VALID_YAML_CONTENT)

    parser = PipelineConfigParser(str(temp_config_file))

    with pytest.raises(CustomException):
        parser.parse()
        

def test_parse_empty_file(temp_config_file: Path) -> None:
    temp_config_file.write_text("   \n  ")

    parser = PipelineConfigParser(str(temp_config_file))

    with pytest.raises(CustomException):
        parser.parse()


def test_parse_invalid_yaml_syntax(temp_config_file: Path) -> None:
    temp_config_file.write_text("invalid: yaml: : syntax:")

    parser = PipelineConfigParser(str(temp_config_file))

    with pytest.raises(CustomException):
        parser.parse()


def test_parse_missing_required_key(
    temp_config_file: Path,
    mock_env_vars: None,
) -> None:
    invalid_yaml = VALID_YAML_CONTENT.replace("pipeline:", "wrong_key:")

    temp_config_file.write_text(invalid_yaml)

    parser = PipelineConfigParser(str(temp_config_file))

    with pytest.raises(CustomException):
        parser.parse()