import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from pipelines.data_pipeline.src.core.config_parser import DataPipelineConfig, PipelineConfigParser
from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.cloud.s3_operations import S3Sync


@pytest.fixture(autouse=True)
def mock_aws_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    bronze_dir = tmp_path / "bronze"
    fs_dir = tmp_path / "feature_store"
    spill_dir = tmp_path / "spill"

    bronze_dir.mkdir(parents=True, exist_ok=True)
    fs_dir.mkdir(parents=True, exist_ok=True)
    spill_dir.mkdir(parents=True, exist_ok=True)

    return tmp_path


@pytest.fixture
def sample_config_yaml_path(temp_workspace: Path) -> str:
    config_data = {
        "pipeline": {
            "name": "continual_learning_data_pipeline_test",
            "version": "1.0.0"
        },
        "storage": {
            "bronze_data_lake": {
                "base_uri": "s3://test-bucket/bronze",
                "datasets": [
                    {"name": "orders", "partition_keys": ["year", "month", "day"]},
                    {"name": "customers", "partition_keys": ["year", "month", "day"]},
                    {"name": "order_payments", "partition_keys": ["year", "month", "day"]}
                ]
            },
            "feature_store": {
                "base_uri": "s3://test-bucket/feature_store",
                "artifact_name": "dataset.parquet",
                "metadata_name": "metadata.json",
                "export_format": "parquet",
                "export_compression": "snappy"
            }
        },
        "compute": {
            "engine": "duckdb",
            "hardware": {
                "threads": 1,
                "memory_limit": "512MB"
            },
            "runtime": {
                "temp_directory": str(temp_workspace / "spill"),
                "extensions": ["httpfs"]
            }
        },
        "business_logic": {
            "target_definition": {
                "churn_window_days": 180,
                "churn_column_name": "target_is_churn",
                "ltv_column_name": "target_180d_ltv"
            },
            "cohort_definition": {
                "minimum_orders": 1,
                "active_status_codes": ["delivered", "shipped"]
            }
        },
        "validation": {
            "enable_schema_checks": True,
            "enable_null_checks": True,
            "strict_mode": True,
            "fail_fast": True
        }
    }

    config_path = temp_workspace / "test_pipeline_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    return str(config_path)


@pytest.fixture
def parsed_config(sample_config_yaml_path: str) -> DataPipelineConfig:
    parser = PipelineConfigParser(sample_config_yaml_path)
    return parser.parse()


@pytest.fixture
def mock_s3_sync() -> MagicMock:
    mock_sync = MagicMock(spec=S3Sync)

    def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        if not str(s3_uri).startswith("s3://"):
            raise ValueError(f"Invalid S3 URI. Must start with 's3://': {s3_uri}")
        parts = str(s3_uri).replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    mock_sync._parse_s3_uri.side_effect = parse_s3_uri
    mock_sync._get_client.return_value = MagicMock()

    return mock_sync


@pytest.fixture
def pipeline_context(parsed_config: DataPipelineConfig, mock_s3_sync: MagicMock):
    with PipelineContext(
        run_id="test_run_001",
        start_date="2016-09-01",
        end_date="2016-10-01",
        config=parsed_config,
        s3_sync=mock_s3_sync
    ) as context:
        yield context