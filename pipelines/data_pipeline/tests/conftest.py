
from pathlib import Path
from typing import Generator, Tuple
from unittest.mock import MagicMock

import pytest

from pipelines.data_pipeline.src.core.config_parser import (
    BusinessLogicConfig,
    BronzeDataLakeConfig,
    CohortDefinitionConfig,
    ComputeConfig,
    DataPipelineConfig,
    DatasetConfig,
    FeatureStoreConfig,
    HardwareConfig,
    PipelineInfoConfig,
    RuntimeConfig,
    StorageConfig,
    TargetDefinitionConfig,
    ValidationConfig,
)
from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.cloud.s3_operations import S3Sync


@pytest.fixture
def mock_s3_sync() -> MagicMock:
    """Provides a mocked S3Sync instance with safely mocked URI parsing."""
    mock = MagicMock(spec=S3Sync)

    def parse_uri_side_effect(uri: str) -> Tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI. Must start with 's3://': {uri}")
        parts = uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    mock._parse_s3_uri.side_effect = parse_uri_side_effect
    mock._get_client.return_value = MagicMock()
    return mock


@pytest.fixture
def dummy_pipeline_config() -> DataPipelineConfig:
    """Provides a fully hydrated, deterministic DataPipelineConfig for testing."""
    return DataPipelineConfig(
        pipeline=PipelineInfoConfig(
            name="test_continual_learning_pipeline",
            version="1.0.0",
        ),
        storage=StorageConfig(
            bronze_data_lake=BronzeDataLakeConfig(
                base_uri="s3://test-bronze-bucket/bronze",
                datasets=[
                    DatasetConfig(
                        name="orders",
                        partition_keys=["year", "month", "day"],
                    ),
                    DatasetConfig(
                        name="customers",
                        partition_keys=["year", "month", "day"],
                    ),
                    DatasetConfig(
                        name="order_payments",
                        partition_keys=["year", "month", "day"],
                    ),
                ],
            ),
            feature_store=FeatureStoreConfig(
                base_uri="s3://test-feature-store-bucket/feature_store",
                artifact_name="test_dataset.parquet",
                metadata_name="test_metadata.json",
                export_format="parquet",
                export_compression="snappy",
            ),
        ),
        compute=ComputeConfig(
            engine="duckdb",
            hardware=HardwareConfig(
                threads=2,
                memory_limit="4GB",
            ),
            runtime=RuntimeConfig(
                temp_directory="/tmp/test_data_pipeline_spill",
                extensions=["httpfs"],
            ),
        ),
        business_logic=BusinessLogicConfig(
            target_definition=TargetDefinitionConfig(
                churn_window_days=180,
                churn_column_name="target_is_churn",
                ltv_column_name="target_180d_ltv",
            ),
            cohort_definition=CohortDefinitionConfig(
                minimum_orders=1,
                active_status_codes=["delivered", "shipped"],
            ),
        ),
        validation=ValidationConfig(
            enable_schema_checks=True,
            enable_null_checks=True,
            strict_mode=True,
            fail_fast=True,
        ),
    )


@pytest.fixture
def mock_pipeline_context(
    dummy_pipeline_config: DataPipelineConfig,
    mock_s3_sync: MagicMock,
) -> MagicMock:
    """Provides a mocked PipelineContext to isolate DuckDB and S3 side effects."""
    mock_ctx = MagicMock(spec=PipelineContext)
    mock_ctx.run_id = "test_run_12345"
    mock_ctx.start_date = "2023-01-01"
    mock_ctx.end_date = "2023-06-01"
    mock_ctx.config = dummy_pipeline_config
    mock_ctx.s3_sync = mock_s3_sync
    mock_ctx.db_con = MagicMock()
    return mock_ctx


@pytest.fixture
def mock_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Injects strictly required environment variables for config parsing tests."""
    monkeypatch.setenv("S3_CUSTOMER_DATABASE", "test-customer-db")
    monkeypatch.setenv("S3_PIPELINE_RUN_ARTIFACTS", "test-pipeline-artifacts")

    yield


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Provides an isolated temporary path for writing dummy YAML configurations."""
    return tmp_path / "pipeline_config.yaml"