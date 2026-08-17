import hashlib
import os
from unittest.mock import MagicMock, patch

import pytest

from pipelines.data_pipeline.src.components.metadata_registry import MetadataRegistry
from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException


def test_get_target_uris(pipeline_context: PipelineContext):
    registry = MetadataRegistry(pipeline_context)
    artifact_uri, metadata_uri = registry._get_target_uris()

    assert artifact_uri == "s3://test-bucket/feature_store/test_run_001/dataset.parquet"
    assert metadata_uri == "s3://test-bucket/feature_store/test_run_001/metadata.json"


def test_extract_parquet_telemetry_success(pipeline_context: PipelineContext):
    mock_con = MagicMock()

    def side_effect(query: str):
        mock_res = MagicMock()
        if "COUNT(*)" in query:
            mock_res.fetchone.return_value = (420,)
        elif "DESCRIBE" in query:
            mock_res.fetchall.return_value = [("customer_id", "VARCHAR"), ("payment_value", "DOUBLE")]
        return mock_res

    mock_con.execute.side_effect = side_effect
    pipeline_context.db_con = mock_con

    registry = MetadataRegistry(pipeline_context)
    row_count, columns_hash = registry._extract_parquet_telemetry(
        "s3://fake-bucket/artifact.parquet"
    )

    assert row_count == 420
    assert len(columns_hash) == 10

    expected_hash = hashlib.md5("customer_idpayment_value".encode("utf-8")).hexdigest()[:10]
    assert columns_hash == expected_hash


def test_extract_parquet_telemetry_db_failure(pipeline_context: PipelineContext):
    mock_con = MagicMock()
    mock_con.execute.side_effect = Exception("Simulated DuckDB read failure")
    pipeline_context.db_con = mock_con

    registry = MetadataRegistry(pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        registry._extract_parquet_telemetry("s3://fake-bucket/artifact.parquet")

    assert "Simulated DuckDB read failure" in str(exc_info.value)


def test_build_metadata_payload(pipeline_context: PipelineContext):
    registry = MetadataRegistry(pipeline_context)

    payload = registry._build_metadata_payload(
        artifact_uri="s3://fake-bucket/artifact.parquet",
        row_count=1000,
        columns_hash="abcdef1234"
    )

    assert payload["run_id"] == "test_run_001"
    assert payload["temporal_bounds"]["start_date"] == "2016-09-01"
    assert payload["temporal_bounds"]["end_date"] == "2016-10-01"
    assert payload["artifact_uri"] == "s3://fake-bucket/artifact.parquet"
    assert payload["schema"]["columns_hash"] == "abcdef1234"
    assert payload["schema"]["row_count"] == 1000
    assert "timestamp_utc" in payload


def test_run_success_and_cleanup(
    pipeline_context: PipelineContext, monkeypatch: pytest.MonkeyPatch
):
    registry = MetadataRegistry(pipeline_context)

    mock_extract = MagicMock(return_value=(500, "testhash12"))
    monkeypatch.setattr(registry, "_extract_parquet_telemetry", mock_extract)

    registry.run()

    pipeline_context.s3_sync.upload_file.assert_called_once()

    call_kwargs = pipeline_context.s3_sync.upload_file.call_args.kwargs
    uploaded_local_path = call_kwargs["local_path"]
    target_s3_uri = call_kwargs["s3_uri"]

    assert target_s3_uri == "s3://test-bucket/feature_store/test_run_001/metadata.json"
    assert not os.path.exists(uploaded_local_path), "Temporary metadata JSON file was not cleaned up."


@patch("os.remove")
def test_run_cleanup_on_upload_failure(
    mock_remove: MagicMock,
    pipeline_context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch
):
    registry = MetadataRegistry(pipeline_context)

    mock_extract = MagicMock(return_value=(500, "testhash12"))
    monkeypatch.setattr(registry, "_extract_parquet_telemetry", mock_extract)

    pipeline_context.s3_sync.upload_file.side_effect = Exception("Simulated S3 Upload Error")

    with pytest.raises(CustomException) as exc_info:
        registry.run()

    assert "Simulated S3 Upload Error" in str(exc_info.value)

    mock_remove.assert_called_once()

    removed_path = mock_remove.call_args[0][0]
    assert removed_path.endswith(".json")