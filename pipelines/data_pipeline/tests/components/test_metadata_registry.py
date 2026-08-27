import hashlib
import os
import pytest
from unittest.mock import MagicMock, patch

from pipelines.data_pipeline.src.components.metadata_registry import MetadataRegistry
from shared_core.exceptions.custom_exception import CustomException


def _create_mock_execute(count_val: int = 42, describe_cols: list | None = None) -> callable:
    if describe_cols is None:
        describe_cols = [("col_a", "INT"), ("col_b", "VARCHAR")]

    def side_effect(query: str) -> MagicMock:
        mock_res = MagicMock()
        upper_query = query.upper()
        if "COUNT(*)" in upper_query:
            mock_res.fetchone.return_value = (count_val,)
        elif "DESCRIBE" in upper_query:
            mock_res.fetchall.return_value = describe_cols
        else:
            mock_res.fetchone.return_value = None
            mock_res.fetchall.return_value = []
        return mock_res

    return side_effect


def test_metadata_registry_initialization(mock_pipeline_context: MagicMock) -> None:
    registry = MetadataRegistry(mock_pipeline_context)
    
    assert registry.base_uri == "s3://test-feature-store-bucket/feature_store"
    assert registry.artifact_name == "test_dataset.parquet"
    assert registry.metadata_name == "test_metadata.json"
    assert registry.context == mock_pipeline_context


def test_get_target_uris(mock_pipeline_context: MagicMock) -> None:
    registry = MetadataRegistry(mock_pipeline_context)
    artifact_uri, metadata_uri = registry._get_target_uris()
    
    expected_artifact = "s3://test-feature-store-bucket/feature_store/test_run_12345/test_dataset.parquet"
    expected_metadata = "s3://test-feature-store-bucket/feature_store/test_run_12345/test_metadata.json"
    
    assert artifact_uri == expected_artifact
    assert metadata_uri == expected_metadata


def test_extract_parquet_telemetry_success(mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = _create_mock_execute(
        count_val=150, 
        describe_cols=[("customer_id", "VARCHAR"), ("payment_value", "DOUBLE")]
    )
    registry = MetadataRegistry(mock_pipeline_context)
    
    row_count, columns_hash = registry._extract_parquet_telemetry("s3://fake/uri")
    
    expected_string = "customer_idpayment_value"
    expected_hash = hashlib.md5(expected_string.encode("utf-8")).hexdigest()[:10]
    
    assert row_count == 150
    assert columns_hash == expected_hash


def test_extract_parquet_telemetry_failure(mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = Exception("DuckDB S3 read error")
    registry = MetadataRegistry(mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        registry._extract_parquet_telemetry("s3://fake/uri")
        
    assert "DuckDB S3 read error" in str(exc_info.value)


@patch("pipelines.data_pipeline.src.components.metadata_registry.datetime")
def test_build_metadata_payload(mock_datetime: MagicMock, mock_pipeline_context: MagicMock) -> None:
    mock_datetime.now.return_value.strftime.return_value = "2023-06-01T12:00:00Z"
    
    registry = MetadataRegistry(mock_pipeline_context)
    payload = registry._build_metadata_payload("s3://fake/uri", 500, "abc123hash")
    
    assert payload["run_id"] == "test_run_12345"
    assert payload["temporal_bounds"]["start_date"] == "2023-01-01"
    assert payload["temporal_bounds"]["end_date"] == "2023-06-01"
    assert payload["artifact_uri"] == "s3://fake/uri"
    assert payload["schema"]["columns_hash"] == "abc123hash"
    assert payload["schema"]["row_count"] == 500
    assert payload["timestamp_utc"] == "2023-06-01T12:00:00Z"


def test_run_success_and_cleanup(mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = _create_mock_execute()
    registry = MetadataRegistry(mock_pipeline_context)
    
    registry.run()
    
    mock_pipeline_context.s3_sync.upload_file.assert_called_once()
    call_kwargs = mock_pipeline_context.s3_sync.upload_file.call_args.kwargs
    
    local_file_path = call_kwargs["local_path"]
    s3_uri = call_kwargs["s3_uri"]
    
    assert s3_uri == "s3://test-feature-store-bucket/feature_store/test_run_12345/test_metadata.json"
    assert not os.path.exists(local_file_path)


@patch("pipelines.data_pipeline.src.components.metadata_registry.os.remove")
def test_run_cleanup_on_upload_failure(mock_remove: MagicMock, mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = _create_mock_execute()
    mock_pipeline_context.s3_sync.upload_file.side_effect = Exception("AWS S3 permissions error")
    
    registry = MetadataRegistry(mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        registry.run()
        
    assert "AWS S3 permissions error" in str(exc_info.value)
    
    mock_remove.assert_called_once()
    removed_path = mock_remove.call_args[0][0]
    assert removed_path.endswith(".json")


@patch("pipelines.data_pipeline.src.components.metadata_registry.os.remove")
def test_run_cleanup_os_error_suppressed(mock_remove: MagicMock, mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = _create_mock_execute()
    mock_remove.side_effect = OSError("File already locked or missing")
    
    registry = MetadataRegistry(mock_pipeline_context)
    
    registry.run()
    
    mock_pipeline_context.s3_sync.upload_file.assert_called_once()
    mock_remove.assert_called_once()