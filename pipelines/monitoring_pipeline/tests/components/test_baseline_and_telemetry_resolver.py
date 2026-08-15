import os
import json
import pytest
from unittest.mock import patch, MagicMock

from pipelines.monitoring_pipeline.src.components.baseline_and_telemetry_resolver import BaselineAndTelemetryResolver
from pipelines.monitoring_pipeline.src.entity.config_entity import BaselineAndTelemetryResolverConfig
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def resolver_config(mock_context):
    """Provides a valid configuration for the Resolver component."""
    return BaselineAndTelemetryResolverConfig.get_config(mock_context)


@pytest.fixture
def resolver(resolver_config, mock_context):
    """
    Provides a configured BaselineAndTelemetryResolver instance.
    The DuckDB connection is mocked to prevent real HTTP/S3 network calls during tests.
    The S3Sync client is mocked internally to avoid real AWS interactions.
    """
    mock_context.duckdb_con = MagicMock()
    instance = BaselineAndTelemetryResolver(config=resolver_config, context=mock_context)
    
    # Isolate S3 operations by directly injecting a mock into the instantiated object.
    # This prevents the botocore 403 Forbidden errors encountered during fixture creation.
    instance.s3_sync = MagicMock()
    
    return instance


@patch("pipelines.monitoring_pipeline.src.components.baseline_and_telemetry_resolver.os.rename")
@patch("pipelines.monitoring_pipeline.src.components.baseline_and_telemetry_resolver.SharedFeatureGenerator")
def test_resolver_successful_execution_path(mock_feature_gen, mock_os_rename, resolver):
    """
    Validates the end-to-end happy path execution of the resolver.
    Ensures all baseline artifacts are resolved and telemetry fetch logic is triggered.
    """
    def mock_download_file(s3_uri, local_path):
        if "model_state.json" in s3_uri:
            with open(local_path, "w") as f:
                json.dump({
                    "champion_run_id": "champion_123",
                    "s3_baseline_metrics_path": "s3://dummy/metrics.json",
                    "s3_reference_distributions_path": "s3://dummy/dist.json",
                    "s3_shap_importance_path": "s3://dummy/shap.json"
                }, f)
        else:
            with open(local_path, "w") as f:
                json.dump({}, f)

    resolver.s3_sync.download_file.side_effect = mock_download_file
    
    mock_feature_gen_instance = mock_feature_gen.return_value
    mock_feature_gen_instance.get_feature_query.return_value = "SELECT * FROM dummy_labels"

    # Run component
    artifact = resolver.run()

    # Assert correct lineage resolution
    assert artifact.champion_run_id == "champion_123"
    assert os.path.exists(artifact.metadata_file_path)

    # Assert DuckDB was commanded to fetch telemetry
    exec_calls = resolver.context.duckdb_con.execute.call_args_list
    assert len(exec_calls) >= 2
    queries = [call[0][0] for call in exec_calls]
    assert any("read_parquet" in q for q in queries)
    
    # Assert os.rename was called since DuckDB didn't physically create the file
    mock_os_rename.assert_called_once()


def test_resolver_synthetic_shap_fallback_generation(resolver):
    """
    Validates that if the active champion pointer is missing the SHAP artifact, 
    the resolver automatically generates a synthetic fallback using the reference distributions 
    to prevent downstream pipeline crashes.
    """
    def mock_download_file(s3_uri, local_path):
        if "model_state.json" in s3_uri:
            with open(local_path, "w") as f:
                json.dump({
                    "champion_run_id": "champion_123",
                    "s3_baseline_metrics_path": "s3://dummy/metrics.json",
                    "s3_reference_distributions_path": "s3://dummy/dist.json"
                    # Intentionally missing s3_shap_importance_path
                }, f)
        elif "dist.json" in local_path:
            with open(local_path, "w") as f:
                json.dump({
                    "distributions": {
                        "feature_1": {}, 
                        "feature_2": {}, 
                        resolver.target_col: {} # Should be excluded
                    }
                }, f)

    resolver.s3_sync.download_file.side_effect = mock_download_file

    resolver._resolve_champion_baselines()

    # Verify synthetic SHAP was created and target columns were excluded
    assert os.path.exists(resolver.config.shap_importance_file_path)
    with open(resolver.config.shap_importance_file_path, "r") as f:
        shap_data = json.load(f)

    assert "feature_importance" in shap_data
    features = [item["feature_name"] for item in shap_data["feature_importance"]]
    assert "feature_1" in features
    assert "feature_2" in features
    assert resolver.target_col not in features


def test_resolver_zero_traffic_current_telemetry_fallback(resolver):
    """
    Validates graceful handling of '0-traffic' days (HTTP 404).
    The resolver must catch the S3 error and execute a fallback COPY query
    to create an empty parquet file with the correct schema.
    """
    def mock_execute(query):
        # Simulate S3 Not Found for the read_parquet command
        if "read_parquet" in query:
            raise Exception("HTTP 404 Not Found")
        return None

    resolver.context.duckdb_con.execute.side_effect = mock_execute

    status = resolver._fetch_current_telemetry(resolver.context.duckdb_con)
    assert status == "SKIPPED_ZERO_TRAFFIC_DAY"

    # Verify the fallback query was issued to duckdb
    exec_calls = resolver.context.duckdb_con.execute.call_args_list
    fallback_query = exec_calls[-1][0][0]
    assert "WHERE 1=0" in fallback_query
    assert resolver.customer_id_col in fallback_query


def test_resolver_immature_system_lookback_fallback(resolver):
    """
    Validates graceful handling of an immature system where the lookback date 
    precedes the system's go-live date (NoSuchKey). Must generate empty parquets.
    """
    def mock_execute(query):
        # Simulate S3 Not Found for lookback telemetry
        if "read_parquet" in query and "lookback_telemetry" in resolver.config.lookback_partition_suffix: # Simplification
            raise Exception("NoSuchKey")
        if "read_parquet" in query: # Catch generic read_parquet just in case
             raise Exception("NoSuchKey")
        return None

    resolver.context.duckdb_con.execute.side_effect = mock_execute

    status = resolver._fetch_lookback_telemetry_and_labels(resolver.context.duckdb_con)
    assert status == "SKIPPED_SYSTEM_IMMATURE"

    # Verify two fallback queries were issued (one for telemetry, one for labels)
    exec_calls = resolver.context.duckdb_con.execute.call_args_list
    queries = [call[0][0] for call in exec_calls]
    
    assert any("WHERE 1=0" in q and resolver.prediction_col in q for q in queries)
    assert any("WHERE 1=0" in q and resolver.target_col in q for q in queries)


def test_resolver_corrupted_pointer_raises_exception(resolver):
    """
    Validates that a corrupted global pointer (missing champion_run_id) 
    fails fast and raises a CustomException.
    """
    def mock_download_file(s3_uri, local_path):
        with open(local_path, "w") as f:
            # Missing champion_run_id completely
            json.dump({
                "s3_baseline_metrics_path": "s3://dummy/metrics.json",
                "s3_reference_distributions_path": "s3://dummy/dist.json"
            }, f)

    resolver.s3_sync.download_file.side_effect = mock_download_file

    with pytest.raises(CustomException):
        resolver._resolve_champion_baselines()