import os
import json
import pytest
from unittest.mock import patch

from pipelines.monitoring_pipeline.src.components.artifact_publisher import ArtifactPublisher
from pipelines.monitoring_pipeline.src.entity.config_entity import ArtifactPublisherConfig
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def publisher_config(mock_context):
    """Provides a valid configuration for the Artifact Publisher."""
    return ArtifactPublisherConfig.get_config(mock_context)


@pytest.fixture
@patch("pipelines.monitoring_pipeline.src.components.artifact_publisher.boto3.client")
def publisher(
    mock_boto3, 
    publisher_config, 
    mock_context, 
    mock_resolver_artifact, 
    mock_drift_artifact, 
    mock_performance_artifact, 
    mock_rule_engine_artifact
):
    """
    Provides a configured ArtifactPublisher instance.
    The S3 boto3 client is mocked to avoid real AWS interactions.
    """
    return ArtifactPublisher(
        config=publisher_config,
        context=mock_context,
        resolver_artifact=mock_resolver_artifact,
        drift_artifact=mock_drift_artifact,
        performance_artifact=mock_performance_artifact,
        rule_engine_artifact=mock_rule_engine_artifact
    )


def test_generate_hive_partition(publisher):
    """Validates accurate temporal parsing to a standard Hive partition format."""
    # Override execution date for explicit testing
    object.__setattr__(publisher.config, 'execution_date', '2026-08-14')
    
    partition = publisher._generate_hive_partition()
    assert partition == "year=2026/month=08/day=14"


def test_generate_hive_partition_invalid_date(publisher):
    """Validates that malformed execution dates trigger a safe pipeline failure."""
    object.__setattr__(publisher.config, 'execution_date', '14-08-2026')  # Wrong format
    
    with pytest.raises(CustomException):
        publisher._generate_hive_partition()


def test_compile_master_metadata(publisher):
    """
    Validates that the publisher successfully aggregates metadata from all upstream
    components into a single Master Execution Ledger JSON file.
    """
    # 1. Create dummy upstream metadata files
    upstream_meta_paths = [
        publisher.resolver_artifact.metadata_file_path,
        publisher.drift_artifact.metadata_file_path,
        publisher.performance_artifact.metadata_file_path,
        publisher.rule_engine_artifact.metadata_file_path
    ]
    
    for i, path in enumerate(upstream_meta_paths):
        with open(path, "w") as f:
            json.dump({f"component_{i}": "success"}, f)

    # 2. Compile master metadata
    master_metadata_path = publisher._compile_master_metadata()

    # 3. Validate master ledger contents
    assert os.path.exists(master_metadata_path)
    with open(master_metadata_path, "r") as f:
        master_data = json.load(f)
        
    assert "orchestration" in master_data
    assert master_data["orchestration"]["run_id"] == publisher.config.run_id
    assert "components" in master_data
    assert "baseline_and_telemetry_resolver" in master_data["components"]
    assert master_data["components"]["baseline_and_telemetry_resolver"]["component_0"] == "success"


def test_build_upload_map(publisher):
    """Validates the correct generation of remote S3 URIs mapped from local file paths."""
    partition_prefix = "year=2026/month=08/day=14"
    master_metadata_path = "/dummy/master.json"
    
    upload_map = publisher._build_upload_map(partition_prefix, master_metadata_path)
    
    # Check that keys correspond to the expected local artifacts
    assert publisher.rule_engine_artifact.monitoring_report_file_path in upload_map
    assert publisher.rule_engine_artifact.need_update_file_path in upload_map
    assert publisher.resolver_artifact.lookback_labels_file_path in upload_map
    assert master_metadata_path in upload_map

    # Check that values form proper S3 URIs
    for s3_uri in upload_map.values():
        assert s3_uri.startswith("s3://mock-bucket/monitoring_output/")
        assert partition_prefix in s3_uri


def test_validate_local_artifacts_missing_file(publisher):
    """Validates that a missing mandatory artifact fails the pipeline fast before upload."""
    invalid_paths = ["/invalid/path/1.json", "/invalid/path/2.json"]
    
    with pytest.raises(CustomException):
        publisher._validate_local_artifacts(invalid_paths)


def test_upload_single_file_invalid_scheme(publisher):
    """Validates protocol enforcement for S3 uploads."""
    local_path = "dummy.json"
    invalid_uri = "gcs://mock-bucket/path/file.json"
    
    with pytest.raises(CustomException):
        publisher._upload_single_file(local_path, invalid_uri)


def test_artifact_publisher_run_e2e(publisher):
    """
    Validates the end-to-end component run. Creates necessary dummy files locally,
    executes the run sequence, and asserts that boto3 upload is called properly.
    """
    # 1. Create dummy local files to pass validation
    paths_to_create = [
        publisher.resolver_artifact.metadata_file_path,
        publisher.drift_artifact.metadata_file_path,
        publisher.performance_artifact.metadata_file_path,
        publisher.rule_engine_artifact.metadata_file_path,
        publisher.rule_engine_artifact.monitoring_report_file_path,
        publisher.rule_engine_artifact.need_update_file_path,
        publisher.resolver_artifact.lookback_labels_file_path
    ]
    for path in paths_to_create:
        # Ensure parent dirs exist just in case tmp_path is flat
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("{}")

    # 2. Execute Run
    artifact = publisher.run()

    # 3. Assert S3 Client Interactions
    # Total uploads expected: audit_report, action_token, matured_evaluation, metadata_ledger
    assert publisher.s3_client.upload_file.call_count == 4
    
    # Inspect arguments of a specific call (e.g., the action token upload)
    upload_calls = publisher.s3_client.upload_file.call_args_list
    bucket_args = [call.kwargs['Bucket'] for call in upload_calls]
    key_args = [call.kwargs['Key'] for call in upload_calls]
    
    assert all(bucket == "mock-bucket" for bucket in bucket_args)
    assert any("action_tokens" in key for key in key_args)
    assert any("audit_reports" in key for key in key_args)
    assert any("metadata" in key for key in key_args)

    # 4. Validate output Artifact schema
    assert artifact.s3_audit_report_uri.startswith("s3://")
    assert artifact.s3_action_token_uri.startswith("s3://")
    assert artifact.s3_matured_evaluation_uri.startswith("s3://")
    assert artifact.s3_metadata_uri.startswith("s3://")