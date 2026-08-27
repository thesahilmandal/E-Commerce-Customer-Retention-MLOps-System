
import pytest
from unittest.mock import MagicMock, patch

from pipelines.monitoring_pipeline.src.components.artifact_publisher import (
    ArtifactPublisher,
)
from pipelines.monitoring_pipeline.src.entity.config_entity import (
    ArtifactPublisherConfig,
)
from pipelines.monitoring_pipeline.src.entity.artifact_entity import (
    ArtifactPublisherArtifact,
)
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def publisher_config(tmp_path) -> ArtifactPublisherConfig:
    """Provides a mocked configuration for the Artifact Publisher."""
    return ArtifactPublisherConfig(
        publisher_root_dir=str(tmp_path),
        s3_bucket_name="test-bucket",
        s3_monitoring_output_prefix="monitoring_output",
        s3_audit_reports_dir="audit",
        s3_action_tokens_dir="tokens",
        s3_matured_evaluations_dir="evals",
        s3_metadata_dir="meta",
        run_id="run_123",
        execution_date="2026-08-27",
    )


@pytest.fixture
def publisher(
    publisher_config: ArtifactPublisherConfig,
    mock_pipeline_context: MagicMock,
    dummy_resolver_artifact: MagicMock,
    dummy_drift_artifact: MagicMock,
    dummy_performance_artifact: MagicMock,
    dummy_rule_engine_artifact: MagicMock,
) -> ArtifactPublisher:
    """Yields a fully initialized ArtifactPublisher with a mocked boto3 S3 client."""
    with patch(
        "pipelines.monitoring_pipeline.src.components.artifact_publisher.boto3.client"
    ) as mock_boto:
        pub = ArtifactPublisher(
            config=publisher_config,
            context=mock_pipeline_context,
            resolver_artifact=dummy_resolver_artifact,
            drift_artifact=dummy_drift_artifact,
            performance_artifact=dummy_performance_artifact,
            rule_engine_artifact=dummy_rule_engine_artifact,
        )

        pub.s3_client = mock_boto.return_value
        return pub


def test_generate_hive_partition_success(
    publisher: ArtifactPublisher,
) -> None:
    partition = publisher._generate_hive_partition()

    assert partition == "year=2026/month=08/day=27"


def test_generate_hive_partition_invalid_date(
    publisher: ArtifactPublisher,
) -> None:
    publisher.config = ArtifactPublisherConfig(
        publisher_root_dir=publisher.config.publisher_root_dir,
        s3_bucket_name="b",
        s3_monitoring_output_prefix="p",
        s3_audit_reports_dir="a",
        s3_action_tokens_dir="t",
        s3_matured_evaluations_dir="m",
        s3_metadata_dir="md",
        run_id="run_123",
        execution_date="invalid-date",
    )

    with pytest.raises(CustomException) as exc_info:
        publisher._generate_hive_partition()

    assert "does not match format" in str(exc_info.value)


@patch(
    "pipelines.monitoring_pipeline.src.components.artifact_publisher.write_json_file"
)
def test_compile_master_metadata(
    mock_write_json: MagicMock,
    publisher: ArtifactPublisher,
) -> None:
    master_path = publisher._compile_master_metadata()

    assert master_path.endswith("master_metadata_run_123.json")
    mock_write_json.assert_called_once()

    saved_content = mock_write_json.call_args[1]["content"]

    assert saved_content["orchestration"]["run_id"] == "run_123"
    assert saved_content["orchestration"]["execution_date"] == "2026-08-27"
    assert "components" in saved_content
    assert (
        "baseline_and_telemetry_resolver"
        in saved_content["components"]
    )
    assert (
        "statistical_drift_calculator"
        in saved_content["components"]
    )


def test_load_json_missing_file(
    publisher: ArtifactPublisher,
) -> None:
    data = publisher._load_json("/path/does/not/exist.json")

    assert data == {}


@patch("builtins.open")
def test_load_json_invalid_format(
    mock_open: MagicMock,
    publisher: ArtifactPublisher,
) -> None:
    mock_open.return_value.__enter__.return_value.read.return_value = (
        "invalid json"
    )

    with patch("os.path.exists", return_value=True):
        with pytest.raises(CustomException) as exc_info:
            publisher._load_json("dummy.json")

        assert "Expecting value" in str(exc_info.value)


def test_build_upload_map(
    publisher: ArtifactPublisher,
) -> None:
    partition = "year=2026/month=08/day=27"
    master_meta = "/tmp/master_metadata.json"

    upload_map = publisher._build_upload_map(
        partition,
        master_meta,
    )

    assert len(upload_map) == 4

    audit_uri = upload_map[
        publisher.rule_engine_artifact.monitoring_report_file_path
    ]

    assert audit_uri == (
        "s3://test-bucket/"
        "monitoring_output/audit/"
        "year=2026/month=08/day=27/"
        "report_run_123.json"
    )

    token_uri = upload_map[
        publisher.rule_engine_artifact.need_update_file_path
    ]

    assert token_uri == (
        "s3://test-bucket/"
        "monitoring_output/tokens/"
        "year=2026/month=08/day=27/"
        "need_update_run_123.json"
    )

    eval_uri = upload_map[
        publisher.resolver_artifact.lookback_labels_file_path
    ]

    assert eval_uri == (
        "s3://test-bucket/"
        "monitoring_output/evals/"
        "year=2026/month=08/day=27/"
        "evaluation_run_123.parquet"
    )

    meta_uri = upload_map[master_meta]

    assert meta_uri == (
        "s3://test-bucket/"
        "monitoring_output/meta/"
        "year=2026/month=08/day=27/"
        "metadata_run_123.json"
    )


@patch("os.path.exists", return_value=True)
def test_validate_local_artifacts_success(
    mock_exists: MagicMock,
    publisher: ArtifactPublisher,
) -> None:
    publisher._validate_local_artifacts(
        ["/path/1", "/path/2"]
    )

    assert mock_exists.call_count == 2


@patch("os.path.exists", return_value=False)
def test_validate_local_artifacts_missing(
    mock_exists: MagicMock,
    publisher: ArtifactPublisher,
) -> None:
    with pytest.raises(CustomException) as exc_info:
        publisher._validate_local_artifacts(
            ["/path/missing"]
        )

    assert "required artifacts are missing locally" in str(
        exc_info.value
    )


def test_upload_single_file_success(
    publisher: ArtifactPublisher,
) -> None:
    local_path = "/tmp/local_file.json"
    s3_uri = "s3://my-bucket/my-prefix/file.json"

    publisher._upload_single_file(
        local_path,
        s3_uri,
    )

    publisher.s3_client.upload_file.assert_called_once_with(
        Filename=local_path,
        Bucket="my-bucket",
        Key="my-prefix/file.json",
    )


def test_upload_single_file_invalid_scheme(
    publisher: ArtifactPublisher,
) -> None:
    with pytest.raises(CustomException) as exc_info:
        publisher._upload_single_file(
            "/tmp/file",
            "gcs://bucket/file",
        )

    assert "Invalid S3 URI scheme" in str(exc_info.value)


def test_upload_single_file_boto3_failure(
    publisher: ArtifactPublisher,
) -> None:
    publisher.s3_client.upload_file.side_effect = Exception(
        "S3 Access Denied"
    )

    with pytest.raises(CustomException) as exc_info:
        publisher._upload_single_file(
            "/tmp/file",
            "s3://bucket/file",
        )

    assert "S3 Access Denied" in str(exc_info.value)


@patch.object(ArtifactPublisher, "_generate_hive_partition")
@patch.object(ArtifactPublisher, "_compile_master_metadata")
@patch.object(ArtifactPublisher, "_build_upload_map")
@patch.object(ArtifactPublisher, "_validate_local_artifacts")
@patch.object(ArtifactPublisher, "_upload_artifacts")
def test_run_success(
    mock_upload: MagicMock,
    mock_validate: MagicMock,
    mock_build_map: MagicMock,
    mock_compile_meta: MagicMock,
    mock_gen_hive: MagicMock,
    publisher: ArtifactPublisher,
) -> None:
    mock_gen_hive.return_value = "year=2026/month=08/day=27"
    mock_compile_meta.return_value = "/tmp/master_meta.json"
    mock_build_map.return_value = {
        "/tmp/local": "s3://bucket/key"
    }

    mock_upload.return_value = {
        "audit_report": "s3://audit",
        "action_token": "s3://token",
        "matured_evaluation": "s3://eval",
        "metadata_ledger": "s3://meta",
    }

    artifact = publisher.run()

    assert isinstance(artifact, ArtifactPublisherArtifact)
    assert artifact.s3_audit_report_uri == "s3://audit"
    assert artifact.s3_action_token_uri == "s3://token"

    mock_gen_hive.assert_called_once()
    mock_compile_meta.assert_called_once()
    mock_build_map.assert_called_once()
    mock_validate.assert_called_once()
    mock_upload.assert_called_once()