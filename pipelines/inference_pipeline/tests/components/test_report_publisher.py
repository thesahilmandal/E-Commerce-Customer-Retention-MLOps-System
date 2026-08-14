import os
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.components.report_publisher import ReportPublisher
from pipelines.inference_pipeline.src.entity.artifact_entity import ReportPublisherArtifact


@pytest.fixture
def mock_upstream_artifacts(temp_workspace):
    """
    Creates isolated dummy files and mock artifacts representing the outputs
    of upstream pipeline components.
    """
    csv_path = temp_workspace / "churn_predictions.csv"
    csv_path.write_text("customer_id,score\nC1,0.8")

    telemetry_path = temp_workspace / "telemetry.parquet"
    telemetry_path.write_text("mock_parquet_data")

    ml_meta = temp_workspace / "ml_meta.json"
    ml_meta.write_text('{"stage": "model_loader", "status": "ok"}')

    fm_meta = temp_workspace / "fm_meta.json"
    fm_meta.write_text('{"stage": "feature_builder", "status": "ok"}')

    rg_meta = temp_workspace / "rg_meta.json"
    rg_meta.write_text('{"stage": "report_generator", "status": "ok"}')

    ml_art = MagicMock()
    ml_art.metadata_file_path = str(ml_meta)

    fm_art = MagicMock()
    fm_art.metadata_file_path = str(fm_meta)

    iv_art = MagicMock()

    rg_art = MagicMock()
    rg_art.csv_report_path = str(csv_path)
    rg_art.telemetry_log_path = str(telemetry_path)
    rg_art.metadata_file_path = str(rg_meta)

    return ml_art, fm_art, iv_art, rg_art


def test_report_publisher_initialization_success(mock_pipeline_context):
    """
    Tests successful initialization of the ReportPublisher component.
    """
    publisher = ReportPublisher(context=mock_pipeline_context)

    assert publisher.context is mock_pipeline_context
    assert publisher.config is not None
    assert "05_report_publisher" in publisher.config.publisher_root_dir


def test_report_publisher_initialization_failure(mock_pipeline_context):
    """
    Tests that ReportPublisher raises a CustomException if configuration extraction fails.
    """
    with patch(
        "pipelines.inference_pipeline.src.entity.config_entity.ReportPublisherConfig.from_context"
    ) as mock_from_context:
        mock_from_context.side_effect = Exception("Config error")
        with pytest.raises(CustomException) as exc_info:
            ReportPublisher(context=mock_pipeline_context)

        assert "Config error" in str(exc_info.value)


def test_report_publisher_run_success(mock_pipeline_context, mock_upstream_artifacts):
    """
    Tests the complete happy path of the Report Publisher workflow.
    Validates pre-publication checks, master ledger compilation, URI partitioning, and S3 streaming.
    """
    ml_art, fm_art, iv_art, rg_art = mock_upstream_artifacts
    publisher = ReportPublisher(context=mock_pipeline_context)

    artifact = publisher.run(
        model_loader_artifact=ml_art,
        feature_matrix_artifact=fm_art,
        validator_artifact=iv_art,
        report_generator_artifact=rg_art
    )

    # Validate output artifact
    assert isinstance(artifact, ReportPublisherArtifact)
    assert artifact.published_business_report_uri.startswith("s3://")
    assert "churn_predictions.csv" in artifact.published_business_report_uri
    assert artifact.published_telemetry_log_uri.startswith("s3://")
    assert "telemetry.parquet" in artifact.published_telemetry_log_uri
    assert os.path.exists(artifact.metadata_file_path)

    # Validate Hive partitioning in URIs (year=YYYY/month=MM/day=DD)
    now = datetime.now(timezone.utc)
    expected_partition = f"year={now.year}/month={now.month:02d}/day={now.day:02d}"
    assert expected_partition in artifact.published_business_report_uri
    assert expected_partition in artifact.published_telemetry_log_uri

    # Validate Master Ledger content
    with open(artifact.metadata_file_path, "r", encoding="utf-8") as f:
        master_ledger = json.load(f)

        assert master_ledger["inference_run_id"] == "test_run_123"
        assert master_ledger["pipeline_status"] == "SUCCESS"
        assert "execution_timestamp_utc" in master_ledger
        assert "publisher_execution_time_seconds" in master_ledger
        assert (
            master_ledger["published_artifact_lineage"]["business_report_s3_uri"]
            == artifact.published_business_report_uri
        )

        # Verify nested stage metadata was successfully aggregated
        stage_telemetry = master_ledger["stage_telemetry"]
        assert stage_telemetry["model_loader"]["stage"] == "model_loader"
        assert (
            stage_telemetry["feature_matrix_builder"]["stage"]
            == "feature_builder"
        )
        assert stage_telemetry["report_generator"]["stage"] == "report_generator"

    # Validate exactly 3 S3 upload calls occurred (CSV, Parquet, JSON Ledger)
    assert mock_pipeline_context.s3_sync.upload_file.call_count == 3


def test_report_publisher_missing_business_report(
    mock_pipeline_context, mock_upstream_artifacts
):
    """
    Tests that publication halts securely if the business report CSV is missing locally.
    """
    ml_art, fm_art, iv_art, rg_art = mock_upstream_artifacts

    # Simulate missing CSV
    os.remove(rg_art.csv_report_path)

    publisher = ReportPublisher(context=mock_pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        publisher.run(
            model_loader_artifact=ml_art,
            feature_matrix_artifact=fm_art,
            validator_artifact=iv_art,
            report_generator_artifact=rg_art
        )

    assert "Business CSV report not found" in str(exc_info.value)
    assert mock_pipeline_context.s3_sync.upload_file.call_count == 0


def test_report_publisher_missing_telemetry_log(
    mock_pipeline_context, mock_upstream_artifacts
):
    """
    Tests that publication halts securely if the MLOps telemetry Parquet is missing locally.
    """
    ml_art, fm_art, iv_art, rg_art = mock_upstream_artifacts

    # Simulate missing telemetry parquet
    os.remove(rg_art.telemetry_log_path)

    publisher = ReportPublisher(context=mock_pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        publisher.run(
            model_loader_artifact=ml_art,
            feature_matrix_artifact=fm_art,
            validator_artifact=iv_art,
            report_generator_artifact=rg_art
        )

    assert "Telemetry log Parquet not found" in str(exc_info.value)
    assert mock_pipeline_context.s3_sync.upload_file.call_count == 0


def test_report_publisher_missing_stage_metadata_fallback(
    mock_pipeline_context, mock_upstream_artifacts
):
    """
    Tests that if an upstream component's metadata JSON is missing, the master ledger
    gracefully records a warning rather than crashing the final publication.
    """
    ml_art, fm_art, iv_art, rg_art = mock_upstream_artifacts

    # Simulate missing metadata for model_loader
    os.remove(ml_art.metadata_file_path)

    publisher = ReportPublisher(context=mock_pipeline_context)

    artifact = publisher.run(
        model_loader_artifact=ml_art,
        feature_matrix_artifact=fm_art,
        validator_artifact=iv_art,
        report_generator_artifact=rg_art
    )

    # Validate that publication still succeeded
    assert os.path.exists(artifact.metadata_file_path)

    # Validate the fallback warning in the ledger
    with open(artifact.metadata_file_path, "r", encoding="utf-8") as f:
        master_ledger = json.load(f)
        assert "warning" in master_ledger["stage_telemetry"]["model_loader"]
        assert (
            "Metadata file not found"
            in master_ledger["stage_telemetry"]["model_loader"]["warning"]
        )

        # Other stages should still be populated
        assert (
            master_ledger["stage_telemetry"]["feature_matrix_builder"]["stage"]
            == "feature_builder"
        )


def test_report_publisher_s3_upload_failure(
    mock_pipeline_context, mock_upstream_artifacts
):
    """
    Tests that an S3 connection/permission failure during the upload streaming process
    is caught and securely wrapped in a CustomException.
    """
    ml_art, fm_art, iv_art, rg_art = mock_upstream_artifacts
    publisher = ReportPublisher(context=mock_pipeline_context)

    # Simulate S3 upload failure on the first call
    mock_pipeline_context.s3_sync.upload_file.side_effect = Exception(
        "AWS STS Token Expired"
    )

    with pytest.raises(CustomException) as exc_info:
        publisher.run(
            model_loader_artifact=ml_art,
            feature_matrix_artifact=fm_art,
            validator_artifact=iv_art,
            report_generator_artifact=rg_art
        )

    assert "AWS STS Token Expired" in str(exc_info.value)
    assert mock_pipeline_context.s3_sync.upload_file.call_count == 1