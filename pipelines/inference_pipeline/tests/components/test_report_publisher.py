import json

import pytest

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, mock_open, call

from pipelines.inference_pipeline.src.components.report_publisher import (
    ReportPublisher,
)
from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    ModelLoaderArtifact,
    FeatureMatrixBuilderArtifact,
    InferenceValidatorArtifact,
    ReportGeneratorArtifact,
)


@patch(
    "pipelines.inference_pipeline.src.components.report_publisher."
    "ReportPublisherConfig.from_context"
)
def test_report_publisher_initialization_failure(
    mock_from_context: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_from_context.side_effect = Exception(
        "Config initialization failed"
    )

    with pytest.raises(CustomException) as exc_info:
        ReportPublisher(context=mock_pipeline_context)

    assert "Config initialization failed" in str(exc_info.value)


@patch.object(ReportPublisher, "_publish_artifacts_to_s3")
@patch.object(ReportPublisher, "_compile_master_inference_ledger")
@patch.object(ReportPublisher, "_get_partition_suffix")
@patch.object(ReportPublisher, "_validate_local_artifacts")
def test_run_success(
    mock_validate: MagicMock,
    mock_get_suffix: MagicMock,
    mock_compile_ledger: MagicMock,
    mock_publish: MagicMock,
    mock_pipeline_context: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
    dummy_inference_validator_artifact: InferenceValidatorArtifact,
    dummy_report_generator_artifact: ReportGeneratorArtifact,
) -> None:
    publisher = ReportPublisher(context=mock_pipeline_context)

    mock_get_suffix.return_value = (
        "year=2026/month=08/day=26"
    )
    mock_compile_ledger.return_value = (
        publisher.config.metadata_file_path
    )

    artifact = publisher.run(
        model_loader_artifact=dummy_model_loader_artifact,
        feature_matrix_artifact=dummy_feature_matrix_artifact,
        validator_artifact=dummy_inference_validator_artifact,
        report_generator_artifact=dummy_report_generator_artifact,
    )

    s3_biz_uri = (
        f"{publisher.config.s3_business_reports_base_uri}"
        "/year=2026/month=08/day=26/"
        "churn_predictions.csv"
    )

    s3_tel_uri = (
        f"{publisher.config.s3_telemetry_logs_base_uri}"
        "/year=2026/month=08/day=26/"
        "telemetry.parquet"
    )

    assert (
        artifact.published_business_report_uri
        == s3_biz_uri
    )
    assert (
        artifact.published_telemetry_log_uri
        == s3_tel_uri
    )
    assert (
        artifact.metadata_file_path
        == publisher.config.metadata_file_path
    )

    mock_validate.assert_called_once_with(
        report_generator_artifact=dummy_report_generator_artifact
    )
    mock_get_suffix.assert_called_once()
    mock_compile_ledger.assert_called_once()
    mock_publish.assert_called_once()


@patch.object(ReportPublisher, "_validate_local_artifacts")
def test_run_failure(
    mock_validate: MagicMock,
    mock_pipeline_context: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
    dummy_inference_validator_artifact: InferenceValidatorArtifact,
    dummy_report_generator_artifact: ReportGeneratorArtifact,
) -> None:
    mock_validate.side_effect = Exception("Validation failed")

    publisher = ReportPublisher(context=mock_pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        publisher.run(
            model_loader_artifact=dummy_model_loader_artifact,
            feature_matrix_artifact=dummy_feature_matrix_artifact,
            validator_artifact=dummy_inference_validator_artifact,
            report_generator_artifact=dummy_report_generator_artifact,
        )

    assert "Validation failed" in str(exc_info.value)


@patch(
    "pipelines.inference_pipeline.src.components.report_publisher."
    "os.path.exists"
)
def test_validate_local_artifacts_success(
    mock_exists: MagicMock,
    mock_pipeline_context: MagicMock,
    dummy_report_generator_artifact: ReportGeneratorArtifact,
) -> None:
    mock_exists.return_value = True

    publisher = ReportPublisher(context=mock_pipeline_context)
    
    mock_exists.reset_mock()  # Reset calls from initialization
    
    publisher._validate_local_artifacts(
        dummy_report_generator_artifact
    )
    
    assert mock_exists.call_count == 2
    
    mock_exists.assert_any_call(
        dummy_report_generator_artifact.csv_report_path
    )
    mock_exists.assert_any_call(
        dummy_report_generator_artifact.telemetry_log_path
    )


@pytest.mark.parametrize(
    "csv_exists, parquet_exists, expected_error",
    [
        (
            False,
            True,
            "Business CSV report not found",
        ),
        (
            True,
            False,
            "Telemetry log Parquet not found",
        ),
    ],
)
@patch(
    "pipelines.inference_pipeline.src.components.report_publisher."
    "os.path.exists"
)
def test_validate_local_artifacts_failures(
    mock_exists: MagicMock,
    csv_exists: bool,
    parquet_exists: bool,
    expected_error: str,
    mock_pipeline_context: MagicMock,
    dummy_report_generator_artifact: ReportGeneratorArtifact,
) -> None:
    def side_effect(path: str) -> bool:
        if (
            path
            == dummy_report_generator_artifact.csv_report_path
        ):
            return csv_exists

        if (
            path
            == dummy_report_generator_artifact.telemetry_log_path
        ):
            return parquet_exists

        return True

    mock_exists.side_effect = side_effect

    publisher = ReportPublisher(context=mock_pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        publisher._validate_local_artifacts(
            dummy_report_generator_artifact
        )

    assert expected_error in str(exc_info.value)


@patch(
    "pipelines.inference_pipeline.src.components.report_publisher."
    "datetime"
)
def test_get_partition_suffix(
    mock_datetime: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_now = datetime(
        2026,
        8,
        26,
        tzinfo=timezone.utc,
    )

    mock_datetime.now.return_value = mock_now

    publisher = ReportPublisher(context=mock_pipeline_context)

    suffix = publisher._get_partition_suffix()

    assert suffix == "year=2026/month=08/day=26"


@patch(
    "pipelines.inference_pipeline.src.components.report_publisher."
    "os.path.exists"
)
def test_compile_master_inference_ledger_success(
    mock_exists: MagicMock,
    mock_pipeline_context: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
    dummy_inference_validator_artifact: InferenceValidatorArtifact,
    dummy_report_generator_artifact: ReportGeneratorArtifact,
) -> None:
    publisher = ReportPublisher(context=mock_pipeline_context)

    def exists_side_effect(path: str) -> bool:
        # Simulate missing model loader metadata,
        # existing metadata for all other stages.
        if (
            path
            == dummy_model_loader_artifact.metadata_file_path
        ):
            return False

        return True

    mock_exists.side_effect = exists_side_effect

    dummy_metadata = {
        "dummy_key": "dummy_value"
    }

    m_open = mock_open(
        read_data=json.dumps(dummy_metadata)
    )

    with patch("builtins.open", m_open):
        ledger_path = (
            publisher._compile_master_inference_ledger(
                model_loader_artifact=dummy_model_loader_artifact,
                feature_matrix_artifact=dummy_feature_matrix_artifact,
                validator_artifact=dummy_inference_validator_artifact,
                report_generator_artifact=dummy_report_generator_artifact,
                execution_time=10.5,
                s3_business_uri="s3://dummy/biz.csv",
                s3_telemetry_uri="s3://dummy/tel.parquet",
            )
        )

    assert (
        ledger_path
        == publisher.config.metadata_file_path
    )

    write_calls = [
        mock_call
        for mock_call in m_open.mock_calls
        if mock_call[0] == "().write"
    ]

    written_data = "".join(
        mock_call[1][0]
        for mock_call in write_calls
    )

    parsed_ledger = json.loads(written_data)

    assert (
        parsed_ledger["inference_run_id"]
        == mock_pipeline_context.run_id
    )
    assert (
        parsed_ledger["pipeline_status"]
        == "SUCCESS"
    )
    assert (
        parsed_ledger[
            "publisher_execution_time_seconds"
        ]
        == 10.5
    )

    assert (
        parsed_ledger["published_artifact_lineage"][
            "business_report_s3_uri"
        ]
        == "s3://dummy/biz.csv"
    )

    assert (
        parsed_ledger["published_artifact_lineage"][
            "telemetry_log_s3_uri"
        ]
        == "s3://dummy/tel.parquet"
    )

    assert (
        "warning"
        in parsed_ledger["stage_telemetry"][
            "model_loader"
        ]
    )

    assert (
        parsed_ledger["stage_telemetry"][
            "feature_matrix_builder"
        ]
        == dummy_metadata
    )

    assert (
        parsed_ledger["stage_telemetry"][
            "report_generator"
        ]
        == dummy_metadata
    )


def test_compile_master_inference_ledger_failure(
    mock_pipeline_context: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
    dummy_inference_validator_artifact: InferenceValidatorArtifact,
    dummy_report_generator_artifact: ReportGeneratorArtifact,
) -> None:
    publisher = ReportPublisher(context=mock_pipeline_context)

    with patch(
        "builtins.open",
        side_effect=Exception("Disk error"),
    ):
        with pytest.raises(CustomException) as exc_info:
            publisher._compile_master_inference_ledger(
                model_loader_artifact=dummy_model_loader_artifact,
                feature_matrix_artifact=dummy_feature_matrix_artifact,
                validator_artifact=dummy_inference_validator_artifact,
                report_generator_artifact=dummy_report_generator_artifact,
                execution_time=10.5,
                s3_business_uri="s3://dummy/biz.csv",
                s3_telemetry_uri="s3://dummy/tel.parquet",
            )

    assert "Disk error" in str(exc_info.value)


def test_publish_artifacts_to_s3_success(
    mock_pipeline_context: MagicMock,
) -> None:
    publisher = ReportPublisher(context=mock_pipeline_context)

    publisher._publish_artifacts_to_s3(
        local_csv_path="/tmp/biz.csv",
        s3_csv_uri="s3://bucket/biz.csv",
        local_telemetry_path="/tmp/tel.parquet",
        s3_telemetry_uri="s3://bucket/tel.parquet",
        local_metadata_path="/tmp/meta.json",
        s3_metadata_uri="s3://bucket/meta.json",
    )

    assert (
        mock_pipeline_context.s3_sync.upload_file.call_count
        == 3
    )

    calls = (
        mock_pipeline_context.s3_sync.upload_file.mock_calls
    )

    expected_calls = [
        call(
            local_path="/tmp/biz.csv",
            s3_uri="s3://bucket/biz.csv",
        ),
        call(
            local_path="/tmp/tel.parquet",
            s3_uri="s3://bucket/tel.parquet",
        ),
        call(
            local_path="/tmp/meta.json",
            s3_uri="s3://bucket/meta.json",
        ),
    ]

    assert calls == expected_calls


def test_publish_artifacts_to_s3_failure(
    mock_pipeline_context: MagicMock,
) -> None:
    publisher = ReportPublisher(context=mock_pipeline_context)

    (
        mock_pipeline_context
        .s3_sync
        .upload_file
        .side_effect
    ) = Exception("AWS Credentials Expired")

    with pytest.raises(CustomException) as exc_info:
        publisher._publish_artifacts_to_s3(
            local_csv_path="/tmp/biz.csv",
            s3_csv_uri="s3://bucket/biz.csv",
            local_telemetry_path="/tmp/tel.parquet",
            s3_telemetry_uri="s3://bucket/tel.parquet",
            local_metadata_path="/tmp/meta.json",
            s3_metadata_uri="s3://bucket/meta.json",
        )

    assert "AWS Credentials Expired" in str(exc_info.value)