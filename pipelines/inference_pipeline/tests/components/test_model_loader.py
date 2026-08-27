import os
import json

import pytest
from unittest.mock import MagicMock, patch, mock_open

from pipelines.inference_pipeline.src.components.model_loader import ModelLoader
from shared_core.exceptions.custom_exception import CustomException


VALID_POINTER_DATA = {
    "run_id": "champion_123",
    "s3_model_path": "s3://bucket/model.pkl",
    "s3_schema_path": "s3://bucket/schema.json",
    "s3_monitoring_baselines_path": "s3://bucket/baselines.json",
    "s3_reference_distributions_path": "s3://bucket/ref.json",
}


@patch(
    "pipelines.inference_pipeline.src.components.model_loader."
    "ModelLoaderConfig.from_context"
)
def test_model_loader_initialization_failure(
    mock_from_context: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_from_context.side_effect = Exception(
        "Config initialization failed"
    )

    with pytest.raises(CustomException) as exc_info:
        ModelLoader(context=mock_pipeline_context)

    assert "Config initialization failed" in str(exc_info.value)


@patch.object(ModelLoader, "_generate_metadata")
@patch.object(ModelLoader, "_download_artifacts")
@patch.object(ModelLoader, "_fetch_and_parse_pointer")
def test_run_success(
    mock_fetch_pointer: MagicMock,
    mock_download: MagicMock,
    mock_generate_metadata: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_fetch_pointer.return_value = VALID_POINTER_DATA

    loader = ModelLoader(context=mock_pipeline_context)
    artifact = loader.run()

    assert artifact.champion_run_id == "champion_123"
    assert artifact.model_file_path == loader.config.model_file_path
    assert artifact.schema_file_path == loader.config.schema_file_path
    assert artifact.metadata_file_path == loader.config.metadata_file_path

    mock_fetch_pointer.assert_called_once()
    mock_download.assert_called_once_with(
        pointer_data=VALID_POINTER_DATA
    )
    mock_generate_metadata.assert_called_once()


@patch.object(ModelLoader, "_fetch_and_parse_pointer")
def test_run_failure(
    mock_fetch_pointer: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_fetch_pointer.side_effect = Exception("Pointer fetch failed")

    loader = ModelLoader(context=mock_pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        loader.run()

    assert "Pointer fetch failed" in str(exc_info.value)


def test_fetch_and_parse_pointer_success(
    mock_pipeline_context: MagicMock,
) -> None:
    loader = ModelLoader(context=mock_pipeline_context)

    m_open = mock_open(
        read_data=json.dumps(VALID_POINTER_DATA)
    )

    with patch("builtins.open", m_open):
        pointer_data = loader._fetch_and_parse_pointer()

    assert pointer_data == VALID_POINTER_DATA

    expected_local_path = os.path.join(
        loader.config.model_loader_root_dir,
        "model_state.json",
    )

    mock_pipeline_context.s3_sync.download_file.assert_called_once_with(
        s3_uri=loader.config.s3_pointer_uri,
        local_path=expected_local_path,
    )


def test_fetch_and_parse_pointer_missing_keys(
    mock_pipeline_context: MagicMock,
) -> None:
    loader = ModelLoader(context=mock_pipeline_context)

    invalid_data = {
        "run_id": "champion_123"
    }

    m_open = mock_open(
        read_data=json.dumps(invalid_data)
    )

    with patch("builtins.open", m_open):
        with pytest.raises(CustomException) as exc_info:
            loader._fetch_and_parse_pointer()

    assert "Missing required keys" in str(exc_info.value)


def test_fetch_and_parse_pointer_download_failure(
    mock_pipeline_context: MagicMock,
) -> None:
    loader = ModelLoader(context=mock_pipeline_context)

    mock_pipeline_context.s3_sync.download_file.side_effect = Exception(
        "S3 bucket access denied"
    )

    with pytest.raises(CustomException) as exc_info:
        loader._fetch_and_parse_pointer()

    assert (
        "Failed to fetch model registry state pointer"
        in str(exc_info.value)
    )


def test_download_artifacts_success(
    mock_pipeline_context: MagicMock,
) -> None:
    loader = ModelLoader(context=mock_pipeline_context)

    loader._download_artifacts(VALID_POINTER_DATA)

    assert (
        mock_pipeline_context.s3_sync.download_file.call_count == 4
    )

    calls = (
        mock_pipeline_context.s3_sync.download_file.call_args_list
    )

    uris_called = [
        call[1]["s3_uri"]
        for call in calls
    ]

    assert (
        VALID_POINTER_DATA["s3_model_path"]
        in uris_called
    )
    assert (
        VALID_POINTER_DATA["s3_schema_path"]
        in uris_called
    )
    assert (
        VALID_POINTER_DATA["s3_monitoring_baselines_path"]
        in uris_called
    )
    assert (
        VALID_POINTER_DATA["s3_reference_distributions_path"]
        in uris_called
    )


def test_download_artifacts_failure(
    mock_pipeline_context: MagicMock,
) -> None:
    loader = ModelLoader(context=mock_pipeline_context)

    mock_pipeline_context.s3_sync.download_file.side_effect = Exception(
        "Network error"
    )

    with pytest.raises(CustomException) as exc_info:
        loader._download_artifacts(VALID_POINTER_DATA)

    assert "Failed to download Model Artifact" in str(exc_info.value)


@patch(
    "pipelines.inference_pipeline.src.components.model_loader."
    "os.path.getsize"
)
def test_generate_metadata_success(
    mock_getsize: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_getsize.return_value = 1024

    loader = ModelLoader(context=mock_pipeline_context)
    execution_time = 1.23

    m_open = mock_open()

    with patch("builtins.open", m_open):
        loader._generate_metadata(
            pointer_data=VALID_POINTER_DATA,
            execution_time=execution_time,
        )

    m_open.assert_called_once_with(
        loader.config.metadata_file_path,
        "w",
        encoding="utf-8",
    )

    handle = m_open()
    written_data = "".join(
        call.args[0]
        for call in handle.write.call_args_list
    )

    parsed_metadata = json.loads(written_data)

    assert parsed_metadata["pipeline_stage"] == "Model Loader"
    assert (
        parsed_metadata["inference_run_id"]
        == mock_pipeline_context.run_id
    )
    assert (
        parsed_metadata["execution_time_seconds"]
        == execution_time
    )
    assert (
        parsed_metadata["model_provenance"]
        == VALID_POINTER_DATA
    )
    assert (
        parsed_metadata["local_artifact_sizes_bytes"]["model"]
        == 1024
    )
    assert (
        parsed_metadata["local_artifact_sizes_bytes"]["schema"]
        == 1024
    )