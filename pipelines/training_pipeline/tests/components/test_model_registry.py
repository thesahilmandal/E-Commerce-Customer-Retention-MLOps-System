import os
from dataclasses import replace
from unittest.mock import MagicMock, mock_open, patch

import pytest

from pipelines.training_pipeline.src.components.model_registry import ModelRegistry
from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataProcessorArtifact,
    ModelEvaluatorArtifact,
    ModelTrainerArtifact,
)
from pipelines.training_pipeline.src.entity.config_entity import (
    ModelRegistryConfig,
)
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def mr_config(
    mock_pipeline_context: MagicMock,
) -> ModelRegistryConfig:
    return ModelRegistryConfig.from_context(mock_pipeline_context)


def test_run_gatekeeper_denied(
    mr_config: ModelRegistryConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
    dummy_model_evaluator_artifact: ModelEvaluatorArtifact,
) -> None:
    rejected_evaluator = replace(
        dummy_model_evaluator_artifact,
        approval_status=False,
    )

    registry = ModelRegistry(
        config=mr_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
        evaluator_artifact=rejected_evaluator,
    )

    with patch.object(registry, "_stage_artifacts") as mock_stage:
        artifact = registry.run()

        assert artifact.deployment_status is False
        assert artifact.s3_model_uri == ""
        assert artifact.metadata_file_path == mr_config.metadata_file_path
        mock_stage.assert_not_called()


@patch(
    "pipelines.training_pipeline.src.components.model_registry.os.makedirs"
)
def test_run_success_approved(
    mock_makedirs: MagicMock,
    mr_config: ModelRegistryConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
    dummy_model_evaluator_artifact: ModelEvaluatorArtifact,
) -> None:
    registry = ModelRegistry(
        config=mr_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
        evaluator_artifact=dummy_model_evaluator_artifact,
    )

    expected_s3_uri = (
        f"{mr_config.s3_models_dir_uri}/"
        f"{mock_pipeline_context.run_id}"
    )

    with (
        patch.object(
            registry,
            "_stage_artifacts",
            return_value={"model.pkl": "/tmp/model.pkl"},
        ),
        patch.object(registry, "_generate_requirements"),
        patch.object(registry, "_generate_deployment_metadata"),
        patch.object(
            registry,
            "_execute_two_phase_commit",
            return_value=expected_s3_uri,
        ),
        patch.object(registry, "_generate_component_metadata"),
    ):
        artifact = registry.run()

        assert artifact.deployment_status is True
        assert artifact.s3_model_uri == expected_s3_uri
        assert artifact.metadata_file_path == mr_config.metadata_file_path

        mock_makedirs.assert_called_once_with(
            mr_config.staging_dir,
            exist_ok=True,
        )


@patch(
    "pipelines.training_pipeline.src.components.model_registry.shutil.copy2"
)
@patch(
    "pipelines.training_pipeline.src.components.model_registry.os.path.exists"
)
def test_stage_artifacts(
    mock_exists: MagicMock,
    mock_copy2: MagicMock,
    mr_config: ModelRegistryConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
    dummy_model_evaluator_artifact: ModelEvaluatorArtifact,
) -> None:
    registry = ModelRegistry(
        config=mr_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
        evaluator_artifact=dummy_model_evaluator_artifact,
    )

    # Simulate some files exist, some do not
    def side_effect_exists(path: str) -> bool:
        if "model.pkl" in path or "schema.json" in path:
            return True
        return False

    mock_exists.side_effect = side_effect_exists

    staged_files = registry._stage_artifacts()

    assert "model.pkl" in staged_files
    assert "schema.json" in staged_files
    assert "evaluation_report.json" not in staged_files

    assert mock_copy2.call_count == 2

    mock_copy2.assert_any_call(
        dummy_model_trainer_artifact.model_file_path,
        os.path.join(
            mr_config.staging_dir,
            "model.pkl",
        ),
    )


def test_generate_requirements(
    mr_config: ModelRegistryConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
    dummy_model_evaluator_artifact: ModelEvaluatorArtifact,
) -> None:
    registry = ModelRegistry(
        config=mr_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
        evaluator_artifact=dummy_model_evaluator_artifact,
    )

    m_open = mock_open()

    with patch("builtins.open", m_open):
        registry._generate_requirements()

    m_open.assert_called_once_with(
        os.path.join(
            mr_config.staging_dir,
            "requirements.txt",
        ),
        "w",
    )

    handle = m_open()
    written_content = handle.write.call_args[0][0]

    assert "scikit-learn" in written_content
    assert "xgboost" in written_content
    assert "pandas" in written_content
    assert "numpy" in written_content


@patch(
    "pipelines.training_pipeline.src.components.model_registry.write_json_file"
)
def test_generate_deployment_metadata(
    mock_write_json: MagicMock,
    mr_config: ModelRegistryConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
    dummy_model_evaluator_artifact: ModelEvaluatorArtifact,
) -> None:
    registry = ModelRegistry(
        config=mr_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
        evaluator_artifact=dummy_model_evaluator_artifact,
    )

    registry._generate_deployment_metadata()

    mock_write_json.assert_called_once()

    args = mock_write_json.call_args[0]

    assert args[0] == os.path.join(
        mr_config.staging_dir,
        "deployment_metadata.json",
    )

    payload = args[1]

    assert payload["run_id"] == mock_pipeline_context.run_id
    assert payload["environment"] == mr_config.deployment_environment
    assert (
        payload["lineage"]["training_dataset_s3_uri"]
        == mock_pipeline_context.training_dataset_s3_uri_path
    )
    assert "deployed_at_utc" in payload


@patch(
    "pipelines.training_pipeline.src.components.model_registry.os.remove"
)
@patch(
    "pipelines.training_pipeline.src.components.model_registry.write_json_file"
)
def test_execute_two_phase_commit(
    mock_write_json: MagicMock,
    mock_remove: MagicMock,
    mr_config: ModelRegistryConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
    dummy_model_evaluator_artifact: ModelEvaluatorArtifact,
) -> None:
    registry = ModelRegistry(
        config=mr_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
        evaluator_artifact=dummy_model_evaluator_artifact,
    )

    staged_files = {
        "model.pkl": "/tmp/staging/model.pkl",
    }

    returned_uri = registry._execute_two_phase_commit(
        staged_files
    )

    expected_vault_uri = (
        f"{mr_config.s3_models_dir_uri}/"
        f"{mock_pipeline_context.run_id}"
    )

    assert returned_uri == expected_vault_uri

    # Phase 1: Sync to Vault
    mock_pipeline_context.s3_sync.sync_folder_to_s3.assert_called_once_with(
        folder=mr_config.staging_dir,
        aws_bucket_url=expected_vault_uri,
    )

    # Phase 2: Overwrite Mutable Pointer
    mock_write_json.assert_called_once()

    payload = mock_write_json.call_args[0][1]

    assert payload["run_id"] == mock_pipeline_context.run_id
    assert (
        payload["s3_model_path"]
        == f"{expected_vault_uri}/model.pkl"
    )
    assert (
        payload["s3_schema_path"]
        == f"{expected_vault_uri}/schema.json"
    )

    mock_pipeline_context.s3_sync.upload_file.assert_called_once()
    mock_remove.assert_called_once()


def test_execute_two_phase_commit_failure_raises(
    mr_config: ModelRegistryConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
    dummy_model_evaluator_artifact: ModelEvaluatorArtifact,
) -> None:
    registry = ModelRegistry(
        config=mr_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
        evaluator_artifact=dummy_model_evaluator_artifact,
    )

    # Simulate network failure during sync
    mock_pipeline_context.s3_sync.sync_folder_to_s3.side_effect = Exception(
        "S3 bucket access denied"
    )

    with pytest.raises(CustomException) as exc_info:
        registry._execute_two_phase_commit({})

    assert "S3 bucket access denied" in str(exc_info.value)