import os
import json
import pytest
from unittest.mock import patch

from pipelines.training_pipeline.src.components.model_registry import ModelRegistry
from pipelines.training_pipeline.src.entity.config_entity import ModelRegistryConfig
from pipelines.training_pipeline.src.entity.artifact_entity import ModelEvaluatorArtifact
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def registry_config(pipeline_context):
    return ModelRegistryConfig.from_context(pipeline_context)


def test_model_registry_initialization(
    pipeline_context,
    registry_config,
    data_processor_artifact,
    model_trainer_artifact,
    model_evaluator_artifact,
):
    registry = ModelRegistry(
        config=registry_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
        evaluator_artifact=model_evaluator_artifact,
    )

    assert registry.config == registry_config
    assert registry.context == pipeline_context
    assert registry.data_artifact == data_processor_artifact
    assert registry.trainer_artifact == model_trainer_artifact
    assert registry.evaluator_artifact == model_evaluator_artifact


def test_model_registry_rejected_gatekeeper(
    pipeline_context,
    registry_config,
    data_processor_artifact,
    model_trainer_artifact,
    model_evaluator_artifact,
):
    # Simulate a rejection from the Model Evaluator
    rejected_evaluator_artifact = ModelEvaluatorArtifact(
        approval_status=False,
        report_file_path=model_evaluator_artifact.report_file_path,
        baseline_performance_metrics_file_path=model_evaluator_artifact.baseline_performance_metrics_file_path,
        metadata_file_path=model_evaluator_artifact.metadata_file_path,
    )

    registry = ModelRegistry(
        config=registry_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
        evaluator_artifact=rejected_evaluator_artifact,
    )

    artifact = registry.run()

    # The registry should short-circuit and return deployment_status=False
    assert artifact.deployment_status is False
    assert artifact.s3_model_uri == ""

    # S3 sync should not have been called
    pipeline_context.s3_sync.sync_folder_to_s3.assert_not_called()
    pipeline_context.s3_sync.upload_file.assert_not_called()


def test_model_registry_stage_artifacts(
    pipeline_context,
    registry_config,
    data_processor_artifact,
    model_trainer_artifact,
    model_evaluator_artifact,
):
    registry = ModelRegistry(
        config=registry_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
        evaluator_artifact=model_evaluator_artifact,
    )

    os.makedirs(registry.config.staging_dir, exist_ok=True)
    staged_files = registry._stage_artifacts()

    # Verify all expected artifacts were mapped and copied
    expected_filenames = [
        "model.pkl",
        "schema.json",
        "reference_feature_distributions.json",
        "baseline_performance_metrics.json",
        "evaluation_report.json",
        "shap_summary.png",
        "shap_feature_importance.json",
    ]

    for filename in expected_filenames:
        assert filename in staged_files
        assert os.path.exists(staged_files[filename])


def test_model_registry_generate_requirements(
    pipeline_context,
    registry_config,
    data_processor_artifact,
    model_trainer_artifact,
    model_evaluator_artifact,
):
    registry = ModelRegistry(
        config=registry_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
        evaluator_artifact=model_evaluator_artifact,
    )

    os.makedirs(registry.config.staging_dir, exist_ok=True)
    registry._generate_requirements()

    req_path = os.path.join(
        registry.config.staging_dir, "requirements.txt"
    )
    assert os.path.exists(req_path)

    with open(req_path, "r") as f:
        content = f.read()

    assert "xgboost" in content
    assert "scikit-learn" in content
    assert "pandas" in content
    assert "pyarrow" in content


def test_model_registry_generate_deployment_metadata(
    pipeline_context,
    registry_config,
    data_processor_artifact,
    model_trainer_artifact,
    model_evaluator_artifact,
):
    registry = ModelRegistry(
        config=registry_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
        evaluator_artifact=model_evaluator_artifact,
    )

    os.makedirs(registry.config.staging_dir, exist_ok=True)
    registry._generate_deployment_metadata()

    meta_path = os.path.join(
        registry.config.staging_dir,
        "deployment_metadata.json",
    )
    assert os.path.exists(meta_path)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    assert metadata["run_id"] == pipeline_context.run_id
    assert metadata["environment"] == registry_config.deployment_environment
    assert (
        metadata["lineage"]["training_dataset_s3_uri"]
        == pipeline_context.training_dataset_s3_uri_path
    )


def test_model_registry_two_phase_commit(
    pipeline_context, registry_config, data_processor_artifact, model_trainer_artifact, model_evaluator_artifact
):
    registry = ModelRegistry(
        config=registry_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
        evaluator_artifact=model_evaluator_artifact,
    )

    staged_files = {"model.pkl": "dummy/path"}

    # Define a side-effect that creates an empty file so os.remove() succeeds during cleanup
    def fake_write_json(filepath, data):
        with open(filepath, 'w') as f:
            f.write('{}')

    with patch(
        "pipelines.training_pipeline.src.components.model_registry.write_json_file", 
        side_effect=fake_write_json
    ) as mock_write:
        s3_vault_uri = registry._execute_two_phase_commit(staged_files)

    expected_vault_uri = f"{registry_config.s3_models_dir_uri}/{pipeline_context.run_id}"
    assert s3_vault_uri == expected_vault_uri

    # Verify Phase 1: Folder sync
    pipeline_context.s3_sync.sync_folder_to_s3.assert_called_once_with(
        folder=registry_config.staging_dir,
        aws_bucket_url=expected_vault_uri
    )

    # Verify Phase 2: Pointer file mutation
    pipeline_context.s3_sync.upload_file.assert_called_once()
    upload_call_args = pipeline_context.s3_sync.upload_file.call_args[0]
    assert upload_call_args[1] == registry_config.s3_pointer_uri

    # Verify state payload content through the intercepted write_json_file call
    write_args = mock_write.call_args[0]
    assert "tmp_model_state.json" in write_args[0]
    
    payload = write_args[1]
    assert payload["run_id"] == pipeline_context.run_id
    assert payload["s3_model_path"] == f"{expected_vault_uri}/model.pkl"
    assert payload["training_dataset_s3_uri"] == pipeline_context.training_dataset_s3_uri_path

def test_model_registry_run_success(
    pipeline_context,
    registry_config,
    data_processor_artifact,
    model_trainer_artifact,
    model_evaluator_artifact,
):
    registry = ModelRegistry(
        config=registry_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
        evaluator_artifact=model_evaluator_artifact,
    )

    artifact = registry.run()

    assert artifact.deployment_status is True
    assert pipeline_context.run_id in artifact.s3_model_uri
    assert os.path.exists(artifact.metadata_file_path)

    # Validate Component Metadata
    with open(artifact.metadata_file_path, "r") as f:
        meta = json.load(f)

    assert meta["deployment_successful"] is True
    assert meta["s3_vault_uri"] == artifact.s3_model_uri


def test_model_registry_run_failure(
    pipeline_context,
    registry_config,
    data_processor_artifact,
    model_trainer_artifact,
    model_evaluator_artifact,
):
    registry = ModelRegistry(
        config=registry_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
        evaluator_artifact=model_evaluator_artifact,
    )

    # Simulate an S3 upload failure
    pipeline_context.s3_sync.sync_folder_to_s3.side_effect = Exception(
        "S3 Access Denied"
    )

    with pytest.raises(CustomException) as excinfo:
        registry.run()

    assert "S3 Access Denied" in str(excinfo.value)