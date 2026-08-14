import os
import json
import pytest
from unittest.mock import patch, MagicMock

from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.components.inference_validator import InferenceValidator
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    InferenceValidatorArtifact,
    ModelLoaderArtifact,
    FeatureMatrixBuilderArtifact
)


@pytest.fixture
def validator_mock_artifacts(dummy_model_schema_path, dummy_builder_schema_path):
    """Provides mock upstream artifacts with valid schema paths."""
    ml_artifact = MagicMock(spec=ModelLoaderArtifact)
    ml_artifact.schema_file_path = dummy_model_schema_path

    fmb_artifact = MagicMock(spec=FeatureMatrixBuilderArtifact)
    fmb_artifact.schema_file_path = dummy_builder_schema_path

    return ml_artifact, fmb_artifact


def test_inference_validator_initialization_success(mock_pipeline_context):
    """Tests successful initialization of InferenceValidator."""
    validator = InferenceValidator(context=mock_pipeline_context)
    assert validator.context is mock_pipeline_context
    assert validator.config is not None
    assert "03_validator" in validator.config.validator_root_dir


def test_inference_validator_initialization_failure(mock_pipeline_context):
    """Tests that initialization failure raises CustomException."""
    with patch("pipelines.inference_pipeline.src.entity.config_entity.InferenceValidatorConfig.from_context") as mock_from_context:
        mock_from_context.side_effect = Exception("Config error")
        with pytest.raises(CustomException) as exc_info:
            InferenceValidator(context=mock_pipeline_context)
        assert "Config error" in str(exc_info.value)


def test_inference_validator_run_success(mock_pipeline_context, validator_mock_artifacts):
    """
    Tests the happy path where both the expected model schema and the 
    actual builder schema align perfectly.
    """
    ml_artifact, fmb_artifact = validator_mock_artifacts
    validator = InferenceValidator(context=mock_pipeline_context)
    
    artifact = validator.run(
        model_loader_artifact=ml_artifact,
        feature_matrix_artifact=fmb_artifact
    )
    
    # Assert Artifact properties
    assert isinstance(artifact, InferenceValidatorArtifact)
    assert artifact.is_valid is True
    assert os.path.exists(artifact.report_file_path)
    
    # Assert Report content
    with open(artifact.report_file_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        assert report["validation_status"] == "PASSED"
        assert len(report["errors"]) == 0
        
    # Assert Metadata content
    with open(validator.config.metadata_file_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        assert metadata["validation_outcome"]["is_valid"] is True


def test_inference_validator_missing_predictive_features(mock_pipeline_context, temp_workspace, dummy_builder_schema_path):
    """
    Tests that a data contract violation (missing predictive features) 
    correctly flags the validation as FAILED without crashing.
    """
    # Create a model schema requiring a feature not present in builder schema
    invalid_model_schema = temp_workspace / "invalid_model_schema.json"
    with open(invalid_model_schema, "w", encoding="utf-8") as f:
        json.dump({
            "features": {
                "numerical": ["f1", "f2", "missing_numerical_feature"],
                "categorical": {"c1": ["A", "B"]}
            }
        }, f)
        
    ml_artifact = MagicMock(spec=ModelLoaderArtifact)
    ml_artifact.schema_file_path = str(invalid_model_schema)
    
    fmb_artifact = MagicMock(spec=FeatureMatrixBuilderArtifact)
    fmb_artifact.schema_file_path = dummy_builder_schema_path
    
    validator = InferenceValidator(context=mock_pipeline_context)
    artifact = validator.run(
        model_loader_artifact=ml_artifact,
        feature_matrix_artifact=fmb_artifact
    )
    
    assert artifact.is_valid is False
    
    with open(artifact.report_file_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        assert report["validation_status"] == "FAILED"
        assert len(report["errors"]) == 1
        assert "missing_numerical_feature" in report["errors"][0]


def test_inference_validator_missing_system_columns(mock_pipeline_context, dummy_model_schema_path, temp_workspace):
    """
    Tests that missing required system columns (e.g., customer_unique_id) 
    fails the validation, as downstream mapping would be impossible.
    """
    invalid_builder_schema = temp_workspace / "invalid_builder_schema.json"
    with open(invalid_builder_schema, "w", encoding="utf-8") as f:
        json.dump([
            {"name": "f1", "physical_type": "DOUBLE"},
            {"name": "f2", "physical_type": "DOUBLE"},
            {"name": "c1", "physical_type": "VARCHAR"}
            # Missing "customer_unique_id" and "snapshot_date"
        ], f)
        
    ml_artifact = MagicMock(spec=ModelLoaderArtifact)
    ml_artifact.schema_file_path = dummy_model_schema_path
    
    fmb_artifact = MagicMock(spec=FeatureMatrixBuilderArtifact)
    fmb_artifact.schema_file_path = str(invalid_builder_schema)
    
    validator = InferenceValidator(context=mock_pipeline_context)
    artifact = validator.run(
        model_loader_artifact=ml_artifact,
        feature_matrix_artifact=fmb_artifact
    )
    
    assert artifact.is_valid is False
    
    with open(artifact.report_file_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        assert report["validation_status"] == "FAILED"
        error_text = str(report["errors"])
        assert "customer_unique_id" in error_text
        assert "snapshot_date" in error_text


def test_inference_validator_schema_load_failure(mock_pipeline_context, temp_workspace):
    """
    Tests that a missing physical schema file raises a CustomException and halts execution.
    """
    ml_artifact = MagicMock(spec=ModelLoaderArtifact)
    ml_artifact.schema_file_path = str(temp_workspace / "does_not_exist.json")
    
    fmb_artifact = MagicMock(spec=FeatureMatrixBuilderArtifact)
    fmb_artifact.schema_file_path = str(temp_workspace / "also_missing.json")
    
    validator = InferenceValidator(context=mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        validator.run(
            model_loader_artifact=ml_artifact,
            feature_matrix_artifact=fmb_artifact
        )
        
    assert "No such file or directory" in str(exc_info.value)


def test_inference_validator_malformed_json(mock_pipeline_context, temp_workspace, dummy_builder_schema_path):
    """
    Tests that structurally invalid JSON in the schema files raises a parsing error.
    """
    malformed_schema = temp_workspace / "malformed_model_schema.json"
    with open(malformed_schema, "w", encoding="utf-8") as f:
        f.write("This is not JSON")
        
    ml_artifact = MagicMock(spec=ModelLoaderArtifact)
    ml_artifact.schema_file_path = str(malformed_schema)
    
    fmb_artifact = MagicMock(spec=FeatureMatrixBuilderArtifact)
    fmb_artifact.schema_file_path = dummy_builder_schema_path
    
    validator = InferenceValidator(context=mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        validator.run(
            model_loader_artifact=ml_artifact,
            feature_matrix_artifact=fmb_artifact
        )
        
    assert "Expecting value" in str(exc_info.value) or "JSON" in str(exc_info.value)