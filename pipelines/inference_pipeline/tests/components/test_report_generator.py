import os
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.components.report_generator import ReportGenerator
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    ReportGeneratorArtifact,
    ModelLoaderArtifact,
    FeatureMatrixBuilderArtifact
)


@pytest.fixture
def generator_mock_artifacts(dummy_model_artifact_path, dummy_feature_matrix_path):
    """Provides mock upstream artifacts mapping to the serialized dummy model and parquet matrix."""
    ml_artifact = MagicMock(spec=ModelLoaderArtifact)
    ml_artifact.model_file_path = dummy_model_artifact_path

    fmb_artifact = MagicMock(spec=FeatureMatrixBuilderArtifact)
    fmb_artifact.feature_matrix_file_path = dummy_feature_matrix_path

    return ml_artifact, fmb_artifact


def test_report_generator_initialization_success(mock_pipeline_context):
    """Tests successful initialization of ReportGenerator."""
    generator = ReportGenerator(context=mock_pipeline_context)
    assert generator.context is mock_pipeline_context
    assert generator.config is not None
    assert "04_report_generator" in generator.config.generator_root_dir


def test_report_generator_initialization_failure(mock_pipeline_context):
    """Tests that initialization failure raises CustomException."""
    with patch("pipelines.inference_pipeline.src.entity.config_entity.ReportGeneratorConfig.from_context") as mock_from_context:
        mock_from_context.side_effect = Exception("Config error")
        with pytest.raises(CustomException) as exc_info:
            ReportGenerator(context=mock_pipeline_context)
        assert "Config error" in str(exc_info.value)


def test_report_generator_run_success_with_monetary_col(mock_pipeline_context, generator_mock_artifacts):
    """
    Tests the happy path where a valid dataset contains both system identifiers 
    and a monetary column, enabling revenue-at-risk prioritization.
    """
    ml_artifact, fmb_artifact = generator_mock_artifacts
    generator = ReportGenerator(context=mock_pipeline_context)
    
    artifact = generator.run(
        model_loader_artifact=ml_artifact,
        feature_matrix_artifact=fmb_artifact
    )
    
    assert isinstance(artifact, ReportGeneratorArtifact)
    assert os.path.exists(artifact.csv_report_path)
    assert os.path.exists(artifact.telemetry_log_path)
    
    # Validate Business CSV Report
    report_df = pd.read_csv(artifact.csv_report_path)
    assert "customer_unique_id" in report_df.columns
    assert "churn_probability" in report_df.columns
    assert "is_churn_risk" in report_df.columns
    assert "revenue_at_risk" in report_df.columns
    
    # Assert sorting (highest revenue at risk first)
    assert report_df["revenue_at_risk"].iloc[0] >= report_df["revenue_at_risk"].iloc[-1]
    
    # Validate Parquet Telemetry Log
    telemetry_df = pd.read_parquet(artifact.telemetry_log_path)
    assert "f1" in telemetry_df.columns  # Original feature retained
    assert "inference_run_id" in telemetry_df.columns
    assert "churn_probability" in telemetry_df.columns
    
    # Validate Metadata Ledger
    with open(artifact.metadata_file_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        assert metadata["business_impact"]["total_customers_scored"] == 3
        assert metadata["business_impact"]["total_revenue_at_risk_flagged"] is not None


def test_report_generator_run_fallback_identifiers(mock_pipeline_context, temp_workspace, dummy_model_artifact_path):
    """
    Tests the fallback behavior when the data lacks both the standard 'customer_unique_id'
    and any monetary columns (resorts to index and probability sorting).
    """
    # Create feature matrix missing system and monetary cols
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "f2": [0.1, 0.2, 0.3],
        "c1": ["A", "B", "A"]
    })
    fallback_parquet_path = temp_workspace / "fallback_features.parquet"
    df.to_parquet(fallback_parquet_path)
    
    ml_artifact = MagicMock(spec=ModelLoaderArtifact)
    ml_artifact.model_file_path = dummy_model_artifact_path
    
    fmb_artifact = MagicMock(spec=FeatureMatrixBuilderArtifact)
    fmb_artifact.feature_matrix_file_path = str(fallback_parquet_path)
    
    generator = ReportGenerator(context=mock_pipeline_context)
    artifact = generator.run(
        model_loader_artifact=ml_artifact,
        feature_matrix_artifact=fmb_artifact
    )
    
    report_df = pd.read_csv(artifact.csv_report_path)
    
    # Should fallback to assigning index as ID
    assert "customer_unique_id" in report_df.columns
    assert list(report_df["customer_unique_id"].sort_values()) == [0, 1, 2]
    
    # Should lack revenue_at_risk and sort by probability instead
    assert "revenue_at_risk" not in report_df.columns
    assert report_df["churn_probability"].iloc[0] >= report_df["churn_probability"].iloc[-1]
    
    with open(artifact.metadata_file_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        assert metadata["business_impact"]["total_revenue_at_risk_flagged"] is None


def test_report_generator_empty_dataframe(mock_pipeline_context, temp_workspace, dummy_model_artifact_path):
    """
    Tests that an empty parquet file raises a ValueError safely.
    """
    empty_parquet_path = temp_workspace / "empty.parquet"
    pd.DataFrame().to_parquet(empty_parquet_path)
    
    ml_artifact = MagicMock(spec=ModelLoaderArtifact)
    ml_artifact.model_file_path = dummy_model_artifact_path
    
    fmb_artifact = MagicMock(spec=FeatureMatrixBuilderArtifact)
    fmb_artifact.feature_matrix_file_path = str(empty_parquet_path)
    
    generator = ReportGenerator(context=mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        generator.run(
            model_loader_artifact=ml_artifact,
            feature_matrix_artifact=fmb_artifact
        )
        
    assert "is empty" in str(exc_info.value).lower()


def test_report_generator_model_load_failure(mock_pipeline_context, temp_workspace, dummy_feature_matrix_path):
    """
    Tests that a corrupted or missing model artifact throws an exception.
    """
    invalid_model_path = temp_workspace / "corrupted_model.pkl"
    invalid_model_path.write_text("Not a real pickle file")
    
    ml_artifact = MagicMock(spec=ModelLoaderArtifact)
    ml_artifact.model_file_path = str(invalid_model_path)
    
    fmb_artifact = MagicMock(spec=FeatureMatrixBuilderArtifact)
    fmb_artifact.feature_matrix_file_path = dummy_feature_matrix_path
    
    generator = ReportGenerator(context=mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        generator.run(
            model_loader_artifact=ml_artifact,
            feature_matrix_artifact=fmb_artifact
        )
        
    # Include the actual Pickle IndexError string in the allowed assertions
    assert any(err in str(exc_info.value).lower() for err in ["unpickling", "magic", "pop from empty list"])


def test_report_generator_prediction_failure(mock_pipeline_context, temp_workspace, dummy_model_artifact_path):
        """
        Tests that a failure during batch inference is securely caught and wrapped.
        """
        bad_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        bad_parquet = temp_workspace / "bad_shape.parquet"
        bad_df.to_parquet(bad_parquet)
        
        ml_artifact = MagicMock(spec=ModelLoaderArtifact)
        ml_artifact.model_file_path = dummy_model_artifact_path
        
        fmb_artifact = MagicMock(spec=FeatureMatrixBuilderArtifact)
        fmb_artifact.feature_matrix_file_path = str(bad_parquet)
        
        generator = ReportGenerator(context=mock_pipeline_context)
        
        # FIX: Explicitly mock the model to raise an exception rather than relying on DummyClassifier
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = Exception("Mocked shape mismatch")
        
        with patch("pipelines.inference_pipeline.src.components.report_generator.joblib.load", return_value=mock_model):
            with pytest.raises(CustomException) as exc_info:
                generator.run(
                    model_loader_artifact=ml_artifact,
                    feature_matrix_artifact=fmb_artifact
                )
            
        assert "batch inference" in str(exc_info.value).lower() or "mocked shape mismatch" in str(exc_info.value).lower()