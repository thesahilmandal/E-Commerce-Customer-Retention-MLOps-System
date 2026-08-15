import os
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from pipelines.monitoring_pipeline.src.components.statistical_drift_calculator import StatisticalDriftCalculator
from pipelines.monitoring_pipeline.src.entity.config_entity import StatisticalDriftCalculatorConfig
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def drift_config(mock_context):
    """Provides a valid configuration for the Statistical Drift Calculator."""
    return StatisticalDriftCalculatorConfig.get_config(mock_context)


@pytest.fixture
def drift_calculator(drift_config, mock_context, mock_resolver_artifact):
    """Provides a configured StatisticalDriftCalculator instance."""
    return StatisticalDriftCalculator(
        config=drift_config,
        context=mock_context,
        resolver_artifact=mock_resolver_artifact
    )


def test_extract_top_shap_features(drift_calculator):
    """
    Validates that SHAP features are extracted, correctly sorted by importance
    descending, and limited to the configured top_k count.
    """
    shap_data = {
        "feature_importance": [
            {"feature_name": "low_importance_feat", "mean_abs_shap_value": 0.05},
            {"feature_name": "high_importance_feat", "mean_abs_shap_value": 0.95},
            {"feature_name": "med_importance_feat", "mean_abs_shap_value": 0.45}
        ]
    }
    
    # Temporarily override config to take top 2 for testing logic bounds
    object.__setattr__(drift_calculator.config, 'top_shap_features_count', 2)
    
    top_features = drift_calculator._extract_top_shap_features(shap_data)
    
    assert len(top_features) == 2
    assert top_features[0] == "high_importance_feat"
    assert top_features[1] == "med_importance_feat"


def test_compute_numerical_psi_with_zero_bin_mitigation(drift_calculator):
    """
    Validates the first-principles numerical PSI calculation, explicitly testing
    the algorithmic mitigation of the Zero-Bin Problem using epsilon clipping.
    """
    reference_stats = {
        "physical_type": "numerical",
        "bin_edges": [0.0, 0.5, 1.0],
        "expected_percentages": [0.5, 0.5]
    }
    
    # Telemetry only has values in the first bin, forcing a 0 count in the second bin
    current_series = pd.Series([0.1, 0.2, 0.3])
    
    psi_score = drift_calculator._compute_numerical_psi(current_series, reference_stats)
    
    # PSI should be > 0 due to shift, but finite (not infinity) due to epsilon clipping
    assert psi_score > 0.0
    assert np.isfinite(psi_score)


def test_compute_numerical_psi_missing_bins(drift_calculator):
    """
    Validates that missing bin definitions in the reference gracefully return 0.0 
    rather than causing division by zero or indexing exceptions.
    """
    reference_stats = {"physical_type": "numerical", "bin_edges": []}
    current_series = pd.Series([0.1, 0.2, 0.3])
    
    psi_score = drift_calculator._compute_numerical_psi(current_series, reference_stats)
    
    assert psi_score == 0.0


def test_compute_categorical_psi_with_unseen_categories(drift_calculator):
    """
    Validates categorical PSI calculation. Unseen categories in production should 
    be dropped/ignored mathematically to align with training dimensions, while missing
    expected categories should trigger epsilon clipping.
    """
    reference_stats = {
        "physical_type": "categorical",
        "categories": ["A", "B", "C"],
        "expected_percentages": [0.4, 0.4, 0.2]
    }
    
    # 'D' is an unseen category; 'C' is missing from current distribution
    current_series = pd.Series(["A", "A", "B", "D"])
    
    psi_score = drift_calculator._compute_categorical_psi(current_series, reference_stats)
    
    assert psi_score > 0.0
    assert np.isfinite(psi_score)


def test_calculate_all_drifts_zero_traffic_graceful_bypass(drift_calculator):
    """
    Validates that if the resolver bypassed telemetry fetching (zero traffic),
    passing an empty DataFrame results in safe 0.0 PSI scores and correct statuses.
    """
    empty_df = pd.DataFrame()
    reference_distributions = {"predicted_probability": {}}
    top_features = ["f1", "f2"]
    
    report = drift_calculator._calculate_all_drifts(
        telemetry_df=empty_df,
        reference_distributions=reference_distributions,
        top_features=top_features
    )
    
    assert report["prediction_drift"]["predicted_probability"]["psi_score"] == 0.0
    assert report["prediction_drift"]["predicted_probability"]["status"] == "EMPTY_TELEMETRY"
    
    for feat in top_features:
        assert report["feature_drift"][feat]["psi_score"] == 0.0
        assert report["feature_drift"][feat]["status"] == "EMPTY_TELEMETRY"


def test_statistical_drift_calculator_run_e2e(drift_calculator, mock_resolver_artifact):
    """
    Validates the end-to-end component run using synthetic baseline files and 
    generated parquet telemetry to ensure IO, JSON parsing, and report generation succeed.
    """
    # 1. Setup mock telemetry parquet
    df = pd.DataFrame({
        "predicted_probability": [0.1, 0.8, 0.9, 0.2],
        "important_feature": [1.0, 2.0, 3.0, 4.0]
    })
    df.to_parquet(mock_resolver_artifact.current_telemetry_file_path)
    
    # 2. Setup mock SHAP json
    with open(mock_resolver_artifact.shap_importance_file_path, "w") as f:
        json.dump({
            "feature_importance": [
                {"feature_name": "important_feature", "mean_abs_shap_value": 0.8}
            ]
        }, f)
        
    # 3. Setup mock reference distribution json
    with open(mock_resolver_artifact.reference_distributions_file_path, "w") as f:
        json.dump({
            "distributions": {
                "predicted_probability": {
                    "physical_type": "numerical", 
                    "bin_edges": [0.0, 0.5, 1.0], 
                    "expected_percentages": [0.5, 0.5]
                },
                "important_feature": {
                    "physical_type": "numerical", 
                    "bin_edges": [0.0, 2.5, 5.0], 
                    "expected_percentages": [0.5, 0.5]
                }
            }
        }, f)
        
    # Run the component
    artifact = drift_calculator.run()
    
    # Assert artifacts were generated
    assert os.path.exists(artifact.drift_report_file_path)
    assert os.path.exists(artifact.metadata_file_path)
    
    # Validate report contents
    with open(artifact.drift_report_file_path, "r") as f:
        report = json.load(f)
        
    assert "prediction_drift" in report
    assert "predicted_probability" in report["prediction_drift"]
    assert report["prediction_drift"]["predicted_probability"]["status"] == "CALCULATED"
    
    assert "feature_drift" in report
    assert "important_feature" in report["feature_drift"]
    assert report["feature_drift"]["important_feature"]["status"] == "CALCULATED"