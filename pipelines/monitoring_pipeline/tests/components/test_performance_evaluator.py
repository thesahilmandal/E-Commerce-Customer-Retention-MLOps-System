import os
import json
import pytest
import numpy as np
import pandas as pd

from pipelines.monitoring_pipeline.src.components.performance_evaluator import PerformanceEvaluator
from pipelines.monitoring_pipeline.src.entity.config_entity import PerformanceEvaluatorConfig


@pytest.fixture
def perf_config(mock_context):
    """Provides a valid configuration for the Performance Evaluator."""
    return PerformanceEvaluatorConfig.get_config(mock_context)


@pytest.fixture
def performance_evaluator(perf_config, mock_context, mock_resolver_artifact):
    """Provides a configured PerformanceEvaluator instance."""
    return PerformanceEvaluator(
        config=perf_config,
        context=mock_context,
        resolver_artifact=mock_resolver_artifact
    )


def test_compute_brier_score(performance_evaluator):
    """
    Validates first-principles Brier Score calculation (Mean Squared Error).
    """
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_prob = np.array([0.8, 0.2, 0.9, 0.1])
    
    # Expected: ((0.8-1)^2 + (0.2-0)^2 + (0.9-1)^2 + (0.1-0)^2) / 4
    # Expected: (0.04 + 0.04 + 0.01 + 0.01) / 4 = 0.10 / 4 = 0.025
    brier_score = performance_evaluator._compute_brier_score(y_true, y_prob)
    
    assert np.isclose(brier_score, 0.025)


def test_compute_log_loss_with_epsilon_clipping(performance_evaluator):
    """
    Validates Log Loss calculation, specifically ensuring that mathematically
    undefined log(0) boundaries are prevented via epsilon clipping.
    """
    y_true = np.array([1.0, 0.0])
    # Extreme probabilities that would normally cause log(0)
    y_prob = np.array([0.0, 1.0])
    
    log_loss = performance_evaluator._compute_log_loss(y_true, y_prob)
    
    # Should not be infinite or NaN
    assert np.isfinite(log_loss)
    assert log_loss > 0.0


def test_compute_realized_roi(performance_evaluator):
    """
    Validates the financial translation of model predictions into realized business value.
    Config assumptions: Campaign Cost=$10, LTV=$150, Save Rate=20%
    """
    # 1 True Positive (Correct Churn Pred), 1 False Positive (Incorrect Churn Pred), 
    # 1 True Negative (Correct Retention), 1 False Negative (Missed Churn)
    y_true = np.array([1, 0, 0, 1])
    y_prob = np.array([0.8, 0.7, 0.2, 0.1])
    
    roi = performance_evaluator._compute_realized_roi(y_true, y_prob)
    
    # Financial breakdown:
    # 2 targeted customers (TP + FP) -> Cost: 2 * $10 = $20
    # 1 correctly identified churner (TP) -> Saved: 1 * 0.20 = 0.2 customers
    # Gross Benefit: 0.2 * $150 = $30
    # Net ROI: $30 - $20 = $10.0
    assert roi == 10.0


def test_evaluate_performance_immature_system(performance_evaluator):
    """
    Validates that a 0-row empty DataFrame (simulating an immature lookback period)
    is handled gracefully, safely skipping evaluation and returning defaults.
    """
    empty_df = pd.DataFrame()
    report = performance_evaluator._evaluate_performance(empty_df)
    
    assert report["is_evaluated"] is False
    assert report["reason"] == "INSUFFICIENT_LOOKBACK_MATURITY"
    assert report["metrics"]["brier_score"] == 0.0


def test_evaluate_performance_nan_handling(performance_evaluator):
    """
    Validates that rows with missing ground truth or predictions are safely dropped
    prior to metric calculation to prevent NaN propagation.
    """
    df_with_nans = pd.DataFrame({
        performance_evaluator.target_col: [1.0, np.nan, 0.0],
        performance_evaluator.prediction_col: [0.8, 0.5, np.nan]
    })
    
    report = performance_evaluator._evaluate_performance(df_with_nans)
    
    assert report["is_evaluated"] is True
    # Only the first row (1.0, 0.8) is valid.
    # Brier for one row: (0.8 - 1.0)^2 = 0.04
    assert np.isclose(report["metrics"]["brier_score"], 0.04)


def test_performance_evaluator_run_e2e(performance_evaluator, mock_resolver_artifact):
    """
    Validates the end-to-end component run. Tests joining separate historical 
    telemetry and label parquet files, calculating metrics, and persisting artifacts.
    """
    # 1. Setup mock telemetry parquet
    telemetry_df = pd.DataFrame({
        performance_evaluator.customer_id_col: ["C1", "C2", "C3", "C4"],
        performance_evaluator.prediction_col: [0.8, 0.2, 0.9, 0.4]
    })
    telemetry_df.to_parquet(mock_resolver_artifact.lookback_telemetry_file_path)
    
    # 2. Setup mock labels parquet (Missing C2, adding C5 to test inner join)
    labels_df = pd.DataFrame({
        performance_evaluator.customer_id_col: ["C1", "C3", "C4", "C5"],
        performance_evaluator.target_col: [1, 1, 0, 1]
    })
    labels_df.to_parquet(mock_resolver_artifact.lookback_labels_file_path)
    
    # Run the component
    artifact = performance_evaluator.run()
    
    # Assert artifacts were generated
    assert os.path.exists(artifact.performance_report_file_path)
    assert os.path.exists(artifact.metadata_file_path)
    
    # Validate report contents
    with open(artifact.performance_report_file_path, "r") as f:
        report = json.load(f)
        
    assert report["is_evaluated"] is True
    assert "metrics" in report
    
    metrics = report["metrics"]
    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert "realized_roi" in metrics