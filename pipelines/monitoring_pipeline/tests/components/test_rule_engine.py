import os
import json
import pytest

from pipelines.monitoring_pipeline.src.components.rule_engine import RuleEngine
from pipelines.monitoring_pipeline.src.entity.config_entity import RuleEngineConfig


@pytest.fixture
def rule_engine_config(mock_context):
    """Provides a valid configuration for the Rule Engine."""
    return RuleEngineConfig.get_config(mock_context)


@pytest.fixture
def rule_engine(
    rule_engine_config, 
    mock_context, 
    mock_resolver_artifact, 
    mock_drift_artifact, 
    mock_performance_artifact
):
    """Provides a configured RuleEngine instance."""
    return RuleEngine(
        config=rule_engine_config,
        context=mock_context,
        resolver_artifact=mock_resolver_artifact,
        drift_artifact=mock_drift_artifact,
        performance_artifact=mock_performance_artifact
    )


def test_evaluate_prediction_drift(rule_engine):
    """
    Validates Condition 1: Triggers if prediction PSI exceeds the threshold (0.20).
    """
    # Test triggering case
    drift_report_triggered = {
        "prediction_drift": {
            "predicted_probability": {"psi_score": 0.25}
        }
    }
    is_triggered, details = rule_engine._evaluate_prediction_drift(drift_report_triggered)
    assert is_triggered is True
    assert details["severity_level"] == "CRITICAL"

    # Test non-triggering case
    drift_report_safe = {
        "prediction_drift": {
            "predicted_probability": {"psi_score": 0.15}
        }
    }
    is_triggered, details = rule_engine._evaluate_prediction_drift(drift_report_safe)
    assert is_triggered is False
    assert details["severity_level"] == "INFO"


def test_evaluate_feature_drift(rule_engine):
    """
    Validates Condition 2: Triggers if >= min_drifted_features (2) 
    exceed the PSI threshold (0.20).
    """
    # Test triggering case (2 features drifted)
    drift_report_triggered = {
        "feature_drift": {
            "feature_1": {"psi_score": 0.25},
            "feature_2": {"psi_score": 0.30},
            "feature_3": {"psi_score": 0.05}
        }
    }
    is_triggered, details = rule_engine._evaluate_feature_drift(drift_report_triggered)
    assert is_triggered is True
    assert details["actual_value"] == 2
    assert "feature_1" in details["drifted_features_list"]

    # Test non-triggering case (Only 1 feature drifted)
    drift_report_safe = {
        "feature_drift": {
            "feature_1": {"psi_score": 0.25},
            "feature_2": {"psi_score": 0.10},
            "feature_3": {"psi_score": 0.05}
        }
    }
    is_triggered, details = rule_engine._evaluate_feature_drift(drift_report_safe)
    assert is_triggered is False
    assert details["actual_value"] == 1


def test_evaluate_performance_degradation_immature_system(rule_engine):
    """
    Validates that Condition 3 gracefully bypasses systems lacking matured evaluation data.
    """
    performance_report = {"is_evaluated": False, "reason": "INSUFFICIENT_LOOKBACK_MATURITY"}
    baseline_metrics = {"brier_score": 0.10}
    
    is_triggered, details = rule_engine._evaluate_performance_degradation(
        performance_report, baseline_metrics
    )
    assert is_triggered is False
    assert details["severity_level"] == "INFO"


def test_evaluate_performance_degradation_logic(rule_engine):
    """
    Validates Condition 3: Triggers if current Brier score exceeds the dynamic boundary.
    (Baseline: 0.10, Threshold Factor: 1.05 -> Boundary: 0.105)
    """
    baseline_metrics = {"global_metrics": {"brier_score": 0.100}}
    
    # Test triggering case (> 0.105)
    performance_report_triggered = {
        "is_evaluated": True,
        "metrics": {"brier_score": 0.110}
    }
    is_triggered, details = rule_engine._evaluate_performance_degradation(
        performance_report_triggered, baseline_metrics
    )
    assert is_triggered is True
    # Using pytest.approx() to mitigate IEEE 754 floating point arithmetic variances (0.10500000000000001)
    assert details["degradation_boundary"] == pytest.approx(0.105)

    # Test non-triggering case (<= 0.105)
    performance_report_safe = {
        "is_evaluated": True,
        "metrics": {"brier_score": 0.102}
    }
    is_triggered, details = rule_engine._evaluate_performance_degradation(
        performance_report_safe, baseline_metrics
    )
    assert is_triggered is False


def test_evaluate_performance_degradation_schema_resolution(rule_engine):
    """
    Validates that baseline Brier score can be robustly resolved from various potential 
    historical training schema locations.
    """
    performance_report = {"is_evaluated": True, "metrics": {"brier_score": 0.150}}
    
    schemas_to_test = [
        {"global_metrics": {"brier_score": 0.10}},
        {"test_metrics": {"brier_score": 0.10}},
        {"brier_score": 0.10}
    ]
    
    for baseline in schemas_to_test:
        is_triggered, details = rule_engine._evaluate_performance_degradation(
            performance_report, baseline
        )
        assert is_triggered is True
        assert details["baseline_value"] == 0.10


def test_formulate_trigger_reasons(rule_engine):
    """
    Validates the standardized trigger string concatenation.
    """
    reason_all = rule_engine._formulate_trigger_reasons(True, True, True)
    assert reason_all == "CRITICAL_PREDICTION_DRIFT | CRITICAL_FEATURE_DRIFT | SEVERE_PERFORMANCE_DEGRADATION"
    
    reason_none = rule_engine._formulate_trigger_reasons(False, False, False)
    assert reason_none == "NONE"


def test_rule_engine_run_e2e(
    rule_engine, 
    mock_resolver_artifact, 
    mock_drift_artifact, 
    mock_performance_artifact
):
    """
    Validates the end-to-end component run. Writes synthetic metrics and evaluates 
    the overall pipeline decision.
    """
    # 1. Setup mock baseline metrics (Resolver)
    with open(mock_resolver_artifact.baseline_metrics_file_path, "w") as f:
        json.dump({"brier_score": 0.10}, f)
        
    # 2. Setup mock drift report (Triggers Condition 1 & 2)
    with open(mock_drift_artifact.drift_report_file_path, "w") as f:
        json.dump({
            "prediction_drift": {"predicted_probability": {"psi_score": 0.30}},
            "feature_drift": {
                "f1": {"psi_score": 0.25}, 
                "f2": {"psi_score": 0.25}
            }
        }, f)
        
    # 3. Setup mock performance report (Does NOT trigger Condition 3)
    with open(mock_performance_artifact.performance_report_file_path, "w") as f:
        json.dump({
            "is_evaluated": True,
            "metrics": {"brier_score": 0.101}
        }, f)
        
    # Run the component
    artifact = rule_engine.run()
    
    # Assert artifacts were generated
    assert os.path.exists(artifact.monitoring_report_file_path)
    assert os.path.exists(artifact.need_update_file_path)
    assert os.path.exists(artifact.metadata_file_path)
    
    # Validate trigger decision output
    assert artifact.need_update is True
    
    with open(artifact.need_update_file_path, "r") as f:
        payload = json.load(f)
        
    assert payload["need_update"] is True
    assert "CRITICAL_PREDICTION_DRIFT" in payload["trigger_reason"]
    assert "CRITICAL_FEATURE_DRIFT" in payload["trigger_reason"]
    assert "SEVERE_PERFORMANCE_DEGRADATION" not in payload["trigger_reason"]