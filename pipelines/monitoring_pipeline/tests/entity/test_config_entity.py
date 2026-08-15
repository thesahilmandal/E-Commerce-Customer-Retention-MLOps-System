import os
import pytest
from unittest.mock import patch

from pipelines.monitoring_pipeline.src.entity.config_entity import (
    BaselineAndTelemetryResolverConfig,
    StatisticalDriftCalculatorConfig,
    PerformanceEvaluatorConfig,
    RuleEngineConfig,
    ArtifactPublisherConfig
)
from shared_core.exceptions.custom_exception import CustomException


def test_baseline_and_telemetry_resolver_config_parsing(mock_context):
    """
    Validates that the BaselineAndTelemetryResolverConfig successfully parses
    parameters, calculates lookback dates, formats partition suffixes, and 
    constructs correct URIs.
    """
    config = BaselineAndTelemetryResolverConfig.get_config(mock_context)
    
    # Verify local directory parsing
    assert config.resolver_root_dir.endswith("01_resolver")
    assert os.path.exists(config.resolver_root_dir)
    
    # Verify temporal logic (execution_date="2026-08-14" - 30 days = "2026-07-15")
    assert config.current_date == "2026-08-14"
    assert config.lookback_date == "2026-07-15"
    assert config.current_partition_suffix == "year=2026/month=08/day=14"
    assert config.lookback_partition_suffix == "year=2026/month=07/day=15"
    assert config.lookback_period_days == 30
    
    # Verify S3 URI construction
    assert config.s3_registry_pointer_uri == "s3://mock-bucket/model_registry/state/model_state.json"
    assert config.s3_telemetry_base_uri == "s3://mock-bucket/telemetry"
    assert config.s3_data_lake_bronze_uri == "s3://mock-bucket/bronze"
    
    # Verify file paths
    assert config.baseline_metrics_file_path.endswith("baseline_metrics.json")
    assert config.reference_distributions_file_path.endswith("ref_dist.json")
    assert config.shap_importance_file_path.endswith("shap.json")


def test_statistical_drift_calculator_config_parsing(mock_context):
    """
    Validates that the StatisticalDriftCalculatorConfig correctly parses
    component-specific and monitoring parameter configurations.
    """
    config = StatisticalDriftCalculatorConfig.get_config(mock_context)
    
    assert config.drift_root_dir.endswith("02_drift")
    assert os.path.exists(config.drift_root_dir)
    
    assert config.drift_report_file_path.endswith("drift_report.json")
    assert config.top_shap_features_count == 5
    assert config.zero_bin_epsilon_psi == 0.0001


def test_performance_evaluator_config_parsing(mock_context):
    """
    Validates that the PerformanceEvaluatorConfig correctly parses
    financial parameters and component configurations.
    """
    config = PerformanceEvaluatorConfig.get_config(mock_context)
    
    assert config.evaluator_root_dir.endswith("03_perf")
    assert os.path.exists(config.evaluator_root_dir)
    
    assert config.performance_report_file_path.endswith("performance_report.json")
    assert config.campaign_cost == 10.0
    assert config.customer_ltv == 150.0
    assert config.intervention_save_rate == 0.20
    assert config.log_loss_epsilon == 1.0e-15


def test_rule_engine_config_parsing(mock_context):
    """
    Validates that the RuleEngineConfig correctly parses threshold parameters
    used for the deterministic retraining decision logic.
    """
    config = RuleEngineConfig.get_config(mock_context)
    
    assert config.rule_engine_root_dir.endswith("04_rule")
    assert os.path.exists(config.rule_engine_root_dir)
    
    assert config.monitoring_report_file_path.endswith("monitoring_report.json")
    assert config.need_update_file_path.endswith("need_update.json")
    
    assert config.prediction_drift_threshold_psi == 0.20
    assert config.feature_drift_threshold_psi == 0.20
    assert config.min_drifted_features_for_retrain == 2
    assert config.brier_degradation_threshold_factor == 1.05


def test_artifact_publisher_config_parsing(mock_context):
    """
    Validates that the ArtifactPublisherConfig correctly parses orchestration 
    metadata and target S3 output prefixes.
    """
    config = ArtifactPublisherConfig.get_config(mock_context)
    
    assert config.publisher_root_dir.endswith("05_pub")
    assert os.path.exists(config.publisher_root_dir)
    
    assert config.run_id == "test_run_123"
    assert config.execution_date == "2026-08-14"
    assert config.s3_bucket_name == "mock-bucket"
    assert config.s3_monitoring_output_prefix == "monitoring_output"
    assert config.s3_audit_reports_dir == "audit_reports"
    assert config.s3_action_tokens_dir == "action_tokens"
    assert config.s3_matured_evaluations_dir == "matured_evaluations"
    assert config.s3_metadata_dir == "monitoring_metadata"


def test_config_empty_dictionary_fallback_defaults(mock_context):
    """
    Validates the defensive 'or {}' logic used throughout the config parsing.
    If the YAML completely lacks expected sections, the dataclasses should fall
    back to their hardcoded default values without raising a KeyError or NoneType error.
    """
    # Empty out the context configuration
    mock_context.config = {}
    
    config = RuleEngineConfig.get_config(mock_context)
    
    # Asserting hardcoded defaults from the get_config method
    assert config.prediction_drift_threshold_psi == 0.20
    assert config.feature_drift_threshold_psi == 0.20
    assert config.min_drifted_features_for_retrain == 2
    assert config.brier_degradation_threshold_factor == 1.05


@patch("pipelines.monitoring_pipeline.src.entity.config_entity.os.makedirs")
def test_config_exception_handling(mock_makedirs, mock_context):
    """
    Validates that a lower-level execution failure (like an OS permission error)
    is securely caught and re-raised as a CustomException by the entity classmethod.
    """
    # Simulate a file system failure when attempting to create the component directory
    mock_makedirs.side_effect = PermissionError("Simulated permission denied")
    
    with pytest.raises(CustomException):
        BaselineAndTelemetryResolverConfig.get_config(mock_context)