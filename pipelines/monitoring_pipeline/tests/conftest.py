import os
import pytest
import duckdb
from unittest.mock import patch

from pipelines.monitoring_pipeline.src.core.context import MonitoringPipelineContext
from pipelines.monitoring_pipeline.src.entity.artifact_entity import (
    BaselineAndTelemetryResolverArtifact,
    StatisticalDriftCalculatorArtifact,
    PerformanceEvaluatorArtifact,
    RuleEngineArtifact,
    ArtifactPublisherArtifact
)


@pytest.fixture
def mock_global_config_dict(tmp_path) -> dict:
    """
    Provides a complete, synthetic global configuration dictionary for testing.
    Uses pytest's tmp_path to ensure all local artifacts are written to a safe,
    ephemeral directory during testing.
    """
    return {
        "project_config": {
            "pipeline_name": "test_monitoring_pipeline",
            "local_artifact_dir": str(tmp_path / "artifacts")
        },
        "data_schema": {
            "customer_id_column": "customer_unique_id",
            "target_column": "target_is_churn",
            "prediction_column": "predicted_probability"
        },
        "cloud_storage": {
            "s3_data_lake_bucket": "mock-bucket",
            "s3_bronze_layer_prefix": "bronze",
            "s3_model_registry_prefix": "model_registry",
            "s3_model_registry_state_dir": "state",
            "s3_model_registry_pointer_file": "model_state.json",
            "s3_inference_telemetry_prefix": "telemetry",
            "s3_monitoring_output_prefix": "monitoring_output",
            "s3_monitoring_audit_reports_dir": "audit_reports",
            "s3_monitoring_action_tokens_dir": "action_tokens",
            "s3_monitoring_matured_evaluations_dir": "matured_evaluations",
            "s3_monitoring_metadata_dir": "monitoring_metadata"
        },
        "monitoring_parameters": {
            "lookback_period_days": 30,
            "top_shap_features_count": 5,
            "zero_bin_epsilon_psi": 0.0001,
            "log_loss_epsilon": 1.0e-15
        },
        "financial_parameters": {
            "campaign_cost": 10.0,
            "customer_ltv": 150.0,
            "intervention_save_rate": 0.20
        },
        "rule_engine_thresholds": {
            "prediction_drift_threshold_psi": 0.20,
            "feature_drift_threshold_psi": 0.20,
            "min_drifted_features_for_retrain": 2,
            "brier_degradation_threshold_factor": 1.05
        },
        "component_config": {
            "baseline_and_telemetry_resolver": {
                "dir_name": "01_resolver",
                "current_telemetry_file": "current_telemetry.parquet",
                "lookback_telemetry_file": "lookback_telemetry.parquet",
                "lookback_labels_file": "lookback_labels.parquet",
                "baseline_metrics_file": "baseline_metrics.json",
                "reference_distributions_file": "ref_dist.json",
                "shap_importance_file": "shap.json"
            },
            "statistical_drift_calculator": {
                "dir_name": "02_drift",
                "drift_report_file": "drift_report.json"
            },
            "performance_evaluator": {
                "dir_name": "03_perf",
                "performance_report_file": "performance_report.json"
            },
            "rule_engine": {
                "dir_name": "04_rule",
                "monitoring_report_file": "monitoring_report.json",
                "need_update_file": "need_update.json"
            },
            "artifact_publisher": {
                "dir_name": "05_pub"
            }
        }
    }


@pytest.fixture
def mock_context(mock_global_config_dict, tmp_path):
    """
    Yields a MonitoringPipelineContext configured for unit testing.
    Bypasses file system configuration loading and circumvents AWS authentication 
    requirements by injecting a pure in-memory DuckDB connection.
    """
    # Create a real dummy config file to naturally pass the os.path.exists check in __init__
    # This avoids globally mocking os.path.exists, which inadvertently breaks os.makedirs.
    dummy_config_path = tmp_path / "dummy_config.yaml"
    dummy_config_path.touch()

    with patch("pipelines.monitoring_pipeline.src.core.context.read_yaml", return_value=mock_global_config_dict):
         
        context = MonitoringPipelineContext(
            run_id="test_run_123",
            execution_date="2026-08-14",
            config_path=str(dummy_config_path)
        )
        
        # Subvert the context manager to skip heavy/flaky AWS DuckDB extensions in unit tests
        with patch.object(context, "__enter__", return_value=context), \
             patch.object(context, "__exit__", return_value=None):
             
            # Manually initialize a safe, standard in-memory connection
            context.duckdb_con = duckdb.connect(database=":memory:")
            
            yield context
            
            # Teardown the in-memory database
            context.duckdb_con.close()


@pytest.fixture
def mock_resolver_artifact(tmp_path) -> BaselineAndTelemetryResolverArtifact:
    """Provides a synthetic BaselineAndTelemetryResolverArtifact with valid temporary paths."""
    return BaselineAndTelemetryResolverArtifact(
        champion_run_id="champion_123",
        baseline_metrics_file_path=str(tmp_path / "baseline_metrics.json"),
        reference_distributions_file_path=str(tmp_path / "ref_dist.json"),
        shap_importance_file_path=str(tmp_path / "shap.json"),
        current_telemetry_file_path=str(tmp_path / "current_telemetry.parquet"),
        lookback_telemetry_file_path=str(tmp_path / "lookback_telemetry.parquet"),
        lookback_labels_file_path=str(tmp_path / "lookback_labels.parquet"),
        metadata_file_path=str(tmp_path / "resolver_meta.json")
    )


@pytest.fixture
def mock_drift_artifact(tmp_path) -> StatisticalDriftCalculatorArtifact:
    """Provides a synthetic StatisticalDriftCalculatorArtifact with valid temporary paths."""
    return StatisticalDriftCalculatorArtifact(
        drift_report_file_path=str(tmp_path / "drift_report.json"),
        metadata_file_path=str(tmp_path / "drift_meta.json")
    )


@pytest.fixture
def mock_performance_artifact(tmp_path) -> PerformanceEvaluatorArtifact:
    """Provides a synthetic PerformanceEvaluatorArtifact with valid temporary paths."""
    return PerformanceEvaluatorArtifact(
        performance_report_file_path=str(tmp_path / "performance_report.json"),
        metadata_file_path=str(tmp_path / "perf_meta.json")
    )


@pytest.fixture
def mock_rule_engine_artifact(tmp_path) -> RuleEngineArtifact:
    """Provides a synthetic RuleEngineArtifact representing a trigger scenario."""
    return RuleEngineArtifact(
        monitoring_report_file_path=str(tmp_path / "monitoring_report.json"),
        need_update_file_path=str(tmp_path / "need_update.json"),
        need_update=True,
        metadata_file_path=str(tmp_path / "rule_engine_meta.json")
    )