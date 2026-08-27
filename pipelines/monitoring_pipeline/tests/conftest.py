from typing import Any, Dict
from unittest.mock import MagicMock

import pandas as pd
import pytest

from pipelines.monitoring_pipeline.src.core.context import MonitoringPipelineContext
from pipelines.monitoring_pipeline.src.entity.artifact_entity import (
    BaselineAndTelemetryResolverArtifact,
    PerformanceEvaluatorArtifact,
    RuleEngineArtifact,
    StatisticalDriftCalculatorArtifact,
)


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically mock critical environment variables across all tests to prevent
    unintended cloud interactions during unit testing.
    """
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv(
        "S3_PIPELINE_RUN_ARTIFACTS",
        "test-pipeline-run-artifacts",
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")


@pytest.fixture
def mock_global_config_dict() -> Dict[str, Any]:
    """Provides a deterministic mock of the parsed global_config.yaml."""
    return {
        "project_config": {
            "pipeline_name": "monitoring_pipeline",
            "local_artifact_dir": "artifacts",
        },
        "data_schema": {
            "customer_id_column": "customer_unique_id",
            "target_column": "target_is_churn",
            "prediction_column": "predicted_probability",
        },
        "cloud_storage": {
            "s3_data_lake_bucket": "test-data-lake",
            "s3_bronze_layer_prefix": "bronze",
            "s3_model_registry_prefix": "model_registry",
            "s3_model_registry_state_dir": "state",
            "s3_model_registry_pointer_file": "model_state.json",
            "s3_inference_telemetry_prefix": (
                "inference_pipeline_artifacts/mlops_telemetry/inference_logs"
            ),
            "s3_monitoring_output_prefix": "monitoring_pipeline_artifacts",
            "s3_monitoring_audit_reports_dir": "audit_reports",
            "s3_monitoring_action_tokens_dir": "action_tokens",
            "s3_monitoring_matured_evaluations_dir": "matured_evaluations",
            "s3_monitoring_metadata_dir": "monitoring_metadata",
        },
        "monitoring_parameters": {
            "lookback_period_days": 30,
            "top_shap_features_count": 2,
            "zero_bin_epsilon_psi": 0.0001,
            "log_loss_epsilon": 1.0e-15,
        },
        "financial_parameters": {
            "campaign_cost": 10.0,
            "customer_ltv": 150.0,
            "intervention_save_rate": 0.20,
        },
        "rule_engine_thresholds": {
            "prediction_drift_threshold_psi": 0.20,
            "feature_drift_threshold_psi": 0.20,
            "min_drifted_features_for_retrain": 1,
            "brier_degradation_threshold_factor": 1.05,
        },
        "component_config": {
            "baseline_and_telemetry_resolver": {
                "dir_name": "01_baseline_and_telemetry_resolver",
                "current_telemetry_file": "current_telemetry.parquet",
                "lookback_telemetry_file": "lookback_telemetry.parquet",
                "lookback_labels_file": "lookback_matured_labels.parquet",
                "baseline_metrics_file": "baseline_performance_metrics.json",
                "reference_distributions_file": (
                    "reference_feature_distributions.json"
                ),
                "shap_importance_file": (
                    "shap_feature_importance_summary.json"
                ),
            },
            "statistical_drift_calculator": {
                "dir_name": "02_statistical_drift_calculator",
                "drift_report_file": "drift_report.json",
            },
            "performance_evaluator": {
                "dir_name": "03_performance_evaluator",
                "performance_report_file": "performance_report.json",
            },
            "rule_engine": {
                "dir_name": "04_rule_engine",
                "monitoring_report_file": "monitoring_report.json",
                "need_update_file": "need_update.json",
            },
            "artifact_publisher": {
                "dir_name": "05_artifact_publisher",
            },
        },
    }


@pytest.fixture
def mock_pipeline_context(
    mock_global_config_dict: Dict[str, Any],
) -> MagicMock:
    """Provides a mocked MonitoringPipelineContext with simulated resources."""
    context = MagicMock(spec=MonitoringPipelineContext)

    context.run_id = "test_monitoring_run_123"
    context.execution_date = "2026-08-26"
    context.config = mock_global_config_dict
    context.root_dir = "/tmp/monitoring/test_monitoring_run_123"

    mock_con = MagicMock()
    context.duckdb_con = mock_con
    context.s3_sync = MagicMock()

    return context


@pytest.fixture
def dummy_resolver_artifact() -> BaselineAndTelemetryResolverArtifact:
    return BaselineAndTelemetryResolverArtifact(
        champion_run_id="train_run_001",
        baseline_metrics_file_path=(
            "/tmp/monitoring/01/baseline_performance_metrics.json"
        ),
        reference_distributions_file_path=(
            "/tmp/monitoring/01/reference_feature_distributions.json"
        ),
        shap_importance_file_path=(
            "/tmp/monitoring/01/shap_feature_importance_summary.json"
        ),
        current_telemetry_file_path=(
            "/tmp/monitoring/01/current_telemetry.parquet"
        ),
        lookback_telemetry_file_path=(
            "/tmp/monitoring/01/lookback_telemetry.parquet"
        ),
        lookback_labels_file_path=(
            "/tmp/monitoring/01/lookback_matured_labels.parquet"
        ),
        metadata_file_path="/tmp/monitoring/01/metadata.json",
    )


@pytest.fixture
def dummy_drift_artifact() -> StatisticalDriftCalculatorArtifact:
    return StatisticalDriftCalculatorArtifact(
        drift_report_file_path="/tmp/monitoring/02/drift_report.json",
        metadata_file_path="/tmp/monitoring/02/metadata.json",
    )


@pytest.fixture
def dummy_performance_artifact() -> PerformanceEvaluatorArtifact:
    return PerformanceEvaluatorArtifact(
        performance_report_file_path=(
            "/tmp/monitoring/03/performance_report.json"
        ),
        metadata_file_path="/tmp/monitoring/03/metadata.json",
    )


@pytest.fixture
def dummy_rule_engine_artifact() -> RuleEngineArtifact:
    return RuleEngineArtifact(
        monitoring_report_file_path=(
            "/tmp/monitoring/04/monitoring_report.json"
        ),
        need_update_file_path="/tmp/monitoring/04/need_update.json",
        need_update=True,
        metadata_file_path="/tmp/monitoring/04/metadata.json",
    )


@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    """Sample dataframe simulating current or lookback telemetry."""
    return pd.DataFrame(
        {
            "customer_unique_id": ["C001", "C002", "C003"],
            "predicted_probability": [0.1, 0.9, 0.4],
            "feature_a": [10.5, 20.1, 15.0],
            "feature_b": ["Category_1", "Category_2", "Category_1"],
        }
    )


@pytest.fixture
def sample_labels_df() -> pd.DataFrame:
    """Sample dataframe simulating matured ground-truth labels."""
    return pd.DataFrame(
        {
            "customer_unique_id": ["C001", "C002", "C003"],
            "target_is_churn": [0, 1, 0],
        }
    )


@pytest.fixture
def sample_reference_distributions() -> Dict[str, Any]:
    """Sample reference distribution dictionary for PSI calculations."""
    return {
        "distributions": {
            "predicted_probability": {
                "physical_type": "numerical",
                "bin_edges": [0.0, 0.33, 0.66, 1.0],
                "expected_percentages": [0.5, 0.3, 0.2],
            },
            "feature_a": {
                "physical_type": "numerical",
                "bin_edges": [0.0, 10.0, 20.0, 30.0],
                "expected_percentages": [0.25, 0.5, 0.25],
            },
            "feature_b": {
                "physical_type": "categorical",
                "categories": ["Category_1", "Category_2"],
                "expected_percentages": [0.6, 0.4],
            },
        }
    }


@pytest.fixture
def sample_shap_importances() -> Dict[str, Any]:
    """Sample SHAP importance array for determining top features."""
    return {
        "feature_importance": [
            {
                "feature_name": "feature_a",
                "mean_abs_shap_value": 0.45,
            },
            {
                "feature_name": "feature_b",
                "mean_abs_shap_value": 0.35,
            },
            {
                "feature_name": "feature_c",
                "mean_abs_shap_value": 0.10,
            },
        ]
    }