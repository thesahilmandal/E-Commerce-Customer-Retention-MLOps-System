from unittest.mock import MagicMock, patch

import pytest

from pipelines.monitoring_pipeline.src.entity.config_entity import (
    ArtifactPublisherConfig,
    BaselineAndTelemetryResolverConfig,
    PerformanceEvaluatorConfig,
    RuleEngineConfig,
    StatisticalDriftCalculatorConfig,
)
from shared_core.exceptions.custom_exception import CustomException


@patch("pipelines.monitoring_pipeline.src.entity.config_entity.os.makedirs")
def test_baseline_and_telemetry_resolver_config_from_context(
    mock_makedirs: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    config = BaselineAndTelemetryResolverConfig.get_config(
        mock_pipeline_context
    )

    assert "01_baseline_and_telemetry_resolver" in config.resolver_root_dir
    assert config.current_date == "2026-08-26"
    assert config.lookback_date == "2026-07-27"
    assert config.current_partition_suffix == "year=2026/month=08/day=26"
    assert config.lookback_partition_suffix == "year=2026/month=07/day=27"

    assert (
        config.s3_registry_pointer_uri
        == "s3://test-pipeline-run-artifacts/"
        "model_registry/state/model_state.json"
    )
    assert (
        config.s3_telemetry_base_uri
        == "s3://test-pipeline-run-artifacts/"
        "inference_pipeline_artifacts/mlops_telemetry/inference_logs"
    )
    assert (
        config.s3_data_lake_bronze_uri
        == "s3://test-pipeline-run-artifacts/bronze"
    )

    assert config.baseline_metrics_file_path.endswith(
        "baseline_performance_metrics.json"
    )
    assert config.reference_distributions_file_path.endswith(
        "reference_feature_distributions.json"
    )
    assert config.shap_importance_file_path.endswith(
        "shap_feature_importance_summary.json"
    )
    assert config.current_telemetry_file_path.endswith(
        "current_telemetry.parquet"
    )
    assert config.lookback_telemetry_file_path.endswith(
        "lookback_telemetry.parquet"
    )
    assert config.lookback_labels_file_path.endswith(
        "lookback_matured_labels.parquet"
    )

    mock_makedirs.assert_called_once_with(
        config.resolver_root_dir,
        exist_ok=True,
    )


@patch("pipelines.monitoring_pipeline.src.entity.config_entity.os.makedirs")
def test_statistical_drift_calculator_config_from_context(
    mock_makedirs: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    config = StatisticalDriftCalculatorConfig.get_config(
        mock_pipeline_context
    )

    assert "02_statistical_drift_calculator" in config.drift_root_dir
    assert config.drift_report_file_path.endswith("drift_report.json")
    assert config.metadata_file_path.endswith("metadata.json")
    assert config.top_shap_features_count == 2
    assert config.zero_bin_epsilon_psi == 0.0001

    mock_makedirs.assert_called_once_with(
        config.drift_root_dir,
        exist_ok=True,
    )


@patch("pipelines.monitoring_pipeline.src.entity.config_entity.os.makedirs")
def test_performance_evaluator_config_from_context(
    mock_makedirs: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    config = PerformanceEvaluatorConfig.get_config(
        mock_pipeline_context
    )

    assert "03_performance_evaluator" in config.evaluator_root_dir
    assert config.performance_report_file_path.endswith(
        "performance_report.json"
    )
    assert config.campaign_cost == 10.0
    assert config.customer_ltv == 150.0
    assert config.intervention_save_rate == 0.20
    assert config.log_loss_epsilon == 1.0e-15

    mock_makedirs.assert_called_once_with(
        config.evaluator_root_dir,
        exist_ok=True,
    )


@patch("pipelines.monitoring_pipeline.src.entity.config_entity.os.makedirs")
def test_rule_engine_config_from_context(
    mock_makedirs: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    config = RuleEngineConfig.get_config(mock_pipeline_context)

    assert "04_rule_engine" in config.rule_engine_root_dir
    assert config.monitoring_report_file_path.endswith(
        "monitoring_report.json"
    )
    assert config.need_update_file_path.endswith("need_update.json")
    assert config.prediction_drift_threshold_psi == 0.20
    assert config.feature_drift_threshold_psi == 0.20
    assert config.min_drifted_features_for_retrain == 1
    assert config.brier_degradation_threshold_factor == 1.05

    mock_makedirs.assert_called_once_with(
        config.rule_engine_root_dir,
        exist_ok=True,
    )


@patch("pipelines.monitoring_pipeline.src.entity.config_entity.os.makedirs")
def test_artifact_publisher_config_from_context(
    mock_makedirs: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    config = ArtifactPublisherConfig.get_config(mock_pipeline_context)

    assert "05_artifact_publisher" in config.publisher_root_dir
    assert config.s3_bucket_name == "test-pipeline-run-artifacts"
    assert (
        config.s3_monitoring_output_prefix
        == "monitoring_pipeline_artifacts"
    )
    assert config.s3_audit_reports_dir == "audit_reports"
    assert config.s3_action_tokens_dir == "action_tokens"
    assert config.s3_matured_evaluations_dir == "matured_evaluations"
    assert config.s3_metadata_dir == "monitoring_metadata"
    assert config.run_id == "test_monitoring_run_123"
    assert config.execution_date == "2026-08-26"

    mock_makedirs.assert_called_once_with(
        config.publisher_root_dir,
        exist_ok=True,
    )


@pytest.mark.parametrize(
    "config_class",
    [
        BaselineAndTelemetryResolverConfig,
        StatisticalDriftCalculatorConfig,
        PerformanceEvaluatorConfig,
        RuleEngineConfig,
        ArtifactPublisherConfig,
    ],
)
@patch("pipelines.monitoring_pipeline.src.entity.config_entity.os.makedirs")
def test_config_initialization_exceptions(
    mock_makedirs: MagicMock,
    config_class: type,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_makedirs.side_effect = OSError("Permission denied")

    with pytest.raises(CustomException) as exc_info:
        config_class.get_config(mock_pipeline_context)

    assert "Permission denied" in str(exc_info.value)