import pytest
from unittest.mock import patch, MagicMock

from pipelines.monitoring_pipeline.src.runner import MonitoringPipelineRunner
from shared_core.exceptions.custom_exception import CustomException


@patch("pipelines.monitoring_pipeline.src.runner.ArtifactPublisher")
@patch("pipelines.monitoring_pipeline.src.runner.ArtifactPublisherConfig")
@patch("pipelines.monitoring_pipeline.src.runner.RuleEngine")
@patch("pipelines.monitoring_pipeline.src.runner.RuleEngineConfig")
@patch("pipelines.monitoring_pipeline.src.runner.PerformanceEvaluator")
@patch("pipelines.monitoring_pipeline.src.runner.PerformanceEvaluatorConfig")
@patch("pipelines.monitoring_pipeline.src.runner.StatisticalDriftCalculator")
@patch("pipelines.monitoring_pipeline.src.runner.StatisticalDriftCalculatorConfig")
@patch("pipelines.monitoring_pipeline.src.runner.BaselineAndTelemetryResolver")
@patch("pipelines.monitoring_pipeline.src.runner.BaselineAndTelemetryResolverConfig")
@patch("pipelines.monitoring_pipeline.src.runner.MonitoringPipelineContext")
def test_monitoring_pipeline_runner_successful_execution_and_lineage(
    mock_context_class,
    mock_res_cfg, mock_res_class,
    mock_drift_cfg, mock_drift_class,
    mock_perf_cfg, mock_perf_class,
    mock_rule_cfg, mock_rule_class,
    mock_pub_cfg, mock_pub_class
):
    """
    Validates the end-to-end orchestration logic. Ensures all components are properly 
    configured, instantiated with correct artifact lineage, and executed sequentially.
    """
    # 1. Setup Context Manager Mock
    mock_context_instance = MagicMock()
    mock_context_class.return_value.__enter__.return_value = mock_context_instance
    
    # 2. Setup component return artifacts to test sequential wiring (lineage)
    res_artifact = MagicMock()
    mock_res_class.return_value.run.return_value = res_artifact
    
    drift_artifact = MagicMock()
    mock_drift_class.return_value.run.return_value = drift_artifact
    
    perf_artifact = MagicMock()
    mock_perf_class.return_value.run.return_value = perf_artifact
    
    rule_artifact = MagicMock()
    rule_artifact.need_update = True  # Triggers the 'RETRAINING TRIGGERED' log branch
    mock_rule_class.return_value.run.return_value = rule_artifact
    
    pub_artifact = MagicMock()
    mock_pub_class.return_value.run.return_value = pub_artifact
    
    # 3. Execute Runner
    runner = MonitoringPipelineRunner(run_id="test_run_123", execution_date="2026-08-14")
    runner.run()
    
    # 4. Assert Context Initialization
    mock_context_class.assert_called_once_with(run_id="test_run_123", execution_date="2026-08-14")
    
    # 5. Assert Component Instantiation and Lineage Wiring
    mock_res_class.assert_called_once_with(
        config=mock_res_cfg.get_config.return_value,
        context=mock_context_instance
    )
    
    mock_drift_class.assert_called_once_with(
        config=mock_drift_cfg.get_config.return_value,
        context=mock_context_instance,
        resolver_artifact=res_artifact
    )
    
    mock_perf_class.assert_called_once_with(
        config=mock_perf_cfg.get_config.return_value,
        context=mock_context_instance,
        resolver_artifact=res_artifact
    )
    
    mock_rule_class.assert_called_once_with(
        config=mock_rule_cfg.get_config.return_value,
        context=mock_context_instance,
        resolver_artifact=res_artifact,
        drift_artifact=drift_artifact,
        performance_artifact=perf_artifact
    )
    
    mock_pub_class.assert_called_once_with(
        config=mock_pub_cfg.get_config.return_value,
        context=mock_context_instance,
        resolver_artifact=res_artifact,
        drift_artifact=drift_artifact,
        performance_artifact=perf_artifact,
        rule_engine_artifact=rule_artifact
    )
    
    # 6. Assert Sequential Execution
    mock_res_class.return_value.run.assert_called_once()
    mock_drift_class.return_value.run.assert_called_once()
    mock_perf_class.return_value.run.assert_called_once()
    mock_rule_class.return_value.run.assert_called_once()
    mock_pub_class.return_value.run.assert_called_once()


@patch("pipelines.monitoring_pipeline.src.runner.MonitoringPipelineContext")
def test_monitoring_pipeline_runner_exception_handling(mock_context_class):
    """
    Validates that fatal failures inside the orchestrator are safely caught 
    and re-raised as CustomExceptions.
    """
    # Simulate a fatal context manager initialization failure
    mock_context_class.return_value.__enter__.side_effect = Exception("Simulated fatal orchestrator failure")
    
    runner = MonitoringPipelineRunner()
    
    with pytest.raises(CustomException) as exc_info:
        runner.run()
        
    assert "Simulated fatal orchestrator failure" in str(exc_info.value)