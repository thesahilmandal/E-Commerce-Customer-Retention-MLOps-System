import sys
import pytest

from unittest.mock import MagicMock, patch

from pipelines.monitoring_pipeline.src.runner import (
    MonitoringPipelineRunner,
    main,
)

from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def mock_pipeline_components() -> dict:
    """
    Mocks all heavy components and configurations for the
    MonitoringPipelineRunner.

    Ensures tests focus strictly on orchestration and control flow.
    """
    with (
        patch(
            "pipelines.monitoring_pipeline.src.runner.MonitoringPipelineContext"
        ) as mock_ctx,
        patch(
            "pipelines.monitoring_pipeline.src.runner.BaselineAndTelemetryResolverConfig"
        ) as mock_res_cfg,
        patch(
            "pipelines.monitoring_pipeline.src.runner.BaselineAndTelemetryResolver"
        ) as mock_res,
        patch(
            "pipelines.monitoring_pipeline.src.runner.StatisticalDriftCalculatorConfig"
        ) as mock_drift_cfg,
        patch(
            "pipelines.monitoring_pipeline.src.runner.StatisticalDriftCalculator"
        ) as mock_drift,
        patch(
            "pipelines.monitoring_pipeline.src.runner.PerformanceEvaluatorConfig"
        ) as mock_perf_cfg,
        patch(
            "pipelines.monitoring_pipeline.src.runner.PerformanceEvaluator"
        ) as mock_perf,
        patch(
            "pipelines.monitoring_pipeline.src.runner.RuleEngineConfig"
        ) as mock_rule_cfg,
        patch(
            "pipelines.monitoring_pipeline.src.runner.RuleEngine"
        ) as mock_rule,
        patch(
            "pipelines.monitoring_pipeline.src.runner.ArtifactPublisherConfig"
        ) as mock_pub_cfg,
        patch(
            "pipelines.monitoring_pipeline.src.runner.ArtifactPublisher"
        ) as mock_pub,
    ):
        # Properly mock the Context Manager __enter__ return value
        mock_ctx_instance = mock_ctx.return_value
        mock_ctx_enter = MagicMock()
        mock_ctx_instance.__enter__.return_value = mock_ctx_enter

        # Mock component run returns
        mock_res.return_value.run.return_value = MagicMock()
        mock_drift.return_value.run.return_value = MagicMock()
        mock_perf.return_value.run.return_value = MagicMock()

        # Mock Rule Engine outcome
        mock_rule_art = MagicMock()
        mock_rule_art.need_update = False
        mock_rule.return_value.run.return_value = mock_rule_art

        # Mock Publisher outcome
        mock_pub_art = MagicMock()
        mock_pub_art.s3_audit_report_uri = "s3://mock-audit"
        mock_pub_art.s3_action_token_uri = "s3://mock-token"
        mock_pub.return_value.run.return_value = mock_pub_art

        yield {
            "context_mgr": mock_ctx,
            "context_enter": mock_ctx_enter,
            "resolver_cfg": mock_res_cfg,
            "resolver": mock_res,
            "drift_cfg": mock_drift_cfg,
            "drift": mock_drift,
            "perf_cfg": mock_perf_cfg,
            "perf": mock_perf,
            "rule_cfg": mock_rule_cfg,
            "rule": mock_rule,
            "rule_art": mock_rule_art,
            "pub_cfg": mock_pub_cfg,
            "pub": mock_pub,
        }


def test_runner_initialization() -> None:
    runner = MonitoringPipelineRunner(
        run_id="init_test",
        execution_date="2026-08-27",
    )

    assert runner.run_id == "init_test"
    assert runner.execution_date == "2026-08-27"


def test_runner_execution_success_no_retrain(
    mock_pipeline_components: dict,
) -> None:
    runner = MonitoringPipelineRunner(
        run_id="test_run_01",
        execution_date="2026-08-27",
    )

    runner.run()

    comps = mock_pipeline_components

    # Assert Context
    comps["context_mgr"].assert_called_once_with(
        run_id="test_run_01",
        execution_date="2026-08-27",
    )

    # Assert configs generated using the correct context
    comps["resolver_cfg"].get_config.assert_called_once_with(
        comps["context_enter"]
    )
    comps["drift_cfg"].get_config.assert_called_once_with(
        comps["context_enter"]
    )
    comps["perf_cfg"].get_config.assert_called_once_with(
        comps["context_enter"]
    )
    comps["rule_cfg"].get_config.assert_called_once_with(
        comps["context_enter"]
    )
    comps["pub_cfg"].get_config.assert_called_once_with(
        comps["context_enter"]
    )

    # Assert all components are instantiated and run
    comps["resolver"].assert_called_once()
    comps["resolver"].return_value.run.assert_called_once()

    comps["drift"].assert_called_once()
    comps["drift"].return_value.run.assert_called_once()

    comps["perf"].assert_called_once()
    comps["perf"].return_value.run.assert_called_once()

    comps["rule"].assert_called_once()
    comps["rule"].return_value.run.assert_called_once()

    comps["pub"].assert_called_once()
    comps["pub"].return_value.run.assert_called_once()


def test_runner_execution_success_with_retrain(
    mock_pipeline_components: dict,
) -> None:
    # Manipulate the rule engine mock to trigger retraining log paths
    mock_pipeline_components["rule_art"].need_update = True

    runner = MonitoringPipelineRunner(
        run_id="test_run_02",
        execution_date="2026-08-27",
    )

    # Should execute successfully without throwing exceptions
    runner.run()

    mock_pipeline_components["pub"].return_value.run.assert_called_once()


def test_runner_execution_component_failure(
    mock_pipeline_components: dict,
) -> None:
    comps = mock_pipeline_components

    # Simulate a critical failure in the Performance Evaluator
    comps["perf"].return_value.run.side_effect = Exception(
        "Simulated Evaluator Crash"
    )

    runner = MonitoringPipelineRunner(
        run_id="test_run_03",
        execution_date="2026-08-27",
    )

    with pytest.raises(CustomException) as exc_info:
        runner.run()

    assert "Simulated Evaluator Crash" in str(exc_info.value)

    # Assert execution halts immediately and subsequent components are not called
    comps["resolver"].return_value.run.assert_called_once()
    comps["drift"].return_value.run.assert_called_once()
    comps["perf"].return_value.run.assert_called_once()
    comps["rule"].return_value.run.assert_not_called()
    comps["pub"].return_value.run.assert_not_called()


@patch(
    "sys.argv",
    [
        "runner.py",
        "--run-id",
        "cli_test_01",
        "--execution-date",
        "2026-08-27",
    ],
)
@patch("pipelines.monitoring_pipeline.src.runner.MonitoringPipelineRunner")
def test_main_cli_success(mock_runner_class: MagicMock) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0

    mock_runner_class.assert_called_once_with(
        run_id="cli_test_01",
        execution_date="2026-08-27",
    )
    mock_runner_class.return_value.run.assert_called_once()


@patch("sys.argv", ["runner.py"])
def test_main_cli_missing_args() -> None:
    # argparse raises SystemExit(2) for missing required arguments
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


@patch(
    "sys.argv",
    [
        "runner.py",
        "--run-id",
        "   ",
        "--execution-date",
        "2026-08-27",
    ],
)
def test_main_cli_empty_run_id() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


@patch(
    "sys.argv",
    [
        "runner.py",
        "--run-id",
        "cli_test_02",
        "--execution-date",
        "27-08-2026",
    ],
)
def test_main_cli_invalid_date_format() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


@patch(
    "sys.argv",
    [
        "runner.py",
        "--run-id",
        "cli_test_03",
        "--execution-date",
        "2026-08-27",
    ],
)
@patch("pipelines.monitoring_pipeline.src.runner.MonitoringPipelineRunner")
def test_main_cli_runner_exception(
    mock_runner_class: MagicMock,
) -> None:
    mock_runner_class.return_value.run.side_effect = CustomException(
        Exception("Pipeline Failed"),
        sys,
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1