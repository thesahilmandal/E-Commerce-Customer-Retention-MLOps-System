import os
import pytest

from typing import Dict, Any
from unittest.mock import MagicMock, patch

from pipelines.monitoring_pipeline.src.components.rule_engine import RuleEngine
from pipelines.monitoring_pipeline.src.entity.config_entity import RuleEngineConfig
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def engine_config(tmp_path: pytest.TempPathFactory) -> RuleEngineConfig:
    """Provides a mocked configuration for the Rule Engine."""
    return RuleEngineConfig(
        rule_engine_root_dir=str(tmp_path),
        monitoring_report_file_path=os.path.join(
            tmp_path,
            "monitoring_report.json",
        ),
        need_update_file_path=os.path.join(
            tmp_path,
            "need_update.json",
        ),
        metadata_file_path=os.path.join(
            tmp_path,
            "meta.json",
        ),
        prediction_drift_threshold_psi=0.20,
        feature_drift_threshold_psi=0.20,
        min_drifted_features_for_retrain=2,
        brier_degradation_threshold_factor=1.05,
    )


@pytest.fixture
def engine(
    engine_config: RuleEngineConfig,
    mock_pipeline_context: MagicMock,
    dummy_resolver_artifact: MagicMock,
    dummy_drift_artifact: MagicMock,
    dummy_performance_artifact: MagicMock,
) -> RuleEngine:
    """Yields a fully initialized RuleEngine."""
    return RuleEngine(
        config=engine_config,
        context=mock_pipeline_context,
        resolver_artifact=dummy_resolver_artifact,
        drift_artifact=dummy_drift_artifact,
        performance_artifact=dummy_performance_artifact,
    )


@patch.object(RuleEngine, "_load_json")
@patch.object(RuleEngine, "_evaluate_prediction_drift")
@patch.object(RuleEngine, "_evaluate_feature_drift")
@patch.object(RuleEngine, "_evaluate_performance_degradation")
@patch.object(RuleEngine, "_save_artifacts")
def test_run_success_no_retrain(
    mock_save: MagicMock,
    mock_eval_perf: MagicMock,
    mock_eval_feat: MagicMock,
    mock_eval_pred: MagicMock,
    mock_load_json: MagicMock,
    engine: RuleEngine,
) -> None:
    mock_load_json.side_effect = [{}, {}, {}]
    mock_eval_pred.return_value = (False, {})
    mock_eval_feat.return_value = (False, {})
    mock_eval_perf.return_value = (False, {})

    artifact = engine.run()

    assert not artifact.need_update
    assert (
        artifact.monitoring_report_file_path
        == engine.config.monitoring_report_file_path
    )

    mock_save.assert_called_once()

    args, _ = mock_save.call_args
    assert args[1] is False
    assert args[2] == "NONE"


@patch.object(RuleEngine, "_load_json")
@patch.object(RuleEngine, "_evaluate_prediction_drift")
@patch.object(RuleEngine, "_evaluate_feature_drift")
@patch.object(RuleEngine, "_evaluate_performance_degradation")
@patch.object(RuleEngine, "_save_artifacts")
def test_run_success_with_retrain(
    mock_save: MagicMock,
    mock_eval_perf: MagicMock,
    mock_eval_feat: MagicMock,
    mock_eval_pred: MagicMock,
    mock_load_json: MagicMock,
    engine: RuleEngine,
) -> None:
    mock_load_json.side_effect = [{}, {}, {}]
    mock_eval_pred.return_value = (True, {"detail": "pred"})
    mock_eval_feat.return_value = (False, {"detail": "feat"})
    mock_eval_perf.return_value = (True, {"detail": "perf"})

    artifact = engine.run()

    assert artifact.need_update is True

    mock_save.assert_called_once()

    args, _ = mock_save.call_args
    assert args[1] is True
    assert "CRITICAL_PREDICTION_DRIFT" in args[2]
    assert "SEVERE_PERFORMANCE_DEGRADATION" in args[2]
    assert "CRITICAL_FEATURE_DRIFT" not in args[2]


@pytest.mark.parametrize(
    "psi_score, expected_trigger, expected_severity",
    [
        (0.15, False, "INFO"),
        (0.20, True, "CRITICAL"),
        (0.25, True, "CRITICAL"),
    ],
)
def test_evaluate_prediction_drift(
    psi_score: float,
    expected_trigger: bool,
    expected_severity: str,
    engine: RuleEngine,
) -> None:
    drift_report = {
        "prediction_drift": {
            engine.prediction_col: {
                "psi_score": psi_score,
            }
        }
    }

    is_triggered, details = engine._evaluate_prediction_drift(
        drift_report
    )

    assert is_triggered == expected_trigger
    assert details["severity_level"] == expected_severity
    assert details["actual_value"] == psi_score


def test_evaluate_prediction_drift_missing_col(
    engine: RuleEngine,
) -> None:
    drift_report: Dict[str, Any] = {
        "prediction_drift": {},
    }

    is_triggered, details = engine._evaluate_prediction_drift(
        drift_report
    )

    assert not is_triggered
    assert details["actual_value"] == 0.0


@pytest.mark.parametrize(
    "f1_psi, f2_psi, expected_trigger",
    [
        (0.10, 0.15, False),
        (0.25, 0.15, False),
        (0.25, 0.25, True),
    ],
)
def test_evaluate_feature_drift(
    f1_psi: float,
    f2_psi: float,
    expected_trigger: bool,
    engine: RuleEngine,
) -> None:
    drift_report = {
        "feature_drift": {
            "f1": {
                "psi_score": f1_psi,
            },
            "f2": {
                "psi_score": f2_psi,
            },
        }
    }

    is_triggered, details = engine._evaluate_feature_drift(
        drift_report
    )

    assert is_triggered == expected_trigger

    if expected_trigger:
        assert details["actual_value"] == 2
        assert "f1" in details["drifted_features_list"]
        assert "f2" in details["drifted_features_list"]


def test_evaluate_performance_degradation_immature(
    engine: RuleEngine,
) -> None:
    perf_report = {
        "is_evaluated": False,
        "reason": "INSUFFICIENT_LOOKBACK_MATURITY",
    }

    is_triggered, details = engine._evaluate_performance_degradation(
        perf_report,
        {},
    )

    assert not is_triggered
    assert details["reason"] == "INSUFFICIENT_LOOKBACK_MATURITY"


@pytest.mark.parametrize(
    "baseline_dict",
    [
        {"global_metrics": {"brier_score": 0.10}},
        {"test_metrics": {"brier_score": 0.10}},
        {"brier_score": 0.10},
    ],
)
def test_evaluate_performance_degradation_schemas_and_trigger(
    baseline_dict: Dict[str, Any],
    engine: RuleEngine,
) -> None:
    perf_report = {
        "is_evaluated": True,
        "metrics": {
            "brier_score": 0.11,
        },
    }

    is_triggered, details = engine._evaluate_performance_degradation(
        perf_report,
        baseline_dict,
    )

    assert is_triggered is True
    assert details["actual_value"] == 0.11
    assert details["baseline_value"] == 0.10
    assert details["degradation_boundary"] == pytest.approx(0.105)


def test_evaluate_performance_degradation_not_triggered(
    engine: RuleEngine,
) -> None:
    perf_report = {
        "is_evaluated": True,
        "metrics": {
            "brier_score": 0.10,
        },
    }

    baseline_dict = {
        "brier_score": 0.10,
    }

    is_triggered, details = engine._evaluate_performance_degradation(
        perf_report,
        baseline_dict,
    )

    assert is_triggered is False


def test_evaluate_performance_degradation_missing_brier(
    engine: RuleEngine,
) -> None:
    perf_report = {
        "is_evaluated": True,
        "metrics": {},
    }

    with pytest.raises(CustomException) as exc_info:
        engine._evaluate_performance_degradation(
            perf_report,
            {"brier_score": 0.10},
        )

    assert "Brier Score missing" in str(exc_info.value)


@pytest.mark.parametrize(
    "c1, c2, c3, expected",
    [
        (
            False,
            False,
            False,
            "NONE",
        ),
        (
            True,
            False,
            False,
            "CRITICAL_PREDICTION_DRIFT",
        ),
        (
            False,
            True,
            False,
            "CRITICAL_FEATURE_DRIFT",
        ),
        (
            False,
            False,
            True,
            "SEVERE_PERFORMANCE_DEGRADATION",
        ),
        (
            True,
            True,
            True,
            "CRITICAL_PREDICTION_DRIFT | "
            "CRITICAL_FEATURE_DRIFT | "
            "SEVERE_PERFORMANCE_DEGRADATION",
        ),
    ],
)
def test_formulate_trigger_reasons(
    c1: bool,
    c2: bool,
    c3: bool,
    expected: str,
    engine: RuleEngine,
) -> None:
    result = engine._formulate_trigger_reasons(
        c1,
        c2,
        c3,
    )

    assert result == expected


@patch(
    "pipelines.monitoring_pipeline.src.components.rule_engine.write_json_file"
)
def test_save_artifacts(
    mock_write_json: MagicMock,
    engine: RuleEngine,
) -> None:
    audit_report = {
        "decision": "mock",
    }

    trigger_reasons = "CRITICAL_PREDICTION_DRIFT"
    need_update = True

    engine._save_artifacts(
        audit_report,
        need_update,
        trigger_reasons,
        execution_time=1.0,
    )

    assert mock_write_json.call_count == 3

    mock_write_json.assert_any_call(
        file_path=engine.config.monitoring_report_file_path,
        content=audit_report,
    )

    payload_call = mock_write_json.call_args_list[1][1]

    assert (
        payload_call["file_path"]
        == engine.config.need_update_file_path
    )

    assert payload_call["content"]["need_update"] is True
    assert (
        payload_call["content"]["trigger_reason"]
        == trigger_reasons
    )

    assert (
        payload_call["content"]["champion_run_id"]
        == engine.resolver_artifact.champion_run_id
    )

    meta_call = mock_write_json.call_args_list[2][1]

    assert (
        meta_call["file_path"]
        == engine.config.metadata_file_path
    )

    assert meta_call["content"]["decision_rendered"] is True