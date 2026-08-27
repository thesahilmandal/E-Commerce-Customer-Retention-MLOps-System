import os

import pytest
import numpy as np
import pandas as pd

from unittest.mock import MagicMock, patch

from pipelines.monitoring_pipeline.src.components.performance_evaluator import (
    PerformanceEvaluator,
)
from pipelines.monitoring_pipeline.src.entity.config_entity import (
    PerformanceEvaluatorConfig,
)


@pytest.fixture
def eval_config(tmp_path: pytest.TempPathFactory) -> PerformanceEvaluatorConfig:
    """Provides a mocked configuration for the Performance Evaluator."""
    return PerformanceEvaluatorConfig(
        evaluator_root_dir=str(tmp_path),
        performance_report_file_path=os.path.join(
            tmp_path,
            "perf_report.json",
        ),
        metadata_file_path=os.path.join(
            tmp_path,
            "meta.json",
        ),
        campaign_cost=10.0,
        customer_ltv=150.0,
        intervention_save_rate=0.20,
        log_loss_epsilon=1e-15,
    )


@pytest.fixture
def evaluator(
    eval_config: PerformanceEvaluatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_resolver_artifact: MagicMock,
) -> PerformanceEvaluator:
    """Yields a fully initialized PerformanceEvaluator."""
    return PerformanceEvaluator(
        config=eval_config,
        context=mock_pipeline_context,
        resolver_artifact=dummy_resolver_artifact,
    )


@patch.object(
    PerformanceEvaluator,
    "_load_and_join_lookback_data",
)
@patch.object(
    PerformanceEvaluator,
    "_evaluate_performance",
)
@patch.object(
    PerformanceEvaluator,
    "_save_reports",
)
def test_run_success(
    mock_save_reports: MagicMock,
    mock_evaluate: MagicMock,
    mock_load_join: MagicMock,
    evaluator: PerformanceEvaluator,
) -> None:
    mock_df = pd.DataFrame({"dummy": [1, 2]})

    mock_load_join.return_value = mock_df
    mock_evaluate.return_value = {
        "is_evaluated": True,
        "metrics": {},
    }

    artifact = evaluator.run()

    assert (
        artifact.performance_report_file_path
        == evaluator.config.performance_report_file_path
    )
    assert (
        artifact.metadata_file_path
        == evaluator.config.metadata_file_path
    )

    mock_load_join.assert_called_once()
    mock_evaluate.assert_called_once_with(mock_df)
    mock_save_reports.assert_called_once()


@patch(
    "pipelines.monitoring_pipeline.src.components.performance_evaluator.pd.read_parquet"
)
def test_load_and_join_lookback_data_success(
    mock_read_parquet: MagicMock,
    evaluator: PerformanceEvaluator,
) -> None:
    # Telemetry
    df1 = pd.DataFrame(
        {
            "customer_unique_id": ["C1", "C2", "C3"],
            "predicted_probability": [0.1, 0.9, 0.5],
        }
    )

    # Labels
    df2 = pd.DataFrame(
        {
            "customer_unique_id": ["C2", "C3", "C4"],
            "target_is_churn": [1, 0, 1],
        }
    )

    mock_read_parquet.side_effect = [df1, df2]

    merged_df = evaluator._load_and_join_lookback_data()

    assert len(merged_df) == 2
    assert "C2" in merged_df["customer_unique_id"].values
    assert "C3" in merged_df["customer_unique_id"].values
    assert "predicted_probability" in merged_df.columns
    assert "target_is_churn" in merged_df.columns


@patch(
    "pipelines.monitoring_pipeline.src.components.performance_evaluator.pd.read_parquet"
)
def test_load_and_join_lookback_data_empty(
    mock_read_parquet: MagicMock,
    evaluator: PerformanceEvaluator,
) -> None:
    mock_read_parquet.side_effect = [
        pd.DataFrame(),
        pd.DataFrame({"col": [1]}),
    ]

    merged_df = evaluator._load_and_join_lookback_data()

    assert merged_df.empty


def test_evaluate_performance_immature_system(
    evaluator: PerformanceEvaluator,
) -> None:
    result = evaluator._evaluate_performance(pd.DataFrame())

    assert result["is_evaluated"] is False
    assert result["reason"] == "INSUFFICIENT_LOOKBACK_MATURITY"
    assert result["metrics"]["brier_score"] == 0.0


def test_evaluate_performance_missing_columns(
    evaluator: PerformanceEvaluator,
) -> None:
    df = pd.DataFrame({"wrong_col": [1, 2]})

    with pytest.raises(ValueError) as exc_info:
        evaluator._evaluate_performance(df)

    assert "Missing required evaluation columns" in str(
        exc_info.value
    )


def test_evaluate_performance_all_nans(
    evaluator: PerformanceEvaluator,
) -> None:
    df = pd.DataFrame(
        {
            evaluator.target_col: [np.nan, np.nan],
            evaluator.prediction_col: [np.nan, np.nan],
        }
    )

    result = evaluator._evaluate_performance(df)

    assert result["is_evaluated"] is False
    assert result["reason"] == "ALL_NAN_EVALUATION_COLUMNS"


@patch.object(
    PerformanceEvaluator,
    "_compute_brier_score",
    return_value=0.15,
)
@patch.object(
    PerformanceEvaluator,
    "_compute_log_loss",
    return_value=0.25,
)
@patch.object(
    PerformanceEvaluator,
    "_compute_realized_roi",
    return_value=15.0,
)
def test_evaluate_performance_success(
    mock_roi: MagicMock,
    mock_log_loss: MagicMock,
    mock_brier: MagicMock,
    evaluator: PerformanceEvaluator,
) -> None:
    df = pd.DataFrame(
        {
            evaluator.target_col: [1, 0, 1],
            evaluator.prediction_col: [0.9, 0.1, 0.8],
        }
    )

    result = evaluator._evaluate_performance(df)

    assert result["is_evaluated"] is True
    assert result["metrics"]["brier_score"] == 0.15
    assert result["metrics"]["log_loss"] == 0.25
    assert result["metrics"]["realized_roi"] == 15.0

    mock_brier.assert_called_once()
    mock_log_loss.assert_called_once()
    mock_roi.assert_called_once()


def test_compute_brier_score(
    evaluator: PerformanceEvaluator,
) -> None:
    y_true = np.array([1, 0])
    y_prob = np.array([0.9, 0.1])

    brier = evaluator._compute_brier_score(
        y_true,
        y_prob,
    )

    assert brier == pytest.approx(
        0.01,
        abs=1e-5,
    )


def test_compute_brier_score_exception(
    evaluator: PerformanceEvaluator,
) -> None:
    brier = evaluator._compute_brier_score(
        np.array(["invalid"]),
        np.array(["types"]),
    )

    assert brier == 0.0


def test_compute_log_loss(
    evaluator: PerformanceEvaluator,
) -> None:
    y_true = np.array([1, 0])
    y_prob = np.array([0.9, 0.1])

    log_loss = evaluator._compute_log_loss(
        y_true,
        y_prob,
    )

    assert log_loss == pytest.approx(
        0.10536,
        abs=1e-4,
    )


def test_compute_log_loss_clipping(
    evaluator: PerformanceEvaluator,
) -> None:
    y_true = np.array([1, 0])
    y_prob = np.array([0.0, 1.0])

    log_loss = evaluator._compute_log_loss(
        y_true,
        y_prob,
    )

    assert log_loss > 0.0
    assert not np.isnan(log_loss)
    assert not np.isinf(log_loss)


def test_compute_log_loss_exception(
    evaluator: PerformanceEvaluator,
) -> None:
    loss = evaluator._compute_log_loss(
        np.array(["invalid"]),
        np.array(["types"]),
    )

    assert loss == 0.0


def test_compute_realized_roi(
    evaluator: PerformanceEvaluator,
) -> None:
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.9, 0.1])

    roi = evaluator._compute_realized_roi(
        y_true,
        y_prob,
    )

    assert roi == 10.0


def test_compute_realized_roi_exception(
    evaluator: PerformanceEvaluator,
) -> None:
    roi = evaluator._compute_realized_roi(
        np.array(["invalid"]),
        np.array(["types"]),
    )

    assert roi == 0.0


@patch(
    "pipelines.monitoring_pipeline.src.components.performance_evaluator.write_json_file"
)
def test_save_reports(
    mock_write_json: MagicMock,
    evaluator: PerformanceEvaluator,
) -> None:
    dummy_report = {"is_evaluated": True}

    evaluator._save_reports(
        dummy_report,
        execution_time=2.0,
        population_size=500,
    )

    assert mock_write_json.call_count == 2

    mock_write_json.assert_any_call(
        file_path=evaluator.config.performance_report_file_path,
        content=dummy_report,
    )

    meta_call_args = mock_write_json.call_args_list[1][1]

    assert (
        meta_call_args["file_path"]
        == evaluator.config.metadata_file_path
    )
    assert (
        meta_call_args["content"]["pipeline_stage"]
        == "Monitoring Performance Evaluator"
    )
    assert (
        meta_call_args["content"]["volumetrics"]["matured_cohort_size"]
        == 500
    )
    assert (
        meta_call_args["content"]["financial_parameters"]["campaign_cost"]
        == 10.0
    )