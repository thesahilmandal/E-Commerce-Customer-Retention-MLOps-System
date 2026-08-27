import os

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from pipelines.monitoring_pipeline.src.components.statistical_drift_calculator import (
    StatisticalDriftCalculator,
)
from pipelines.monitoring_pipeline.src.entity.config_entity import (
    StatisticalDriftCalculatorConfig,
)
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def drift_config(
    tmp_path: pytest.TempPathFactory,
) -> StatisticalDriftCalculatorConfig:
    return StatisticalDriftCalculatorConfig(
        drift_root_dir=str(tmp_path),
        drift_report_file_path=os.path.join(
            tmp_path,
            "drift_report.json",
        ),
        metadata_file_path=os.path.join(
            tmp_path,
            "metadata.json",
        ),
        top_shap_features_count=2,
        zero_bin_epsilon_psi=0.0001,
    )


@pytest.fixture
def drift_calc(
    drift_config: StatisticalDriftCalculatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_resolver_artifact: MagicMock,
) -> StatisticalDriftCalculator:
    return StatisticalDriftCalculator(
        config=drift_config,
        context=mock_pipeline_context,
        resolver_artifact=dummy_resolver_artifact,
    )


@patch.object(StatisticalDriftCalculator, "_load_current_telemetry")
@patch.object(StatisticalDriftCalculator, "_load_json")
@patch.object(StatisticalDriftCalculator, "_extract_top_shap_features")
@patch.object(StatisticalDriftCalculator, "_calculate_all_drifts")
@patch.object(StatisticalDriftCalculator, "_save_reports")
def test_run_success(
    mock_save_reports: MagicMock,
    mock_calc_drifts: MagicMock,
    mock_extract_shap: MagicMock,
    mock_load_json: MagicMock,
    mock_load_telemetry: MagicMock,
    drift_calc: StatisticalDriftCalculator,
) -> None:
    mock_load_telemetry.return_value = pd.DataFrame(
        {"dummy": [1, 2]}
    )

    mock_load_json.side_effect = [
        {"feature_importance": []},
        {"distributions": {"feature_a": {}}},
    ]

    mock_extract_shap.return_value = ["feature_a"]

    mock_calc_drifts.return_value = {
        "prediction_drift": {},
        "feature_drift": {},
    }

    artifact = drift_calc.run()

    assert (
        artifact.drift_report_file_path
        == drift_calc.config.drift_report_file_path
    )
    assert (
        artifact.metadata_file_path
        == drift_calc.config.metadata_file_path
    )

    mock_load_telemetry.assert_called_once()
    assert mock_load_json.call_count == 2
    mock_extract_shap.assert_called_once()
    mock_calc_drifts.assert_called_once()
    mock_save_reports.assert_called_once()


def test_run_fallback_flat_distributions(
    drift_calc: StatisticalDriftCalculator,
    sample_telemetry_df: pd.DataFrame,
) -> None:
    with (
        patch.object(
            drift_calc,
            "_load_current_telemetry",
            return_value=sample_telemetry_df,
        ),
        patch.object(
            drift_calc,
            "_load_json",
        ) as mock_load_json,
        patch.object(
            drift_calc,
            "_extract_top_shap_features",
            return_value=[],
        ),
        patch.object(
            drift_calc,
            "_calculate_all_drifts",
            return_value={},
        ) as mock_calc,
        patch.object(
            drift_calc,
            "_save_reports",
        ),
    ):
        mock_load_json.side_effect = [
            {"feature_importance": []},
            {"feature_a": {"physical_type": "numerical"}},
        ]

        drift_calc.run()

        _, kwargs = mock_calc.call_args

        assert kwargs["reference_distributions"] == {
            "feature_a": {
                "physical_type": "numerical",
            }
        }


def test_extract_top_shap_features_success(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    shap_data = {
        "feature_importance": [
            {
                "feature_name": "feat_low",
                "mean_abs_shap_value": 0.1,
            },
            {
                "feature_name": "feat_high",
                "mean_abs_shap_value": 0.9,
            },
            {
                "feature_name": "feat_mid",
                "mean_abs_shap_value": 0.5,
            },
        ]
    }

    top_features = drift_calc._extract_top_shap_features(
        shap_data
    )

    assert len(top_features) == 2
    assert top_features == [
        "feat_high",
        "feat_mid",
    ]


def test_extract_top_shap_features_missing_data(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    with pytest.raises(CustomException) as exc_info:
        drift_calc._extract_top_shap_features({})

    assert (
        "missing the 'feature_importance' array"
        in str(exc_info.value)
    )


def test_calculate_all_drifts_empty_telemetry(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    empty_df = pd.DataFrame()
    top_features = [
        "feature_a",
        "feature_b",
    ]

    report = drift_calc._calculate_all_drifts(
        empty_df,
        {},
        top_features,
    )

    assert (
        report["prediction_drift"][
            drift_calc.prediction_col
        ]["psi_score"]
        == 0.0
    )

    assert (
        report["prediction_drift"][
            drift_calc.prediction_col
        ]["status"]
        == "EMPTY_TELEMETRY"
    )

    for feature in top_features:
        assert (
            report["feature_drift"][feature]["psi_score"]
            == 0.0
        )
        assert (
            report["feature_drift"][feature]["status"]
            == "EMPTY_TELEMETRY"
        )


def test_calculate_all_drifts_missing_columns(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    telemetry_df = pd.DataFrame(
        {"unrelated_col": [1, 2]}
    )

    reference_distributions = {
        drift_calc.prediction_col: {},
        "feature_a": {},
    }

    top_features = ["feature_a"]

    report = drift_calc._calculate_all_drifts(
        telemetry_df,
        reference_distributions,
        top_features,
    )

    assert (
        report["prediction_drift"][
            drift_calc.prediction_col
        ]["status"]
        == "MISSING_DATA"
    )

    assert (
        report["feature_drift"]["feature_a"]["status"]
        == "MISSING_IN_TELEMETRY"
    )


def test_calculate_all_drifts_success(
    drift_calc: StatisticalDriftCalculator,
    sample_telemetry_df: pd.DataFrame,
) -> None:
    reference_distributions = {
        drift_calc.prediction_col: {
            "physical_type": "numerical",
            "bin_edges": [0, 1],
            "expected_percentages": [1.0],
        },
        "feature_a": {
            "physical_type": "numerical",
            "bin_edges": [0, 100],
            "expected_percentages": [1.0],
        },
    }

    top_features = [
        "feature_a",
        "feature_missing_in_baseline",
    ]

    sample_telemetry_df[
        "feature_missing_in_baseline"
    ] = [0.0, 0.0, 0.0]

    with patch.object(
        drift_calc,
        "_compute_psi_router",
        return_value=0.15,
    ) as mock_router:
        report = drift_calc._calculate_all_drifts(
            sample_telemetry_df,
            reference_distributions,
            top_features,
        )

        assert (
            report["prediction_drift"][
                drift_calc.prediction_col
            ]["psi_score"]
            == 0.15
        )

        assert (
            report["feature_drift"]["feature_a"]["psi_score"]
            == 0.15
        )

        assert (
            report["feature_drift"][
                "feature_missing_in_baseline"
            ]["status"]
            == "MISSING_IN_BASELINE"
        )

        assert mock_router.call_count == 2


def test_compute_psi_router(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    with (
        patch.object(
            drift_calc,
            "_compute_numerical_psi",
            return_value=1.0,
        ) as mock_num,
        patch.object(
            drift_calc,
            "_compute_categorical_psi",
            return_value=2.0,
        ) as mock_cat,
    ):
        series = pd.Series([1])

        result_categorical = drift_calc._compute_psi_router(
            series,
            {"physical_type": "categorical"},
        )

        assert result_categorical == 2.0
        mock_cat.assert_called_once()

        result_numerical = drift_calc._compute_psi_router(
            series,
            {"physical_type": "numerical"},
        )

        assert result_numerical == 1.0
        mock_num.assert_called_once()


def test_compute_numerical_psi_perfect_match(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    reference_stats = {
        "bin_edges": [
            0.0,
            10.0,
            20.0,
        ],
        "expected_percentages": [
            0.5,
            0.5,
        ],
    }

    current_series = pd.Series(
        [5.0, 15.0]
    )

    psi = drift_calc._compute_numerical_psi(
        current_series,
        reference_stats,
    )

    assert psi == pytest.approx(
        0.0,
        abs=1e-4,
    )


def test_compute_numerical_psi_severe_drift(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    reference_stats = {
        "bin_edges": [
            0.0,
            10.0,
            20.0,
        ],
        "expected_percentages": [
            0.5,
            0.5,
        ],
    }

    current_series = pd.Series(
        [5.0, 5.0]
    )

    psi = drift_calc._compute_numerical_psi(
        current_series,
        reference_stats,
    )

    assert psi > 0.1


def test_compute_numerical_psi_edge_cases(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    empty_series = pd.Series(
        dtype=float
    )

    assert (
        drift_calc._compute_numerical_psi(
            empty_series,
            {
                "bin_edges": [0, 1],
                "expected_percentages": [1.0],
            },
        )
        == 0.0
    )

    invalid_stats = {
        "bin_edges": [],
    }

    assert (
        drift_calc._compute_numerical_psi(
            pd.Series([1.0]),
            invalid_stats,
        )
        == 0.0
    )


def test_compute_categorical_psi_perfect_match(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    reference_stats = {
        "categories": [
            "A",
            "B",
        ],
        "expected_percentages": [
            0.5,
            0.5,
        ],
    }

    current_series = pd.Series(
        ["A", "B"]
    )

    psi = drift_calc._compute_categorical_psi(
        current_series,
        reference_stats,
    )

    assert psi == pytest.approx(
        0.0,
        abs=1e-4,
    )


def test_compute_categorical_psi_severe_drift(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    reference_stats = {
        "categories": [
            "A",
            "B",
        ],
        "expected_percentages": [
            0.5,
            0.5,
        ],
    }

    current_series = pd.Series(
        ["A", "A", "C"]
    )

    psi = drift_calc._compute_categorical_psi(
        current_series,
        reference_stats,
    )

    assert psi > 0.1


def test_compute_categorical_psi_edge_cases(
    drift_calc: StatisticalDriftCalculator,
) -> None:
    empty_series = pd.Series(
        dtype=str
    )

    assert (
        drift_calc._compute_categorical_psi(
            empty_series,
            {
                "categories": ["A"],
                "expected_percentages": [1.0],
            },
        )
        == 0.0
    )

    invalid_stats = {
        "categories": [],
    }

    assert (
        drift_calc._compute_categorical_psi(
            pd.Series(["A"]),
            invalid_stats,
        )
        == 0.0
    )


@patch(
    "pipelines.monitoring_pipeline.src.components.statistical_drift_calculator.write_json_file"
)
def test_save_reports(
    mock_write_json: MagicMock,
    drift_calc: StatisticalDriftCalculator,
) -> None:
    dummy_report = {
        "test": "data",
    }

    drift_calc._save_reports(
        dummy_report,
        execution_time=1.5,
        population_size=100,
    )

    assert mock_write_json.call_count == 2

    mock_write_json.assert_any_call(
        file_path=drift_calc.config.drift_report_file_path,
        content=dummy_report,
    )

    meta_call_args = mock_write_json.call_args_list[1][1]

    assert (
        meta_call_args["file_path"]
        == drift_calc.config.metadata_file_path
    )

    assert (
        meta_call_args["content"]["pipeline_stage"]
        == "Monitoring Statistical Drift Calculator"
    )

    assert (
        meta_call_args["content"]["volumetrics"][
            "scored_population_size"
        ]
        == 100
    )