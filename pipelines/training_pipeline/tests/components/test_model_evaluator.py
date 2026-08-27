import json
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from pipelines.training_pipeline.src.components.model_evaluator import (
    ModelEvaluator,
)
from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataProcessorArtifact,
    ModelTrainerArtifact,
)
from pipelines.training_pipeline.src.entity.config_entity import (
    ModelEvaluatorConfig,
)


@pytest.fixture
def me_config(
    mock_pipeline_context: MagicMock,
) -> ModelEvaluatorConfig:
    return ModelEvaluatorConfig.from_context(mock_pipeline_context)


@patch(
    "pipelines.training_pipeline.src.components.model_evaluator.joblib.load"
)
def test_run_success(
    mock_joblib_load: MagicMock,
    me_config: ModelEvaluatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
) -> None:
    evaluator = ModelEvaluator(
        config=me_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
    )

    mock_challenger = MagicMock()
    mock_joblib_load.return_value = mock_challenger

    mock_metrics = {
        "log_loss": 0.3,
        "roc_auc": 0.85,
        "brier_score": 0.1,
        "eroi": 0.08,
        "optimal_threshold": 0.5,
    }

    with (
        patch.object(
            evaluator,
            "_load_test_data",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch.object(
            evaluator,
            "_evaluate_model",
            return_value=mock_metrics,
        ),
        patch.object(
            evaluator,
            "_fetch_champion_model",
            return_value=None,
        ),
        patch.object(
            evaluator,
            "_execute_hysteresis_duel",
            return_value=True,
        ),
        patch.object(evaluator, "_generate_artifacts"),
        patch.object(evaluator, "_generate_metadata"),
    ):
        artifact = evaluator.run()

        assert artifact.approval_status is True
        assert artifact.report_file_path == me_config.report_file_path
        assert (
            artifact.baseline_performance_metrics_file_path
            == me_config.baseline_performance_metrics_file_path
        )

        mock_joblib_load.assert_called_once_with(
            dummy_model_trainer_artifact.model_file_path
        )


@patch(
    "pipelines.training_pipeline.src.components.model_evaluator.pd.read_parquet"
)
def test_load_test_data(
    mock_read_parquet: MagicMock,
    me_config: ModelEvaluatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
) -> None:
    evaluator = ModelEvaluator(
        config=me_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
    )

    mock_df = pd.DataFrame({"target": [0, 1]})
    mock_read_parquet.return_value = mock_df

    X_test, y_test = evaluator._load_test_data()

    assert mock_read_parquet.call_count == 2
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, np.ndarray)


def test_evaluate_model(
    me_config: ModelEvaluatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
) -> None:
    evaluator = ModelEvaluator(
        config=me_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
    )

    mock_model = MagicMock()

    # Predicts class 1 probabilities perfectly
    mock_model.predict_proba.return_value = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ]
    )

    X_test = pd.DataFrame({"feature": [1, 2]})
    y_test = np.array([0, 1])

    with patch.object(
        evaluator,
        "_calculate_eroi_and_threshold",
        return_value=(10.5, 0.45),
    ):
        metrics = evaluator._evaluate_model(
            mock_model,
            X_test,
            y_test,
        )

    assert "log_loss" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert metrics["roc_auc"] == 1.0
    assert metrics["eroi"] == 10.5
    assert metrics["optimal_threshold"] == 0.45


def test_calculate_eroi_and_threshold(
    me_config: ModelEvaluatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
) -> None:
    evaluator = ModelEvaluator(
        config=me_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
    )

    # 4 users. LTV=500, save=0.1 -> Rev per save = 50. Cost = 10.
    y_true = np.array([1, 1, 0, 0])
    y_probs = np.array([0.9, 0.4, 0.8, 0.1])

    eroi, threshold = evaluator._calculate_eroi_and_threshold(
        y_true,
        y_probs,
    )

    assert isinstance(eroi, float)
    assert isinstance(threshold, float)

    # Ensure threshold is selected correctly and EROI is calculated
    assert 0.01 <= threshold <= 0.99
    assert eroi > -100.0


@patch(
    "pipelines.training_pipeline.src.components.model_evaluator.os.remove"
)
@patch(
    "pipelines.training_pipeline.src.components.model_evaluator.joblib.load"
)
def test_fetch_champion_model_success(
    mock_joblib_load: MagicMock,
    mock_remove: MagicMock,
    me_config: ModelEvaluatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
) -> None:
    evaluator = ModelEvaluator(
        config=me_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
    )

    mock_state = {
        "s3_model_path": "s3://test/champion.pkl",
    }

    mock_champion = MagicMock()
    mock_joblib_load.return_value = mock_champion

    with patch(
        "builtins.open",
        mock_open(read_data=json.dumps(mock_state)),
    ):
        model = evaluator._fetch_champion_model()

    assert model == mock_champion
    assert (
        mock_pipeline_context.s3_sync.download_file.call_count == 2
    )
    assert mock_joblib_load.called
    assert mock_remove.call_count == 2


def test_fetch_champion_model_cold_start(
    me_config: ModelEvaluatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
) -> None:
    evaluator = ModelEvaluator(
        config=me_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
    )

    # Simulate pointer file not existing (Cold Start)
    mock_pipeline_context.s3_sync.download_file.side_effect = Exception(
        "Not found"
    )

    model = evaluator._fetch_champion_model()

    assert model is None


@pytest.mark.parametrize(
    "champion_eroi, challenger_eroi, expected_result",
    [
        (None, 0.06, True),  # Cold Start Success (>= 0.05 min)
        (None, 0.04, False),  # Cold Start Fail
        (0.10, 0.13, True),  # Duel Success (0.13 >= 0.10 + 0.02)
        (0.10, 0.11, False),  # Duel Fail (0.11 < 0.10 + 0.02)
    ],
)
def test_execute_hysteresis_duel(
    champion_eroi: float,
    challenger_eroi: float,
    expected_result: bool,
    me_config: ModelEvaluatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
) -> None:
    evaluator = ModelEvaluator(
        config=me_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
    )

    challenger_metrics = {
        "eroi": challenger_eroi,
    }

    champion_metrics = (
        {"eroi": champion_eroi}
        if champion_eroi is not None
        else None
    )

    result = evaluator._execute_hysteresis_duel(
        challenger_metrics,
        champion_metrics,
    )

    assert result is expected_result


@patch(
    "pipelines.training_pipeline.src.components.model_evaluator.write_json_file"
)
def test_generate_artifacts_approved(
    mock_write_json: MagicMock,
    me_config: ModelEvaluatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
) -> None:
    evaluator = ModelEvaluator(
        config=me_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
    )

    challenger_metrics = {
        "log_loss": 0.1,
        "roc_auc": 0.9,
        "brier_score": 0.1,
        "eroi": 0.2,
        "optimal_threshold": 0.5,
    }

    evaluator._generate_artifacts(
        challenger_metrics,
        None,
        is_approved=True,
    )

    # 1 for report, 1 for baseline metrics
    assert mock_write_json.call_count == 2

    report_args = mock_write_json.call_args_list[0][0]

    assert report_args[0] == me_config.report_file_path
    assert report_args[1]["approval_status"] is True

    baseline_args = mock_write_json.call_args_list[1][0]

    assert (
        baseline_args[0]
        == me_config.baseline_performance_metrics_file_path
    )
    assert baseline_args[1]["metrics"]["log_loss"] == 0.1


@patch(
    "pipelines.training_pipeline.src.components.model_evaluator.write_json_file"
)
def test_generate_artifacts_rejected(
    mock_write_json: MagicMock,
    me_config: ModelEvaluatorConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
    dummy_model_trainer_artifact: ModelTrainerArtifact,
) -> None:
    evaluator = ModelEvaluator(
        config=me_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
        trainer_artifact=dummy_model_trainer_artifact,
    )

    challenger_metrics = {
        "log_loss": 0.2,
        "roc_auc": 0.8,
        "brier_score": 0.2,
        "eroi": 0.1,
        "optimal_threshold": 0.5,
    }

    champion_metrics = {
        "log_loss": 0.1,
        "roc_auc": 0.9,
        "brier_score": 0.1,
        "eroi": 0.3,
        "optimal_threshold": 0.5,
    }

    evaluator._generate_artifacts(
        challenger_metrics,
        champion_metrics,
        is_approved=False,
    )

    assert mock_write_json.call_count == 2

    baseline_args = mock_write_json.call_args_list[1][0]

    # Since Challenger was rejected, baseline should fallback to Champion
    assert baseline_args[1]["metrics"]["log_loss"] == 0.1