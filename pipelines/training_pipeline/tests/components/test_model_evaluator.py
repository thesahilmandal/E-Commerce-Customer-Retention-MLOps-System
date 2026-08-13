import os
import json
import joblib
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.dummy import DummyClassifier

from pipelines.training_pipeline.src.components.model_evaluator import ModelEvaluator
from pipelines.training_pipeline.src.entity.config_entity import ModelEvaluatorConfig
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def evaluator_config(pipeline_context):
    return ModelEvaluatorConfig.from_context(pipeline_context)


def test_model_evaluator_initialization(
    pipeline_context,
    evaluator_config,
    data_processor_artifact,
    model_trainer_artifact,
):
    evaluator = ModelEvaluator(
        config=evaluator_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
    )

    assert evaluator.config == evaluator_config
    assert evaluator.context == pipeline_context
    assert evaluator.data_artifact == data_processor_artifact
    assert evaluator.trainer_artifact == model_trainer_artifact


def test_calculate_eroi_and_threshold(
    pipeline_context,
    evaluator_config,
    data_processor_artifact,
    model_trainer_artifact,
):
    evaluator = ModelEvaluator(
        config=evaluator_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
    )

    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_probs = np.array([0.9, 0.8, 0.4, 0.1, 0.6, 0.2])

    # Based on config: cost=10.0, ltv=500.0, save_rate=0.10 (Revenue per save = 50.0)
    # At threshold 0.5:
    # y_pred = [1, 1, 0, 0, 1, 0] -> TP=3, FP=0
    # Interventions = 3. Cost = 30. Revenue = 150. ROI = 120. Normalized = 120 / 6 = 20.0

    best_eroi, best_threshold = evaluator._calculate_eroi_and_threshold(
        y_true, y_probs
    )

    assert best_eroi > 0
    assert 0.01 <= best_threshold <= 0.99
    assert np.isclose(best_eroi, 20.0)


def test_evaluate_model(
    pipeline_context,
    evaluator_config,
    data_processor_artifact,
    model_trainer_artifact,
):
    evaluator = ModelEvaluator(
        config=evaluator_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
    )

    X_test = pd.DataFrame({"feat1": [1, 2, 3, 4]})
    y_test = np.array([0, 1, 0, 1])

    mock_model = MagicMock()
    # Predicts perfect probabilities
    mock_model.predict_proba.return_value = np.array(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
    )

    metrics = evaluator._evaluate_model(mock_model, X_test, y_test)

    assert "log_loss" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "eroi" in metrics
    assert "optimal_threshold" in metrics

    assert metrics["roc_auc"] == 1.0


def test_fetch_champion_model_success(
    pipeline_context,
    evaluator_config,
    data_processor_artifact,
    model_trainer_artifact,
):
    evaluator = ModelEvaluator(
        config=evaluator_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
    )

    # Create a dummy trained model
    dummy_model = DummyClassifier(strategy="prior")
    dummy_model.fit(np.array([[0], [1]]), np.array([0, 1]))

    def mock_download(remote_uri, local_path):
        if remote_uri == evaluator_config.s3_pointer_uri:
            with open(local_path, "w") as f:
                json.dump(
                    {"s3_model_path": "s3://bucket/champion.pkl"},
                    f,
                )
        elif remote_uri == "s3://bucket/champion.pkl":
            joblib.dump(dummy_model, local_path)
        else:
            raise Exception("Unexpected URI")

    evaluator.context.s3_sync.download_file.side_effect = mock_download

    champion = evaluator._fetch_champion_model()

    assert champion is not None
    assert hasattr(champion, "predict_proba")

    # Ensure temporary files were cleaned up
    local_state_path = os.path.join(
        evaluator.config.model_evaluator_dir,
        "tmp_model_state.json",
    )
    local_champion_path = os.path.join(
        evaluator.config.model_evaluator_dir,
        "champion_model.pkl",
    )
    assert not os.path.exists(local_state_path)
    assert not os.path.exists(local_champion_path)


def test_fetch_champion_model_cold_start(
    pipeline_context,
    evaluator_config,
    data_processor_artifact,
    model_trainer_artifact,
):
    evaluator = ModelEvaluator(
        config=evaluator_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
    )

    # Simulate S3 file not found
    evaluator.context.s3_sync.download_file.side_effect = Exception(
        "Not Found"
    )

    champion = evaluator._fetch_champion_model()
    assert champion is None


def test_execute_hysteresis_duel_cold_start(
    pipeline_context,
    evaluator_config,
    data_processor_artifact,
    model_trainer_artifact,
):
    evaluator = ModelEvaluator(
        config=evaluator_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
    )

    # min_eroi_threshold is 0.05 from fixture

    # Approved
    is_approved = evaluator._execute_hysteresis_duel(
        {"eroi": 0.10}, None
    )
    assert is_approved is True

    # Rejected
    is_rejected = evaluator._execute_hysteresis_duel(
        {"eroi": 0.01}, None
    )
    assert is_rejected is False


def test_execute_hysteresis_duel_champion_exists(
    pipeline_context,
    evaluator_config,
    data_processor_artifact,
    model_trainer_artifact,
):
    evaluator = ModelEvaluator(
        config=evaluator_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
    )

    # margin is 0.02 from fixture
    champion_metrics = {"eroi": 0.50}

    # Approved: Challenger beats Champion + Margin (0.50 + 0.02 = 0.52)
    is_approved = evaluator._execute_hysteresis_duel(
        {"eroi": 0.55}, champion_metrics
    )
    assert is_approved is True

    # Rejected: Challenger is better, but doesn't beat the margin
    is_rejected_margin = evaluator._execute_hysteresis_duel(
        {"eroi": 0.51}, champion_metrics
    )
    assert is_rejected_margin is False

    # Rejected: Challenger is worse
    is_rejected_worse = evaluator._execute_hysteresis_duel(
        {"eroi": 0.40}, champion_metrics
    )
    assert is_rejected_worse is False


def test_model_evaluator_run_success(
    pipeline_context,
    evaluator_config,
    data_processor_artifact,
    model_trainer_artifact,
):
    evaluator = ModelEvaluator(
        config=evaluator_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
    )

    # Force _fetch_champion_model to return None (Cold Start)
    with patch.object(
        evaluator,
        "_fetch_champion_model",
        return_value=None,
    ):
        artifact = evaluator.run()

    assert artifact.approval_status is not None
    assert artifact.report_file_path == evaluator_config.report_file_path

    assert os.path.exists(artifact.report_file_path)
    assert os.path.exists(artifact.metadata_file_path)

    if artifact.approval_status:
        assert os.path.exists(
            artifact.baseline_performance_metrics_file_path
        )

        with open(
            artifact.baseline_performance_metrics_file_path, "r"
        ) as f:
            baselines = json.load(f)

        assert "metrics" in baselines
        assert "log_loss" in baselines["metrics"]


def test_model_evaluator_run_failure(
    pipeline_context,
    evaluator_config,
    data_processor_artifact,
    model_trainer_artifact,
):
    evaluator = ModelEvaluator(
        config=evaluator_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
        trainer_artifact=model_trainer_artifact,
    )

    # Simulate a critical failure during test data loading
    with patch.object(
        evaluator,
        "_load_test_data",
        side_effect=ValueError("Corrupt Parquet File"),
    ):
        with pytest.raises(CustomException) as excinfo:
            evaluator.run()

        assert "Corrupt Parquet File" in str(excinfo.value)