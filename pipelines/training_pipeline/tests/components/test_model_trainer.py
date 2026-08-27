import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from pipelines.training_pipeline.src.components.model_trainer import ModelTrainer
from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataProcessorArtifact,
)
from pipelines.training_pipeline.src.entity.config_entity import ModelTrainerConfig
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def mt_config(
    mock_pipeline_context: MagicMock,
) -> ModelTrainerConfig:
    return ModelTrainerConfig.from_context(mock_pipeline_context)


@patch("pipelines.training_pipeline.src.components.model_trainer.joblib.dump")
def test_run_success(
    mock_joblib_dump: MagicMock,
    mt_config: ModelTrainerConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
) -> None:
    trainer = ModelTrainer(
        config=mt_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
    )

    with (
        patch.object(
            trainer,
            "_load_data",
            return_value=(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ),
        ),
        patch.object(
            trainer,
            "_optimize_hyperparameters",
            return_value={"mock": "params"},
        ),
        patch.object(
            trainer,
            "_train_and_calibrate",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch.object(
            trainer,
            "_construct_mega_pipeline",
            return_value=MagicMock(),
        ),
        patch.object(trainer, "_generate_shap_artifacts"),
        patch.object(trainer, "_generate_reference_distributions"),
        patch.object(trainer, "_generate_metadata"),
    ):
        artifact = trainer.run()

        assert artifact.model_file_path == mt_config.model_file_path
        assert (
            artifact.shap_summary_file_path
            == mt_config.shap_summary_file_path
        )
        assert artifact.metadata_file_path == mt_config.metadata_file_path

        mock_joblib_dump.assert_called_once()


@patch(
    "pipelines.training_pipeline.src.components.model_trainer.pd.read_parquet"
)
def test_load_data_success(
    mock_read_parquet: MagicMock,
    mt_config: ModelTrainerConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
) -> None:
    trainer = ModelTrainer(
        config=mt_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
    )

    mock_df = pd.DataFrame({"dummy": [1, 2]})
    mock_read_parquet.return_value = mock_df

    X_train, y_train, X_val, y_val = trainer._load_data()

    assert mock_read_parquet.call_count == 4
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, np.ndarray)


@patch(
    "pipelines.training_pipeline.src.components.model_trainer.optuna.create_study"
)
def test_optimize_hyperparameters(
    mock_create_study: MagicMock,
    mt_config: ModelTrainerConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
) -> None:
    trainer = ModelTrainer(
        config=mt_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
    )

    mock_study = MagicMock()
    mock_study.best_params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
    }
    mock_create_study.return_value = mock_study

    params = trainer._optimize_hyperparameters(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    mock_create_study.assert_called_once()
    mock_study.optimize.assert_called_once()

    assert params["n_estimators"] == 100
    assert params["learning_rate"] == 0.05
    assert params["enable_categorical"] is True
    assert params["tree_method"] == "hist"
    assert params["random_state"] == mt_config.random_state
    assert params["n_jobs"] == -1


@patch(
    "pipelines.training_pipeline.src.components.model_trainer.CalibratedClassifierCV"
)
@patch(
    "pipelines.training_pipeline.src.components.model_trainer.XGBClassifier"
)
def test_train_and_calibrate(
    mock_xgb: MagicMock,
    mock_calibrated: MagicMock,
    mt_config: ModelTrainerConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
) -> None:
    trainer = ModelTrainer(
        config=mt_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
    )

    mock_xgb_instance = MagicMock()
    mock_xgb.return_value = mock_xgb_instance

    mock_calibrated_instance = MagicMock()
    mock_calibrated.return_value = mock_calibrated_instance

    X_train = pd.DataFrame()
    y_train = np.array([])

    best_params = {
        "n_estimators": 50,
        "random_state": 42,
    }

    cal_model, base_model = trainer._train_and_calibrate(
        best_params,
        X_train,
        y_train,
    )

    mock_xgb.assert_called_with(**best_params)
    mock_xgb_instance.fit.assert_called_once()
    mock_calibrated_instance.fit.assert_called_once()

    assert cal_model == mock_calibrated_instance
    assert base_model == mock_xgb_instance


@patch(
    "pipelines.training_pipeline.src.components.model_trainer.joblib.load"
)
def test_construct_mega_pipeline(
    mock_joblib_load: MagicMock,
    mt_config: ModelTrainerConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
) -> None:
    trainer = ModelTrainer(
        config=mt_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
    )

    mock_preprocessor = MagicMock()
    mock_joblib_load.return_value = mock_preprocessor

    mock_calibrated_model = MagicMock()

    pipeline = trainer._construct_mega_pipeline(mock_calibrated_model)

    mock_joblib_load.assert_called_once_with(
        dummy_data_processor_artifact.preprocessor_file_path
    )

    assert pipeline.steps[0][0] == "preprocessor"
    assert pipeline.steps[0][1] == mock_preprocessor
    assert pipeline.steps[1][0] == "model"
    assert pipeline.steps[1][1] == mock_calibrated_model


@patch(
    "pipelines.training_pipeline.src.components.model_trainer.write_json_file"
)
@patch("pipelines.training_pipeline.src.components.model_trainer.plt")
@patch("pipelines.training_pipeline.src.components.model_trainer.shap")
def test_generate_shap_artifacts(
    mock_shap: MagicMock,
    mock_plt: MagicMock,
    mock_write_json: MagicMock,
    mt_config: ModelTrainerConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
) -> None:
    trainer = ModelTrainer(
        config=mt_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
    )

    mock_base_model = MagicMock()
    mock_explainer = MagicMock()

    mock_explainer.shap_values.return_value = np.array(
        [
            [0.1, -0.2],
            [-0.1, 0.4],
            [0.0, 0.1],
            [0.2, -0.1],
            [-0.2, 0.2],
        ]
    )

    mock_shap.TreeExplainer.return_value = mock_explainer

    X_sample = pd.DataFrame(
        {
            "feat1": [1] * 5,
            "feat2": [2] * 5,
        }
    )

    trainer._generate_shap_artifacts(
        mock_base_model,
        X_sample,
    )

    mock_plt.figure.assert_called_once()
    mock_shap.summary_plot.assert_called_once()
    mock_plt.savefig.assert_called_once_with(
        mt_config.shap_summary_file_path,
        dpi=300,
        bbox_inches="tight",
    )

    mock_write_json.assert_called_once()

    args = mock_write_json.call_args[0]

    assert args[0] == mt_config.shap_feature_importance_file_path

    importances = args[1]

    assert list(importances.keys())[0] == "feat2"
    assert list(importances.keys())[1] == "feat1"
    assert np.isclose(importances["feat2"], 0.2)
    assert np.isclose(importances["feat1"], 0.12)


@patch(
    "pipelines.training_pipeline.src.components.model_trainer.write_json_file"
)
def test_generate_reference_distributions(
    mock_write_json: MagicMock,
    mt_config: ModelTrainerConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
) -> None:
    trainer = ModelTrainer(
        config=mt_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
    )

    df = pd.DataFrame(
        {
            "num_feat": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat_feat": pd.Series(
                ["A", "B", "A", "A", "C"],
                dtype="category",
            ),
        }
    )

    trainer._generate_reference_distributions(df)

    mock_write_json.assert_called_once()

    args = mock_write_json.call_args[0]

    assert (
        args[0]
        == mt_config.reference_feature_distributions_file_path
    )

    payload = args[1]

    assert payload["num_feat"]["type"] == "numerical"
    assert payload["num_feat"]["mean"] == 3.0
    assert payload["num_feat"]["missing_rate"] == 0.0

    assert payload["cat_feat"]["type"] == "categorical"
    assert payload["cat_feat"]["frequencies"]["A"] == 0.6
    assert payload["cat_feat"]["frequencies"]["B"] == 0.2


def test_run_top_level_exception_handling(
    mt_config: ModelTrainerConfig,
    mock_pipeline_context: MagicMock,
    dummy_data_processor_artifact: DataProcessorArtifact,
) -> None:
    trainer = ModelTrainer(
        config=mt_config,
        context=mock_pipeline_context,
        data_artifact=dummy_data_processor_artifact,
    )

    with patch.object(
        trainer,
        "_load_data",
        side_effect=Exception("Data missing"),
    ):
        with pytest.raises(CustomException) as exc_info:
            trainer.run()

        assert "Data missing" in str(exc_info.value)