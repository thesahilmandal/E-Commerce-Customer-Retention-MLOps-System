import os
import json
import joblib
import pytest
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from pipelines.training_pipeline.src.components.model_trainer import ModelTrainer
from pipelines.training_pipeline.src.entity.config_entity import ModelTrainerConfig
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def trainer_config(pipeline_context):
    return ModelTrainerConfig.from_context(pipeline_context)


def test_model_trainer_initialization(
    pipeline_context, trainer_config, data_processor_artifact
):
    trainer = ModelTrainer(
        config=trainer_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
    )

    assert trainer.config == trainer_config
    assert trainer.context == pipeline_context
    assert trainer.data_artifact == data_processor_artifact


def test_model_trainer_load_data(
    pipeline_context, trainer_config, data_processor_artifact
):
    trainer = ModelTrainer(
        config=trainer_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
    )

    X_train, y_train, X_val, y_val = trainer._load_data()

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_val, np.ndarray)
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)


def test_model_trainer_train_and_calibrate(
    pipeline_context, trainer_config, data_processor_artifact
):
    trainer = ModelTrainer(
        config=trainer_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
    )

    X_train, y_train, _, _ = trainer._load_data()

    best_params = {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "enable_categorical": True,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }

    calibrated_model, base_model = trainer._train_and_calibrate(
        best_params, X_train, y_train
    )

    assert isinstance(calibrated_model, CalibratedClassifierCV)
    assert isinstance(base_model, XGBClassifier)


def test_model_trainer_construct_mega_pipeline(
    pipeline_context, trainer_config, data_processor_artifact
):
    trainer = ModelTrainer(
        config=trainer_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
    )

    X_train, y_train, _, _ = trainer._load_data()

    dummy_xgb = XGBClassifier(
        enable_categorical=True,
        tree_method="hist"
    )
    dummy_xgb.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(estimator=dummy_xgb, cv=2)
    calibrated.fit(X_train, y_train)

    mega_pipeline = trainer._construct_mega_pipeline(calibrated)

    assert isinstance(mega_pipeline, Pipeline)
    assert "preprocessor" in mega_pipeline.named_steps
    assert "model" in mega_pipeline.named_steps


def test_model_trainer_generate_reference_distributions(
    pipeline_context, trainer_config, data_processor_artifact
):
    trainer = ModelTrainer(
        config=trainer_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
    )

    X_train, _, _, _ = trainer._load_data()

    trainer._generate_reference_distributions(X_train)

    assert os.path.exists(
        trainer_config.reference_feature_distributions_file_path
    )

    with open(
        trainer_config.reference_feature_distributions_file_path, "r"
    ) as f:
        dist_data = json.load(f)

    assert "num_feature_1" in dist_data
    assert dist_data["num_feature_1"]["type"] == "numerical"
    assert "mean" in dist_data["num_feature_1"]

    assert "cat_feature_1" in dist_data
    assert dist_data["cat_feature_1"]["type"] == "categorical"
    assert "frequencies" in dist_data["cat_feature_1"]


def test_model_trainer_run_success(
    pipeline_context, trainer_config, data_processor_artifact
):
    trainer = ModelTrainer(
        config=trainer_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
    )

    artifact = trainer.run()

    assert artifact.model_file_path == trainer_config.model_file_path
    assert (
        artifact.shap_summary_file_path
        == trainer_config.shap_summary_file_path
    )
    assert (
        artifact.shap_feature_importance_file_path
        == trainer_config.shap_feature_importance_file_path
    )
    assert (
        artifact.reference_feature_distributions_file_path
        == trainer_config.reference_feature_distributions_file_path
    )
    assert artifact.metadata_file_path == trainer_config.metadata_file_path

    assert os.path.exists(artifact.model_file_path)
    assert os.path.exists(artifact.shap_summary_file_path)
    assert os.path.exists(artifact.shap_feature_importance_file_path)
    assert os.path.exists(
        artifact.reference_feature_distributions_file_path
    )
    assert os.path.exists(artifact.metadata_file_path)

    loaded_pipeline = joblib.load(artifact.model_file_path)
    assert isinstance(loaded_pipeline, Pipeline)


def test_model_trainer_run_failure(
    pipeline_context, trainer_config, data_processor_artifact
):
    # Corrupt data artifact path to simulate loading failure
    data_processor_artifact = data_processor_artifact.__class__(
        preprocessor_file_path=data_processor_artifact.preprocessor_file_path,
        schema_file_path=data_processor_artifact.schema_file_path,
        metadata_file_path=data_processor_artifact.metadata_file_path,
        x_train_file_path="non_existent_x_train.parquet",
        y_train_file_path=data_processor_artifact.y_train_file_path,
        x_val_file_path=data_processor_artifact.x_val_file_path,
        y_val_file_path=data_processor_artifact.y_val_file_path,
        x_test_file_path=data_processor_artifact.x_test_file_path,
        y_test_file_path=data_processor_artifact.y_test_file_path,
    )

    trainer = ModelTrainer(
        config=trainer_config,
        context=pipeline_context,
        data_artifact=data_processor_artifact,
    )

    with pytest.raises(CustomException) as excinfo:
        trainer.run()

    assert (
        "Failed to load training/validation datasets" in str(excinfo.value)
        or "No such file or directory" in str(excinfo.value)
    )