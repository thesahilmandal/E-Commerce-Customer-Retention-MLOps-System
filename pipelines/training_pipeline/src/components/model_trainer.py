"""
Model Trainer Module for the Training Pipeline.

This module is responsible for algorithmic optimization and probability calibration.
It utilizes Optuna for hyperparameter tuning, trains an XGBoost model, applies 
Isotonic Regression for probability calibration, extracts SHAP explainability artifacts,
and bundles the model with the pre-fitted Categorical Schema Enforcer into a single 
Scikit-Learn Mega-Pipeline for deployment.
"""

import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib
import optuna
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

from pipelines.training_pipeline.src.core.context import PipelineContext
from pipelines.training_pipeline.src.entity.config_entity import ModelTrainerConfig
from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataProcessorArtifact,
    ModelTrainerArtifact,
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


# Ensure matplotlib does not attempt to open GUI windows in headless environments
plt.switch_backend("Agg")


class ModelTrainer:
    """
    Model Trainer component for the Training Pipeline.

    Responsibilities:
    - Load the typed Data Processor artifacts (Train/Val sets).
    - Execute Optuna hyperparameter optimization using XGBoost.
    - Apply early stopping during tuning to optimize compute efficiency.
    - Calibrate the best model using CalibratedClassifierCV to ensure output 
      probabilities reflect real-world likelihoods (crucial for ROI calculations).
    - Generate global SHAP feature importance artifacts.
    - Compute reference feature distributions for downstream Data Drift monitoring.
    - Bundle the preprocessor and calibrated model into a single deployment pipeline.
    """

    def __init__(
        self,
        config: ModelTrainerConfig,
        context: PipelineContext,
        data_artifact: DataProcessorArtifact,
    ) -> None:
        """
        Initializes the Model Trainer component.
        """
        try:
            self.config = config
            self.context = context
            self.data_artifact = data_artifact

            # Suppress Optuna spam in production logs
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            logging.info("Training Pipeline: Model Trainer component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Model Trainer component.")
            raise CustomException(e, sys) from e

    def run(self) -> ModelTrainerArtifact:
        """
        Executes the model training, tuning, and calibration stage.
        """
        try:
            logging.info("Starting Model Trainer execution.")
            start_time = time.time()

            # 1. Load Datasets
            X_train, y_train, X_val, y_val = self._load_data()

            # 2. Hyperparameter Optimization
            logging.info("Initiating Optuna hyperparameter tuning (%d trials).", self.config.optuna_n_trials)
            best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
            logging.info("Optuna tuning completed. Best Params: %s", best_params)

            # 3. Final Model Training & Calibration
            calibrated_model, uncalibrated_base_model = self._train_and_calibrate(
                best_params, X_train, y_train
            )

            # 4. Construct the Scikit-Learn Mega-Pipeline
            mega_pipeline = self._construct_mega_pipeline(calibrated_model)

            # 5. Save Model Artifact
            joblib.dump(mega_pipeline, self.config.model_file_path)
            logging.info("Mega-Pipeline serialized successfully to: %s", self.config.model_file_path)

            # 6. Generate Explainability & Monitoring Artifacts
            self._generate_shap_artifacts(uncalibrated_base_model, X_train)
            self._generate_reference_distributions(X_train)

            # 7. Generate Telemetry Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(execution_time, best_params)

            artifact = ModelTrainerArtifact(
                model_file_path=self.config.model_file_path,
                shap_summary_file_path=self.config.shap_summary_file_path,
                shap_feature_importance_file_path=self.config.shap_feature_importance_file_path,
                reference_feature_distributions_file_path=self.config.reference_feature_distributions_file_path,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("Model Trainer execution completed successfully.")
            return artifact

        except Exception as e:
            logging.exception("Model Trainer run failed.")
            raise CustomException(e, sys) from e

    def _load_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """Loads training and validation datasets using PyArrow."""
        try:
            logging.debug("Loading feature matrices and target vectors from disk.")
            
            X_train = pd.read_parquet(self.data_artifact.x_train_file_path, engine="pyarrow")
            y_train = pd.read_parquet(self.data_artifact.y_train_file_path, engine="pyarrow").values.ravel()
            
            X_val = pd.read_parquet(self.data_artifact.x_val_file_path, engine="pyarrow")
            y_val = pd.read_parquet(self.data_artifact.y_val_file_path, engine="pyarrow").values.ravel()

            return X_train, y_train, X_val, y_val

        except Exception as e:
            logging.exception("Failed to load training/validation datasets.")
            raise CustomException(e, sys) from e

    def _optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Runs an Optuna study to find the best XGBoost hyperparameters."""
        try:
            search_space = self.config.hyperparameter_search_space

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", search_space["n_estimators"]["min"], search_space["n_estimators"]["max"], step=search_space["n_estimators"]["step"]
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", search_space["learning_rate"]["min"], search_space["learning_rate"]["max"], log=True
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth", search_space["max_depth"]["min"], search_space["max_depth"]["max"]
                    ),
                    "min_child_weight": trial.suggest_int(
                        "min_child_weight", search_space["min_child_weight"]["min"], search_space["min_child_weight"]["max"]
                    ),
                    "subsample": trial.suggest_float(
                        "subsample", search_space["subsample"]["min"], search_space["subsample"]["max"]
                    ),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", search_space["colsample_bytree"]["min"], search_space["colsample_bytree"]["max"]
                    ),
                    "gamma": trial.suggest_float(
                        "gamma", search_space["gamma"]["min"], search_space["gamma"]["max"]
                    ),
                    "enable_categorical": True,
                    "tree_method": "hist",
                    "random_state": self.config.random_state,
                    "n_jobs": -1,
                    # XGBoost >= 2.0 API: early_stopping_rounds must be passed to the constructor
                    "early_stopping_rounds": self.config.early_stopping_rounds,
                }

                model = XGBClassifier(**params)
                
                # Fit the model using the updated API signature
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                preds = model.predict_proba(X_val)[:, 1]
                return float(log_loss(y_val, preds))

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.config.random_state))
            study.optimize(objective, n_trials=self.config.optuna_n_trials)

            best_params = study.best_params
            
            # Update with static parameters for downstream training.
            # Notice we explicitly DO NOT add early_stopping_rounds here, because the final 
            # CalibratedClassifierCV does not natively support passing eval_set to the base estimator.
            best_params.update({
                "enable_categorical": True,
                "tree_method": "hist",
                "random_state": self.config.random_state,
                "n_jobs": -1
            })
            
            return best_params

        except Exception as e:
            logging.exception("Failed to execute Optuna hyperparameter optimization.")
            raise CustomException(e, sys) from e

    def _train_and_calibrate(
        self, best_params: Dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray
    ) -> Tuple[CalibratedClassifierCV, XGBClassifier]:
        """
        Trains the final XGBoost model and applies Isotonic Calibration.
        Returns both the Calibrated model (for deployment) and the Base model (for SHAP).
        """
        try:
            logging.info("Training base XGBoost model with best hyperparameters.")
            base_model = XGBClassifier(**best_params)
            base_model.fit(X_train, y_train)

            logging.info("Applying %s calibration using %d-fold cross-validation.", 
                         self.config.calibration_method, self.config.calibration_cv_folds)
            
            # The CalibratedClassifierCV wraps the base estimator. With cv>1, it trains
            # additional cloned base models on cross-validation folds of the training set.
            calibrated_model = CalibratedClassifierCV(
                estimator=XGBClassifier(**best_params),
                method=self.config.calibration_method,
                cv=self.config.calibration_cv_folds
            )
            calibrated_model.fit(X_train, y_train)

            return calibrated_model, base_model

        except Exception as e:
            logging.exception("Failed to train and calibrate the final model.")
            raise CustomException(e, sys) from e

    def _construct_mega_pipeline(self, calibrated_model: CalibratedClassifierCV) -> Pipeline:
        """
        Combines the pre-fitted CategoricalSchemaEnforcer and the Calibrated 
        XGBoost model into a single, cohesive Scikit-Learn Pipeline.
        """
        try:
            logging.info("Loading pre-fitted CategoricalSchemaEnforcer.")
            preprocessor = joblib.load(self.data_artifact.preprocessor_file_path)

            mega_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", calibrated_model)
            ])
            return mega_pipeline

        except Exception as e:
            logging.exception("Failed to construct the deployment Mega-Pipeline.")
            raise CustomException(e, sys) from e

    def _generate_shap_artifacts(self, base_model: XGBClassifier, X_sample: pd.DataFrame) -> None:
        """
        Extracts SHAP values using the underlying, uncalibrated base model. 
        Calibrated models (ensembles) cannot be easily explained directly by TreeExplainer.
        """
        try:
            logging.info("Generating SHAP feature importance artifacts.")
            
            # Downsample for SHAP computation efficiency if the dataset is massive
            if len(X_sample) > 10000:
                X_sample = X_sample.sample(10000, random_state=self.config.random_state)
            
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_sample)

            # 1. Generate and save the Visual Summary Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(self.config.shap_summary_file_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Extract Mean Absolute SHAP values as a JSON dictionary
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importances = dict(zip(X_sample.columns, mean_abs_shap.tolist()))
            
            # Sort by importance (descending)
            sorted_importances = {
                k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
            }
            
            write_json_file(self.config.shap_feature_importance_file_path, sorted_importances)
            logging.info("SHAP artifacts generated and saved.")

        except Exception as e:
            logging.exception("Failed to generate SHAP artifacts.")
            raise CustomException(e, sys) from e

    def _generate_reference_distributions(self, X_train: pd.DataFrame) -> None:
        """
        Calculates the statistical distributions of features in the training set.
        This strict JSON contract acts as the baseline for downstream Data Drift detection.
        """
        try:
            logging.info("Calculating reference feature distributions for Monitoring Pipeline.")
            
            distributions: Dict[str, Any] = {}
            for col in X_train.columns:
                series = X_train[col]
                if pd.api.types.is_numeric_dtype(series):
                    distributions[col] = {
                        "type": "numerical",
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "missing_rate": float(series.isnull().mean())
                    }
                else:
                    # Categorical/Object
                    counts = series.value_counts(normalize=True).to_dict()
                    # Convert keys to string for JSON serialization
                    distributions[col] = {
                        "type": "categorical",
                        "frequencies": {str(k): float(v) for k, v in counts.items()},
                        "missing_rate": float(series.isnull().mean())
                    }

            write_json_file(self.config.reference_feature_distributions_file_path, distributions)

        except Exception as e:
            logging.exception("Failed to generate reference feature distributions.")
            raise CustomException(e, sys) from e

    def _generate_metadata(self, execution_time: float, best_params: Dict[str, Any]) -> None:
        """Generates observability telemetry for the Model Trainer stage."""
        try:
            metadata = {
                "pipeline_stage": "Model Training & Calibration",
                "execution_time_seconds": execution_time,
                "hyperparameters_selected": best_params,
                "calibration_method": self.config.calibration_method,
                "optuna_trials_executed": self.config.optuna_n_trials,
                "experiment_name": self.config.mlflow_experiment_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(self.config.metadata_file_path, metadata)
            logging.info("Model Trainer metadata generated successfully.")

        except Exception as e:
            logging.exception("Failed to generate model trainer metadata.")
            raise CustomException(e, sys) from e