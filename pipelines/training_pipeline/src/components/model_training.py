"""
Model Training Module for the Training Pipeline.

This module orchestrates the core algorithmic workload. It implements hyperparameter
tuning via Optuna with native XGBoost early stopping to maximize compute efficiency.
It resolves temporal data leakage during Isotonic Regression calibration by enforcing
a TimeSeriesSplit. Finally, it natively tracks all parameters, metrics, and models
via MLflow for complete production auditability and reproducibility.
"""

import os
import sys
import time
import warnings
import platform
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import sklearn
import xgboost
import mlflow
import mlflow.xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


from pipelines.training_pipeline.src import constants
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from pipelines.training_pipeline.src.entity.artifact_entity import (
    FeatureTransformationArtifact,
    ModelTrainingArtifact,
)
from pipelines.training_pipeline.src.entity.config_entity import ModelTrainingConfig
from shared_core.utils.main_utils import write_json_file

warnings.filterwarnings("ignore")


class ModelTraining:
    """
    Model Trainer component for the Training Pipeline.

    Responsibilities:
    - Load transformed datasets and the stateful schema enforcer (preprocessor).
    - Initialize MLflow tracking to centrally log parameters, metrics, and models.
    - Perform cost-aware hyperparameter tuning using Optuna, heavily optimized via 
      XGBoost native early stopping to prevent compute waste.
    - Calibrate the best XGBoost model using Isotonic Regression, strictly enforcing 
      temporal ordering via TimeSeriesSplit to prevent leakage.
    - Assemble a self-contained Scikit-Learn deployment pipeline.
    - Generate global business explainability plots using SHAP.
    - Extract and save JSON contracts for downstream monitoring (PSI and SHAP baseline).
    - Generate observability metadata including data provenance and environment state.
    """

    def __init__(
        self,
        config: ModelTrainingConfig,
        transformation_artifact: FeatureTransformationArtifact,
    ) -> None:
        """
        Initialize the Model Trainer component.
        """
        try:
            self.config = config
            self.transformation_artifact = transformation_artifact

            os.makedirs(self.config.model_trainer_root_dir, exist_ok=True)

            logging.info(
                "Training Pipeline: Model Trainer component initialized."
            )

        except Exception as e:
            logging.exception(
                "Failed to initialize Model Trainer component."
            )
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> ModelTrainingArtifact:
        """
        Execute the model training, calibration, and explainability pipeline.
        """
        try:
            logging.info("Starting Model Training Pipeline.")
            start_time = time.time()

            # 1. Initialize MLflow Experiment Context
            # In a true FAANG environment, tracking_uri is set via environment variable. 
            # We safely configure the local fallback here.
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
            mlflow.set_experiment(self.config.mlflow_experiment_name)

            # Encase the entire training process in a parent MLflow run
            with mlflow.start_run(run_name=f"XGB_CT_Run_{int(time.time())}") as parent_run:
                
                # 2. Load PyArrow Parquet Data and Preprocessor
                X_train, y_train = self._load_data(
                    self.transformation_artifact.x_train_file_path,
                    self.transformation_artifact.y_train_file_path,
                )

                X_val, y_val = self._load_data(
                    self.transformation_artifact.x_val_file_path,
                    self.transformation_artifact.y_val_file_path,
                )

                preprocessor = joblib.load(
                    self.transformation_artifact.preprocessor_file_path
                )

                # Extract data provenance shapes
                train_rows, num_features = X_train.shape
                val_rows, _ = X_val.shape

                # Calculate class imbalance weight dynamically
                imbalance_weight = float(
                    (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                )

                logging.info(
                    "Calculated scale_pos_weight: %.2f",
                    imbalance_weight,
                )

                # 3. Hyperparameter Tuning (Optuna)
                best_params = self._optimize_hyperparameters(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    imbalance_weight=imbalance_weight,
                )
                mlflow.log_params(best_params)

                # 4. Train Base Model & Calibrate (Zero Leakage)
                calibrated_model, base_xgb, final_val_loss = self._train_and_calibrate(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    best_params=best_params,
                    imbalance_weight=imbalance_weight,
                )
                mlflow.log_metric("calibrated_val_log_loss", final_val_loss)

                # 5. Assemble Deployment Pipeline
                mega_pipeline = Pipeline(
                    steps=[
                        ("schema_enforcer", preprocessor),
                        ("model", calibrated_model),
                    ]
                )

                # Log the uncalibrated base estimator for MLflow natively
                mlflow.xgboost.log_model(base_xgb, "base_xgb_estimator")

                # 6. Business Explainability (SHAP) & JSON Baseline Extraction
                self._generate_shap_summary(
                    model=base_xgb,
                    X_val=X_val,
                )
                mlflow.log_artifact(self.config.shap_summary_file_path)
                mlflow.log_artifact(self.config.shap_feature_importance_summary_file_path)

                # 7. Generate Reference Feature Distributions (for downstream PSI check)
                self._generate_reference_feature_distributions(
                    X_train=X_train,
                    calibrated_model=calibrated_model,
                )
                mlflow.log_artifact(self.config.reference_feature_distributions_file_path)

                # 8. Serialize Artifacts
                joblib.dump(
                    mega_pipeline,
                    self.config.model_file_path,
                )

                # 9. Generate Metadata
                execution_time = round(time.time() - start_time, 2)

                self._generate_metadata(
                    best_params=best_params,
                    imbalance_weight=imbalance_weight,
                    execution_time=execution_time,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    num_features=num_features,
                )

                # 10. Package Artifact
                artifact = ModelTrainingArtifact(
                    model_file_path=self.config.model_file_path,
                    shap_summary_file_path=self.config.shap_summary_file_path,
                    metadata_file_path=self.config.metadata_file_path,
                    reference_feature_distributions_file_path=self.config.reference_feature_distributions_file_path,
                    shap_feature_importance_summary_file_path=self.config.shap_feature_importance_summary_file_path,
                )

                logging.info(
                    "Model Training completed successfully: %s",
                    artifact,
                )

                return artifact

        except Exception as e:
            logging.exception("Model Training run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DATA LOADING
    # ==========================================================
    def _load_data(
        self,
        x_path: str,
        y_path: str,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load feature matrices and target vectors from Parquet files
        utilizing the high-performance PyArrow engine to prevent OOM spikes.
        """
        try:
            X = pd.read_parquet(x_path, engine="pyarrow")
            y = pd.read_parquet(y_path, engine="pyarrow")
            
            # The target array must be standard numpy for Scikit-Learn/XGBoost
            y_arr = y.values.ravel().astype(np.int32)
            return X, y_arr

        except Exception as e:
            logging.exception("Failed to load training datasets.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # HYPERPARAMETER TUNING (OPTUNA)
    # ==========================================================
    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        imbalance_weight: float,
    ) -> Dict[str, Any]:
        """
        Run Optuna study with XGBoost pruning AND native early stopping
        to aggressively terminate underperforming trials and save compute resources.
        """
        try:
            logging.info(
                "Starting Optuna hyperparameter optimization with native Early Stopping."
            )

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        100,
                        800,
                        step=100,
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.2,
                        log=True,
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth",
                        3,
                        9,
                    ),
                    "min_child_weight": trial.suggest_int(
                        "min_child_weight",
                        1,
                        10,
                    ),
                    "subsample": trial.suggest_float(
                        "subsample",
                        0.6,
                        1.0,
                    ),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree",
                        0.6,
                        1.0,
                    ),
                    "gamma": trial.suggest_float(
                        "gamma",
                        0.0,
                        5.0,
                    ),
                    "tree_method": "hist",
                    "enable_categorical": True,
                    "scale_pos_weight": imbalance_weight,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "random_state": 42,
                    "n_jobs": -1,
                    # Natively halt tree construction if validation log-loss 
                    # doesn't improve for 30 consecutive rounds.
                    "early_stopping_rounds": 30,
                }

                # Pruning callback handles halting entirely unpromising *trials*
                pruning_callback = (
                    optuna.integration.XGBoostPruningCallback(
                        trial,
                        "validation_0-logloss",
                    )
                )

                model = XGBClassifier(
                    **params,
                    callbacks=[pruning_callback],
                )

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                predictions = model.predict_proba(X_val)[:, 1]

                # Save the exact iteration where early stopping halted
                trial.set_user_attr("optimal_n_estimators", model.best_iteration)

                return log_loss(y_val, predictions)

            study = optuna.create_study(
                direction="minimize",
                study_name="XGBoost_Churn_Tuning",
            )

            study.optimize(
                objective,
                n_trials=30,
            )

            logging.info(
                "Optuna optimization completed. "
                "Best Log Loss: %.4f",
                study.best_value,
            )

            best_params = study.best_params
            optimal_trees = study.best_trial.user_attrs.get("optimal_n_estimators")
            
            if optimal_trees:
                best_params["n_estimators"] = optimal_trees

            return best_params

        except Exception as e:
            logging.exception(
                "Failed during Optuna optimization."
            )
            raise CustomException(e, sys) from e

    # ==========================================================
    # MODEL TRAINING & CALIBRATION (ZERO LEAKAGE)
    # ==========================================================
    def _train_and_calibrate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        best_params: Dict[str, Any],
        imbalance_weight: float,
    ) -> Tuple[CalibratedClassifierCV, XGBClassifier, float]:
        """
        Train the final XGBoost model and apply Isotonic calibration.
        Strictly enforces TimeSeriesSplit to prevent bitemporal leakage.
        """
        try:
            logging.info(
                "Training base XGBoost model with optimal parameters."
            )

            # Initialize base XGBoost model
            base_xgb = XGBClassifier(
                **best_params,
                tree_method="hist",
                enable_categorical=True,
                scale_pos_weight=imbalance_weight,
                objective="binary:logistic",
                random_state=42,
                n_jobs=-1,
                # We do not use early_stopping during the final calibration fit, 
                # as CalibratedClassifierCV manages the internal splits.
            )

            base_xgb.fit(X_train, y_train)

            # Apply Isotonic Regression calibration using TimeSeriesSplit
            # This is critical. Using standard CV shuffles data, leaking future 
            # OOT snapshots into past snapshots, artificially inflating probabilities.
            logging.info(
                "Applying Isotonic Calibration with TimeSeriesSplit to prevent leakage."
            )
            
            tscv = TimeSeriesSplit(n_splits=5)
            
            calibrated_model = CalibratedClassifierCV(
                estimator=base_xgb,
                method="isotonic",
                cv=tscv,
            )

            calibrated_model.fit(X_train, y_train)

            # Calculate final log-loss on the validation holdout set
            val_probs = calibrated_model.predict_proba(X_val)[:, 1]
            final_val_loss = float(log_loss(y_val, val_probs))

            return calibrated_model, base_xgb, final_val_loss

        except Exception as e:
            logging.exception(
                "Failed to train and calibrate model."
            )
            raise CustomException(e, sys) from e

    # ==========================================================
    # BUSINESS EXPLAINABILITY (SHAP)
    # ==========================================================
    def _generate_shap_summary(
        self,
        model: XGBClassifier,
        X_val: pd.DataFrame,
    ) -> None:
        """
        Generate global feature importance explanations and extract the exact
        SHAP JSON schema contract required by the downstream Monitoring Pipeline.
        """
        try:
            logging.info(
                "Generating SHAP feature importance summary and downstream JSON artifact."
            )

            # Use representative validation sample for speed
            sample_size = min(2000, len(X_val))

            X_sample = X_val.sample(
                n=sample_size,
                random_state=42,
            )

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # 1. Generate and save the PNG Summary Plot
            plt.figure(figsize=(10, 8))

            shap.summary_plot(
                shap_values,
                X_sample,
                show=False,
            )

            plt.tight_layout()
            plt.savefig(
                self.config.shap_summary_file_path,
                dpi=300,
            )
            plt.close()

            logging.info("SHAP summary plot saved successfully.")

            # 2. Extract and format data for the JSON Artifact
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = []
            
            for i, col in enumerate(X_sample.columns):
                feature_importance.append({
                    "feature_name": col,
                    "mean_abs_shap_value": float(mean_abs_shap[i])
                })
            
            # Sort descending by importance
            feature_importance.sort(key=lambda x: x["mean_abs_shap_value"], reverse=True)
            
            # Assign ranks
            for rank, item in enumerate(feature_importance, start=1):
                item["rank"] = rank

            # Infer the champion_run_id logically based on directory structure
            run_id = os.path.basename(os.path.dirname(self.config.model_trainer_root_dir))
            
            shap_metadata = {
                "metadata": {
                    "champion_run_id": run_id,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "explainer_type": "TreeExplainer"
                },
                "feature_importance": feature_importance
            }

            write_json_file(
                file_path=self.config.shap_feature_importance_summary_file_path,
                content=shap_metadata
            )
            
            logging.info("SHAP feature importance JSON artifact saved successfully.")

        except Exception as e:
            logging.exception(
                "Failed to generate SHAP summary artifacts."
            )
            raise CustomException(e, sys) from e

    # ==========================================================
    # DOWNSTREAM MONITORING CONTRACTS (PSI)
    # ==========================================================
    def _generate_reference_feature_distributions(
        self,
        X_train: pd.DataFrame,
        calibrated_model: CalibratedClassifierCV,
    ) -> None:
        """
        Generates the reference distribution bins for all numerical and categorical 
        features, as well as the predicted probabilities. This is strictly required 
        by the Monitoring Pipeline to compute Population Stability Index (PSI).
        """
        try:
            logging.info("Generating reference feature distributions for PSI monitoring.")
            
            run_id = os.path.basename(os.path.dirname(self.config.model_trainer_root_dir))
            distributions: Dict[str, Any] = {}

            # Process original training features
            for col in X_train.columns:
                series = X_train[col].dropna()
                
                # Check for numerics (excluding boolean)
                if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                    # Compute unique decile edges
                    edges = np.unique(np.percentile(series, np.arange(0, 101, 10)))
                    
                    if len(edges) > 1:
                        counts, _ = np.histogram(series, bins=edges)
                        percentages = (counts / counts.sum()).tolist()
                        distributions[col] = {
                            "physical_type": "numerical",
                            "bin_edges": edges.tolist(),
                            "expected_percentages": [float(p) for p in percentages]
                        }
                    else:
                        # Fallback for constant features
                        distributions[col] = {
                            "physical_type": "numerical",
                            "bin_edges": [float(edges[0]), float(edges[0]) + 1e-5],
                            "expected_percentages": [1.0]
                        }
                else:
                    # Treat as categorical
                    val_counts = series.value_counts(normalize=True)
                    top_cats = val_counts[val_counts >= 0.01]  # Keep >= 1%
                    other_pct = val_counts[val_counts < 0.01].sum()
                    
                    cats = [str(idx) for idx in top_cats.index.tolist()]
                    pcts = [float(p) for p in top_cats.tolist()]
                    
                    if other_pct > 0:
                        cats.append("OTHER")
                        pcts.append(float(other_pct))
                        
                    distributions[col] = {
                        "physical_type": "categorical",
                        "categories": cats,
                        "expected_percentages": pcts
                    }

            # Process predicted probability output space
            probs = calibrated_model.predict_proba(X_train)[:, 1]
            prob_edges = np.arange(0.0, 1.1, 0.1)
            prob_counts, _ = np.histogram(probs, bins=prob_edges)
            prob_percentages = (prob_counts / prob_counts.sum()).tolist()
            
            distributions["predicted_probability"] = {
                "physical_type": "numerical",
                "bin_edges": prob_edges.tolist(),
                "expected_percentages": [float(p) for p in prob_percentages]
            }

            payload = {
                "metadata": {
                    "champion_run_id": run_id,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat()
                },
                "distributions": distributions
            }

            write_json_file(
                file_path=self.config.reference_feature_distributions_file_path,
                content=payload
            )
            logging.info("Reference feature distributions JSON artifact saved successfully.")

        except Exception as e:
            logging.exception("Failed to generate reference feature distributions.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY METADATA
    # ==========================================================
    def _generate_metadata(
        self,
        best_params: Dict[str, Any],
        imbalance_weight: float,
        execution_time: float,
        train_rows: int,
        val_rows: int,
        num_features: int,
    ) -> None:
        """
        Generate telemetry metadata for downstream auditing,
        reproducibility, and observability.
        """
        try:
            logging.info(
                "Generating Model Trainer observability metadata."
            )

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Model Training & Calibration",
                "execution_time_seconds": execution_time,
                "data_provenance": {
                    "train_rows": train_rows,
                    "val_rows": val_rows,
                    "feature_count": num_features,
                    "train_snapshots_used": (
                        constants.TRAIN_SNAPSHOT
                    ),
                    "val_snapshot_used": (
                        constants.VAL_SNAPSHOT
                    ),
                },
                "environment_state": {
                    "python_version": (
                        platform.python_version()
                    ),
                    "xgboost_version": (
                        xgboost.__version__
                    ),
                    "scikit_learn_version": (
                        sklearn.__version__
                    ),
                },
                "model_architecture": {
                    "base_estimator": (
                        "XGBClassifier "
                        "(Hist Gradient Boosting with Early Stopping)"
                    ),
                    "calibration_method": (
                        "Isotonic Regression"
                    ),
                    "calibration_cv_folds": "5 (TimeSeriesSplit)",
                    "scale_pos_weight": (
                        imbalance_weight
                    ),
                },
                "optimal_hyperparameters": best_params,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            write_json_file(
                file_path=self.config.metadata_file_path,
                content=metadata,
            )

        except Exception as e:
            logging.exception(
                "Failed to generate metadata."
            )
            raise CustomException(e, sys) from e