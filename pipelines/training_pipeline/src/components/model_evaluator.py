"""
Model Evaluator Module for the Training Pipeline.

This module acts as the automated business gatekeeper. It evaluates the newly 
trained Challenger model against a holdout Test set to calculate both statistical 
and business metrics (Expected Return on Investment - EROI). It then fetches the 
currently deployed Champion model from S3, evaluates it on the exact same Test set, 
and executes a Hysteresis Duel. The Challenger is only approved for deployment if 
it beats the Champion by a strictly defined EROI margin, preventing unnecessary 
model churn.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

from pipelines.training_pipeline.src.core.context import PipelineContext
from pipelines.training_pipeline.src.entity.config_entity import ModelEvaluatorConfig
from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataProcessorArtifact,
    ModelTrainerArtifact,
    ModelEvaluatorArtifact,
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class ModelEvaluator:
    """
    Model Evaluator component for the Training Pipeline.

    Responsibilities:
    - Load the Test set processed by the Data Processor.
    - Evaluate the Challenger model's statistical performance (Log Loss, AUC).
    - Perform Threshold Optimization to maximize Expected ROI (EROI) based on 
      dynamic business assumptions (LTV, Campaign Cost).
    - Fetch the active production Champion from S3 (if one exists).
    - Perform an apples-to-apples evaluation of the Champion on the current Test set.
    - Execute the Hysteresis Duel to determine deployment eligibility.
    - Generate baseline performance JSON contracts for downstream monitoring.
    """

    def __init__(
        self,
        config: ModelEvaluatorConfig,
        context: PipelineContext,
        data_artifact: DataProcessorArtifact,
        trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        """
        Initializes the Model Evaluator component.
        """
        try:
            self.config = config
            self.context = context
            self.data_artifact = data_artifact
            self.trainer_artifact = trainer_artifact

            logging.info("Training Pipeline: Model Evaluator component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Model Evaluator component.")
            raise CustomException(e, sys) from e

    def run(self) -> ModelEvaluatorArtifact:
        """
        Executes the evaluation, comparison, and gatekeeping stage.
        """
        try:
            logging.info("Starting Model Evaluator execution.")
            start_time = time.time()

            # 1. Load Test Data and Challenger Model
            X_test, y_test = self._load_test_data()
            challenger_model = joblib.load(self.trainer_artifact.model_file_path)

            # 2. Evaluate Challenger
            logging.info("Evaluating Challenger Model on Test Set.")
            challenger_metrics = self._evaluate_model(challenger_model, X_test, y_test)
            logging.info("Challenger EROI: $%.2f", challenger_metrics["eroi"])

            # 3. Fetch and Evaluate Champion
            champion_model = self._fetch_champion_model()
            champion_metrics = None

            if champion_model is not None:
                logging.info("Evaluating Champion Model on current Test Set.")
                champion_metrics = self._evaluate_model(champion_model, X_test, y_test)
                logging.info("Champion EROI: $%.2f", champion_metrics["eroi"])
            else:
                logging.warning("No Champion model found. Proceeding with Cold Start Evaluation.")

            # 4. Execute the Gatekeeper Duel
            is_approved = self._execute_hysteresis_duel(challenger_metrics, champion_metrics)

            # 5. Generate Artifacts and Metadata
            self._generate_artifacts(challenger_metrics, champion_metrics, is_approved)
            
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(execution_time, is_approved, challenger_metrics, champion_metrics)

            artifact = ModelEvaluatorArtifact(
                approval_status=is_approved,
                report_file_path=self.config.report_file_path,
                baseline_performance_metrics_file_path=self.config.baseline_performance_metrics_file_path,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("Model Evaluator execution completed successfully. Approval: %s", is_approved)
            return artifact

        except Exception as e:
            logging.exception("Model Evaluator run failed.")
            raise CustomException(e, sys) from e

    def _load_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Loads the strictly typed Test set using PyArrow."""
        try:
            logging.debug("Loading Test feature matrix and target vector.")
            X_test = pd.read_parquet(self.data_artifact.x_test_file_path, engine="pyarrow")
            y_test = pd.read_parquet(self.data_artifact.y_test_file_path, engine="pyarrow").values.ravel()
            return X_test, y_test

        except Exception as e:
            logging.exception("Failed to load Test datasets.")
            raise CustomException(e, sys) from e

    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Computes standard statistical metrics and custom business metrics (EROI) 
        for a given model on the Test set.
        """
        try:
            # Generate probabilities
            y_probs = model.predict_proba(X_test)[:, 1]

            # 1. Statistical Metrics
            logloss_val = log_loss(y_test, y_probs)
            auc_val = roc_auc_score(y_test, y_probs)
            brier_val = brier_score_loss(y_test, y_probs)

            # 2. Business Metrics (Threshold Optimization for Expected ROI)
            eroi_val, optimal_threshold = self._calculate_eroi_and_threshold(y_test, y_probs)

            return {
                "log_loss": float(logloss_val),
                "roc_auc": float(auc_val),
                "brier_score": float(brier_val),
                "eroi": float(eroi_val),
                "optimal_threshold": float(optimal_threshold),
            }

        except Exception as e:
            logging.exception("Failed to evaluate model metrics.")
            raise CustomException(e, sys) from e

    def _calculate_eroi_and_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
        """
        Sweeps through probability thresholds to find the decision boundary that 
        maximizes the Expected Return on Investment (EROI).
        
        EROI = (True Positives * Customer LTV * Save Rate) - ((True Positives + False Positives) * Campaign Cost)
        """
        try:
            best_eroi = -np.inf
            best_threshold = 0.5

            cost = self.config.campaign_cost
            ltv = self.config.customer_ltv
            save_rate = self.config.intervention_save_rate
            revenue_per_save = ltv * save_rate

            # Vectorized threshold sweep (0.01 to 0.99)
            thresholds = np.linspace(0.01, 0.99, 99)
            
            for threshold in thresholds:
                # Classify based on threshold
                y_pred = (y_probs >= threshold).astype(int)
                
                # Calculate Confusion Matrix elements
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                
                # Calculate Financials
                total_interventions = tp + fp
                total_cost = total_interventions * cost
                total_revenue = tp * revenue_per_save
                
                # Calculate average ROI per user in the test set to normalize scale
                roi = total_revenue - total_cost
                normalized_eroi = roi / len(y_true)

                if normalized_eroi > best_eroi:
                    best_eroi = normalized_eroi
                    best_threshold = threshold

            return best_eroi, best_threshold

        except Exception as e:
            logging.exception("Failed to calculate EROI and optimal threshold.")
            raise CustomException(e, sys) from e

    def _fetch_champion_model(self) -> Optional[Any]:
        """
        Attempts to fetch the currently deployed Champion model from S3 using the 
        global pointer file. Returns None if no active model exists.
        """
        try:
            local_state_path = os.path.join(self.config.model_evaluator_dir, "tmp_model_state.json")
            
            # Download pointer file
            try:
                self.context.s3_sync.download_file(self.config.s3_pointer_uri, local_state_path)
            except Exception:
                logging.info("No active model_state.json found at %s. (Cold Start)", self.config.s3_pointer_uri)
                return None

            # Read pointer to get Champion URI
            with open(local_state_path, "r") as f:
                state_data = json.load(f)
            
            s3_model_path = state_data.get("s3_model_path")
            if not s3_model_path:
                logging.warning("model_state.json found but missing s3_model_path. Proceeding as Cold Start.")
                return None

            # Download Champion model
            local_champion_path = os.path.join(self.config.model_evaluator_dir, "champion_model.pkl")
            logging.info("Downloading Champion model from %s", s3_model_path)
            self.context.s3_sync.download_file(s3_model_path, local_champion_path)

            champion_model = joblib.load(local_champion_path)
            
            # Cleanup temp files
            os.remove(local_state_path)
            os.remove(local_champion_path)

            return champion_model

        except Exception as e:
            logging.warning("Non-fatal error fetching Champion model. Falling back to Cold Start. Details: %s", str(e))
            return None

    def _execute_hysteresis_duel(self, challenger_metrics: Dict[str, Any], champion_metrics: Optional[Dict[str, Any]]) -> bool:
        """
        Evaluates whether the Challenger model warrants a production deployment.
        Applies a Hysteresis margin to prevent flipping models over trivial gains.
        """
        try:
            challenger_eroi = challenger_metrics["eroi"]
            min_threshold = self.config.min_eroi_threshold

            # Scenario 1: Cold Start (No Champion)
            if champion_metrics is None:
                if challenger_eroi >= min_threshold:
                    logging.info(
                        "Cold Start Approved. Challenger EROI (%.4f) >= Minimum Threshold (%.4f).", 
                        challenger_eroi, min_threshold
                    )
                    return True
                else:
                    logging.warning(
                        "Cold Start Rejected. Challenger EROI (%.4f) < Minimum Threshold (%.4f).", 
                        challenger_eroi, min_threshold
                    )
                    return False

            # Scenario 2: Active Champion exists (Hysteresis Duel)
            champion_eroi = champion_metrics["eroi"]
            margin = self.config.eroi_hysteresis_margin
            target_eroi = champion_eroi + margin

            if challenger_eroi >= target_eroi:
                logging.info(
                    "Challenger Approved. Challenger EROI (%.4f) >= Champion EROI (%.4f) + Margin (%.4f).",
                    challenger_eroi, champion_eroi, margin
                )
                return True
            else:
                logging.warning(
                    "Challenger Rejected. Challenger EROI (%.4f) failed to clear Hysteresis target (%.4f).",
                    challenger_eroi, target_eroi
                )
                return False

        except Exception as e:
            logging.exception("Failed to execute Hysteresis duel.")
            raise CustomException(e, sys) from e

    def _generate_artifacts(self, challenger_metrics: Dict[str, Any], champion_metrics: Optional[Dict[str, Any]], is_approved: bool) -> None:
        """
        Generates the detailed evaluation report and the baseline monitoring contracts.
        """
        try:
            # 1. Evaluation Report (Full context of the duel)
            report_payload = {
                "approval_status": is_approved,
                "business_assumptions": {
                    "campaign_cost": self.config.campaign_cost,
                    "customer_ltv": self.config.customer_ltv,
                    "intervention_save_rate": self.config.intervention_save_rate,
                    "eroi_hysteresis_margin": self.config.eroi_hysteresis_margin,
                },
                "challenger_metrics": challenger_metrics,
                "champion_metrics": champion_metrics if champion_metrics else "None (Cold Start)",
            }
            write_json_file(self.config.report_file_path, report_payload)

            # 2. Baseline Performance Metrics (Strict contract for Monitoring Pipeline)
            # Only generate baselines for the APPROVED model to prevent monitoring false metrics.
            baseline_metrics = challenger_metrics if is_approved else champion_metrics
            
            if baseline_metrics:
                baseline_payload = {
                    "metrics": {
                        "log_loss": baseline_metrics["log_loss"],
                        "roc_auc": baseline_metrics["roc_auc"],
                        "brier_score": baseline_metrics["brier_score"],
                    },
                    "operational": {
                        "optimal_decision_threshold": baseline_metrics["optimal_threshold"],
                        "expected_roi_per_user": baseline_metrics["eroi"],
                    }
                }
                write_json_file(self.config.baseline_performance_metrics_file_path, baseline_payload)
            
            logging.info("Evaluation artifacts generated successfully.")

        except Exception as e:
            logging.exception("Failed to generate evaluation artifacts.")
            raise CustomException(e, sys) from e

    def _generate_metadata(
        self, 
        execution_time: float, 
        is_approved: bool, 
        challenger_metrics: Dict[str, Any], 
        champion_metrics: Optional[Dict[str, Any]]
    ) -> None:
        """Generates observability telemetry for the Model Evaluator stage."""
        try:
            metadata = {
                "pipeline_stage": "Model Evaluation & Gatekeeping",
                "execution_time_seconds": execution_time,
                "approval_decision": is_approved,
                "metrics_summary": {
                    "challenger_eroi": challenger_metrics["eroi"],
                    "champion_eroi": champion_metrics["eroi"] if champion_metrics else None,
                },
                "run_context": {
                    "run_id": self.context.run_id,
                    "dataset_s3_uri": self.context.training_dataset_s3_uri_path
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(self.config.metadata_file_path, metadata)
            logging.info("Model Evaluator metadata generated successfully.")

        except Exception as e:
            logging.exception("Failed to generate evaluator metadata.")
            raise CustomException(e, sys) from e