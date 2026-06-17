import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import log_loss, brier_score_loss

from shared_core import constants
from pipelines.training_pipeline.src.entity.config_entity import ModelEvaluationConfig
from pipelines.training_pipeline.src.entity.artifact_entity import (
    ModelTrainingArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact,
)
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class ModelEvaluation:
    """
    Model Evaluation component for the Training Pipeline.

    Responsibilities:
    - Act as the strict, automated gatekeeper before production deployment.
    - Calculate global statistical metrics (Log Loss, Brier Score) for the Challenger model.
    - Perform Slice-Based Evaluation (e.g., performance on High-Value customers).
    - Translate probability outputs into business metrics (Expected ROI).
    - Retrieve the current Champion model from the updated S3 Registry structure (Cold Start handling).
    - Execute Champion vs. Challenger Duel using defined Hysteresis margins.
    - Generate an immutable, FAANG-grade evaluation report including data provenance and slice definitions.
    """

    def __init__(
        self,
        config: ModelEvaluationConfig,
        trainer_artifact: ModelTrainingArtifact,
        ingestion_artifact: DataIngestionArtifact,
    ) -> None:
        """
        Initializes the Model Evaluation component.
        """
        try:
            self.config = config
            self.trainer_artifact = trainer_artifact
            self.ingestion_artifact = ingestion_artifact
            self.s3_sync = S3Sync()

            # Dynamic S3 URIs aligned with the updated model registry architecture
            self.s3_registry_base_uri = f"s3://{constants.S3_BUCKET_NAME}/{constants.S3_MODEL_REGISTRY_DIR_NAME}"
            self.s3_pointer_uri = f"{self.s3_registry_base_uri}/model_state.json"

            os.makedirs(self.config.model_evaluation_root_dir, exist_ok=True)
            logging.info("Training Pipeline: Model Evaluation component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Model Evaluation component.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> ModelEvaluationArtifact:
        """
        Executes the evaluation pipeline and determines production promotion status.
        """
        try:
            logging.info("Starting Model Evaluation Pipeline.")
            start_time = time.time()

            # 1. Load Data and Challenger Model
            X_test, y_test = self._load_data_from_ingestion(
                self.ingestion_artifact.test_data_path
            )
            test_set_size = len(X_test)
            logging.info("Test set loaded with %d records.", test_set_size)

            challenger_model = joblib.load(self.trainer_artifact.model_file_path)

            # 2. Evaluate Challenger Model
            logging.info("Calculating metrics for Challenger model.")
            challenger_metrics = self._calculate_metrics(challenger_model, X_test, y_test)
            logging.info("Challenger Metrics: %s", challenger_metrics)

            # Phase A: Absolute Threshold Check
            if challenger_metrics["eroi"] < self.config.min_eroi_threshold:
                logging.warning(
                    "Challenger failed absolute minimum EROI threshold (%.4f < %.4f).",
                    challenger_metrics["eroi"],
                    self.config.min_eroi_threshold,
                )
                approval_status = False
                champion_metrics = None
            else:
                # 3. Retrieve Champion Model (Cold Start Check)
                champion_data = self._get_production_champion()

                # Phase B & C: The Duel
                if champion_data is None:
                    logging.info("COLD START: No Champion found in Registry. Promoting Challenger.")
                    approval_status = True
                    champion_metrics = None
                else:
                    champion_model, champion_registered_metrics = champion_data
                    logging.info("Champion model retrieved. Commencing Duel on Test Set.")
                    
                    champion_metrics = self._calculate_metrics(champion_model, X_test, y_test)
                    logging.info("Champion Metrics (Current Test Set): %s", champion_metrics)

                    approval_status = self._duel_models(challenger_metrics, champion_metrics)

            # 4. Generate Reports and Metadata
            self._generate_reports(
                challenger_metrics=challenger_metrics,
                champion_metrics=champion_metrics,
                approval_status=approval_status,
                test_set_size=test_set_size
            )

            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(approval_status, execution_time, test_set_size)

            # 5. Package Artifact
            artifact = ModelEvaluationArtifact(
                report_file_path=self.config.report_file_path,
                metadata_file_path=self.config.metadata_file_path,
                approval_status=approval_status,
            )

            logging.info("Model Evaluation completed. Approval Status: %s", approval_status)
            return artifact

        except Exception as e:
            logging.exception("Model Evaluation run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DATA AND MODEL LOADING
    # ==========================================================
    def _load_data_from_ingestion(self, file_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Loads test dataset and splits X and y, dropping system metadata columns
        to ensure the feature matrix matches the trained model schema perfectly.
        """
        try:
            df = pd.read_parquet(file_path)
            y = df[constants.TARGET_COLUMN].values
            
            cols_to_drop = constants.SYSTEM_COLUMNS_TO_DROP + [constants.TARGET_COLUMN]
            cols_to_drop = [c for c in cols_to_drop if c in df.columns]
            X = df.drop(columns=cols_to_drop)
            
            return X, y
        except Exception as e:
            logging.exception("Failed to load test datasets from ingestion path.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # METRICS & BUSINESS LOGIC
    # ==========================================================
    def _calculate_metrics(self, model: Any, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Calculates Global, Slice-based, and Business metrics (EROI).
        """
        try:
            y_proba = model.predict_proba(X)[:, 1]

            # 1. Global Statistical Metrics
            global_log_loss = float(log_loss(y, y_proba))
            global_brier = float(brier_score_loss(y, y_proba))

            # 2. Slice-Based Evaluation (High-Value Cohort)
            # Identifying the top 10% of customers by monetary total
            if "monetary_total" in X.columns:
                threshold_90th = X["monetary_total"].quantile(0.90)
                high_value_mask = X["monetary_total"] >= threshold_90th
                
                if high_value_mask.sum() > 0:
                    slice_brier = float(brier_score_loss(y[high_value_mask], y_proba[high_value_mask]))
                else:
                    slice_brier = global_brier
            else:
                logging.warning("Slice column 'monetary_total' missing. Falling back to global Brier.")
                slice_brier = global_brier

            # 3. Business Metric: Expected ROI (EROI)
            eroi = self._calculate_expected_roi(y, y_proba)

            return {
                "log_loss": global_log_loss,
                "brier_score": global_brier,
                "high_value_slice_brier_score": slice_brier,
                "eroi": eroi,
            }

        except Exception as e:
            logging.exception("Error calculating evaluation metrics.")
            raise CustomException(e, sys) from e

    def _calculate_expected_roi(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Simulates a retention campaign to translate model probabilities into financial ROI.
        Assumptions: 
        - Cost to target a predicted churner: $10
        - LTV saved if successful: $500
        - Intervention success rate: 10%
        """
        try:
            campaign_cost = 10.0
            ltv = 500.0
            save_rate = 0.10

            # Expected Value of targeting user i: P(Churn) * LTV * Save_Rate - Cost
            expected_values = (y_proba * ltv * save_rate) - campaign_cost
            
            # Action: Only target users with positive expected value
            target_mask = expected_values > 0
            
            if not np.any(target_mask):
                return 0.0
            
            total_cost = np.sum(target_mask) * campaign_cost
            
            # Actual Revenue Saved = (Actual Churners in Target Group) * Save_Rate * LTV
            actual_churners_targeted = np.sum(y_true[target_mask])
            total_revenue_saved = actual_churners_targeted * save_rate * ltv
            
            eroi = (total_revenue_saved - total_cost) / total_cost
            return float(eroi)

        except Exception as e:
            logging.exception("Error calculating Expected ROI.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # S3 REGISTRY INTERACTION
    # ==========================================================
    def _get_production_champion(self) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Checks the S3 Model Registry for a currently deployed Champion model using the
        new atomic model_state.json pointer. Returns the loaded model and its baseline metrics, 
        or None if Cold Start.
        """
        try:
            local_pointer_path = os.path.join(self.config.model_evaluation_root_dir, "tmp_pointer.json")
            
            try:
                logging.info("Checking S3 for Production Champion pointer at: %s", self.s3_pointer_uri)
                self.s3_sync.download_file(self.s3_pointer_uri, local_pointer_path)
            except Exception:
                logging.info("Production pointer not found. Assuming Cold Start scenario.")
                return None

            with open(local_pointer_path, "r") as f:
                pointer_data = json.load(f)
            
            champion_run_id = pointer_data.get("champion_run_id")
            s3_champion_model_uri = pointer_data.get("s3_model_path")
            
            if not s3_champion_model_uri:
                raise ValueError("Registry pointer file is corrupted: missing 's3_model_path' key.")

            champion_metrics = {
                "eroi_baseline": pointer_data.get("eroi_baseline"),
                "log_loss_baseline": pointer_data.get("log_loss_baseline")
            }

            logging.info("Champion found (Run ID: %s). Downloading artifact from: %s", champion_run_id, s3_champion_model_uri)
            
            local_champion_path = os.path.join(self.config.model_evaluation_root_dir, "tmp_champion_model.pkl")
            
            self.s3_sync.download_file(s3_champion_model_uri, local_champion_path)
            champion_model = joblib.load(local_champion_path)

            # Cleanup temp files
            os.remove(local_pointer_path)
            os.remove(local_champion_path)

            return champion_model, champion_metrics

        except Exception as e:
            logging.exception("Error retrieving Champion model from S3.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # HYSTERESIS LOGIC (THE DUEL)
    # ==========================================================
    def _duel_models(self, challenger_metrics: Dict[str, float], champion_metrics: Dict[str, float]) -> bool:
        """
        Executes Hysteresis evaluation. Challenger must beat Champion's EROI by a margin 
        to justify the operational risk of a new deployment.
        """
        try:
            challenger_eroi = challenger_metrics["eroi"]
            champion_eroi = champion_metrics["eroi"]
            
            required_eroi = champion_eroi + self.config.eroi_hysteresis_margin

            logging.info("Duel Results -> Challenger EROI: %.4f | Champion EROI: %.4f", challenger_eroi, champion_eroi)
            logging.info("Required EROI to promote (Margin=%.4f): %.4f", self.config.eroi_hysteresis_margin, required_eroi)

            if challenger_eroi >= required_eroi:
                logging.info("Challenger wins duel. Promotion approved.")
                return True
            else:
                logging.warning("Challenger failed to surpass Hysteresis margin. Promotion denied.")
                return False

        except Exception as e:
            logging.exception("Error executing Champion vs. Challenger duel.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # AUDIT TRAIL & METADATA GENERATION
    # ==========================================================
    def _generate_reports(
        self, 
        challenger_metrics: Dict[str, float], 
        champion_metrics: Optional[Dict[str, float]], 
        approval_status: bool,
        test_set_size: int
    ) -> None:
        """
        Generates the definitive business audit trail for the evaluation run, 
        including data provenance and explicit slice definitions.
        """
        try:
            logging.info("Generating Model Evaluation Report.")

            report = {
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                "approval_status": approval_status,
                "data_provenance": {
                    "test_set_size_rows": test_set_size,
                    "evaluation_snapshot_used": constants.TEST_SNAPSHOT
                },
                "slice_definitions": {
                    "high_value_cohort": "monetary_total >= 90th percentile (Top 10% by revenue)"
                },
                "business_logic": {
                    "minimum_eroi_threshold": self.config.min_eroi_threshold,
                    "eroi_hysteresis_margin": self.config.eroi_hysteresis_margin
                },
                "challenger_metrics": challenger_metrics,
                "champion_metrics_on_current_test_set": champion_metrics if champion_metrics else "N/A - Cold Start"
            }

            write_json_file(file_path=self.config.report_file_path, content=report)

        except Exception as e:
            logging.exception("Failed to generate evaluation report.")
            raise CustomException(e, sys) from e

    def _generate_metadata(self, approval_status: bool, execution_time: float, test_set_size: int) -> None:
        """
        Generates standard telemetry metadata for the pipeline component.
        """
        try:
            metadata: Dict[str, Any] = {
                "pipeline_stage": "Model Evaluation",
                "execution_time_seconds": execution_time,
                "deployment_approved": approval_status,
                "test_set_size_rows": test_set_size,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(file_path=self.config.metadata_file_path, content=metadata)

        except Exception as e:
            logging.exception("Failed to generate evaluation metadata.")
            raise CustomException(e, sys) from e