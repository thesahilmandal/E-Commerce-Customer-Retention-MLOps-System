import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd

from pipelines.monitoring_pipeline.src import constants
from pipelines.monitoring_pipeline.src.entity.config_entity import PerformanceEvaluatorConfig
from pipelines.monitoring_pipeline.src.entity.artifact_entity import (
    BaselineAndTelemetryResolverArtifact,
    PerformanceEvaluatorArtifact
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class PerformanceEvaluator:
    """
    Performance Evaluator Component.

    Responsibilities:
    - Operate as a reactive, label-dependent diagnostic engine.
    - Merge historical predictions (T-30 telemetry) with newly matured ground-truth labels.
    - Compute strictly calibrated probability metrics: Brier Score and Log Loss.
    - Estimate Realized Return on Investment (ROI) to evaluate business impact.
    - Handle immature system states gracefully if the lookback period has not yet elapsed.
    """

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        resolver_artifact: BaselineAndTelemetryResolverArtifact
    ) -> None:
        """
        Initializes the Performance Evaluator.
        """
        try:
            self.config = config
            self.resolver_artifact = resolver_artifact
            logging.info("Monitoring Pipeline: Performance Evaluator initialized.")
        except Exception as e:
            logging.exception("Failed to initialize Performance Evaluator.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> PerformanceEvaluatorArtifact:
        """
        Executes the label-dependent performance evaluation workflow.

        Returns:
            PerformanceEvaluatorArtifact: Local paths to the performance report and metadata.
        """
        try:
            logging.info("Starting label-dependent performance evaluation.")
            start_time = time.time()

            # 1. Load and join historical telemetry with matured labels
            merged_df = self._load_and_join_lookback_data()

            # 2. Evaluate state maturity and calculate metrics
            performance_report = self._evaluate_performance(merged_df)

            # 3. Generate Artifacts
            execution_time = round(time.time() - start_time, 2)
            scored_population = len(merged_df) if performance_report.get("is_evaluated") else 0
            self._save_reports(performance_report, execution_time, scored_population)

            # 4. Package Artifact
            artifact = PerformanceEvaluatorArtifact(
                performance_report_file_path=self.config.performance_report_file_path,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("Performance evaluation completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Critical Failure: Performance Evaluator run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DATA LOADING & JOINING
    # ==========================================================
    def _load_and_join_lookback_data(self) -> pd.DataFrame:
        """
        Loads the historical predictions and matured outcomes, joining them into a 
        single evaluation DataFrame in memory.
        """
        try:
            telemetry_path = self.resolver_artifact.lookback_telemetry_file_path
            labels_path = self.resolver_artifact.lookback_labels_file_path

            telemetry_df = pd.read_parquet(telemetry_path)
            labels_df = pd.read_parquet(labels_path)

            if telemetry_df.empty or labels_df.empty:
                return pd.DataFrame()

            # The out-of-core resolver logic handles the primary join, but we merge locally 
            # to align rows perfectly just in case of partitioned discrepancies.
            merged_df = pd.merge(
                telemetry_df, 
                labels_df, 
                on=constants.CUSTOMER_ID_COLUMN, 
                how="inner"
            )
            
            logging.info(f"Loaded and joined lookback cohort containing {len(merged_df)} matured records.")
            return merged_df

        except Exception as e:
            logging.exception("Failed to load and join lookback datasets.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # METRICS CALCULATION
    # ==========================================================
    def _evaluate_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Determines cohort maturity and calculates statistical and business metrics.
        """
        # Guardrail: Check for immature system state (dummy records from Resolver)
        if len(df) == 0 or (len(df) == 1 and df.iloc[0].get(constants.CUSTOMER_ID_COLUMN) == 'dummy'):
            logging.warning("System immature: Lookback cohort not found. Skipping reactive evaluation.")
            return {
                "is_evaluated": False,
                "reason": "INSUFFICIENT_LOOKBACK_MATURITY",
                "metrics": {
                    "brier_score": 0.0,
                    "log_loss": 0.0,
                    "realized_roi": 0.0
                }
            }

        target_col = constants.TARGET_COLUMN
        pred_col = "predicted_probability"

        if target_col not in df.columns or pred_col not in df.columns:
            raise ValueError(f"Missing required evaluation columns. Expected: {target_col}, {pred_col}")

        y_true = df[target_col].values.astype(float)
        y_prob = df[pred_col].values.astype(float)

        # 1. Calculate strictly proper scoring rules from first principles
        brier_score = self._compute_brier_score(y_true, y_prob)
        log_loss = self._compute_log_loss(y_true, y_prob)

        # 2. Calculate Business Value (Realized ROI)
        realized_roi = self._compute_realized_roi(y_true, y_prob)

        logging.info(
            f"Evaluation Complete | Brier: {brier_score:.4f} | "
            f"Log Loss: {log_loss:.4f} | Realized ROI: ${realized_roi:.2f}"
        )

        return {
            "is_evaluated": True,
            "metrics": {
                "brier_score": brier_score,
                "log_loss": log_loss,
                "realized_roi": realized_roi
            }
        }

    def _compute_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        First principles calculation of Brier Score (Mean Squared Error for probabilities).
        Formula: (1/N) * sum( (y_prob - y_true)^2 )
        """
        return float(np.mean((y_prob - y_true) ** 2))

    def _compute_log_loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        First principles calculation of Log Loss (Cross-Entropy Loss).
        Incorporates epsilon clipping to mathematically prevent undefined log(0) exceptions.
        Formula: -(1/N) * sum( y_true * log(y_prob) + (1 - y_true) * log(1 - y_prob) )
        """
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
        return float(loss)

    def _compute_realized_roi(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculates the actual business impact generated by the model on this historical cohort.
        Uses a standard 0.5 binary threshold to simulate targeting decisions.
        """
        threshold = 0.5
        y_pred = (y_prob >= threshold).astype(int)

        # True Positives: Customers we correctly predicted would churn and intervened with.
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        # False Positives: Customers we thought would churn, but wouldn't have (wasted intervention cost).
        fp = np.sum((y_true == 0) & (y_pred == 1))

        # Financial Math
        # We only save a fraction of True Positives based on intervention effectiveness.
        saved_customers = tp * self.config.intervention_save_rate
        gross_benefit = saved_customers * self.config.customer_ltv
        
        # We incur costs for every customer we target (TP + FP).
        total_campaign_cost = (tp + fp) * self.config.campaign_cost
        
        net_roi = gross_benefit - total_campaign_cost
        return float(net_roi)

    # ==========================================================
    # ARTIFACT & METADATA GENERATION
    # ==========================================================
    def _save_reports(self, performance_report: Dict[str, Any], execution_time: float, population_size: int) -> None:
        """
        Persists the performance evaluation calculations and generates operational metadata.
        """
        try:
            # 1. Save Performance Report
            write_json_file(file_path=self.config.performance_report_file_path, content=performance_report)
            logging.info(f"Performance report saved to: {self.config.performance_report_file_path}")

            # 2. Save Operational Metadata
            metadata: Dict[str, Any] = {
                "pipeline_stage": "Monitoring Performance Evaluator",
                "execution_time_seconds": execution_time,
                "volumetrics": {
                    "matured_cohort_size": population_size
                },
                "financial_parameters": {
                    "campaign_cost": self.config.campaign_cost,
                    "customer_ltv": self.config.customer_ltv,
                    "intervention_save_rate": self.config.intervention_save_rate
                },
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            
        except Exception as e:
            logging.exception("Failed to write performance reports and metadata to disk.")
            raise CustomException(e, sys) from e