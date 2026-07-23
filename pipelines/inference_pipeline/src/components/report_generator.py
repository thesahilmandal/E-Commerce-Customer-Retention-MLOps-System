import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import pandas as pd

from pipelines.inference_pipeline.src.core.context import InferenceContext
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    FeatureMatrixBuilderArtifact,
    ModelLoadingArtifact,
    ReportGenerationArtifact,
)
from pipelines.inference_pipeline.src.entity.config_entity import ReportGeneratorConfig
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class ReportGenerator:
    """
    Report Generator Component.

    Executes batch inference using the validated Champion model and materialized Feature Matrix.
    Produces two distinct data products:
    1. A filtered, sorted CSV report for business stakeholders.
    2. A dense, feature-complete Parquet telemetry log for downstream MLOps monitoring.
    """

    def __init__(self, config: ReportGeneratorConfig, context: InferenceContext) -> None:
        """
        Initializes the ReportGenerator component.

        Args:
            config (ReportGeneratorConfig): Configuration containing thresholds and output paths.
            context (InferenceContext): Centralized pipeline execution context.
        """
        try:
            self.config = config
            self.context = context

            os.makedirs(os.path.dirname(self.config.local_business_report_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.local_telemetry_log_path), exist_ok=True)

            logging.info("ReportGenerator component initialized.")
        except Exception as e:
            logging.exception("Failed to initialize ReportGenerator component.")
            raise CustomException(e, sys) from e

    def run(
        self,
        model_artifact: ModelLoadingArtifact,
        feature_artifact: FeatureMatrixBuilderArtifact,
    ) -> ReportGenerationArtifact:
        """
        Loads the model, executes scoring, and generates dual-track output reports.

        Args:
            model_artifact (ModelLoadingArtifact): Artifact containing the model path and run ID.
            feature_artifact (FeatureMatrixBuilderArtifact): Artifact containing the feature matrix.

        Returns:
            ReportGenerationArtifact: Paths to the generated business CSV and telemetry Parquet.
        """
        try:
            logging.info("Starting batch inference scoring and report generation.")
            start_time = time.time()

            # 1. Load Model and Feature Matrix
            logging.debug("Loading champion model from disk.")
            model = joblib.load(model_artifact.model_file_path)

            logging.debug("Loading feature matrix into memory for scoring.")
            df = pd.read_parquet(feature_artifact.feature_matrix_file_path)

            if df.empty:
                raise ValueError("Feature matrix is empty. Cannot perform inference.")

            # 2. Isolate Features
            features_df = df.drop(columns=self.config.system_columns_to_drop, errors="ignore")
            
            # 3. Execute Inference
            logging.info("Executing model.predict_proba() on %d records.", len(features_df))
            probabilities = model.predict_proba(features_df)[:, 1]

            # 4. Generate MLOps Telemetry Log (Parquet)
            # Retain all original features and system columns, append inference metadata
            telemetry_df = df.copy()
            telemetry_df["champion_run_id"] = model_artifact.champion_run_id
            telemetry_df["prediction_probability"] = probabilities
            telemetry_df["inference_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            
            telemetry_df.to_parquet(self.config.local_telemetry_log_path, index=False)
            logging.info("Telemetry log saved to: %s", self.config.local_telemetry_log_path)

            # 5. Generate Business Report (CSV)
            # Filter for at-risk customers and calculate proxy business metrics
            business_df = pd.DataFrame({
                "customer_unique_id": df.get("customer_unique_id", df.index),
                "snapshot_date": feature_artifact.snapshot_date,
                "churn_probability": probabilities
            })

            # Calculate Revenue at Risk if the monetary proxy feature exists
            if "monetary_total" in df.columns:
                business_df["expected_revenue_risk"] = df["monetary_total"] * probabilities
                business_df = business_df.sort_values(by="expected_revenue_risk", ascending=False)
            else:
                business_df = business_df.sort_values(by="churn_probability", ascending=False)

            # Apply probability threshold for business actionability
            business_df = business_df[business_df["churn_probability"] >= self.config.probability_threshold]
            
            business_df.to_csv(self.config.local_business_report_path, index=False)
            logging.info("Business report saved to: %s", self.config.local_business_report_path)

            # 6. Record execution metadata in central context ledger
            execution_time = round(time.time() - start_time, 2)
            telemetry: Dict[str, Any] = {
                "total_scored_records": len(df),
                "at_risk_customers_identified": len(business_df),
                "probability_threshold_applied": self.config.probability_threshold,
                "local_business_report_path": self.config.local_business_report_path,
                "local_telemetry_log_path": self.config.local_telemetry_log_path,
                "execution_time_seconds": execution_time,
            }
            self.context.add_metadata("ReportGenerator", telemetry)

            # 7. Package and return artifact
            artifact = ReportGenerationArtifact(
                business_report_file_path=self.config.local_business_report_path,
                telemetry_log_file_path=self.config.local_telemetry_log_path,
            )

            logging.info("ReportGenerator execution completed successfully.")
            return artifact

        except Exception as e:
            logging.exception("Critical Failure in ReportGenerator component.")
            raise CustomException(e, sys) from e