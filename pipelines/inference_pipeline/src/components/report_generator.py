import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import pandas as pd
import numpy as np

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from pipelines.inference_pipeline.src.core.context import InferencePipelineContext
from pipelines.inference_pipeline.src.entity.config_entity import ReportGeneratorConfig
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    ReportGeneratorArtifact,
    ModelLoaderArtifact,
    FeatureMatrixBuilderArtifact
)


class ReportGenerator:
    """
    Report Generator Component.

    Responsibilities:
    - Loads the serialized production model and the materialized feature matrix.
    - Isolates system/entity columns from the predictive features to prevent shape mismatches.
    - Executes batch inference to generate churn probabilities.
    - Applies business logic (probability thresholds) to flag at-risk entities.
    - Generates a stakeholder-friendly Business Report (CSV) focused on actionable insights.
    - Generates an exhaustive MLOps Telemetry Log (Parquet) for downstream drift detection.
    """

    def __init__(self, context: InferencePipelineContext) -> None:
        """
        Initializes the Report Generator component.

        Args:
            context (InferencePipelineContext): The centralized execution context.
        """
        try:
            self.context = context
            self.config = ReportGeneratorConfig.from_context(context)
            self._business_impact_metrics: Dict[str, Any] = {}
            logging.info("Inference Pipeline: Report Generator component initialized.")
        except Exception as e:
            logging.exception("Failed to initialize Report Generator component.")
            raise CustomException(e, sys) from e

    def run(
        self,
        model_loader_artifact: ModelLoaderArtifact,
        feature_matrix_artifact: FeatureMatrixBuilderArtifact
    ) -> ReportGeneratorArtifact:
        """
        Executes the batch inference and dual-artifact generation workflow.

        Args:
            model_loader_artifact (ModelLoaderArtifact): Artifact containing the model path.
            feature_matrix_artifact (FeatureMatrixBuilderArtifact): Artifact containing the feature matrix path.

        Returns:
            ReportGeneratorArtifact: Artifact containing local paths to the CSV report, 
                                     Parquet telemetry log, and metadata.
        """
        try:
            logging.info("Starting Report Generation Sequence.")
            start_time = time.time()

            # 1. Load Artifacts into Memory
            model = self._load_model(model_loader_artifact.model_file_path)
            feature_matrix_df = self._load_data(feature_matrix_artifact.feature_matrix_file_path)

            # 2. Execute Batch Inference
            probabilities = self._generate_predictions(model=model, df=feature_matrix_df)

            # 3. Generate Dual Outputs (Business vs. Engineering)
            self._generate_business_report(df=feature_matrix_df, probabilities=probabilities)
            self._generate_telemetry_log(df=feature_matrix_df, probabilities=probabilities)

            # 4. Generate Operational Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(execution_time=execution_time)

            # 5. Package Artifact
            artifact = ReportGeneratorArtifact(
                csv_report_path=self.config.csv_report_path,
                telemetry_log_path=self.config.telemetry_log_path,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("Report Generation completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Critical Failure inside Report Generator execution routine.")
            raise CustomException(e, sys) from e

    def _load_model(self, model_path: str) -> Any:
        """Loads the serialized Scikit-Learn / XGBoost pipeline from disk."""
        try:
            logging.debug("Loading predictive model from: %s", model_path)
            return joblib.load(model_path)
        except Exception as e:
            logging.exception("Failed to load the model artifact.")
            raise CustomException(e, sys) from e

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Loads the materialized Parquet feature matrix into memory."""
        try:
            logging.debug("Loading feature matrix from: %s", data_path)
            df = pd.read_parquet(data_path)
            if df.empty:
                raise ValueError(f"The loaded feature matrix at {data_path} is empty.")
            return df
        except Exception as e:
            logging.exception("Failed to load the feature matrix.")
            raise CustomException(e, sys) from e

    def _generate_predictions(self, model: Any, df: pd.DataFrame) -> np.ndarray:
        """
        Drops system columns to match the expected model schema and executes inference.
        """
        try:
            logging.info("Executing batch inference on %d records.", len(df))
            
            # Isolate the predictive features by dropping system metadata and entity IDs
            cols_to_drop = [col for col in self.config.system_columns_to_drop if col in df.columns]
            inference_features = df.drop(columns=cols_to_drop)

            # Generate probabilities for the positive class (Churn = 1)
            probabilities = model.predict_proba(inference_features)[:, 1]
            return probabilities
            
        except Exception as e:
            logging.exception("Failed during batch inference execution.")
            raise CustomException(e, sys) from e

    def _generate_business_report(self, df: pd.DataFrame, probabilities: np.ndarray) -> None:
        """
        Constructs the stakeholder-facing CSV report. 
        Focuses strictly on identifiers, probabilities, and actionable business metrics.
        """
        try:
            logging.info("Constructing business-facing CSV report.")
            
            report_df = pd.DataFrame()
            
            # 1. Attach primary business identifiers
            if "customer_unique_id" in df.columns:
                report_df["customer_unique_id"] = df["customer_unique_id"]
            else:
                logging.warning("customer_unique_id not found in data. Using index as identifier.")
                report_df["customer_unique_id"] = df.index

            # 2. Attach prediction outputs
            report_df["churn_probability"] = probabilities
            report_df["is_churn_risk"] = (probabilities >= self.config.probability_threshold).astype(int)

            # 3. Apply Business Value Metric (Revenue at Risk)
            # Attempt to locate a monetary column to calculate financial risk
            monetary_col = next(
                (col for col in df.columns if "monetary" in col.lower() or "value" in col.lower()), 
                None
            )
            
            if monetary_col:
                report_df["revenue_at_risk"] = (report_df["churn_probability"] * df[monetary_col]).round(2)
                sort_by_col = "revenue_at_risk"
            else:
                sort_by_col = "churn_probability"

            # 4. Sort to prioritize highest risk/value entities first
            report_df = report_df.sort_values(by=sort_by_col, ascending=False).reset_index(drop=True)

            # 5. Persist to Disk
            report_df.to_csv(self.config.csv_report_path, index=False)
            logging.debug("Business report materialized at: %s", self.config.csv_report_path)

            # 6. Capture metrics for the Master Inference Ledger
            self._business_impact_metrics = {
                "total_customers_scored": len(report_df),
                "total_churners_flagged": int(report_df["is_churn_risk"].sum()),
                "total_revenue_at_risk_flagged": float(report_df["revenue_at_risk"].sum()) if monetary_col else None
            }

        except Exception as e:
            logging.exception("Failed to generate the business report.")
            raise CustomException(e, sys) from e

    def _generate_telemetry_log(self, df: pd.DataFrame, probabilities: np.ndarray) -> None:
        """
        Constructs the MLOps-facing Parquet telemetry log.
        Combines the full feature matrix with the prediction outputs for drift analysis.
        """
        try:
            logging.info("Constructing MLOps telemetry Parquet log.")
            
            telemetry_df = df.copy()
            telemetry_df["inference_run_id"] = self.context.run_id
            telemetry_df["churn_probability"] = probabilities
            telemetry_df["is_churn_risk"] = (probabilities >= self.config.probability_threshold).astype(int)

            telemetry_df.to_parquet(self.config.telemetry_log_path, index=False, compression="snappy")
            logging.debug("Telemetry log materialized at: %s", self.config.telemetry_log_path)

        except Exception as e:
            logging.exception("Failed to generate the MLOps telemetry log.")
            raise CustomException(e, sys) from e

    def _generate_metadata(self, execution_time: float) -> None:
        """
        Generates operational metadata including the aggregated business impact metrics.
        """
        try:
            metadata: Dict[str, Any] = {
                "pipeline_stage": "Report Generator",
                "inference_run_id": self.context.run_id,
                "execution_time_seconds": execution_time,
                "business_impact": self._business_impact_metrics,
                "artifacts_generated": {
                    "business_report": self.config.csv_report_path,
                    "telemetry_log": self.config.telemetry_log_path
                },
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }

            with open(self.config.metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
                
            logging.debug("Report Generator metadata saved to: %s", self.config.metadata_file_path)

        except Exception as e:
            logging.exception("Failed to generate Report Generator metadata.")
            raise CustomException(e, sys) from e