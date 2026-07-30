import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from pipelines.monitoring_pipeline.src.core.context import MonitoringPipelineContext
from pipelines.monitoring_pipeline.src.entity.config_entity import StatisticalDriftCalculatorConfig
from pipelines.monitoring_pipeline.src.entity.artifact_entity import (
BaselineAndTelemetryResolverArtifact,
StatisticalDriftCalculatorArtifact
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file

class StatisticalDriftCalculator:
    """
    Statistical Drift Calculator Component.

    ```
    Responsibilities:
    - Operate as a proactive, label-independent diagnostic engine.
    - Identify the top N most influential predictive features using training SHAP baselines.
    - Compute the Population Stability Index (PSI) for the predicted probability distribution.
    - Compute the PSI for the identified top N features to detect covariate shift.
    - Gracefully handle both numerical and categorical feature distributions based on the
    physical types recorded during model training.
    - Guarantee robust statistical evaluation by enforcing strict bin edge / category continuity 
    from the reference baseline and algorithmically mitigating the Zero-Bin Problem.
    """

    def __init__(
        self,
        config: StatisticalDriftCalculatorConfig,
        context: MonitoringPipelineContext,
        resolver_artifact: BaselineAndTelemetryResolverArtifact
    ) -> None:
        """
        Initializes the Statistical Drift Calculator.

        Args:
            config (StatisticalDriftCalculatorConfig): Component-specific configuration.
            context (MonitoringPipelineContext): Global pipeline context containing shared resources.
            resolver_artifact (BaselineAndTelemetryResolverArtifact): Artifacts from the resolution stage.
        """
        try:
            self.config = config
            self.context = context
            self.resolver_artifact = resolver_artifact
            
            data_schema = self.context.config.get("data_schema", {})
            self.prediction_col = data_schema.get("prediction_column", "predicted_probability")
            
            logging.info("Monitoring Pipeline: Statistical Drift Calculator initialized.")
        except Exception as e:
            logging.exception("Failed to initialize Statistical Drift Calculator.")
            raise CustomException(e, sys) from e

    def run(self) -> StatisticalDriftCalculatorArtifact:
        """
        Executes the PSI-based statistical drift calculation workflow.

        Returns:
            StatisticalDriftCalculatorArtifact: Local paths to the drift report and metadata.
        """
        try:
            logging.info("Starting label-independent statistical drift calculation.")
            start_time = time.time()

            # 1. Load Data & Baselines
            telemetry_df = self._load_current_telemetry()
            shap_importances = self._load_json(self.resolver_artifact.shap_importance_file_path)
            reference_distributions_raw = self._load_json(self.resolver_artifact.reference_distributions_file_path)

            # Safely extract the distributions dictionary from the nested production JSON schema
            reference_distributions = reference_distributions_raw.get("distributions", {})
            if not reference_distributions:
                logging.warning("Reference baseline JSON missing 'distributions' key. Attempting flat fallback.")
                reference_distributions = reference_distributions_raw

            # 2. Identify Target Features
            top_features = self._extract_top_shap_features(shap_importances)

            # 3. Calculate Drift
            drift_report = self._calculate_all_drifts(
                telemetry_df=telemetry_df,
                reference_distributions=reference_distributions,
                top_features=top_features
            )

            # 4. Generate Artifacts
            execution_time = round(time.time() - start_time, 2)
            self._save_reports(drift_report, execution_time, len(telemetry_df))

            # 5. Package Artifact
            artifact = StatisticalDriftCalculatorArtifact(
                drift_report_file_path=self.config.drift_report_file_path,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("Statistical drift calculation completed successfully.")
            return artifact

        except Exception as e:
            logging.exception("Critical Failure: Statistical Drift Calculator run failed.")
            raise CustomException(e, sys) from e

    def _load_current_telemetry(self) -> pd.DataFrame:
        """
        Loads the daily proactive inference telemetry directly into a Pandas DataFrame.
        Returns an empty DataFrame gracefully if the file is structurally empty.
        """
        try:
            path = self.resolver_artifact.current_telemetry_file_path
            df = pd.read_parquet(path)
            logging.info("Loaded current telemetry with %d records.", len(df))
            return df
        except Exception as e:
            logging.exception("Failed to load telemetry parquet from: %s", self.resolver_artifact.current_telemetry_file_path)
            raise CustomException(e, sys) from e

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Safely loads a JSON baseline artifact.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.exception("Failed to load baseline JSON from: %s", file_path)
            raise CustomException(e, sys) from e

    def _extract_top_shap_features(self, shap_data: Dict[str, Any]) -> List[str]:
        """
        Parses the nested SHAP importance JSON artifact to identify the top N predictive features.
        Expects a schema containing a 'feature_importance' list of dictionaries.
        """
        try:
            feature_list = shap_data.get("feature_importance", [])
            if not feature_list:
                raise ValueError("SHAP baseline data is missing the 'feature_importance' array or it is empty.")

            # Sort by mean_abs_shap_value descending to guarantee correct ordering
            sorted_features = sorted(
                feature_list, 
                key=lambda item: float(item.get("mean_abs_shap_value", 0.0)), 
                reverse=True
            )
            
            top_k = self.config.top_shap_features_count
            top_feature_names = [
                feature.get("feature_name") 
                for feature in sorted_features[:top_k] 
                if feature.get("feature_name")
            ]
            
            logging.info("Resolved top %d SHAP features for drift monitoring: %s", top_k, top_feature_names)
            return top_feature_names

        except Exception as e:
            logging.exception("Failed to parse SHAP baseline data.")
            raise CustomException(e, sys) from e

    def _calculate_all_drifts(
        self, 
        telemetry_df: pd.DataFrame, 
        reference_distributions: Dict[str, Any], 
        top_features: List[str]
    ) -> Dict[str, Any]:
        """
        Iterates over the target prediction column and top features to compute individual PSI metrics.
        """
        drift_report: Dict[str, Any] = {
            "prediction_drift": {},
            "feature_drift": {}
        }

        # Handle empty telemetry scenario safely
        if telemetry_df.empty:
            logging.warning("Telemetry DataFrame is empty. Logging 0.0 PSI for all components.")
            drift_report["prediction_drift"][self.prediction_col] = {"psi_score": 0.0, "status": "EMPTY_TELEMETRY"}
            for feature in top_features:
                drift_report["feature_drift"][feature] = {"psi_score": 0.0, "status": "EMPTY_TELEMETRY"}
            return drift_report

        # 1. Prediction Drift (Target = predicted_probability)
        if self.prediction_col in telemetry_df.columns and self.prediction_col in reference_distributions:
            ref_stats = reference_distributions[self.prediction_col]
            psi_val = self._compute_psi_router(telemetry_df[self.prediction_col], ref_stats)
            drift_report["prediction_drift"][self.prediction_col] = {
                "psi_score": round(psi_val, 4),
                "status": "CALCULATED"
            }
        else:
            logging.warning("Could not compute prediction drift. Missing '%s' in telemetry or baselines.", self.prediction_col)
            drift_report["prediction_drift"][self.prediction_col] = {
                "psi_score": 0.0,
                "status": "MISSING_DATA"
            }

        # 2. Feature Drift (Top N SHAP Features)
        for feature in top_features:
            if feature not in telemetry_df.columns:
                logging.warning("Feature '%s' not found in telemetry. Skipping PSI.", feature)
                drift_report["feature_drift"][feature] = {"psi_score": 0.0, "status": "MISSING_IN_TELEMETRY"}
                continue
            
            if feature not in reference_distributions:
                logging.warning("Feature '%s' not found in reference baseline. Skipping PSI.", feature)
                drift_report["feature_drift"][feature] = {"psi_score": 0.0, "status": "MISSING_IN_BASELINE"}
                continue

            ref_stats = reference_distributions[feature]
            psi_val = self._compute_psi_router(telemetry_df[feature], ref_stats)
            
            drift_report["feature_drift"][feature] = {
                "psi_score": round(psi_val, 4),
                "status": "CALCULATED"
            }

        return drift_report

    def _compute_psi_router(
        self, 
        current_series: pd.Series, 
        reference_stats: Dict[str, Any]
    ) -> float:
        """
        Routes the PSI calculation to the appropriate method based on the physical type 
        of the feature recorded in the baseline distributions.
        """
        physical_type = reference_stats.get("physical_type", "numerical").lower()
        
        if physical_type == "categorical":
            return self._compute_categorical_psi(current_series, reference_stats)
        else:
            return self._compute_numerical_psi(current_series, reference_stats)

    def _compute_numerical_psi(
        self, 
        current_series: pd.Series, 
        reference_stats: Dict[str, Any]
    ) -> float:
        """
        First-principles Population Stability Index (PSI) calculation for numerical features.
        Enforces strict bin consistency with the training reference and applies 
        epsilon clipping to algorithmically mitigate the Zero-Bin problem.
        """
        try:
            expected_pct = np.array(reference_stats.get("expected_percentages", []))
            bin_edges = reference_stats.get("bin_edges", [])
            
            if len(expected_pct) == 0 or len(bin_edges) == 0:
                logging.warning("Numerical reference stats missing bin definitions. Returning 0.0 PSI.")
                return 0.0

            valid_data = current_series.dropna().values
            if len(valid_data) == 0:
                return 0.0

            # Ensure new inference data that falls slightly outside the exact min/max 
            # of the training distribution does not trigger out-of-bounds indexing errors.
            edges = list(bin_edges)
            if len(edges) > 1:
                edges[0] = -np.inf
                edges[-1] = np.inf

            # Perform binning using the frozen reference edges
            counts, _ = np.histogram(valid_data, bins=edges)
            total_count = np.sum(counts)

            if total_count == 0:
                return 0.0

            actual_pct = counts / total_count

            # The Zero-Bin Problem Mitigation
            epsilon = self.config.zero_bin_epsilon_psi
            actual_pct_clipped = np.maximum(actual_pct, epsilon)
            expected_pct_clipped = np.maximum(expected_pct, epsilon)

            # Re-normalize
            actual_pct_norm = actual_pct_clipped / np.sum(actual_pct_clipped)
            expected_pct_norm = expected_pct_clipped / np.sum(expected_pct_clipped)

            # PSI Formula
            psi_components = (actual_pct_norm - expected_pct_norm) * np.log(actual_pct_norm / expected_pct_norm)
            return float(np.sum(psi_components))

        except Exception:
            logging.exception("Mathematical error encountered during numerical PSI calculation.")
            return 0.0

    def _compute_categorical_psi(
        self, 
        current_series: pd.Series, 
        reference_stats: Dict[str, Any]
    ) -> float:
        """
        First-principles Population Stability Index (PSI) calculation for categorical features.
        Aligns strictly with the reference categories and applies epsilon clipping.
        """
        try:
            expected_pct = np.array(reference_stats.get("expected_percentages", []))
            categories = reference_stats.get("categories", [])
            
            if len(expected_pct) == 0 or len(categories) == 0:
                logging.warning("Categorical reference stats missing category definitions. Returning 0.0 PSI.")
                return 0.0

            valid_data = current_series.dropna()
            if len(valid_data) == 0:
                return 0.0

            # Use Pandas Categorical to guarantee alignment with the reference categories sequence.
            # Any values in current_series not present in `categories` become NaN and are omitted 
            # from value_counts, protecting against dimension mismatches.
            cat_series = pd.Categorical(valid_data, categories=categories)
            counts = cat_series.value_counts(dropna=True).values
            
            total_count = np.sum(counts)
            if total_count == 0:
                return 0.0

            actual_pct = counts / total_count

            # The Zero-Bin Problem Mitigation
            epsilon = self.config.zero_bin_epsilon_psi
            actual_pct_clipped = np.maximum(actual_pct, epsilon)
            expected_pct_clipped = np.maximum(expected_pct, epsilon)

            # Re-normalize
            actual_pct_norm = actual_pct_clipped / np.sum(actual_pct_clipped)
            expected_pct_norm = expected_pct_clipped / np.sum(expected_pct_clipped)

            # PSI Formula
            psi_components = (actual_pct_norm - expected_pct_norm) * np.log(actual_pct_norm / expected_pct_norm)
            return float(np.sum(psi_components))

        except Exception:
            logging.exception("Mathematical error encountered during categorical PSI calculation.")
            return 0.0

    def _save_reports(self, drift_report: Dict[str, Any], execution_time: float, population_size: int) -> None:
        """
        Persists the drift calculations and generates component observability metadata.
        """
        try:
            # 1. Save Drift Report
            write_json_file(file_path=self.config.drift_report_file_path, content=drift_report)
            logging.info("Statistical drift report saved to: %s", self.config.drift_report_file_path)

            # 2. Save Operational Metadata
            metadata: Dict[str, Any] = {
                "pipeline_stage": "Monitoring Statistical Drift Calculator",
                "execution_time_seconds": execution_time,
                "volumetrics": {
                    "scored_population_size": population_size
                },
                "configuration_applied": {
                    "top_shap_features_monitored": self.config.top_shap_features_count,
                    "zero_bin_epsilon": self.config.zero_bin_epsilon_psi
                },
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "orchestration": {
                    "run_id": self.context.run_id
                }
            }
            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            
        except Exception as e:
            logging.exception("Failed to write drift reports and metadata to disk.")
            raise CustomException(e, sys) from e