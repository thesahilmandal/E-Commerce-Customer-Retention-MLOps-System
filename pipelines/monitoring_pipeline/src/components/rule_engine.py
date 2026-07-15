import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

from pipelines.monitoring_pipeline.src.entity.config_entity import RuleEngineConfig
from pipelines.monitoring_pipeline.src.entity.artifact_entity import (
    BaselineAndTelemetryResolverArtifact,
    StatisticalDriftCalculatorArtifact,
    PerformanceEvaluatorArtifact,
    RuleEngineArtifact
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class RuleEngine:
    """
    Rule Engine Component for the Monitoring Pipeline.

    Responsibilities:
    - Operate as the deterministic decision boundary for the MLOps lifecycle.
    - Evaluate label-independent drift metrics and label-dependent performance metrics
      against strict predefined configuration thresholds.
    - Trigger model retraining based on three explicit conditions:
        1. Critical Prediction Drift
        2. Critical Feature Drift
        3. Severe Performance Degradation
    - Output an immutable consolidated monitoring report and a boolean trigger payload
      (`need_update`) to safely decouple monitoring from continual learning.
    """

    def __init__(
        self,
        config: RuleEngineConfig,
        resolver_artifact: BaselineAndTelemetryResolverArtifact,
        drift_artifact: StatisticalDriftCalculatorArtifact,
        performance_artifact: PerformanceEvaluatorArtifact
    ) -> None:
        """
        Initializes the Rule Engine.
        """
        try:
            self.config = config
            self.resolver_artifact = resolver_artifact
            self.drift_artifact = drift_artifact
            self.performance_artifact = performance_artifact
            
            logging.info("Monitoring Pipeline: Rule Engine component initialized.")
            
        except Exception as e:
            logging.exception("Failed to initialize Rule Engine.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> RuleEngineArtifact:
        """
        Executes the deterministic rule evaluation workflow.

        Returns:
            RuleEngineArtifact: Paths to the monitoring report, trigger payload, and metadata.
        """
        try:
            logging.info("Starting rule evaluation and retraining decision logic.")
            start_time = time.time()

            # 1. Load Upstream Metrics and Baselines
            baseline_metrics = self._load_json(self.resolver_artifact.baseline_metrics_file_path)
            drift_report = self._load_json(self.drift_artifact.drift_report_file_path)
            performance_report = self._load_json(self.performance_artifact.performance_report_file_path)

            # 2. Evaluate Deterministic Conditions
            cond1_triggered, cond1_details = self._evaluate_prediction_drift(drift_report)
            cond2_triggered, cond2_details = self._evaluate_feature_drift(drift_report)
            cond3_triggered, cond3_details = self._evaluate_performance_degradation(
                performance_report, baseline_metrics
            )

            # 3. Consolidate Decision
            need_update = bool(cond1_triggered or cond2_triggered or cond3_triggered)

            # 4. Generate Audit Report
            audit_report = {
                "decision": {
                    "need_update": need_update,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "champion_run_id": self.resolver_artifact.champion_run_id
                },
                "evaluations": {
                    "condition_1_prediction_drift": cond1_details,
                    "condition_2_feature_drift": cond2_details,
                    "condition_3_performance_degradation": cond3_details
                }
            }

            # 5. Persist Artifacts
            execution_time = round(time.time() - start_time, 2)
            self._save_artifacts(audit_report, need_update, execution_time)

            # 6. Package and Return Artifact
            artifact = RuleEngineArtifact(
                monitoring_report_file_path=self.config.monitoring_report_file_path,
                need_update_file_path=self.config.need_update_file_path,
                need_update=need_update,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info(
                "Rule Engine execution completed successfully. Retraining required: %s", 
                need_update
            )
            return artifact

        except Exception as e:
            logging.exception("Critical Failure: Rule Engine run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # FILE I/O
    # ==========================================================
    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Safely loads a JSON artifact."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.exception(f"Failed to load JSON artifact at {file_path}")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DETERMINISTIC DECISION RULES
    # ==========================================================
    def _evaluate_prediction_drift(self, drift_report: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Condition 1: Critical Prediction Drift
        Triggered if the PSI for the predicted probability exceeds the configured threshold.
        """
        try:
            pred_drift = drift_report.get("prediction_drift", {}).get("predicted_probability", {})
            psi_score = pred_drift.get("psi_score", 0.0)
            threshold = self.config.prediction_drift_threshold_psi

            is_triggered = psi_score >= threshold

            details = {
                "is_triggered": is_triggered,
                "metric": "predicted_probability_psi",
                "actual_value": psi_score,
                "threshold": threshold
            }
            
            if is_triggered:
                logging.warning(f"Condition 1 Triggered: Prediction Drift PSI ({psi_score}) >= {threshold}")
            
            return is_triggered, details

        except Exception as e:
            logging.exception("Failed to evaluate Condition 1 (Prediction Drift).")
            raise CustomException(e, sys) from e

    def _evaluate_feature_drift(self, drift_report: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Condition 2: Critical Feature Drift
        Triggered if the PSI for at least N of the Top SHAP-ranked features exceeds the configured threshold.
        """
        try:
            feature_drift_dict = drift_report.get("feature_drift", {})
            threshold = self.config.feature_drift_threshold_psi
            min_features = self.config.min_drifted_features_for_retrain

            drifted_features = []
            for feature, data in feature_drift_dict.items():
                if data.get("psi_score", 0.0) >= threshold:
                    drifted_features.append(feature)

            drifted_count = len(drifted_features)
            is_triggered = drifted_count >= min_features

            details = {
                "is_triggered": is_triggered,
                "metric": "drifted_top_features_count",
                "actual_value": drifted_count,
                "threshold": min_features,
                "drifted_features_list": drifted_features,
                "psi_threshold_used": threshold
            }

            if is_triggered:
                logging.warning(
                    f"Condition 2 Triggered: {drifted_count} top features drifted, "
                    f"exceeding threshold of {min_features}."
                )

            return is_triggered, details

        except Exception as e:
            logging.exception("Failed to evaluate Condition 2 (Feature Drift).")
            raise CustomException(e, sys) from e

    def _evaluate_performance_degradation(
        self, 
        performance_report: Dict[str, Any], 
        baseline_metrics: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Condition 3: Severe Performance Degradation
        Triggered if the Brier Score on the matured lookback cohort has degraded by more 
        than the configured relative percentage against the training baseline Brier Score.
        Note: Brier Score is an error metric, so a higher score implies degradation.
        """
        try:
            # If the lookback period has not elapsed, the system skips this condition.
            if not performance_report.get("is_evaluated", False):
                logging.info("System immature for reactive evaluation. Skipping Condition 3.")
                return False, {
                    "is_triggered": False,
                    "reason": "INSUFFICIENT_LOOKBACK_MATURITY"
                }

            current_brier = performance_report.get("metrics", {}).get("brier_score")
            baseline_brier = baseline_metrics.get("global_metrics", {}).get("brier_score")

            if current_brier is None or baseline_brier is None:
                raise ValueError("Brier Score missing from performance report or baselines.")

            # Calculate degradation threshold boundary
            degradation_factor = self.config.brier_degradation_threshold
            degradation_boundary = baseline_brier * degradation_factor

            is_triggered = current_brier > degradation_boundary

            details = {
                "is_triggered": is_triggered,
                "metric": "brier_score_degradation",
                "actual_value": current_brier,
                "baseline_value": baseline_brier,
                "degradation_boundary": degradation_boundary,
                "relative_degradation_threshold_factor": degradation_factor
            }

            if is_triggered:
                logging.warning(
                    f"Condition 3 Triggered: Current Brier Score ({current_brier:.4f}) "
                    f"exceeds degradation boundary ({degradation_boundary:.4f})."
                )

            return is_triggered, details

        except Exception as e:
            logging.exception("Failed to evaluate Condition 3 (Performance Degradation).")
            raise CustomException(e, sys) from e

    # ==========================================================
    # ARTIFACT PERSISTENCE
    # ==========================================================
    def _save_artifacts(self, audit_report: Dict[str, Any], need_update: bool, execution_time: float) -> None:
        """
        Saves the consolidated monitoring report, the deterministic trigger payload, and metadata.
        """
        try:
            # 1. Save Full Audit Report
            write_json_file(file_path=self.config.monitoring_report_file_path, content=audit_report)
            logging.info(f"Consolidated monitoring audit report saved to: {self.config.monitoring_report_file_path}")

            # 2. Save Deterministic Trigger Payload
            trigger_payload = {"need_update": need_update}
            write_json_file(file_path=self.config.need_update_file_path, content=trigger_payload)
            logging.debug(f"Retraining trigger payload saved to: {self.config.need_update_file_path}")

            # 3. Save Operational Metadata
            metadata: Dict[str, Any] = {
                "pipeline_stage": "Monitoring Rule Engine",
                "execution_time_seconds": execution_time,
                "decision_rendered": need_update,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            
        except Exception as e:
            logging.exception("Failed to write Rule Engine artifacts to disk.")
            raise CustomException(e, sys) from e