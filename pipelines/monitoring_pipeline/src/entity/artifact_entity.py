from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineAndTelemetryResolverArtifact:
    """
    Artifact containing local paths to the resolved champion model baselines, 
    current proactive telemetry, historical reactive telemetry, and matured 
    ground-truth labels required for downstream statistical monitoring.
    """
    champion_run_id: str
    baseline_metrics_file_path: str
    reference_distributions_file_path: str
    shap_importance_file_path: str
    current_telemetry_file_path: str
    lookback_telemetry_file_path: str
    lookback_labels_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nBaselineAndTelemetryResolverArtifact(\n"
            f"  champion_run_id = {self.champion_run_id}\n"
            f"  baseline_metrics_file_path = {self.baseline_metrics_file_path}\n"
            f"  reference_distributions_file_path = {self.reference_distributions_file_path}\n"
            f"  shap_importance_file_path = {self.shap_importance_file_path}\n"
            f"  current_telemetry_file_path = {self.current_telemetry_file_path}\n"
            f"  lookback_telemetry_file_path = {self.lookback_telemetry_file_path}\n"
            f"  lookback_labels_file_path = {self.lookback_labels_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class StatisticalDriftCalculatorArtifact:
    """
    Artifact containing the path to the generated label-independent drift report
    (PSI metrics for features and predictions) and the component metadata.
    """
    drift_report_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nStatisticalDriftCalculatorArtifact(\n"
            f"  drift_report_file_path = {self.drift_report_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class PerformanceEvaluatorArtifact:
    """
    Artifact containing the path to the generated label-dependent performance report
    (Brier Score, Log Loss, and Realized ROI) and the component metadata.
    """
    performance_report_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nPerformanceEvaluatorArtifact(\n"
            f"  performance_report_file_path = {self.performance_report_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class RuleEngineArtifact:
    """
    Artifact representing the final state of the Monitoring Pipeline. Contains the 
    paths to the consolidated immutable audit report, the deterministic trigger 
    payload, the trigger boolean itself, and the component metadata.
    """
    monitoring_report_file_path: str
    need_update_file_path: str
    need_update: bool
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nRuleEngineArtifact(\n"
            f"  monitoring_report_file_path = {self.monitoring_report_file_path}\n"
            f"  need_update_file_path = {self.need_update_file_path}\n"
            f"  need_update = {self.need_update}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )