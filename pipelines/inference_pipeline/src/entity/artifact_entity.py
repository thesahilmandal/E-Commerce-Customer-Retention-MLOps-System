from dataclasses import dataclass


@dataclass(frozen=True)
class ModelLoadingArtifact:
    """
    Artifact containing the local paths to the dynamically resolved production model,
    its structural schema contract, and the specific registry run ID.
    """
    model_file_path: str
    schema_file_path: str
    champion_run_id: str


@dataclass(frozen=True)
class FeatureMatrixBuilderArtifact:
    """
    Artifact containing local paths to the generated feature matrix required for inference,
    its schema definition, and the temporal snapshot bound used.
    """
    feature_matrix_file_path: str
    schema_file_path: str
    snapshot_date: str


@dataclass(frozen=True)
class InferenceValidationArtifact:
    """
    Artifact containing the structural data contract validation result and the local
    path to the generated validation audit report.
    """
    is_valid: bool
    validation_report_file_path: str


@dataclass(frozen=True)
class ReportGenerationArtifact:
    """
    Artifact containing the definitive local paths for the generated business-facing
    CSV report and the engineering Parquet telemetry log.
    """
    business_report_file_path: str
    telemetry_log_file_path: str


@dataclass(frozen=True)
class ReportPublishingArtifact:
    """
    Artifact representing the final state of the Inference Pipeline. Contains the
    cloud URIs of the securely vaulted reports and the central metadata manifest.
    """
    published_business_report_uri: str
    published_telemetry_log_uri: str
    published_metadata_manifest_uri: str