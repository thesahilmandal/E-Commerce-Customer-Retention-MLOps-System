from dataclasses import dataclass


@dataclass(frozen=True)
class ModelLoaderArtifact:
    """
    Artifact containing the local paths to the dynamically resolved production model,
    its structural schema contract, the specific registry run ID, and component metadata.
    """
    model_file_path: str
    schema_file_path: str
    champion_run_id: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nModelLoaderArtifact(\n"
            f"  model_file_path = {self.model_file_path}\n"
            f"  schema_file_path = {self.schema_file_path}\n"
            f"  champion_run_id = {self.champion_run_id}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class FeatureMatrixBuilderArtifact:
    """
    Artifact containing local paths to the generated feature matrix required for inference,
    its schema definition, pipeline metadata, and the temporal snapshot bound used.
    """
    feature_matrix_file_path: str
    schema_file_path: str
    metadata_file_path: str
    snapshot_date: str

    def __str__(self) -> str:
        return (
            "\nFeatureMatrixBuilderArtifact(\n"
            f"  feature_matrix_file_path = {self.feature_matrix_file_path}\n"
            f"  schema_file_path = {self.schema_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  snapshot_date = {self.snapshot_date}\n"
            ")"
        )


@dataclass(frozen=True)
class InferenceValidatorArtifact:
    """
    Artifact representing the state of the structural data contract verification.
    Determines whether downstream prediction scoring is authorized to proceed.
    """
    is_valid: bool
    report_file_path: str

    def __str__(self) -> str:
        return (
            "\nInferenceValidatorArtifact(\n"
            f"  is_valid = {self.is_valid}\n"
            f"  report_file_path = {self.report_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ReportGeneratorArtifact:
    """
    Artifact containing the definitive local paths for the generated Publisher artifacts:
    the business-facing CSV report, the engineering Parquet telemetry log, and the metadata.
    """
    csv_report_path: str
    telemetry_log_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nReportGeneratorArtifact(\n"
            f"  csv_report_path = {self.csv_report_path}\n"
            f"  telemetry_log_path = {self.telemetry_log_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ReportPublisherArtifact:
    """
    Artifact representing the final state of the Inference Pipeline. Contains the
    cloud URIs of the published artifacts and the local path to the Master Inference Ledger.
    """
    published_business_report_uri: str
    published_telemetry_log_uri: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nReportPublisherArtifact(\n"
            f"  published_business_report_uri = {self.published_business_report_uri}\n"
            f"  published_telemetry_log_uri = {self.published_telemetry_log_uri}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )