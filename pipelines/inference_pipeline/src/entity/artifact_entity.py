from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceModelLoaderArtifact:
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
            "\nInferenceModelLoaderArtifact(\n"
            f"  model_file_path = {self.model_file_path}\n"
            f"  schema_file_path = {self.schema_file_path}\n"
            f"  champion_run_id = {self.champion_run_id}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class InferenceInputFeatureMatrixBuilderArtifact:
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
            "\nInferenceInputFeatureMatrixBuilderArtifact(\n"
            f"  feature_matrix_file_path = {self.feature_matrix_file_path}\n"
            f"  schema_file_path = {self.schema_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  snapshot_date = {self.snapshot_date}\n"
            ")"
        )


@dataclass(frozen=True)
class InferenceValidatorArtifact:
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
class InferenceReportGeneratorArtifact:
    """
    Artifact containing the definitive paths for the three generated Publisher artifacts:
    the business-facing CSV report, the engineering Parquet telemetry log, and the JSON metadata.
    """
    csv_report_path: str
    telemetry_log_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nInferenceReportGeneratorArtifact(\n"
            f"  csv_report_path = {self.csv_report_path}\n"
            f"  telemetry_log_path = {self.telemetry_log_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class InferenceReportPublisherArtifact:
    """
    Artifact representing the final state of the Inference Pipeline. Contains the
    cloud URIs of the securely vaulted reports and the local path to the upload manifest.
    """
    published_business_report_uri: str
    published_telemetry_log_uri: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nInferenceReportPublisherArtifact(\n"
            f"  published_business_report_uri = {self.published_business_report_uri}\n"
            f"  published_telemetry_log_uri = {self.published_telemetry_log_uri}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )