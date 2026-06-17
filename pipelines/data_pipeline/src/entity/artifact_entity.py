from dataclasses import dataclass


@dataclass(frozen=True)
class DataExtractorArtifact:
    """
    Artifact containing paths for the Extractor component outputs.
    """
    raw_data_dir_path: str
    raw_data_schema_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataPipelineExtractorArtifact(\n"
            f"  raw_data_dir_path = {self.raw_data_dir_path}\n"
            f"  raw_data_schema_file_path = {self.raw_data_schema_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class DataValidationArtifact:
    """
    Artifact containing the validation report path and boolean status.
    """
    report_file_path: str
    is_valid: bool

    def __str__(self) -> str:
        return (
            "\nDataPipelineValidatorArtifact(\n"
            f"  report_file_path = {self.report_file_path}\n"
            f"  is_valid = {self.is_valid}\n"
            ")"
        )


@dataclass(frozen=True)
class DataTransformationArtifact:
    """
    Artifact containing paths for the Transformer component outputs.
    Contains the path to the definitive out-of-core generated Parquet file.
    """
    transformed_data_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataPipelineTransformerArtifact(\n"
            f"  transformed_data_file_path = {self.transformed_data_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class DataLoadingArtifact:
    """
    Artifact containing remote (S3) path for the exported feature store
    and local path for the loader's telemetry metadata.
    """
    s3_file_uri: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataPipelineLoaderArtifact(\n"
            f"  s3_file_uri = {self.s3_file_uri}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )