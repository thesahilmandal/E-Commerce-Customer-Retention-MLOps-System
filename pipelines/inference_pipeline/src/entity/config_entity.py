from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModelLoaderConfig:
    """
    Configuration for the Model Loader component.
    Specifies S3 registry URIs and local target paths for deployment artifacts.
    """
    s3_pointer_uri: str
    local_model_file_path: str
    local_schema_file_path: str


@dataclass(frozen=True)
class FeatureMatrixBuilderConfig:
    """
    Configuration for the Feature Matrix Builder component.
    Specifies the S3 data lake URI, temporal snapshot bound, and local output paths.
    """
    s3_data_lake_uri: str
    snapshot_date: str
    local_feature_matrix_file_path: str
    local_schema_file_path: str


@dataclass(frozen=True)
class InferenceValidatorConfig:
    """
    Configuration for the Inference Validator component.
    Specifies the local path for the structural data contract audit report.
    """
    local_validation_report_path: str


@dataclass(frozen=True)
class ReportGeneratorConfig:
    """
    Configuration for the Report Generator component.
    Contains business logic parameters and local output paths for scoring artifacts.
    """
    probability_threshold: float
    system_columns_to_drop: List[str]
    local_business_report_path: str
    local_telemetry_log_path: str


@dataclass(frozen=True)
class ReportPublisherConfig:
    """
    Configuration for the Report Publisher component.
    Specifies S3 destination URIs, execution metadata, and local manifest paths.
    """
    run_id: str
    target_date: str
    local_metadata_manifest_path: str
    s3_business_reports_base_uri: str
    s3_telemetry_logs_base_uri: str
    s3_metadata_manifest_base_uri: str