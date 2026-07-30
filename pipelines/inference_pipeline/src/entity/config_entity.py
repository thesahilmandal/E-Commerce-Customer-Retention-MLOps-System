import os
import sys
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from pipelines.inference_pipeline.src.core.context import InferencePipelineContext


@dataclass(frozen=True)
class ModelLoaderConfig:
    """
    Configuration entity for the Model Loader component.
    
    Provides the local directory structure for storing downloaded artifacts 
    and the single S3 URI pointer to resolve the current production model.
    It intentionally excludes direct S3 paths to the model or its schemas,
    as these are dynamically resolved at runtime from the state pointer.
    """
    model_loader_root_dir: str
    model_file_path: str
    schema_file_path: str
    baseline_metrics_file_path: str
    reference_distributions_file_path: str
    metadata_file_path: str
    s3_pointer_uri: str

    @classmethod
    def from_context(cls, context: InferencePipelineContext) -> "ModelLoaderConfig":
        try:
            sys_cfg = context.config.get_system_config()
            comp_cfg = context.config.get_components_config()["model_loader"]
            cloud_cfg = context.config.get_cloud_storage_config()

            root_dir = os.path.join(
                sys_cfg["artifact_dir"],
                sys_cfg["pipeline_name"],
                context.run_id,
                comp_cfg["dir_name"]
            )
            os.makedirs(root_dir, exist_ok=True)

            registry_cfg = cloud_cfg["model_registry"]
            s3_pointer_uri = (
                f"s3://{registry_cfg['bucket_name']}/"
                f"{registry_cfg['registry_dir']}/"
                f"{registry_cfg['state_dir']}/"
                f"{registry_cfg['pointer_file']}"
            )

            return cls(
                model_loader_root_dir=root_dir,
                model_file_path=os.path.join(root_dir, comp_cfg["model_file"]),
                schema_file_path=os.path.join(root_dir, comp_cfg["schema_file"]),
                baseline_metrics_file_path=os.path.join(root_dir, comp_cfg["baseline_metrics_file"]),
                reference_distributions_file_path=os.path.join(root_dir, comp_cfg["reference_distributions_file"]),
                metadata_file_path=os.path.join(root_dir, comp_cfg["metadata_file"]),
                s3_pointer_uri=s3_pointer_uri
            )
        except Exception as e:
            logging.exception("Failed to initialize ModelLoaderConfig from context.")
            raise CustomException(e, sys) from e


@dataclass(frozen=True)
class FeatureMatrixBuilderConfig:
    """Configuration entity for the Feature Matrix Builder component."""
    builder_root_dir: str
    feature_matrix_file_path: str
    schema_file_path: str
    metadata_file_path: str
    s3_data_lake_uri: str
    snapshot_date: str

    @classmethod
    def from_context(cls, context: InferencePipelineContext) -> "FeatureMatrixBuilderConfig":
        try:
            sys_cfg = context.config.get_system_config()
            comp_cfg = context.config.get_components_config()["feature_matrix_builder"]
            cloud_cfg = context.config.get_cloud_storage_config()

            root_dir = os.path.join(
                sys_cfg["artifact_dir"],
                sys_cfg["pipeline_name"],
                context.run_id,
                comp_cfg["dir_name"]
            )
            os.makedirs(root_dir, exist_ok=True)

            bucket = cloud_cfg["data_lake"]["database_name"]
            bronze_dir = cloud_cfg["data_lake"]["bronze_dir"]
            s3_data_lake_uri = f"s3://{bucket}/{bronze_dir}"

            # Dynamically resolve T-1 Snapshot Date anchored in UTC
            snapshot_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

            return cls(
                builder_root_dir=root_dir,
                feature_matrix_file_path=os.path.join(root_dir, comp_cfg["feature_matrix_file"]),
                schema_file_path=os.path.join(root_dir, comp_cfg["schema_file"]),
                metadata_file_path=os.path.join(root_dir, comp_cfg["metadata_file"]),
                s3_data_lake_uri=s3_data_lake_uri,
                snapshot_date=snapshot_date
            )
        except Exception as e:
            logging.exception("Failed to initialize FeatureMatrixBuilderConfig from context.")
            raise CustomException(e, sys) from e


@dataclass(frozen=True)
class InferenceValidatorConfig:
    """Configuration entity for the Inference Validator component."""
    validator_root_dir: str
    report_file_path: str
    metadata_file_path: str

    @classmethod
    def from_context(cls, context: InferencePipelineContext) -> "InferenceValidatorConfig":
        try:
            sys_cfg = context.config.get_system_config()
            comp_cfg = context.config.get_components_config()["inference_validator"]

            root_dir = os.path.join(
                sys_cfg["artifact_dir"],
                sys_cfg["pipeline_name"],
                context.run_id,
                comp_cfg["dir_name"]
            )
            os.makedirs(root_dir, exist_ok=True)

            return cls(
                validator_root_dir=root_dir,
                report_file_path=os.path.join(root_dir, comp_cfg["report_file"]),
                metadata_file_path=os.path.join(root_dir, comp_cfg["metadata_file"])
            )
        except Exception as e:
            logging.exception("Failed to initialize InferenceValidatorConfig from context.")
            raise CustomException(e, sys) from e


@dataclass(frozen=True)
class ReportGeneratorConfig:
    """Configuration entity for the Report Generator component."""
    generator_root_dir: str
    run_id: str
    probability_threshold: float
    system_columns_to_drop: List[str]
    csv_report_path: str
    telemetry_log_path: str
    metadata_file_path: str

    def __post_init__(self):
        if not (0.0 <= self.probability_threshold <= 1.0):
            raise ValueError(f"probability_threshold must be between 0 and 1, got {self.probability_threshold}")

    @classmethod
    def from_context(cls, context: InferencePipelineContext) -> "ReportGeneratorConfig":
        try:
            sys_cfg = context.config.get_system_config()
            comp_cfg = context.config.get_components_config()["report_generator"]
            biz_cfg = context.config.get_business_logic_config()

            root_dir = os.path.join(
                sys_cfg["artifact_dir"],
                sys_cfg["pipeline_name"],
                context.run_id,
                comp_cfg["dir_name"]
            )
            os.makedirs(root_dir, exist_ok=True)

            return cls(
                generator_root_dir=root_dir,
                run_id=context.run_id,
                probability_threshold=float(biz_cfg["churn_probability_threshold"]),
                system_columns_to_drop=biz_cfg.get("system_columns_to_drop", []),
                csv_report_path=os.path.join(root_dir, comp_cfg["csv_report_file"]),
                telemetry_log_path=os.path.join(root_dir, comp_cfg["telemetry_file"]),
                metadata_file_path=os.path.join(root_dir, comp_cfg["metadata_file"])
            )
        except Exception as e:
            logging.exception("Failed to initialize ReportGeneratorConfig from context.")
            raise CustomException(e, sys) from e


@dataclass(frozen=True)
class ReportPublisherConfig:
    """Configuration entity for the Report Publisher component."""
    publisher_root_dir: str
    run_id: str
    s3_business_reports_base_uri: str
    s3_telemetry_logs_base_uri: str
    s3_metadata_base_uri: str
    metadata_file_path: str

    @classmethod
    def from_context(cls, context: InferencePipelineContext) -> "ReportPublisherConfig":
        try:
            sys_cfg = context.config.get_system_config()
            comp_cfg = context.config.get_components_config()["report_publisher"]
            cloud_cfg = context.config.get_cloud_storage_config()

            root_dir = os.path.join(
                sys_cfg["artifact_dir"],
                sys_cfg["pipeline_name"],
                context.run_id,
                comp_cfg["dir_name"]
            )
            os.makedirs(root_dir, exist_ok=True)

            bucket = cloud_cfg["model_registry"]["bucket_name"]
            base_art = cloud_cfg["inference_outputs"]["base_artifact_dir"]
            
            s3_business_base = f"s3://{bucket}/{base_art}/{cloud_cfg['inference_outputs']['business_reports_dir']}"
            s3_telemetry_base = f"s3://{bucket}/{base_art}/{cloud_cfg['inference_outputs']['mlops_telemetry_dir']}"
            s3_metadata_base = f"s3://{bucket}/{base_art}/{cloud_cfg['inference_outputs']['metadata_dir']}"

            return cls(
                publisher_root_dir=root_dir,
                run_id=context.run_id,
                s3_business_reports_base_uri=s3_business_base,
                s3_telemetry_logs_base_uri=s3_telemetry_base,
                s3_metadata_base_uri=s3_metadata_base,
                metadata_file_path=os.path.join(root_dir, comp_cfg["metadata_file"])
            )
        except Exception as e:
            logging.exception("Failed to initialize ReportPublisherConfig from context.")
            raise CustomException(e, sys) from e