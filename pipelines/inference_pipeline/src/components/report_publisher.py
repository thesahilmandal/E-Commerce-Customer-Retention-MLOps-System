import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from pipelines.inference_pipeline.src.core.context import InferencePipelineContext
from pipelines.inference_pipeline.src.entity.config_entity import ReportPublisherConfig
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    ReportPublisherArtifact,
    ReportGeneratorArtifact,
    ModelLoaderArtifact,
    FeatureMatrixBuilderArtifact,
    InferenceValidatorArtifact
)


class ReportPublisher:
    """
    Report Publisher Component.

    Responsibilities:
    - Validates local existence and integrity of all generated inference outputs.
    - Aggregates stage-level metadata from all upstream components into a single,
      comprehensive Master Inference Ledger.
    - Constructs standard Hive-partitioned S3 target URIs (year/month/day).
    - Streams the Business Report (CSV), MLOps Telemetry Log (Parquet), and Master 
      Inference Ledger (JSON) to their designated Cloud Data Lake directories.
    - Enforces idempotent overwrites for repeatable, scheduled batch execution.
    """

    def __init__(self, context: InferencePipelineContext) -> None:
        """
        Initializes the Report Publisher component.

        Args:
            context (InferencePipelineContext): The centralized execution context.
        """
        try:
            self.context = context
            self.config = ReportPublisherConfig.from_context(context)
            logging.info("Inference Pipeline: Report Publisher component initialized.")
        except Exception as e:
            logging.exception("Failed to initialize Report Publisher component.")
            raise CustomException(e, sys) from e

    def run(
        self,
        model_loader_artifact: ModelLoaderArtifact,
        feature_matrix_artifact: FeatureMatrixBuilderArtifact,
        validator_artifact: InferenceValidatorArtifact,
        report_generator_artifact: ReportGeneratorArtifact
    ) -> ReportPublisherArtifact:
        """
        Executes the master ledger consolidation and cloud publication workflow.

        Args:
            model_loader_artifact (ModelLoaderArtifact): Artifact from ModelLoader.
            feature_matrix_artifact (FeatureMatrixBuilderArtifact): Artifact from FeatureMatrixBuilder.
            validator_artifact (InferenceValidatorArtifact): Artifact from InferenceValidator.
            report_generator_artifact (ReportGeneratorArtifact): Artifact from ReportGenerator.

        Returns:
            ReportPublisherArtifact: Artifact containing published cloud URIs and master ledger path.
        """
        try:
            logging.info("Starting Report Publication Sequence.")
            start_time = time.time()

            # 1. Pre-publication Validation
            self._validate_local_artifacts(report_generator_artifact=report_generator_artifact)

            # 2. Construct Hive-partitioned S3 Target URIs
            partition_suffix = self._get_partition_suffix()
            s3_business_uri = f"{self.config.s3_business_reports_base_uri}/{partition_suffix}/churn_predictions.csv"
            s3_telemetry_uri = f"{self.config.s3_telemetry_logs_base_uri}/{partition_suffix}/telemetry.parquet"
            s3_metadata_uri = f"{self.config.s3_metadata_base_uri}/{partition_suffix}/run_{self.config.run_id}_metadata.json"

            # 3. Consolidate Master Inference Ledger
            master_ledger_path = self._compile_master_inference_ledger(
                model_loader_artifact=model_loader_artifact,
                feature_matrix_artifact=feature_matrix_artifact,
                validator_artifact=validator_artifact,
                report_generator_artifact=report_generator_artifact,
                execution_time=round(time.time() - start_time, 2),
                s3_business_uri=s3_business_uri,
                s3_telemetry_uri=s3_telemetry_uri
            )

            # 4. Upload Artifacts to Cloud Storage
            self._publish_artifacts_to_s3(
                local_csv_path=report_generator_artifact.csv_report_path,
                s3_csv_uri=s3_business_uri,
                local_telemetry_path=report_generator_artifact.telemetry_log_path,
                s3_telemetry_uri=s3_telemetry_uri,
                local_metadata_path=master_ledger_path,
                s3_metadata_uri=s3_metadata_uri
            )

            # 5. Package Artifact
            artifact = ReportPublisherArtifact(
                published_business_report_uri=s3_business_uri,
                published_telemetry_log_uri=s3_telemetry_uri,
                metadata_file_path=master_ledger_path
            )

            logging.info("Report Publication completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Critical Failure inside Report Publisher execution routine.")
            raise CustomException(e, sys) from e

    def _validate_local_artifacts(self, report_generator_artifact: ReportGeneratorArtifact) -> None:
        """Validates that all required upstream local artifacts exist before publishing."""
        try:
            if not os.path.exists(report_generator_artifact.csv_report_path):
                raise FileNotFoundError(f"Business CSV report not found at {report_generator_artifact.csv_report_path}")

            if not os.path.exists(report_generator_artifact.telemetry_log_path):
                raise FileNotFoundError(f"Telemetry log Parquet not found at {report_generator_artifact.telemetry_log_path}")

            logging.debug("All required local artifact files verified successfully.")
        except Exception as e:
            logging.exception("Pre-publication validation failed.")
            raise CustomException(e, sys) from e

    def _get_partition_suffix(self) -> str:
        """Generates standard UTC Hive partition path suffix (year=YYYY/month=MM/day=DD)."""
        now = datetime.now(timezone.utc)
        return f"year={now.year}/month={now.month:02d}/day={now.day:02d}"

    def _compile_master_inference_ledger(
        self,
        model_loader_artifact: ModelLoaderArtifact,
        feature_matrix_artifact: FeatureMatrixBuilderArtifact,
        validator_artifact: InferenceValidatorArtifact,
        report_generator_artifact: ReportGeneratorArtifact,
        execution_time: float,
        s3_business_uri: str,
        s3_telemetry_uri: str
    ) -> str:
        """
        Aggregates stage-level metadata files from all pipeline stages into a unified Master Ledger.
        """
        try:
            logging.info("Compiling Master Inference Ledger.")

            stage_metadata = {}
            metadata_sources = {
                "model_loader": model_loader_artifact.metadata_file_path,
                "feature_matrix_builder": feature_matrix_artifact.metadata_file_path,
                "report_generator": report_generator_artifact.metadata_file_path
            }

            for stage_name, path in metadata_sources.items():
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        stage_metadata[stage_name] = json.load(f)
                else:
                    stage_metadata[stage_name] = {"warning": f"Metadata file not found at {path}"}

            master_ledger: Dict[str, Any] = {
                "inference_run_id": self.context.run_id,
                "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pipeline_status": "SUCCESS",
                "publisher_execution_time_seconds": execution_time,
                "published_artifact_lineage": {
                    "business_report_s3_uri": s3_business_uri,
                    "telemetry_log_s3_uri": s3_telemetry_uri
                },
                "stage_telemetry": stage_metadata
            }

            with open(self.config.metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(master_ledger, f, indent=4)

            logging.debug("Master Inference Ledger written locally to: %s", self.config.metadata_file_path)
            return self.config.metadata_file_path

        except Exception as e:
            logging.exception("Failed to compile Master Inference Ledger.")
            raise CustomException(e, sys) from e

    def _publish_artifacts_to_s3(
        self,
        local_csv_path: str,
        s3_csv_uri: str,
        local_telemetry_path: str,
        s3_telemetry_uri: str,
        local_metadata_path: str,
        s3_metadata_uri: str
    ) -> None:
        """Streams all local artifacts to their respective Hive-partitioned S3 paths."""
        try:
            logging.info("Uploading Business Report to S3: %s", s3_csv_uri)
            self.context.s3_sync.upload_file(local_path=local_csv_path, s3_uri=s3_csv_uri)

            logging.info("Uploading MLOps Telemetry Log to S3: %s", s3_telemetry_uri)
            self.context.s3_sync.upload_file(local_path=local_telemetry_path, s3_uri=s3_telemetry_uri)

            logging.info("Uploading Master Inference Ledger to S3: %s", s3_metadata_uri)
            self.context.s3_sync.upload_file(local_path=local_metadata_path, s3_uri=s3_metadata_uri)

            logging.info("All inference artifacts successfully published to Cloud Data Lake.")

        except Exception as e:
            logging.exception("Failed to publish artifacts to S3.")
            raise CustomException(e, sys) from e