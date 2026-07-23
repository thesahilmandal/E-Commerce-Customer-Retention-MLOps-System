import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

from pipelines.inference_pipeline.src.core.context import InferenceContext
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    ReportGenerationArtifact,
    ReportPublishingArtifact,
)
from pipelines.inference_pipeline.src.entity.config_entity import ReportPublisherConfig
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class ReportPublisher:
    """
    Report Publisher Component.

    Responsible for securely publishing the generated business reports, operational 
    telemetry logs, and centralized metadata manifests to their final Amazon S3 
    destinations using an idempotent, Hive-partitioned directory structure.
    """

    def __init__(self, config: ReportPublisherConfig, context: InferenceContext) -> None:
        """
        Initializes the ReportPublisher component.

        Args:
            config (ReportPublisherConfig): Configuration containing target S3 URIs.
            context (InferenceContext): Centralized pipeline execution context.
        """
        try:
            self.config = config
            self.context = context
            self.s3_sync = S3Sync()

            os.makedirs(os.path.dirname(self.config.local_metadata_manifest_path), exist_ok=True)
            logging.info("ReportPublisher component initialized.")
        except Exception as e:
            logging.exception("Failed to initialize ReportPublisher component.")
            raise CustomException(e, sys) from e

    def run(self, report_artifact: ReportGenerationArtifact) -> ReportPublishingArtifact:
        """
        Publishes all local artifacts to S3 and generates the final execution manifest.

        Args:
            report_artifact (ReportGenerationArtifact): Artifacts from the scoring phase.

        Returns:
            ReportPublishingArtifact: S3 URIs of the successfully published artifacts.
        """
        try:
            logging.info("Starting artifact publishing to Amazon S3.")
            start_time = time.time()

            # 1. Resolve Hive partitions
            target_date = datetime.strptime(self.config.target_date, "%Y-%m-%d")
            year_part = f"year={target_date.strftime('%Y')}"
            month_part = f"month={target_date.strftime('%m')}"
            day_part = f"day={target_date.strftime('%d')}"

            # 2. Build Target S3 URIs
            business_s3_uri = self._build_partitioned_uri(
                self.config.s3_business_reports_base_uri,
                year_part, month_part, day_part,
                "churn_predictions.csv"
            )
            
            telemetry_s3_uri = self._build_partitioned_uri(
                self.config.s3_telemetry_logs_base_uri,
                year_part, month_part, day_part,
                "telemetry.parquet"
            )
            
            manifest_s3_uri = self._build_partitioned_uri(
                self.config.s3_metadata_manifest_base_uri,
                year_part, month_part, day_part,
                f"{self.config.run_id}_metadata.json"
            )

            # 3. Publish Artifacts
            logging.debug("Publishing business report to: %s", business_s3_uri)
            self.s3_sync.upload_file(
                local_path=report_artifact.business_report_file_path,
                s3_uri=business_s3_uri
            )

            logging.debug("Publishing telemetry log to: %s", telemetry_s3_uri)
            self.s3_sync.upload_file(
                local_path=report_artifact.telemetry_log_file_path,
                s3_uri=telemetry_s3_uri
            )

            # 4. Generate and Publish Metadata Manifest
            # Record publisher telemetry before writing the manifest
            execution_time = round(time.time() - start_time, 2)
            telemetry: Dict[str, Any] = {
                "published_business_report_uri": business_s3_uri,
                "published_telemetry_log_uri": telemetry_s3_uri,
                "published_metadata_manifest_uri": manifest_s3_uri,
                "execution_time_seconds": execution_time,
            }
            self.context.add_metadata("ReportPublisher", telemetry)

            logging.info("Writing centralized metadata manifest to local workspace.")
            write_json_file(self.config.local_metadata_manifest_path, self.context.metadata_ledger)

            logging.debug("Publishing metadata manifest to: %s", manifest_s3_uri)
            self.s3_sync.upload_file(
                local_path=self.config.local_metadata_manifest_path,
                s3_uri=manifest_s3_uri
            )

            # 5. Return Output Artifact
            artifact = ReportPublishingArtifact(
                published_business_report_uri=business_s3_uri,
                published_telemetry_log_uri=telemetry_s3_uri,
                published_metadata_manifest_uri=manifest_s3_uri,
            )

            logging.info("ReportPublisher execution completed successfully.")
            return artifact

        except Exception as e:
            logging.exception("Critical Failure in ReportPublisher component.")
            raise CustomException(e, sys) from e

    @staticmethod
    def _build_partitioned_uri(
        base_uri: str, year: str, month: str, day: str, filename: str
    ) -> str:
        """
        Safely constructs a Hive-partitioned S3 URI.
        """
        # Strip trailing slashes to prevent double slashes in S3 keys
        clean_base = base_uri.rstrip("/")
        uri_parts = [clean_base, year, month, day, filename]
        return "/".join(uri_parts)