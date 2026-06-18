import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any

from pipelines.inference_pipeline.src.entity.config_entity import ReportPublishingConfig
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    InferenceReportGeneratorArtifact,
    InferenceReportPublisherArtifact
)
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class ReportPublishing:
    """
    Report Publisher Component.

    Responsibilities:
    - Act as the delivery layer for the Inference Pipeline.
    - Validate the existence of local artifacts produced by the Report Generator.
    - Determine the dynamic Hive-style partition path based on the current UTC execution date.
    - Upload the Customer Churn CSV Report to the designated business S3 bucket.
    - Upload the Telemetry Log Parquet file to the designated MLOps S3 bucket.
    - Generate a final `metadata.json` shipping manifest to ensure end-to-end traceability.
    """

    def __init__(
        self,
        config: ReportPublishingConfig,
        generator_artifact: InferenceReportGeneratorArtifact
    ) -> None:
        """
        Initializes the Report Publisher component.
        """
        try:
            self.config = config
            self.generator_artifact = generator_artifact
            self.s3_sync = S3Sync()

            logging.info("Inference Pipeline: Report Publisher component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Report Publisher component.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> InferenceReportPublisherArtifact:
        """
        Executes the file validation, partitioning, S3 uploading, and metadata generation workflow.

        Returns:
            InferenceReportPublisherArtifact: Artifact containing the final S3 URIs.
        """
        try:
            logging.info("Starting S3 Artifact Publishing Sequence.")
            start_time = time.time()

            # 1. Validate Input Artifacts
            self._validate_artifacts_exist()

            # 2. Determine Hive Partition Key
            partition_suffix, partition_dict = self._get_hive_partition_suffix()

            # 3. Publish Business Artifact (CSV)
            published_csv_uri = self._publish_to_s3(
                local_path=self.generator_artifact.csv_report_path,
                base_s3_uri=self.config.s3_business_reports_base_uri,
                partition_suffix=partition_suffix,
                file_name=os.path.basename(self.generator_artifact.csv_report_path)
            )

            # 4. Publish Engineering Artifact (Parquet)
            published_parquet_uri = self._publish_to_s3(
                local_path=self.generator_artifact.telemetry_log_path,
                base_s3_uri=self.config.s3_telemetry_logs_base_uri,
                partition_suffix=partition_suffix,
                file_name=os.path.basename(self.generator_artifact.telemetry_log_path)
            )

            # 5. Generate Shipping Manifest Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(
                published_csv_uri=published_csv_uri,
                published_parquet_uri=published_parquet_uri,
                partition_dict=partition_dict,
                execution_time=execution_time
            )

            # 6. Package Artifact
            artifact = InferenceReportPublisherArtifact(
                published_business_report_uri=published_csv_uri,
                published_telemetry_log_uri=published_parquet_uri,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("Report Publisher execution completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Critical Failure inside Report Publisher execution routine.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # VALIDATION & PREPARATION
    # ==========================================================
    def _validate_artifacts_exist(self) -> None:
        """
        Ensures the local files promised by the Report Generator actually exist on disk
        before attempting to open network connections.
        """
        try:
            logging.debug("Validating existence of local artifacts prior to upload.")
            
            if not os.path.exists(self.generator_artifact.csv_report_path):
                raise FileNotFoundError(f"Business report not found at: {self.generator_artifact.csv_report_path}")
            
            if not os.path.exists(self.generator_artifact.telemetry_log_path):
                raise FileNotFoundError(f"Telemetry log not found at: {self.generator_artifact.telemetry_log_path}")

            logging.info("All required local artifacts verified.")

        except Exception as e:
            logging.exception("Artifact validation failed.")
            raise CustomException(e, sys) from e

    def _get_hive_partition_suffix(self) -> tuple[str, Dict[str, str]]:
        """
        Calculates the Hive partition string based on the current UTC date.
        
        Returns:
            tuple[str, Dict[str, str]]: The URI suffix and a dictionary of the extracted keys for logging.
        """
        try:
            now = datetime.now(timezone.utc)
            year = str(now.year)
            month = f"{now.month:02d}"
            day = f"{now.day:02d}"

            partition_suffix = f"year={year}/month={month}/day={day}"
            partition_dict = {"year": year, "month": month, "day": day}
            
            logging.debug("Generated Hive partition suffix: %s", partition_suffix)
            return partition_suffix, partition_dict

        except Exception as e:
            logging.exception("Failed to generate Hive partition suffix.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # CLOUD UPLOADS
    # ==========================================================
    def _publish_to_s3(self, local_path: str, base_s3_uri: str, partition_suffix: str, file_name: str) -> str:
        """
        Uploads a single local file to S3, appending the partition suffix to the base URI.
        
        Args:
            local_path (str): The local file to upload.
            base_s3_uri (str): The root S3 destination.
            partition_suffix (str): The calculated Hive partition string.
            file_name (str): The destination file name.
            
        Returns:
            str: The final complete S3 URI of the uploaded file.
        """
        try:
            destination_uri = f"{base_s3_uri}/{partition_suffix}/{file_name}"
            logging.info("Uploading %s to %s", file_name, destination_uri)
            
            self.s3_sync.upload_file(local_path=local_path, s3_uri=destination_uri)
            
            logging.info("Upload successful for %s", file_name)
            return destination_uri

        except Exception as e:
            logging.exception("Failed to publish artifact to S3: %s", file_name)
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY & METADATA
    # ==========================================================
    def _generate_metadata(
        self, 
        published_csv_uri: str, 
        published_parquet_uri: str, 
        partition_dict: Dict[str, str], 
        execution_time: float
    ) -> None:
        """
        Generates the definitive shipping manifest documenting the success of the upload operations.
        """
        try:
            logging.info("Generating Publisher Shipping Manifest (metadata.json).")

            csv_size = os.path.getsize(self.generator_artifact.csv_report_path)
            parquet_size = os.path.getsize(self.generator_artifact.telemetry_log_path)

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Inference Publisher (S3 Sync)",
                "inference_run_id": self.config.run_id,
                "execution_time_seconds": execution_time,
                "storage_context": {
                    "partition_strategy": "hive",
                    "target_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "partition_keys": partition_dict
                },
                "published_artifacts": [
                    {
                        "artifact_type": "business_report_csv",
                        "source_local_path": self.generator_artifact.csv_report_path,
                        "destination_s3_uri": published_csv_uri,
                        "file_size_bytes": csv_size,
                        "upload_status": "SUCCESS"
                    },
                    {
                        "artifact_type": "telemetry_log_parquet",
                        "source_local_path": self.generator_artifact.telemetry_log_path,
                        "destination_s3_uri": published_parquet_uri,
                        "file_size_bytes": parquet_size,
                        "upload_status": "SUCCESS"
                    }
                ],
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }

            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            logging.debug("Shipping manifest saved successfully to: %s", self.config.metadata_file_path)

        except Exception as e:
            logging.exception("Failed to generate Publisher metadata manifest.")
            raise CustomException(e, sys) from e