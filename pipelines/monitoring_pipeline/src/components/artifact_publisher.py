import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import boto3
from urllib.parse import urlparse

from pipelines.monitoring_pipeline.src.core.context import MonitoringPipelineContext
from pipelines.monitoring_pipeline.src.entity.config_entity import ArtifactPublisherConfig
from pipelines.monitoring_pipeline.src.entity.artifact_entity import (
BaselineAndTelemetryResolverArtifact,
StatisticalDriftCalculatorArtifact,
PerformanceEvaluatorArtifact,
RuleEngineArtifact,
ArtifactPublisherArtifact
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file

class ArtifactPublisher:
    """
    Artifact Publisher Component for the Monitoring Pipeline.

    ```
    Responsibilities:
    - Publish curated, high-value monitoring artifacts to a standardized, 
    Hive-partitioned Amazon S3 directory structure.
    - Guarantee deterministic uploads tied strictly to the orchestrator's `run_id` 
    and `execution_date` to ensure idempotency and traceability.
    - Compile individual component metadata ledgers into a single, comprehensive 
    Master Execution Ledger for system observability.
    - Provide a robust, verifiable output artifact containing the final S3 URIs 
    of all published files to be consumed by downstream automation.
    """

    def __init__(
        self,
        config: ArtifactPublisherConfig,
        context: MonitoringPipelineContext,
        resolver_artifact: BaselineAndTelemetryResolverArtifact,
        drift_artifact: StatisticalDriftCalculatorArtifact,
        performance_artifact: PerformanceEvaluatorArtifact,
        rule_engine_artifact: RuleEngineArtifact
    ) -> None:
        """
        Initializes the Artifact Publisher.

        Args:
            config (ArtifactPublisherConfig): Component-specific configuration.
            context (MonitoringPipelineContext): Global pipeline context containing shared resources.
            resolver_artifact (BaselineAndTelemetryResolverArtifact): Resolver component outputs.
            drift_artifact (StatisticalDriftCalculatorArtifact): Drift component outputs.
            performance_artifact (PerformanceEvaluatorArtifact): Performance evaluation outputs.
            rule_engine_artifact (RuleEngineArtifact): Rule engine decision outputs.
        """
        try:
            self.config = config
            self.context = context
            self.resolver_artifact = resolver_artifact
            self.drift_artifact = drift_artifact
            self.performance_artifact = performance_artifact
            self.rule_engine_artifact = rule_engine_artifact
            
            # Initialize S3 client for precise, idempotent file uploads
            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            self.s3_client = boto3.client("s3", region_name=aws_region)
            
            logging.info("Monitoring Pipeline: Artifact Publisher initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Artifact Publisher.")
            raise CustomException(e, sys) from e

    def run(self) -> ArtifactPublisherArtifact:
        """
        Executes the artifact publishing workflow.

        Returns:
            ArtifactPublisherArtifact: S3 URIs of all published production artifacts.
        """
        try:
            logging.info("Starting production artifact publishing phase.")
            start_time = time.time()

            # 1. Generate Hive Partition Prefix
            partition_prefix = self._generate_hive_partition()

            # 2. Compile Master Metadata Ledger
            master_metadata_path = self._compile_master_metadata()

            # 3. Define Upload Map (Local Path -> S3 URI)
            upload_map = self._build_upload_map(partition_prefix, master_metadata_path)

            # 4. Validate Local Artifacts
            self._validate_local_artifacts(list(upload_map.keys()))

            # 5. Upload Artifacts to S3
            published_uris = self._upload_artifacts(upload_map)

            # 6. Generate Publisher Artifact
            artifact = ArtifactPublisherArtifact(
                s3_audit_report_uri=published_uris["audit_report"],
                s3_action_token_uri=published_uris["action_token"],
                s3_matured_evaluation_uri=published_uris["matured_evaluation"],
                s3_metadata_uri=published_uris["metadata_ledger"]
            )

            execution_time = round(time.time() - start_time, 2)
            logging.info(
                "Artifact Publisher completed successfully in %.2f seconds.", 
                execution_time
            )
            return artifact

        except Exception as e:
            logging.exception("Critical Failure: Artifact Publisher run failed.")
            raise CustomException(e, sys) from e

    def _generate_hive_partition(self) -> str:
        """
        Generates a standardized Hive partition string based on the execution date.
        Format: year=YYYY/month=MM/day=DD
        """
        try:
            exec_date_obj = datetime.strptime(self.config.execution_date, "%Y-%m-%d")
            partition = (
                f"year={exec_date_obj.year}/"
                f"month={exec_date_obj.month:02d}/"
                f"day={exec_date_obj.day:02d}"
            )
            logging.debug("Generated Hive partition prefix: %s", partition)
            return partition
        except ValueError as e:
            logging.error("Invalid execution_date format. Expected YYYY-MM-DD.")
            raise CustomException(e, sys) from e

    def _compile_master_metadata(self) -> str:
        """
        Aggregates individual component metadata files into a single Master Execution Ledger.
        
        Returns:
            str: Local path to the generated master metadata file.
        """
        try:
            logging.info("Compiling Master Execution Metadata Ledger.")
            
            master_metadata: Dict[str, Any] = {
                "orchestration": {
                    "run_id": self.config.run_id,
                    "execution_date": self.config.execution_date,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()
                },
                "components": {
                    "baseline_and_telemetry_resolver": self._load_json(self.resolver_artifact.metadata_file_path),
                    "statistical_drift_calculator": self._load_json(self.drift_artifact.metadata_file_path),
                    "performance_evaluator": self._load_json(self.performance_artifact.metadata_file_path),
                    "rule_engine": self._load_json(self.rule_engine_artifact.metadata_file_path)
                }
            }

            master_metadata_path = os.path.join(
                self.config.publisher_root_dir, 
                f"master_metadata_{self.config.run_id}.json"
            )
            write_json_file(file_path=master_metadata_path, content=master_metadata)
            
            return master_metadata_path

        except Exception as e:
            logging.exception("Failed to compile Master Metadata Ledger.")
            raise CustomException(e, sys) from e

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Safely loads a JSON artifact."""
        try:
            if not os.path.exists(file_path):
                logging.warning("Metadata file not found: %s. Proceeding with empty block.", file_path)
                return {}
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.exception("Failed to parse JSON file: %s", file_path)
            raise CustomException(e, sys) from e

    def _build_upload_map(self, partition_prefix: str, master_metadata_path: str) -> Dict[str, str]:
        """
        Maps the local file paths to their deterministic, versioned S3 destination URIs.
        """
        base_s3_uri = f"s3://{self.config.s3_bucket_name}/{self.config.s3_monitoring_output_prefix}"
        run_id = self.config.run_id

        # S3 URIs mapping
        s3_audit_uri = (
            f"{base_s3_uri}/{self.config.s3_audit_reports_dir}/"
            f"{partition_prefix}/report_{run_id}.json"
        )
        s3_token_uri = (
            f"{base_s3_uri}/{self.config.s3_action_tokens_dir}/"
            f"{partition_prefix}/need_update_{run_id}.json"
        )
        s3_evaluation_uri = (
            f"{base_s3_uri}/{self.config.s3_matured_evaluations_dir}/"
            f"{partition_prefix}/evaluation_{run_id}.parquet"
        )
        s3_metadata_uri = (
            f"{base_s3_uri}/{self.config.s3_metadata_dir}/"
            f"{partition_prefix}/metadata_{run_id}.json"
        )

        return {
            self.rule_engine_artifact.monitoring_report_file_path: s3_audit_uri,
            self.rule_engine_artifact.need_update_file_path: s3_token_uri,
            self.resolver_artifact.lookback_labels_file_path: s3_evaluation_uri,
            master_metadata_path: s3_metadata_uri
        }

    def _validate_local_artifacts(self, local_paths: List[str]) -> None:
        """
        Validates the existence of all local files prior to beginning the upload sequence.
        Fails fast if any critical artifact is missing.
        """
        try:
            missing_files = [path for path in local_paths if not os.path.exists(path)]
            if missing_files:
                raise FileNotFoundError(
                    f"The following required artifacts are missing locally: {missing_files}"
                )
            logging.info("All required artifacts validated locally.")
        except Exception as e:
            logging.exception("Local artifact validation failed.")
            raise CustomException(e, sys) from e

    def _upload_artifacts(self, upload_map: Dict[str, str]) -> Dict[str, str]:
        """
        Iterates over the upload map and securely transfers files to S3.
        
        Returns:
            Dict[str, str]: A map of logical artifact names to their S3 URIs.
        """
        published_uris = {}
        logical_names = [
            "audit_report", 
            "action_token", 
            "matured_evaluation", 
            "metadata_ledger"
        ]

        try:
            for logical_name, (local_path, s3_uri) in zip(logical_names, upload_map.items()):
                self._upload_single_file(local_path, s3_uri)
                published_uris[logical_name] = s3_uri

            logging.info("Successfully published all %d artifacts to S3.", len(upload_map))
            return published_uris

        except Exception as e:
            logging.exception("Failed during the artifact upload sequence.")
            raise CustomException(e, sys) from e

    def _upload_single_file(self, local_path: str, s3_uri: str) -> None:
        """
        Uploads a single file to an S3 URI using boto3 for precise control.
        """
        try:
            parsed_uri = urlparse(s3_uri)
            if parsed_uri.scheme != "s3":
                raise ValueError(f"Invalid S3 URI scheme: {s3_uri}")

            bucket = parsed_uri.netloc
            key = parsed_uri.path.lstrip("/")

            logging.debug("Uploading %s to s3://%s/%s", os.path.basename(local_path), bucket, key)
            
            # Using boto3 directly guarantees idempotency and precise key placement
            self.s3_client.upload_file(Filename=local_path, Bucket=bucket, Key=key)

        except Exception as e:
            logging.error("Failed to upload file %s to %s", local_path, s3_uri)
            raise CustomException(e, sys) from e