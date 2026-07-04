import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import duckdb

from pipelines.monitoring_pipeline.src import constants
from pipelines.monitoring_pipeline.src.entity.config_entity import BaselineAndTelemetryResolverConfig
from pipelines.monitoring_pipeline.src.entity.artifact_entity import BaselineAndTelemetryResolverArtifact
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class BaselineAndTelemetryResolver:
    """
    Baseline & Telemetry Resolver Component for the Monitoring Pipeline.

    Responsibilities:
    - Load the immutable training baselines (Reference distributions, SHAP importance, Metrics) 
      from the active Champion model in the S3 Registry.
    - Fetch today's proactive inference telemetry directly from the Hive-partitioned S3 MLOps bucket.
    - Fetch historical reactive telemetry (e.g., T-30 days) and perform an out-of-core join 
      against today's matured ground-truth labels in the Bronze Data Lake.
    - Persist resolved datasets locally for downstream label-independent and label-dependent evaluation.
    """

    def __init__(self, config: BaselineAndTelemetryResolverConfig) -> None:
        """
        Initializes the Baseline and Telemetry Resolver component.
        """
        try:
            self.config = config
            self.s3_sync = S3Sync()
            
            # Temporary local path for fetching the global model pointer
            self.local_pointer_path = os.path.join(
                self.config.resolver_root_dir, "temp_model_state.json"
            )

            logging.info("Monitoring Pipeline: Baseline & Telemetry Resolver initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Baseline & Telemetry Resolver.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> BaselineAndTelemetryResolverArtifact:
        """
        Executes the resolution workflow for baselines and out-of-core telemetry joins.

        Returns:
            BaselineAndTelemetryResolverArtifact: Local paths to resolved artifacts.
        """
        try:
            logging.info("Starting Baseline & Telemetry Resolution phase.")
            start_time = time.time()

            # 1. Resolve Active Baselines from S3 Registry
            champion_run_id = self._resolve_champion_baselines()

            # 2. Establish DuckDB Connection with Native AWS Support
            con = self._initialize_duckdb_s3_connection()

            try:
                # 3. Fetch Today's Proactive Telemetry
                self._fetch_current_telemetry(con)

                # 4. Fetch Historical Telemetry & Matured Labels (The Lookback Join)
                lookback_status = self._fetch_lookback_telemetry_and_labels(con)

                # 5. Generate Component Metadata
                execution_time = round(time.time() - start_time, 2)
                self._generate_metadata(champion_run_id, lookback_status, execution_time)

            finally:
                con.close()
                logging.debug("DuckDB S3 connection closed safely.")

            # 6. Package and Return Artifact
            artifact = BaselineAndTelemetryResolverArtifact(
                champion_run_id=champion_run_id,
                baseline_metrics_file_path=self.config.baseline_metrics_file_path,
                reference_distributions_file_path=self.config.reference_distributions_file_path,
                shap_importance_file_path=self.config.shap_importance_file_path,
                current_telemetry_file_path=self.config.current_telemetry_file_path,
                lookback_telemetry_file_path=self.config.lookback_telemetry_file_path,
                lookback_labels_file_path=self.config.lookback_labels_file_path,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("Baseline & Telemetry Resolution completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Critical Failure: Baseline & Telemetry Resolver run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # BASELINE RESOLUTION
    # ==========================================================
    def _resolve_champion_baselines(self) -> str:
        """
        Queries the global model pointer to locate the active Champion, then 
        downloads its immutable baseline metrics, reference distributions, and SHAP stats.
        """
        try:
            logging.info("Resolving global active pointer at: %s", self.config.s3_registry_pointer_uri)

            # Download pointer
            self.s3_sync.download_file(
                s3_uri=self.config.s3_registry_pointer_uri, 
                local_path=self.local_pointer_path
            )

            # Parse state
            with open(self.local_pointer_path, "r") as f:
                pointer_data = json.load(f)
            
            os.remove(self.local_pointer_path)

            champion_run_id = pointer_data.get("champion_run_id")
            if not champion_run_id:
                raise ValueError("Corrupted Registry Pointer: Missing 'champion_run_id'.")

            logging.info("Active Champion resolved -> Run ID: %s. Downloading baselines...", champion_run_id)

            # Download core baselines
            baselines = {
                "s3_baseline_metrics_path": self.config.baseline_metrics_file_path,
                "s3_reference_distributions_path": self.config.reference_distributions_file_path,
                "s3_shap_importance_path": self.config.shap_importance_file_path
            }

            for s3_key, local_dest in baselines.items():
                s3_uri = pointer_data.get(s3_key)
                if not s3_uri:
                    raise ValueError(f"Corrupted Registry Pointer: Missing '{s3_key}'.")
                
                self.s3_sync.download_file(s3_uri=s3_uri, local_path=local_dest)
                logging.debug("Downloaded %s successfully.", os.path.basename(local_dest))

            return champion_run_id

        except Exception as e:
            logging.exception("Failed to resolve and download champion baselines.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DUCKDB ENGINE & OUT-OF-CORE QUERIES
    # ==========================================================
    def _initialize_duckdb_s3_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Initializes an ephemeral DuckDB connection configured with AWS extensions.
        """
        try:
            logging.info("Initializing DuckDB with native AWS credential auto-discovery.")
            con = duckdb.connect(database=":memory:")
            
            con.execute("INSTALL httpfs;")
            con.execute("LOAD httpfs;")
            con.execute("INSTALL aws;")
            con.execute("LOAD aws;")

            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            con.execute(f"SET s3_region='{aws_region}';")
            con.execute("CALL load_aws_credentials();")
            
            con.execute("PRAGMA threads=4;")
            con.execute("PRAGMA memory_limit='4GB';")

            return con

        except Exception as e:
            logging.exception("Failed to initialize DuckDB S3 connection.")
            raise CustomException(e, sys) from e

    def _fetch_current_telemetry(self, con: duckdb.DuckDBPyConnection) -> None:
        """
        Streams today's proactive telemetry directly from S3 to local Parquet.
        """
        try:
            s3_current_telemetry_uri = (
                f"{self.config.s3_telemetry_base_uri}/"
                f"{self.config.current_partition_suffix}/"
                f"telemetry_log.parquet"
            )
            
            logging.info(
                "Streaming current proactive telemetry for drift detection: %s", 
                s3_current_telemetry_uri
            )

            query = f"""
                COPY (SELECT * FROM read_parquet('{s3_current_telemetry_uri}')) 
                TO '{self.config.current_telemetry_file_path}' 
                (FORMAT PARQUET, COMPRESSION 'snappy');
            """
            con.execute(query)
            logging.info("Current telemetry materialized successfully.")

        except Exception as e:
            # If current telemetry is missing, monitoring cannot proactively run.
            logging.exception("Failed to fetch today's telemetry. Did the Inference Pipeline execute?")
            raise CustomException(e, sys) from e

    def _fetch_lookback_telemetry_and_labels(self, con: duckdb.DuckDBPyConnection) -> str:
        """
        Performs the Lookback Join: Fetches T-30 days telemetry and joins it 
        against today's Bronze Data Lake to resolve matured outcomes.
        Returns a status string indicating if the lookback data was found.
        """
        try:
            s3_lookback_telemetry_uri = (
                f"{self.config.s3_telemetry_base_uri}/"
                f"{self.config.lookback_partition_suffix}/"
                f"telemetry_log.parquet"
            )

            logging.info("Attempting to fetch reactive lookback telemetry from: %s", s3_lookback_telemetry_uri)

            # Step 1: Attempt to download lookback telemetry
            try:
                con.execute(f"""
                    COPY (SELECT * FROM read_parquet('{s3_lookback_telemetry_uri}')) 
                    TO '{self.config.current_telemetry_file_path}.tmp' 
                    (FORMAT PARQUET, COMPRESSION 'snappy');
                """)
                # Move from tmp to official path
                os.rename(
                    f"{self.config.current_telemetry_file_path}.tmp", 
                    self.config.lookback_telemetry_file_path
                )
                lookback_found = True
            except Exception as s3_err:
                error_msg = str(s3_err)
                if "HTTP" in error_msg or "No files found" in error_msg or "NoSuchKey" in error_msg:
                    logging.warning(
                        "Lookback telemetry not found. System age may be less than %d days. "
                        "Skipping label-dependent evaluation.", 
                        constants.MONITORING_LOOKBACK_DAYS
                    )
                    lookback_found = False
                else:
                    raise

            # Step 2: Handle Matured Labels Extraction
            if lookback_found:
                logging.info(
                    "Executing out-of-core lookback join against S3 Data Lake to retrieve matured labels."
                )
                
                s3_lake_pattern = f"{self.config.s3_data_lake_bronze_uri}/**/*.parquet"
                
                # We extract the ground-truth outcomes as they stand *today* for the customers 
                # scored *30 days ago*.
                join_query = f"""
                    COPY (
                        SELECT 
                            t.{constants.CUSTOMER_ID_COLUMN}, 
                            l.{constants.TARGET_COLUMN}
                        FROM read_parquet('{self.config.lookback_telemetry_file_path}') AS t
                        INNER JOIN read_parquet('{s3_lake_pattern}') AS l
                            ON t.{constants.CUSTOMER_ID_COLUMN} = l.{constants.CUSTOMER_ID_COLUMN}
                        WHERE l.snapshot_date = '{self.config.current_date}'
                    ) TO '{self.config.lookback_labels_file_path}' 
                    (FORMAT PARQUET, COMPRESSION 'snappy');
                """
                con.execute(join_query)
                logging.info("Lookback join completed. Matured labels materialized.")
                return "MATURED_EVALUATION_READY"

            else:
                # Create empty sentinel files so downstream components don't crash, 
                # but rather gracefully skip processing.
                con.execute(f"""
                    COPY (
                        SELECT 'dummy' AS {constants.CUSTOMER_ID_COLUMN}, 
                        0.0 AS predicted_probability WHERE 1=0
                    ) TO '{self.config.lookback_telemetry_file_path}' (FORMAT PARQUET);
                """)
                con.execute(f"""
                    COPY (
                        SELECT 'dummy' AS {constants.CUSTOMER_ID_COLUMN}, 
                        0 AS {constants.TARGET_COLUMN} WHERE 1=0
                    ) TO '{self.config.lookback_labels_file_path}' (FORMAT PARQUET);
                """)
                return "SKIPPED_SYSTEM_IMMATURE"

        except Exception as e:
            logging.exception("Failed to execute lookback telemetry resolution and label joining.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # METADATA GENERATION
    # ==========================================================
    def _generate_metadata(
        self, 
        champion_run_id: str, 
        lookback_status: str, 
        execution_time: float
    ) -> None:
        """
        Generates standard operational telemetry metadata for data lineage tracking.
        """
        try:
            logging.info("Generating Resolver operational metadata.")

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Monitoring Baseline & Telemetry Resolver",
                "execution_time_seconds": execution_time,
                "resolved_state": {
                    "pointer_uri_polled": self.config.s3_registry_pointer_uri,
                    "champion_run_id_loaded": champion_run_id,
                },
                "temporal_bounds": {
                    "evaluation_date_utc": self.config.current_date,
                    "lookback_anchor_date_utc": self.config.lookback_date,
                    "lookback_period_days": constants.MONITORING_LOOKBACK_DAYS,
                    "lookback_evaluation_status": lookback_status
                },
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            logging.debug("Resolver metadata securely saved to: %s", self.config.metadata_file_path)

        except Exception as e:
            logging.exception("Failed to generate operational metadata.")
            raise CustomException(e, sys) from e