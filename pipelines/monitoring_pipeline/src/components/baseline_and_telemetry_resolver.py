import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any

import duckdb

from pipelines.monitoring_pipeline.src.core.context import MonitoringPipelineContext
from pipelines.monitoring_pipeline.src.entity.config_entity import BaselineAndTelemetryResolverConfig
from pipelines.monitoring_pipeline.src.entity.artifact_entity import BaselineAndTelemetryResolverArtifact
from shared_core.cloud.s3_operations import S3Sync
from shared_core.features.shared_feature import SharedFeatureGenerator
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class BaselineAndTelemetryResolver:
    """
    Baseline & Telemetry Resolver Component for the Monitoring Pipeline.
    """

    def __init__(
        self, 
        config: BaselineAndTelemetryResolverConfig, 
        context: MonitoringPipelineContext
    ) -> None:
        try:
            self.config = config
            self.context = context
            self.s3_sync = S3Sync()
            
            data_schema = self.context.config.get("data_schema", {})
            self.customer_id_col = data_schema.get("customer_id_column", "customer_unique_id")
            self.target_col = data_schema.get("target_column", "target_is_churn")
            self.prediction_col = data_schema.get("prediction_column", "predicted_probability")

            self.local_pointer_path = os.path.join(
                self.config.resolver_root_dir, "temp_model_state.json"
            )

            logging.info("Monitoring Pipeline: Baseline & Telemetry Resolver initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Baseline & Telemetry Resolver.")
            raise CustomException(e, sys) from e

    def run(self) -> BaselineAndTelemetryResolverArtifact:
        try:
            logging.info("Starting Baseline & Telemetry Resolution phase.")
            start_time = time.time()

            champion_run_id = self._resolve_champion_baselines()

            con = self.context.duckdb_con
            if not con:
                raise RuntimeError("Shared DuckDB connection is not initialized in the context.")

            current_status = self._fetch_current_telemetry(con)
            lookback_status = self._fetch_lookback_telemetry_and_labels(con)

            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(champion_run_id, current_status, lookback_status, execution_time)

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

            logging.info("Baseline & Telemetry Resolution completed successfully.")
            return artifact

        except Exception as e:
            logging.exception("Critical Failure: Baseline & Telemetry Resolver run failed.")
            raise CustomException(e, sys) from e

    def _resolve_champion_baselines(self) -> str:
        try:
            logging.info("Resolving global active pointer at: %s", self.config.s3_registry_pointer_uri)

            self.s3_sync.download_file(
                s3_uri=self.config.s3_registry_pointer_uri, 
                local_path=self.local_pointer_path
            )

            with open(self.local_pointer_path, "r", encoding="utf-8") as f:
                pointer_data = json.load(f)
            
            if os.path.exists(self.local_pointer_path):
                os.remove(self.local_pointer_path)

            champion_run_id = pointer_data.get("champion_run_id") or pointer_data.get("run_id")
            if not champion_run_id:
                raise ValueError("Corrupted Registry Pointer: Missing 'champion_run_id' or 'run_id'.")

            logging.info("Active Champion resolved -> Run ID: %s. Downloading baselines...", champion_run_id)

            metrics_uri = pointer_data.get("s3_baseline_metrics_path") or pointer_data.get("s3_monitoring_baselines_path")
            dist_uri = pointer_data.get("s3_reference_distributions_path")
            
            if not metrics_uri or not dist_uri:
                raise ValueError("Corrupted Registry Pointer: Missing required baseline URIs.")

            self.s3_sync.download_file(s3_uri=metrics_uri, local_path=self.config.baseline_metrics_file_path)
            self.s3_sync.download_file(s3_uri=dist_uri, local_path=self.config.reference_distributions_file_path)
            
            shap_uri = pointer_data.get("s3_shap_importance_path")
            if shap_uri:
                self.s3_sync.download_file(s3_uri=shap_uri, local_path=self.config.shap_importance_file_path)
            else:
                logging.warning("SHAP artifact missing from registry pointer. Generating synthetic SHAP fallback to prevent downstream drift calculation failure.")
                self._generate_synthetic_shap()

            return champion_run_id

        except Exception as e:
            logging.exception("Failed to resolve and download champion baselines.")
            raise CustomException(e, sys) from e

    def _generate_synthetic_shap(self) -> None:
        try:
            with open(self.config.reference_distributions_file_path, "r", encoding="utf-8") as f:
                ref_data = json.load(f)
            
            distributions = ref_data.get("distributions", ref_data)
            exclude_cols = {self.target_col, self.prediction_col, self.customer_id_col}
            features = [k for k in distributions.keys() if k not in exclude_cols]
            
            synthetic_shap = []
            for i, feat in enumerate(features):
                synthetic_shap.append({
                    "feature_name": feat,
                    "mean_abs_shap_value": 1.0 / (i + 1)
                })
            
            with open(self.config.shap_importance_file_path, "w", encoding="utf-8") as f:
                json.dump({"feature_importance": synthetic_shap}, f, indent=4)
                
        except Exception as e:
            logging.exception("Failed to generate synthetic SHAP artifact.")
            raise CustomException(e, sys) from e

    def _fetch_current_telemetry(self, con: duckdb.DuckDBPyConnection) -> str:
        """
        Fetches today's proactive telemetry. If no inference was run today (404 Not Found),
        safely generates a 0-row schema to avoid pipeline crashes (0-traffic handling).
        """
        try:
            s3_current_telemetry_uri = (
                f"{self.config.s3_telemetry_base_uri}/"
                f"{self.config.current_partition_suffix}/"
                f"telemetry_log.parquet"
            )
            logging.info("Attempting to stream current proactive telemetry: %s", s3_current_telemetry_uri)
            
            try:
                query = f"COPY (SELECT * FROM read_parquet('{s3_current_telemetry_uri}')) TO '{self.config.current_telemetry_file_path}' (FORMAT PARQUET, COMPRESSION 'snappy');"
                con.execute(query)
                return "CURRENT_TELEMETRY_RESOLVED"
            except Exception as s3_err:
                error_msg = str(s3_err).lower()
                if "http" in error_msg or "404" in error_msg or "no files found" in error_msg or "nosuchkey" in error_msg:
                    logging.warning("Current telemetry not found (HTTP 404). This implies 0 traffic or Inference Pipeline did not run for this date.")
                    logging.info("Generating empty 0-row Parquet schema for graceful downstream drift bypass.")
                    
                    fallback_query = f"""
                        COPY (
                            SELECT CAST(NULL AS VARCHAR) AS {self.customer_id_col}, 
                                   CAST(NULL AS DOUBLE) AS {self.prediction_col} 
                            WHERE 1=0
                        ) TO '{self.config.current_telemetry_file_path}' (FORMAT PARQUET, COMPRESSION 'snappy');
                    """
                    con.execute(fallback_query)
                    return "SKIPPED_ZERO_TRAFFIC_DAY"
                else:
                    raise

        except Exception as e:
            logging.exception("Failed to execute current telemetry resolution.")
            raise CustomException(e, sys) from e

    def _fetch_lookback_telemetry_and_labels(self, con: duckdb.DuckDBPyConnection) -> str:
        try:
            s3_lookback_telemetry_uri = f"{self.config.s3_telemetry_base_uri}/{self.config.lookback_partition_suffix}/telemetry_log.parquet"
            logging.info("Attempting to fetch reactive lookback telemetry from: %s", s3_lookback_telemetry_uri)

            lookback_found = False
            try:
                con.execute(f"COPY (SELECT * FROM read_parquet('{s3_lookback_telemetry_uri}')) TO '{self.config.current_telemetry_file_path}.tmp' (FORMAT PARQUET, COMPRESSION 'snappy');")
                os.rename(f"{self.config.current_telemetry_file_path}.tmp", self.config.lookback_telemetry_file_path)
                lookback_found = True
            except Exception as s3_err:
                error_msg = str(s3_err).lower()
                if "http" in error_msg or "no files found" in error_msg or "nosuchkey" in error_msg:
                    logging.warning("Lookback telemetry not found. System age may be less than %d days.", self.config.lookback_period_days)
                else:
                    raise

            if lookback_found:
                logging.info("Executing out-of-core lookback join against S3 Data Lake to retrieve matured labels.")
                feature_generator = SharedFeatureGenerator(data_dir=self.config.s3_data_lake_bronze_uri, is_partitioned=True)
                label_query = feature_generator.get_feature_query(self.config.lookback_date)
                join_query = f"""
                    COPY (
                        SELECT t.{self.customer_id_col}, l.{self.target_col}
                        FROM read_parquet('{self.config.lookback_telemetry_file_path}') AS t
                        INNER JOIN ({label_query}) AS l ON t.{self.customer_id_col} = l.{self.customer_id_col}
                    ) TO '{self.config.lookback_labels_file_path}' (FORMAT PARQUET, COMPRESSION 'snappy');
                """
                con.execute(join_query)
                return "MATURED_EVALUATION_READY"
            else:
                logging.info("Generating empty 0-row Parquet schemas for graceful downstream bypass.")
                con.execute(f"COPY (SELECT CAST(NULL AS VARCHAR) AS {self.customer_id_col}, CAST(NULL AS DOUBLE) AS {self.prediction_col} WHERE 1=0) TO '{self.config.lookback_telemetry_file_path}' (FORMAT PARQUET, COMPRESSION 'snappy');")
                con.execute(f"COPY (SELECT CAST(NULL AS VARCHAR) AS {self.customer_id_col}, CAST(NULL AS INTEGER) AS {self.target_col} WHERE 1=0) TO '{self.config.lookback_labels_file_path}' (FORMAT PARQUET, COMPRESSION 'snappy');")
                return "SKIPPED_SYSTEM_IMMATURE"

        except Exception as e:
            logging.exception("Failed to execute lookback telemetry resolution.")
            raise CustomException(e, sys) from e

    def _generate_metadata(self, champion_run_id: str, current_status: str, lookback_status: str, execution_time: float) -> None:
        try:
            metadata: Dict[str, Any] = {
                "pipeline_stage": "Monitoring Baseline & Telemetry Resolver",
                "execution_time_seconds": execution_time,
                "resolved_state": {
                    "pointer_uri_polled": self.config.s3_registry_pointer_uri,
                    "champion_run_id_loaded": champion_run_id,
                },
                "temporal_bounds": {
                    "evaluation_date_utc": self.config.current_date,
                    "current_telemetry_status": current_status,
                    "lookback_anchor_date_utc": self.config.lookback_date,
                    "lookback_period_days": self.config.lookback_period_days,
                    "lookback_evaluation_status": lookback_status
                },
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "orchestration": {"run_id": self.context.run_id}
            }
            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
        except Exception as e:
            logging.exception("Failed to generate operational metadata.")
            raise CustomException(e, sys) from e