"""
Metadata Registry Component for the Continual Learning Data Pipeline.

This module is responsible for generating the strict, contract-compliant 
metadata required by the Continual Learning pipeline and downstream consumers 
(e.g., Training, Monitoring). It extracts telemetry (row counts, schema hashes) 
directly from the finalized Parquet file in the S3 Feature Store via DuckDB, 
and registers the metadata artifact to S3.
"""

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class MetadataRegistry:
    """
    Metadata Registry Execution Stage.

    Responsibilities:
    - Extract structural telemetry (row count, column names) from the generated 
      Feature Store Parquet artifact directly via S3 without local downloading.
    - Compute a deterministic hash of the schema to facilitate downstream data 
      drift and schema compatibility checks.
    - Construct the precise, standardized JSON metadata payload expected by the 
      Continual Learning pipeline.
    - Persist the metadata payload into the specific run's Feature Store directory.
    """

    def __init__(self, context: PipelineContext) -> None:
        """
        Initializes the Metadata Registry component.

        Args:
            context (PipelineContext): The injected pipeline execution context.
        """
        self.context = context
        self.con = self.context.db_con
        
        feature_store_cfg = self.context.config.storage.feature_store
        self.base_uri = feature_store_cfg.base_uri.rstrip("/")
        self.artifact_name = feature_store_cfg.artifact_name
        self.metadata_name = feature_store_cfg.metadata_name

        logging.info("MetadataRegistry initialized.")

    def _get_target_uris(self) -> Tuple[str, str]:
        """
        Constructs the S3 URIs for the dataset artifact and the metadata file.

        Returns:
            Tuple[str, str]: A tuple containing (artifact_s3_uri, metadata_s3_uri).
        """
        run_uri_prefix = f"{self.base_uri}/{self.context.run_id}"
        artifact_uri = f"{run_uri_prefix}/{self.artifact_name}"
        metadata_uri = f"{run_uri_prefix}/{self.metadata_name}"
        
        return artifact_uri, metadata_uri

    def _extract_parquet_telemetry(self, artifact_uri: str) -> Tuple[int, str]:
        """
        Extracts the row count and computes the schema hash directly from the 
        Parquet file stored in S3, leveraging DuckDB's optimized metadata reading.

        Args:
            artifact_uri (str): The S3 URI of the target Parquet artifact.

        Returns:
            Tuple[int, str]: A tuple containing (row_count, columns_hash).
        """
        logging.debug("Extracting Parquet telemetry via DuckDB for URI: %s", artifact_uri)
        
        try:
            # DuckDB optimizes COUNT(*) by reading the Parquet footer directly.
            count_query = f"SELECT COUNT(*) FROM read_parquet('{artifact_uri}')"
            count_res = self.con.execute(count_query).fetchone()
            row_count = count_res[0] if count_res else 0

            # Extract schema structure
            describe_query = f"DESCRIBE SELECT * FROM read_parquet('{artifact_uri}') LIMIT 1"
            describe_res = self.con.execute(describe_query).fetchall()
            columns = [row[0] for row in describe_res]

            # Generate a deterministic hash of the column structure (first 10 chars)
            columns_string = "".join(columns)
            columns_hash = hashlib.md5(columns_string.encode("utf-8")).hexdigest()[:10]
            
            logging.debug("Extracted telemetry - Rows: %d, Hash: %s", row_count, columns_hash)
            return row_count, columns_hash

        except Exception as exc:
            logging.error("Failed to extract Parquet telemetry from %s", artifact_uri)
            raise CustomException(exc, sys) from exc

    def _build_metadata_payload(
        self, artifact_uri: str, row_count: int, columns_hash: str
    ) -> Dict[str, Any]:
        """
        Constructs the precise JSON metadata payload adhering to the contract 
        expected by the Continual Learning pipeline.

        Args:
            artifact_uri (str): S3 URI of the generated dataset.
            row_count (int): Total records in the dataset.
            columns_hash (str): Deterministic MD5 hash of the schema columns.

        Returns:
            Dict[str, Any]: The fully formatted metadata dictionary.
        """
        return {
            "run_id": self.context.run_id,
            "temporal_bounds": {
                "start_date": self.context.start_date,
                "end_date": self.context.end_date,
            },
            "artifact_uri": artifact_uri,
            "schema": {
                "columns_hash": columns_hash,
                "row_count": row_count,
            },
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def run(self) -> None:
        """
        Executes the Metadata Registry process to generate and upload telemetry.

        Raises:
            CustomException: If telemetry extraction or S3 upload fails.
        """
        logging.info("Starting Metadata Registry stage.")

        artifact_uri, metadata_uri = self._get_target_uris()
        tmp_file_path = None

        try:
            row_count, columns_hash = self._extract_parquet_telemetry(artifact_uri)
            
            payload = self._build_metadata_payload(
                artifact_uri=artifact_uri,
                row_count=row_count,
                columns_hash=columns_hash,
            )

            # Persist payload securely to an ephemeral local file for upload
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump(payload, tmp, indent=2)
                tmp_file_path = tmp.name

            logging.info("Uploading registry metadata to S3 URI: %s", metadata_uri)
            
            self.context.s3_sync.upload_file(
                local_path=tmp_file_path,
                s3_uri=metadata_uri
            )
            
            logging.info(
                "Metadata Registry completed successfully. Registered %d rows for run: %s",
                row_count,
                self.context.run_id,
            )

        except Exception as exc:
            logging.exception("Failed during Metadata Registry stage.")
            raise CustomException(exc, sys) from exc
        
        finally:
            # Ensure local ephemeral artifact is cleaned up
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                    logging.debug("Ephemeral metadata file cleaned up locally.")
                except OSError as cleanup_exc:
                    logging.warning(
                        "Failed to clean up temporary metadata file %s: %s", 
                        tmp_file_path, 
                        cleanup_exc
                    )