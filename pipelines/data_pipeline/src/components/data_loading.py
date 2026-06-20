import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any

import pyarrow.parquet as pq

from pipelines.data_pipeline.src.entity.config_entity import DataLoadingConfig
from pipelines.data_pipeline.src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataLoadingArtifact,
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file
from shared_core.cloud.s3_operations import S3Sync


class DataLoading:
    """
    Loader component for persisting the Master Feature Panel to AWS S3.

    Responsibilities:
    - Act as a zero-copy pass-through layer to prevent disk storage duplication.
    - Receive the out-of-core generated Parquet file path from Transformer.
    - Upload the Parquet file directly to the AWS S3 Feature Store via injected S3 utility.
    - Extract Parquet metadata (row counts) without loading data into memory.
    - Generate observability telemetry and bitemporal lineage tracking data.
    """

    def __init__(
        self,
        config: DataLoadingConfig,
        transformer_artifact: DataTransformationArtifact,
        s3_sync: S3Sync,
    ) -> None:
        """
        Initializes Loader with required configuration and artifacts.
        
        Args:
            config (DataLoadingConfig): Loader configuration paths.
            transformer_artifact (DataTransformationArtifact): Details of the transformed data.
            s3_sync (S3Sync): Injected S3 synchronization client for cloud operations.
        """
        self.config = config
        self.transformer_artifact = transformer_artifact
        
        # The definitive local path is strictly owned by the Transformer
        self.source_parquet_path = self.transformer_artifact.transformed_data_file_path
        self.s3_sync = s3_sync

        logging.info("Loader initialized successfully (Zero-Copy Architecture).")

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataLoadingArtifact:
        """
        Executes the data loading and S3 upload process out-of-core.

        Returns:
            DataLoadingArtifact: Details of remote S3 path and local metadata.
        """
        logging.info("Starting Data Loader pipeline (Local to S3 Pass-Through).")
        start_time = time.time()

        # 1. Upload to AWS S3 directly from Transformer's artifact directory
        self._upload_to_s3()

        # 2. Extract lightweight metrics and generate metadata
        execution_time = round(time.time() - start_time, 2)
        self._generate_metadata(execution_time)

        # 3. Package Artifact
        artifact = DataLoadingArtifact(
            s3_file_uri=self.config.s3_master_panel_uri,
            metadata_file_path=self.config.metadata_file_path,
        )

        logging.info("Loader artifact created successfully: %s", artifact)
        return artifact

    # ==========================================================
    # CLOUD OPERATIONS
    # ==========================================================
    def _upload_to_s3(self) -> None:
        """Uploads the Parquet file directly from the Transformer artifact directory to S3."""
        logging.info(
            "Uploading Master Panel directly from %s to S3 URI: %s", 
            self.source_parquet_path,
            self.config.s3_master_panel_uri
        )
        self.s3_sync.upload_file(
            local_path=self.source_parquet_path,
            s3_uri=self.config.s3_master_panel_uri,
        )

    # ==========================================================
    # OBSERVABILITY
    # ==========================================================
    def _generate_metadata(self, execution_time: float) -> None:
        """
        Extracts file metrics (size, row count) directly from Parquet metadata 
        to prevent Out-Of-Memory (OOM) issues, and saves JSON observability data.
        """
        logging.info("Generating Loader telemetry and bitemporal lineage...")

        try:
            # Get file size directly from the source file
            file_size_bytes = os.path.getsize(self.source_parquet_path)
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

            # Efficiently read total rows from Parquet footer without loading data into RAM
            parquet_metadata = pq.read_metadata(self.source_parquet_path)
            total_rows = parquet_metadata.num_rows
        except Exception as exc:
            logging.error("Failed to read Parquet metadata from %s.", self.source_parquet_path)
            raise CustomException(exc, sys) from exc

        metadata: Dict[str, Any] = {
            "pipeline_stage": "Loader",
            "architecture": "zero_copy_pass_through",
            "execution_time_seconds": execution_time,
            "storage": {
                "format": "parquet",
                "compression": "snappy",
                "file_size_mb": file_size_mb,
                "total_rows_saved": total_rows,
            },
            "lineage": {
                "source_local_path": self.source_parquet_path,
                "s3_uri": self.config.s3_master_panel_uri,
                "bucket": self.config.s3_bucket_name,
                "feature_store_prefix": self.config.s3_feature_store_dir,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_json_file(file_path=self.config.metadata_file_path, content=metadata)
        
        logging.info(
            "Loader metadata saved. Total size uploaded: %s MB (%s rows).", 
            file_size_mb, 
            total_rows
        )