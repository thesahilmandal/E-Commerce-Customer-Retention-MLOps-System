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
    - Upload the Parquet file directly to the AWS S3 Feature Store using native boto3.
    - Extract Parquet metadata (row counts) without loading data into memory.
    - Generate observability telemetry and bitemporal lineage tracking data.
    """

    def __init__(
        self,
        config: DataLoadingConfig,
        transformer_artifact: DataTransformationArtifact,
    ) -> None:
        """
        Initializes Loader with required configuration and artifacts.
        """
        try:
            self.config: DataLoadingConfig = config
            self.transformer_artifact: DataTransformationArtifact = transformer_artifact
            
            # The definitive local path is strictly owned by the Transformer
            self.source_parquet_path: str = self.transformer_artifact.transformed_data_file_path
            self.s3_sync: S3Sync = S3Sync()

            logging.info("Loader initialized successfully (Zero-Copy Architecture).")

        except Exception as e:
            logging.exception("Error during Loader initialization.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataLoadingArtifact:
        """
        Executes the data loading and S3 upload process out-of-core.

        Returns:
            DataPipelineLoaderArtifact: Details of remote S3 path and local metadata.
        """
        try:
            logging.info("Starting Data Loader pipeline (Local to S3 Pass-Through).")
            start_time: float = time.time()

            # 1. Upload to AWS S3 directly from Transformer's artifact directory
            self._upload_to_s3()

            # 2. Extract lightweight metrics and generate metadata
            execution_time: float = round(time.time() - start_time, 2)
            self._generate_metadata(execution_time)

            # 3. Package Artifact
            artifact = DataLoadingArtifact(
                s3_file_uri=self.config.s3_master_panel_uri,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("Loader artifact created successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Error during Loader run.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # CLOUD OPERATIONS
    # ==========================================================
    def _upload_to_s3(self) -> None:
        """Uploads the Parquet file directly from the Transformer artifact directory to S3."""
        try:
            logging.info(
                "Uploading Master Panel directly from %s to S3 URI: %s", 
                self.source_parquet_path,
                self.config.s3_master_panel_uri
            )
            self.s3_sync.upload_file(
                local_path=self.source_parquet_path,
                s3_uri=self.config.s3_master_panel_uri,
            )
        except Exception as e:
            logging.exception("Failed to upload Master Panel to S3.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY
    # ==========================================================
    def _generate_metadata(self, execution_time: float) -> None:
        """
        Extracts file metrics (size, row count) directly from Parquet metadata 
        to prevent Out-Of-Memory (OOM) issues, and saves JSON observability data.
        """
        try:
            logging.info("Generating Loader telemetry and bitemporal lineage...")

            # Get file size directly from the source file
            file_size_bytes: int = os.path.getsize(self.source_parquet_path)
            file_size_mb: float = round(file_size_bytes / (1024 * 1024), 2)

            # Efficiently read total rows from Parquet footer without loading data into RAM
            parquet_metadata = pq.read_metadata(self.source_parquet_path)
            total_rows: int = parquet_metadata.num_rows

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

        except Exception as e:
            logging.exception("Failed to generate Loader metadata.")
            raise CustomException(e, sys) from e