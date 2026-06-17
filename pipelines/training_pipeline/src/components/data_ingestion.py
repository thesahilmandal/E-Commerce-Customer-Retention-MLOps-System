import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import polars as pl

from pipelines.training_pipeline.src.entity.config_entity import DataIngestionConfig
from pipelines.training_pipeline.src.entity.artifact_entity import DataIngestionArtifact
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.cloud.s3_operations import S3Sync
from shared_core.utils.main_utils import write_json_file


class DataIngestion:
    """
    Data Ingestion component for the Training Pipeline.

    Responsibilities:
    - Securely fetch the out-of-core Master Feature Panel from AWS S3.
    - Utilize Polars for memory-safe, lazy-evaluated data loading.
    - Perform Out-Of-Time (OOT) splitting based on bitemporal snapshot dates 
      to strictly prevent target leakage.
    - Persist Train, Validation, and Test datasets locally as Parquet files.
    - Generate observability metadata including row counts and class imbalance ratios.
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initializes the Data Ingestion component.

        Args:
            config (TrainingPipelineDataIngestionConfig): Configuration object.
        """
        try:
            self.config = config
            self.s3_sync = S3Sync()
            
            # Temporary local path for the downloaded master panel
            self.local_master_panel_path = os.path.join(
                self.config.data_ingestion_root_dir, 
                "master_panel_downloaded.parquet"
            )

            logging.info("Training Pipeline: Data Ingestion component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Data Ingestion component.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataIngestionArtifact:
        """
        Executes the data ingestion and OOT splitting pipeline.

        Returns:
            TrainingPipelineDataIngestionArtifact: Artifact containing paths to the splits.
        """
        try:
            logging.info("Starting Out-Of-Time (OOT) Data Ingestion process.")
            start_time = time.time()

            # Step 1: Download Master Panel from S3
            self._download_data_from_s3()

            # Step 2: Perform Memory-Safe OOT Splitting using Polars
            row_counts, imbalance_ratio = self._perform_oot_split()

            # Step 3: Generate Execution Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(row_counts, imbalance_ratio, execution_time)

            # Step 4: Clean up temporary master panel to save disk space
            if os.path.exists(self.local_master_panel_path):
                os.remove(self.local_master_panel_path)
                logging.debug("Cleaned up temporary master panel file.")

            # Step 5: Package Artifact
            artifact = DataIngestionArtifact(
                train_data_path=self.config.train_data_path,
                val_data_path=self.config.val_data_path,
                test_data_path=self.config.test_data_path,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("Data Ingestion completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Data Ingestion run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # CLOUD OPERATIONS
    # ==========================================================
    def _download_data_from_s3(self) -> None:
        """
        Downloads the Master Panel Parquet file from the S3 feature store.
        """
        try:
            logging.info(
                "Downloading master panel from S3 URI: %s to %s", 
                self.config.s3_master_panel_uri, 
                self.local_master_panel_path
            )
            self.s3_sync.download_file(
                s3_uri=self.config.s3_master_panel_uri,
                local_path=self.local_master_panel_path
            )
            logging.info("Successfully downloaded master panel.")

        except Exception as e:
            logging.exception("Failed to download master panel from S3.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # MEMORY-SAFE OOT SPLITTING (POLARS)
    # ==========================================================
    def _perform_oot_split(self) -> Tuple[Dict[str, int], float]:
        """
        Lazily evaluates the master panel and splits it into Train/Val/Test 
        sets strictly by bitemporal snapshot dates to prevent leakage.

        Returns:
            Tuple[Dict[str, int], float]: Dictionary of row counts and the training class imbalance ratio.
        """
        try:
            logging.info("Executing Out-Of-Time (OOT) splits via Polars lazy evaluation.")

            # Initiate LazyFrame to prevent loading the entire dataset into RAM
            lazy_df = pl.scan_parquet(self.local_master_panel_path)

            # --------------------------------------------------
            # 1. Train Split (Earliest Historical Data)
            # --------------------------------------------------
            train_df = lazy_df.filter(
                pl.col("snapshot_date").is_in(self.config.train_snapshots)
            ).collect()
            
            train_df.write_parquet(self.config.train_data_path)
            train_rows = train_df.height
            logging.info("Train split saved: %s rows.", train_rows)

            # Calculate Class Imbalance Ratio (Negative Class / Positive Class)
            # Ensures we can dynamically pass this to XGBoost's scale_pos_weight
            positive_class_count = train_df["target_is_churn"].sum()
            negative_class_count = train_rows - positive_class_count
            
            imbalance_ratio = 1.0
            if positive_class_count > 0:
                imbalance_ratio = float(negative_class_count / positive_class_count)

            # Free memory explicitly before next operations
            del train_df

            # --------------------------------------------------
            # 2. Validation Split (Next Sequential Snapshot)
            # --------------------------------------------------
            val_df = lazy_df.filter(
                pl.col("snapshot_date") == self.config.val_snapshot
            ).collect()
            
            val_df.write_parquet(self.config.val_data_path)
            val_rows = val_df.height
            logging.info("Validation split saved: %s rows.", val_rows)
            del val_df

            # --------------------------------------------------
            # 3. Test Split (Most Recent Snapshot)
            # --------------------------------------------------
            test_df = lazy_df.filter(
                pl.col("snapshot_date") == self.config.test_snapshot
            ).collect()
            
            test_df.write_parquet(self.config.test_data_path)
            test_rows = test_df.height
            logging.info("Test split saved: %s rows.", test_rows)
            del test_df

            row_counts = {
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
                "total_rows": train_rows + val_rows + test_rows
            }

            return row_counts, round(imbalance_ratio, 4)

        except Exception as e:
            logging.exception("Failed to perform OOT split using Polars.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY METADATA
    # ==========================================================
    def _generate_metadata(
        self, 
        row_counts: Dict[str, int], 
        imbalance_ratio: float, 
        execution_time: float
    ) -> None:
        """
        Generates and saves execution observability telemetry.

        Args:
            row_counts (Dict[str, int]): Row counts for each split.
            imbalance_ratio (float): Calculated scale_pos_weight for XGBoost.
            execution_time (float): Component execution time in seconds.
        """
        try:
            logging.info("Generating Data Ingestion observability metadata.")

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Training Data Ingestion",
                "execution_time_seconds": execution_time,
                "s3_source_uri": self.config.s3_master_panel_uri,
                "split_logic": "Out-Of-Time (OOT) Bitemporal Splitting",
                "split_definitions": {
                    "train_snapshots": self.config.train_snapshots,
                    "val_snapshot": self.config.val_snapshot,
                    "test_snapshot": self.config.test_snapshot
                },
                "data_profiles": {
                    "row_counts": row_counts,
                    "training_class_imbalance_ratio": imbalance_ratio
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            logging.info("Data Ingestion metadata securely saved at: %s", self.config.metadata_file_path)

        except Exception as e:
            logging.exception("Failed to generate metadata.")
            raise CustomException(e, sys) from e