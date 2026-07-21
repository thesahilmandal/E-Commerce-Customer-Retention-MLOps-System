import os
import sys
import shutil
from typing import Dict

import duckdb

from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class HivePartitionGenerator:
    """
    Utility script to transform flat raw Olist datasets into structured, 
    temporally partitioned datasets (year/month/day) within the Bronze Data Lake.

    Leverages S3Sync for robust AWS network I/O and DuckDB for high-performance 
    relational joins, extracting the temporal ingestion boundaries from the 
    orders table to uniformly partition all dependent entities.
    """

    def __init__(self) -> None:
        """
        Initializes the Hive Partition Generator.
        """
        try:
            logging.info("Initializing HivePartitionGenerator.")
            self.s3_sync = S3Sync()
            self.con = duckdb.connect(database=":memory:")
            
            # Performance optimizations for local out-of-core processing
            self.con.execute("PRAGMA threads=4;")
            self.con.execute("PRAGMA memory_limit='4GB';")
            
            self.local_workspace = os.path.join(os.getcwd(), "temp_hive_workspace")
        except Exception as e:
            logging.exception("Failed to initialize HivePartitionGenerator.")
            raise CustomException(e, sys) from e

    def _prepare_workspace(self) -> None:
        """Cleans and creates an ephemeral local workspace for processing."""
        if os.path.exists(self.local_workspace):
            shutil.rmtree(self.local_workspace)
        os.makedirs(self.local_workspace, exist_ok=True)

    def _cleanup_workspace(self) -> None:
        """Removes the ephemeral local workspace to ensure idempotency."""
        if os.path.exists(self.local_workspace):
            shutil.rmtree(self.local_workspace)

    def run_conversion(self) -> None:
        """
        Orchestrates the extraction, temporal partition key generation, and 
        Hive-partitioned materialization for the core Olist datasets.
        """
        try:
            logging.info("Starting production-grade temporal Hive partitioning process.")
            self._prepare_workspace()

            local_raw_orders = os.path.join(self.local_workspace, "raw_orders.parquet")
            local_raw_customers = os.path.join(self.local_workspace, "raw_customers.parquet")
            local_raw_payments = os.path.join(self.local_workspace, "raw_payments.parquet")

            # 1. Download raw datasets locally to facilitate cross-table temporal joins
            logging.info("Downloading raw Olist datasets from S3...")
            self.s3_sync.download_file("s3://ml-platform-production/raw_data/olist_orders_dataset.parquet", local_raw_orders)
            self.s3_sync.download_file("s3://ml-platform-production/raw_data/olist_customers_dataset.parquet", local_raw_customers)
            self.s3_sync.download_file("s3://ml-platform-production/raw_data/olist_order_payments_dataset.parquet", local_raw_payments)

            # 2. Define configurations to inject year/month/day boundaries uniformly
            dataset_configs: Dict[str, Dict[str, str]] = {
                "orders": {
                    "destination": "s3://company-central-data-lake/bronze/orders",
                    "select_query": f"""
                        SELECT *, 
                               strftime(CAST(order_purchase_timestamp AS TIMESTAMP), '%Y') AS year, 
                               strftime(CAST(order_purchase_timestamp AS TIMESTAMP), '%m') AS month,
                               strftime(CAST(order_purchase_timestamp AS TIMESTAMP), '%d') AS day
                        FROM read_parquet('{local_raw_orders}')
                    """
                },
                "customers": {
                    "destination": "s3://company-central-data-lake/bronze/customers",
                    "select_query": f"""
                        SELECT c.*, 
                               strftime(CAST(o.order_purchase_timestamp AS TIMESTAMP), '%Y') AS year, 
                               strftime(CAST(o.order_purchase_timestamp AS TIMESTAMP), '%m') AS month,
                               strftime(CAST(o.order_purchase_timestamp AS TIMESTAMP), '%d') AS day
                        FROM read_parquet('{local_raw_customers}') AS c
                        INNER JOIN read_parquet('{local_raw_orders}') AS o 
                            ON c.customer_id = o.customer_id
                    """
                },
                "order_payments": {
                    "destination": "s3://company-central-data-lake/bronze/order_payments",
                    "select_query": f"""
                        SELECT p.*, 
                               strftime(CAST(o.order_purchase_timestamp AS TIMESTAMP), '%Y') AS year, 
                               strftime(CAST(o.order_purchase_timestamp AS TIMESTAMP), '%m') AS month,
                               strftime(CAST(o.order_purchase_timestamp AS TIMESTAMP), '%d') AS day
                        FROM read_parquet('{local_raw_payments}') AS p
                        INNER JOIN read_parquet('{local_raw_orders}') AS o 
                            ON p.order_id = o.order_id
                    """
                }
            }

            for dataset_name, config in dataset_configs.items():
                logging.info("Processing unified temporal partitioning for: [%s]", dataset_name)
                
                local_partitioned_dir = os.path.join(self.local_workspace, f"partitioned_{dataset_name}")
                os.makedirs(local_partitioned_dir, exist_ok=True)
                
                # Execute out-of-core DuckDB transformation
                copy_query = f"""
                    COPY ({config["select_query"]}) 
                    TO '{local_partitioned_dir}' 
                    (FORMAT PARQUET, PARTITION_BY (year, month, day), COMPRESSION 'snappy', OVERWRITE_OR_IGNORE 1);
                """
                logging.debug("Executing DuckDB COPY query for %s", dataset_name)
                self.con.execute(copy_query)
                
                # Upload materialized partitioned directory to S3
                logging.info("Uploading partitioned dataset to Bronze Data Lake: %s", config["destination"])
                self.s3_sync.sync_folder_to_s3(
                    folder=local_partitioned_dir, 
                    aws_bucket_url=config["destination"]
                )
                logging.info("Successfully completed partitioning for [%s].", dataset_name)

            logging.info("All Olist datasets successfully migrated to day-level temporal Hive partitions.")

        except Exception as e:
            logging.exception("Critical error encountered during temporal Hive partitioning execution.")
            raise CustomException(e, sys) from e
        finally:
            self._cleanup_workspace()
            self.con.close()
            logging.info("DuckDB ephemeral session and workspaces closed safely.")


if __name__ == "__main__":
    generator = HivePartitionGenerator()
    generator.run_conversion()