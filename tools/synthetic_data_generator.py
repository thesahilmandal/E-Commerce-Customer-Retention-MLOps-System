"""
Cloud-Native Synthetic Data Generator for Enterprise ML Testing.

This script simulates an upstream Data Engineering batch ETL process. It dynamically
determines the previous day (T-1), validates idempotency via S3 _SUCCESS flags,
and generates realistic, schema-compliant customer, order, and payment records.
It uses a deterministic user pool to ensure historical continuity (allowing downstream
rolling-window feature engineering) and streams data directly to an Amazon S3 Data Lake
using strict Hive-style partitioning (year/month/day).

Requirements:
    pip install pandas pyarrow s3fs python-dotenv

Usage:
    python synthetic_data_generator.py
    # Or override the target date:
    python synthetic_data_generator.py --target-date 2026-06-11
"""

import argparse
import gc
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from dotenv import load_dotenv

# Load AWS Credentials securely from .env file
load_dotenv()

# Fallback for custom logging/exceptions if running outside the main project tree
try:
    from src.custom_exception import CustomException
    from src.custom_logging import logging
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    class CustomException(Exception):
        """Fallback custom exception."""
        pass


class SyntheticDataGeneratorConfig:
    """
    Configuration dataclass for the S3-Native Synthetic Data Generator.
    Responsible for establishing target dates, entity volumes, and cloud targets.
    """

    def __init__(
        self,
        target_date_str: Optional[str] = None,
        num_customers: int = 1000,
        num_orders: int = 1500,
        output_base_dir: str = "s3://company-central-data-lake/bronze",
    ) -> None:
        """
        Initializes configuration, automatically defaulting to T-1 (yesterday) if no date is provided.
        """
        try:
            if target_date_str:
                self.target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
                logging.info("Using explicitly provided target date: %s", self.target_date)
            else:
                # Automatically determine T-1 (Yesterday) in UTC
                self.target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
                logging.info("Automatically determined T-1 target date: %s", self.target_date)

            self.num_customers = num_customers
            self.num_orders = num_orders

            if not output_base_dir.startswith("s3://"):
                raise ValueError(f"output_base_dir must start with 's3://'. Got: {output_base_dir}")
            
            self.output_base_dir = output_base_dir.rstrip("/")
            self.states = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "CE", "PE", "DF"]

            if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
                logging.warning("AWS credentials missing from environment. s3fs will attempt IAM Role fallback.")

        except Exception as e:
            logging.exception("Failed to initialize SyntheticDataGeneratorConfig.")
            raise CustomException(e, sys) from e


class SyntheticDataGenerator:
    """
    Generates structured, schema-compliant synthetic data and persists it
    directly to AWS S3 as Hive-partitioned Parquet files with atomic overwrite.
    """

    def __init__(self, config: SyntheticDataGeneratorConfig) -> None:
        self.config = config

        try:
            self.s3_fs = s3fs.S3FileSystem()
            logging.info("S3 File System connected successfully.")
        except Exception as e:
            logging.error("Failed to initialize S3 FileSystem. Check AWS credentials and network context.")
            raise CustomException(e, sys) from e

    def _get_partition_prefix(self, table_name: str) -> str:
        """
        Constructs the standard Hive-partitioned S3 URI for a given table and target date.
        """
        return (
            f"{self.config.output_base_dir}/{table_name}/"
            f"year={self.config.target_date.year}/"
            f"month={self.config.target_date.month:02d}/"
            f"day={self.config.target_date.day:02d}"
        )

    def _verify_run_requirement(self) -> bool:
        """
        Checks the S3 bucket destination path for a safe ingestion target state.
        Returns True if execution is required, False if the target date is already complete.
        """
        try:
            # Check the orders table as the primary source of truth for completion
            target_path = f"{self._get_partition_prefix('olist_orders_dataset')}/_SUCCESS"
            
            if self.s3_fs.exists(target_path):
                logging.info("Target data partition (%s) already verified on S3. Exiting gracefully.", target_path)
                return False
            return True

        except Exception as e:
            logging.exception("Failed to verify run requirement on S3.")
            raise CustomException(e, sys) from e

    def _get_stable_customer_pool(self) -> List[str]:
        """
        Generates a deterministic pool of global customer IDs. 
        This ensures historical continuity across daily runs, which is mathematically 
        required for downstream rolling-window feature engineering (e.g., LTV, Recency).
        """
        pool = []
        rng = random.Random(42)  # Fixed seed guarantees the exact same UUIDs every execution
        for _ in range(self.config.num_customers):
            pool.append(str(uuid.UUID(int=rng.getrandbits(128))))
        return pool

    def _generate_core_entities(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generates the raw data lists for Customers, Orders, and Payments.
        Ensures temporal accuracy (strictly bound to target date) and relational integrity.
        """
        try:
            logging.info("Generating base entities: %d unique customers...", self.config.num_customers)
            
            global_customers = self._get_stable_customer_pool()
            
            orders_data: List[Dict[str, Any]] = []
            customers_data: List[Dict[str, Any]] = []
            payments_data: List[Dict[str, Any]] = []

            # Temporal boundaries strictly locked to the target_date
            start_of_target_day = datetime.combine(self.config.target_date, datetime.min.time())

            # Partition keys explicitly defined for Parquet dataset writing
            year_val = str(self.config.target_date.year)
            month_val = f"{self.config.target_date.month:02d}"
            day_val = f"{self.config.target_date.day:02d}"

            logging.info("Generating %d orders and simulating purchasing behavior...", self.config.num_orders)

            for _ in range(self.config.num_orders):
                order_id = str(uuid.uuid4())
                transactional_customer_id = str(uuid.uuid4())
                global_customer_id = random.choice(global_customers)

                # Strict Temporal Logic: Adds random seconds to the start of the target day.
                # Guarantees the timestamp never spills over into T-2 or T.
                seconds_forward = random.randint(0, 86399)
                purchase_ts = start_of_target_day + timedelta(seconds=seconds_forward)

                # Delivery Logic
                est_delivery_days = random.randint(5, 15)
                est_delivery_ts = purchase_ts + timedelta(days=est_delivery_days)

                # Status & Actual Delivery
                is_canceled = random.random() < 0.05
                if is_canceled:
                    order_status = "canceled"
                    act_delivery_ts = None
                else:
                    order_status = "delivered"
                    act_delivery_days = random.randint(3, 25)
                    act_delivery_ts = purchase_ts + timedelta(days=act_delivery_days)

                # Append Order
                orders_data.append({
                    "order_id": order_id,
                    "customer_id": transactional_customer_id,
                    "order_purchase_timestamp": purchase_ts,
                    "order_estimated_delivery_date": est_delivery_ts,
                    "order_delivered_customer_date": act_delivery_ts,
                    "order_status": order_status,
                    "year": year_val,
                    "month": month_val,
                    "day": day_val
                })

                # Append Customer Profile mapping
                customers_data.append({
                    "customer_id": transactional_customer_id,
                    "customer_unique_id": global_customer_id,
                    "customer_state": random.choice(self.config.states),
                    "year": year_val,
                    "month": month_val,
                    "day": day_val
                })

                # Append Payments
                num_payments = random.randint(1, 3)
                base_value = round(random.uniform(10.0, 500.0), 2)
                for _ in range(num_payments):
                    payments_data.append({
                        "order_id": order_id,
                        "payment_value": round(base_value / num_payments, 2),
                        "year": year_val,
                        "month": month_val,
                        "day": day_val
                    })

            return pd.DataFrame(orders_data), pd.DataFrame(customers_data), pd.DataFrame(payments_data)

        except Exception as e:
            logging.exception("Failed to generate core entities.")
            raise CustomException(e, sys) from e

    def _atomic_partition_cleanup_s3(self, table_name: str) -> None:
        """
        Removes the specific target date directory in S3 to ensure idempotency.
        Prevents data duplication if a pipeline is rerun after partial failure.
        """
        try:
            partition_prefix = self._get_partition_prefix(table_name)
            
            if self.s3_fs.exists(partition_prefix):
                self.s3_fs.rm(partition_prefix, recursive=True)
                logging.debug("Cleaned existing S3 partition for atomic overwrite: %s", partition_prefix)
                
        except Exception as e:
            logging.exception("Failed to perform atomic partition cleanup in S3.")
            raise CustomException(e, sys) from e

    def _write_hive_partitioned_parquet(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Cleans existing target partitions atomically in S3, then streams the DataFrame directly 
        to the S3 bucket using Hive-style partitioning (year=.../month=.../day=...).
        """
        try:
            self._atomic_partition_cleanup_s3(table_name)

            output_root = f"{self.config.output_base_dir}/{table_name}"
            table = pa.Table.from_pandas(df)

            pq.write_to_dataset(
                table,
                root_path=output_root,
                partition_cols=["year", "month", "day"],
                compression="snappy",
                existing_data_behavior="overwrite_or_ignore",
                filesystem=self.s3_fs
            )
            logging.info("Successfully streamed %s partitioned by year/month/day to S3.", table_name)

        except Exception as e:
            logging.exception("Failed to write partitioned Parquet dataset to S3 for table: %s", table_name)
            raise CustomException(e, sys) from e

    def _write_success_flags(self) -> None:
        """
        Writes a _SUCCESS file to all target partitions to mark safe completion.
        This signals downstream pipelines (like Inference) that the partition is ready to read.
        """
        try:
            tables = ["olist_orders_dataset", "olist_customers_dataset", "olist_order_payments_dataset"]
            for table_name in tables:
                success_flag_path = f"{self._get_partition_prefix(table_name)}/_SUCCESS"
                with self.s3_fs.open(success_flag_path, "w") as f:
                    f.write("")
            logging.info("Idempotency _SUCCESS flags written to all target partitions.")
            
        except Exception as e:
            logging.exception("Failed to write _SUCCESS flags to S3 partitions.")
            raise CustomException(e, sys) from e

    def run(self) -> None:
        """
        Executes the data generation and S3 streaming pipeline.
        Manages memory aggressively for stable execution in restricted environments.
        """
        try:
            logging.info("===================================================")
            logging.info("STARTING S3-NATIVE SYNTHETIC DATA GENERATION (ETL)")
            logging.info("===================================================")

            if not self._verify_run_requirement():
                return

            df_orders, df_customers, df_payments = self._generate_core_entities()

            self._write_hive_partitioned_parquet(df_orders, "olist_orders_dataset")
            del df_orders
            gc.collect()

            self._write_hive_partitioned_parquet(df_customers, "olist_customers_dataset")
            del df_customers
            gc.collect()

            self._write_hive_partitioned_parquet(df_payments, "olist_order_payments_dataset")
            del df_payments
            gc.collect()

            self._write_success_flags()

            logging.info("===================================================")
            logging.info("S3-NATIVE DATA GENERATION COMPLETED SUCCESSFULLY")
            logging.info("===================================================")

        except Exception as e:
            logging.critical("Synthetic Data Generator terminated due to an error.", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate daily synthetic transaction data to AWS S3 for ML pipeline testing. Defaults to T-1."
    )
    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="Optional anchor date for execution (YYYY-MM-DD). If omitted, defaults to Yesterday (T-1).",
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        default=1000,
        help="Number of unique global customers to simulate from the stable pool.",
    )
    parser.add_argument(
        "--num-orders",
        type=int,
        default=1500,
        help="Number of total orders to simulate for the target date.",
    )
    parser.add_argument(
        "--s3-output-dir",
        type=str,
        default="s3://company-central-data-lake/bronze",
        help="S3 root directory for output Parquet files.",
    )

    args = parser.parse_args()

    try:
        config = SyntheticDataGeneratorConfig(
            target_date_str=args.target_date,
            num_customers=args.num_customers,
            num_orders=args.num_orders,
            output_base_dir=args.s3_output_dir,
        )
        generator = SyntheticDataGenerator(config)
        generator.run()
    except Exception as e:
        logging.error("Pipeline initialization failed: %s", str(e))
        sys.exit(1)