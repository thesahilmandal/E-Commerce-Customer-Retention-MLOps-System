"""
Shared Feature Engineering Module.

This module serves as the single source of truth for generating the customer
feature matrix. It is consumed by both the Continuous Training (CT) Data Pipeline
and the Ephemeral Batch Inference Pipeline to mathematically guarantee zero
training-serving skew.
"""

import sys
from datetime import datetime
from typing import Optional

import duckdb

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class SharedFeatureGenerator:
    """
    State-agnostic feature generator utilizing DuckDB for out-of-core SQL execution.
    
    Responsibilities:
    - Centralize core feature definitions (Recency, Frequency, Monetary, etc.).
    - Abstract away underlying physical storage (Flat files vs. Hive partitions).
    - Enforce strict point-in-time constraints to prevent target leakage.
    - Provide the primary key (`customer_unique_id`) for downstream business routing.
    """

    def __init__(self, data_dir: str, is_partitioned: bool = False) -> None:
        """
        Initializes the Feature Generator.

        Args:
            data_dir (str): Root directory or S3 URI containing the Parquet tables.
            is_partitioned (bool): Set to True if data is stored in Hive-style
                                   partitions (e.g., year=YYYY/month=MM).
        """
        self.data_dir = data_dir.rstrip("/")
        self.is_partitioned = is_partitioned

        logging.info(
            "SharedFeatureGenerator initialized. Data directory: %s | Partitioned: %s",
            self.data_dir,
            self.is_partitioned,
        )

    def _validate_snapshot_date(self, snapshot_date: str) -> None:
        """
        Ensures the snapshot_date adheres to the strict ISO format expected by DuckDB.
        """
        try:
            # Accepts both date and datetime string formats
            if len(snapshot_date) == 10:
                datetime.strptime(snapshot_date, "%Y-%m-%d")
            else:
                datetime.strptime(snapshot_date, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            logging.error("Invalid snapshot_date format: %s", snapshot_date)
            raise ValueError(
                f"snapshot_date must be 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'. Got: {snapshot_date}"
            ) from exc

    def _build_table_path(self, base_table_name: str) -> str:
        """
        Constructs the DuckDB read_parquet path, abstracting physical storage layouts.
        """
        if self.is_partitioned:
            # Recursively matches all parquet files inside Hive partition directories
            return f"{self.data_dir}/{base_table_name}/**/*.parquet"
        
        # Falls back to standard flat file layout
        return f"{self.data_dir}/{base_table_name}.parquet"

    def get_feature_query(self, snapshot_date: str) -> str:
        """
        Generates the point-in-time SQL query for the feature matrix.

        Args:
            snapshot_date (str): The execution cutoff timestamp.

        Returns:
            str: The fully formatted DuckDB SQL query string.
        """
        try:
            self._validate_snapshot_date(snapshot_date)

            orders_path = self._build_table_path("olist_orders_dataset")
            customers_path = self._build_table_path("olist_customers_dataset")
            payments_path = self._build_table_path("olist_order_payments_dataset")

            hive_flag = "true" if self.is_partitioned else "false"

            logging.debug("Generating shared feature query for snapshot: %s", snapshot_date)

            return f"""
            WITH valid_orders AS (
                -- 1. Apply strict temporal bounding to prevent data leakage
                SELECT 
                    o.order_id,
                    o.customer_id,
                    TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) AS purchase_ts,
                    TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) AS est_delivery_ts,
                    
                    -- Mask future delivery timestamps to prevent target leakage
                    CASE 
                        WHEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) <= TIMESTAMP '{snapshot_date}' 
                        THEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) 
                        ELSE NULL 
                    END AS act_delivery_ts,
                    
                    -- Mask future status updates to prevent target leakage
                    CASE 
                        WHEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) > TIMESTAMP '{snapshot_date}' 
                        THEN 'processing'
                        ELSE o.order_status
                    END AS order_status
                FROM read_parquet('{orders_path}', hive_partitioning={hive_flag}) o
                WHERE TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) < TIMESTAMP '{snapshot_date}'
            ),
            customer_mapping AS (
                -- 2. Map transactional customer_id to global customer_unique_id
                SELECT customer_id, customer_unique_id, customer_state
                FROM read_parquet('{customers_path}', hive_partitioning={hive_flag})
            ),
            payments AS (
                -- 3. Aggregate payments at the order level
                SELECT order_id, SUM(payment_value) AS order_payment_value
                FROM read_parquet('{payments_path}', hive_partitioning={hive_flag})
                GROUP BY order_id
            ),
            enriched_transactions AS (
                -- 4. Denormalize bounded transactions
                SELECT 
                    cm.customer_unique_id,
                    cm.customer_state,
                    vo.order_id,
                    vo.purchase_ts,
                    vo.est_delivery_ts,
                    vo.act_delivery_ts,
                    vo.order_status,
                    COALESCE(p.order_payment_value, 0) AS payment_value
                FROM valid_orders vo
                JOIN customer_mapping cm ON vo.customer_id = cm.customer_id
                LEFT JOIN payments p ON vo.order_id = p.order_id
            )
            -- 5. Final Aggregation (Analytical Base Table / Feature Matrix)
            SELECT
                customer_unique_id,
                MAX(customer_state) AS customer_state,
                
                -- Recency & Tenure
                DATE_DIFF('day', MAX(purchase_ts), TIMESTAMP '{snapshot_date}') AS recency_days,
                DATE_DIFF('day', MIN(purchase_ts), TIMESTAMP '{snapshot_date}') AS tenure_days,
                
                -- Frequency
                COUNT(DISTINCT order_id) AS frequency,
                
                -- Monetary
                SUM(payment_value) AS monetary_total,
                SUM(payment_value) / NULLIF(COUNT(DISTINCT order_id), 0) AS aov,
                
                -- Behavioral / Operational Risk Factors
                MAX(DATE_DIFF('day', est_delivery_ts, act_delivery_ts)) AS max_delivery_delay_days,
                MAX(CASE WHEN act_delivery_ts IS NULL THEN 1 ELSE 0 END) AS has_undelivered_order,
                SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) AS total_canceled_orders,
                
                -- Lineage & Temporal Metadata
                '{snapshot_date}' AS snapshot_date
            FROM enriched_transactions
            GROUP BY customer_unique_id
            -- Active cohort filter: Only score users who have interacted with the platform
            HAVING COUNT(DISTINCT order_id) >= 1
            """

        except Exception as e:
            logging.exception("Failed to generate feature query.")
            raise CustomException(e, sys) from e

    def execute_query(
        self, con: duckdb.DuckDBPyConnection, snapshot_date: str
    ) -> duckdb.DuckDBPyRelation:
        """
        Utility method to generate and execute the query using a provided DuckDB connection.

        Args:
            con (duckdb.DuckDBPyConnection): Active DuckDB connection.
            snapshot_date (str): The execution cutoff timestamp.

        Returns:
            duckdb.DuckDBPyRelation: The DuckDB relation containing the feature matrix.
                                     Can be converted to Pandas (.df()) or Parquet (.to_parquet()).
        """
        try:
            sql_query = self.get_feature_query(snapshot_date)
            logging.info("Executing shared feature query for snapshot: %s", snapshot_date)
            
            relation = con.execute(sql_query)
            return relation

        except Exception as e:
            logging.exception("Failed to execute feature query via DuckDB.")
            raise CustomException(e, sys) from e