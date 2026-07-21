"""
Shared Feature Engineering Module.

This module serves as the single source of truth for generating the customer
feature matrix. It is consumed by both the Continuous Training (CT) Data Pipeline
and the Ephemeral Batch Inference Pipeline to mathematically guarantee zero
training-serving skew.
"""

import sys
from datetime import datetime

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class SharedFeatureGenerator:
    """
    State-agnostic feature generator utilizing DuckDB for out-of-core SQL execution.

    Responsibilities:
    - Centralize core business logic and feature definitions (Recency, Frequency, Monetary, etc.).
    - Abstract away underlying physical storage (AWS S3 Hive partitions).
    - Leverage Hive partition pushdown to prevent full S3 table scans (Compute Efficiency).
    - Enforce strict temporal bounding (start_date to end_date) to prevent data leakage.
    - Provide the primary key (`customer_unique_id`) for downstream ML routing.
    """

    def __init__(self, bronze_base_uri: str) -> None:
        """
        Initializes the Shared Feature Generator.

        Args:
            bronze_base_uri (str): Base S3 URI for the Bronze Data Lake
                                   (e.g., 's3://company-central-data-lake/bronze').
        """
        self.bronze_base_uri = bronze_base_uri.rstrip("/")
        logging.info(
            "SharedFeatureGenerator initialized. Bronze Data Lake URI: %s",
            self.bronze_base_uri,
        )

    def _validate_date_format(self, date_str: str, field_name: str) -> None:
        """
        Ensures the provided date adheres to the strict ISO format expected by DuckDB.

        Args:
            date_str (str): The date string to validate.
            field_name (str): The name of the field (for logging/error context).

        Raises:
            ValueError: If the date format is invalid.
        """
        try:
            if len(date_str) == 10:
                datetime.strptime(date_str, "%Y-%m-%d")
            else:
                datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            error_msg = (
                f"Invalid format for {field_name}. Must be 'YYYY-MM-DD' or "
                f"'YYYY-MM-DD HH:MM:SS'. Got: {date_str}"
            )
            logging.error(error_msg)
            raise ValueError(error_msg) from exc

    def _build_table_path(self, table_name: str) -> str:
        """
        Constructs the DuckDB read_parquet path for a given table in the Bronze Data Lake.
        Utilizes wildcard matching to recursively target all Parquet files within Hive partitions.

        Args:
            table_name (str): The logical name of the dataset (e.g., 'orders').

        Returns:
            str: The fully qualified S3 wildcard path.
        """
        return f"{self.bronze_base_uri}/{table_name}/**/*.parquet"

    def get_feature_query(self, start_date: str, end_date: str) -> str:
        """
        Generates the point-in-time SQL query for the feature matrix utilizing
        DuckDB's optimal Hive partition pushdown capabilities.

        Args:
            start_date (str): The beginning of the historical feature window.
            end_date (str): The execution cutoff timestamp (features are calculated AS OF this date).

        Returns:
            str: The fully formatted DuckDB SQL query string.

        Raises:
            CustomException: If query generation fails due to input validation or formatting errors.
        """
        try:
            self._validate_date_format(start_date, "start_date")
            self._validate_date_format(end_date, "end_date")

            # Validate logical temporal ordering
            if start_date >= end_date:
                raise ValueError(f"start_date ({start_date}) must be strictly before end_date ({end_date}).")

            orders_path = self._build_table_path("orders")
            customers_path = self._build_table_path("customers")
            payments_path = self._build_table_path("order_payments")

            logging.debug(
                "Generating shared feature query for temporal window: [%s to %s)",
                start_date,
                end_date,
            )

            # DuckDB Hive Partitioning Optimization Note:
            # We explicitly cast and filter on the virtual partition columns (year, month, day) 
            # to push the predicates down to the S3 network layer, avoiding full bucket scans.
            
            return f"""
            WITH valid_orders AS (
                -- 1. Apply strict temporal bounding and Hive partition pruning to prevent data leakage
                SELECT 
                    o.order_id,
                    o.customer_id,
                    TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) AS purchase_ts,
                    TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) AS est_delivery_ts,
                    
                    -- Mask future delivery timestamps to prevent target leakage in the feature set
                    CASE 
                        WHEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) <= TIMESTAMP '{end_date}' 
                        THEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) 
                        ELSE NULL 
                    END AS act_delivery_ts,
                    
                    -- Mask future status updates to prevent target leakage
                    CASE 
                        WHEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) > TIMESTAMP '{end_date}' 
                        THEN 'processing'
                        ELSE o.order_status
                    END AS order_status
                FROM read_parquet('{orders_path}', hive_partitioning=true) o
                WHERE 
                    -- Hive partition pushdown optimization
                    CAST(year || '-' || month || '-' || day AS DATE) >= CAST('{start_date}' AS DATE)
                    AND CAST(year || '-' || month || '-' || day AS DATE) <= CAST('{end_date}' AS DATE)
                    
                    -- Exact point-in-time timestamp bounding
                    AND TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) >= TIMESTAMP '{start_date}'
                    AND TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) < TIMESTAMP '{end_date}'
            ),
            customer_mapping AS (
                -- 2. Map transactional customer_id to global customer_unique_id with partition pruning
                SELECT customer_id, customer_unique_id, customer_state
                FROM read_parquet('{customers_path}', hive_partitioning=true)
                WHERE 
                    CAST(year || '-' || month || '-' || day AS DATE) >= CAST('{start_date}' AS DATE)
                    AND CAST(year || '-' || month || '-' || day AS DATE) <= CAST('{end_date}' AS DATE)
            ),
            payments AS (
                -- 3. Aggregate payments at the order level with partition pruning
                SELECT order_id, SUM(payment_value) AS order_payment_value
                FROM read_parquet('{payments_path}', hive_partitioning=true)
                WHERE 
                    CAST(year || '-' || month || '-' || day AS DATE) >= CAST('{start_date}' AS DATE)
                    AND CAST(year || '-' || month || '-' || day AS DATE) <= CAST('{end_date}' AS DATE)
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
                    COALESCE(p.order_payment_value, 0.0) AS payment_value
                FROM valid_orders vo
                JOIN customer_mapping cm ON vo.customer_id = cm.customer_id
                LEFT JOIN payments p ON vo.order_id = p.order_id
            )
            -- 5. Final Aggregation (Analytical Base Table / Feature Matrix)
            SELECT
                customer_unique_id,
                MAX(customer_state) AS customer_state,
                
                -- Recency & Tenure relative to the specific pipeline execution window
                DATE_DIFF('day', MAX(purchase_ts), TIMESTAMP '{end_date}') AS recency_days,
                DATE_DIFF('day', MIN(purchase_ts), TIMESTAMP '{end_date}') AS tenure_days,
                
                -- Frequency metrics
                COUNT(DISTINCT order_id) AS frequency,
                
                -- Monetary metrics
                SUM(payment_value) AS monetary_total,
                SUM(payment_value) / NULLIF(COUNT(DISTINCT order_id), 0) AS aov,
                
                -- Behavioral / Operational Risk Factors
                MAX(DATE_DIFF('day', est_delivery_ts, act_delivery_ts)) AS max_delivery_delay_days,
                MAX(CASE WHEN act_delivery_ts IS NULL THEN 1 ELSE 0 END) AS has_undelivered_order,
                SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) AS total_canceled_orders,
                
                -- Lineage & Temporal Metadata
                '{end_date}' AS snapshot_date
            FROM enriched_transactions
            GROUP BY customer_unique_id
            -- Active cohort filter: Only score users who have interacted with the platform in this window
            HAVING COUNT(DISTINCT order_id) >= 1
            """

        except Exception as exc:
            logging.exception("Failed to generate shared feature query.")
            raise CustomException(exc, sys) from exc