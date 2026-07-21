"""
Feature Materializer Component for the Continual Learning Data Pipeline.

This module combines point-in-time feature generation with forward-looking 
target label computation across the specified temporal window. It executes out-of-core 
DuckDB queries directly against the S3 Bronze Data Lake via httpfs and exports 
the resulting analytical base table directly to the S3 Feature Store without 
spilling intermediate data to local disk.
"""

import sys
import time

from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException
from shared_core.features.shared_feature import SharedFeatureGenerator
from shared_core.logging.custom_logging import logging


class FeatureMaterializer:
    """
    Feature Materializer Execution Stage.

    Responsibilities:
    - Invoke SharedFeatureGenerator to obtain zero-skew historical feature SQL.
    - Construct forward-looking target label SQL (churn and LTV over target window).
    - Compose and execute unified DuckDB Lakehouse query.
    - Stream compressed Parquet dataset directly from S3 Bronze Lake to S3 Feature Store.
    """

    def __init__(self, context: PipelineContext) -> None:
        """
        Initializes the Feature Materializer component.

        Args:
            context (PipelineContext): The injected pipeline execution context.
        """
        self.context = context
        self.con = self.context.db_con
        self.bronze_uri = self.context.config.storage.bronze_data_lake.base_uri.rstrip("/")
        
        feature_store_cfg = self.context.config.storage.feature_store
        self.s3_output_uri = (
            f"{feature_store_cfg.base_uri.rstrip('/')}/"
            f"{self.context.run_id}/"
            f"{feature_store_cfg.artifact_name}"
        )
        
        self.feature_generator = SharedFeatureGenerator(bronze_base_uri=self.bronze_uri)
        self.target_cfg = self.context.config.business_logic.target_definition
        self.cohort_cfg = self.context.config.business_logic.cohort_definition

        logging.info(
            "FeatureMaterializer initialized. S3 Target Output URI: %s",
            self.s3_output_uri,
        )

    def _build_target_query(self, end_date: str) -> str:
        """
        Generates the forward-looking target SQL query (Churn and LTV).

        Args:
            end_date (str): The execution cutoff timestamp separating past features from future targets.

        Returns:
            str: DuckDB SQL query string calculating target variables.
        """
        orders_path = f"{self.bronze_uri}/orders/**/*.parquet"
        customers_path = f"{self.bronze_uri}/customers/**/*.parquet"
        payments_path = f"{self.bronze_uri}/order_payments/**/*.parquet"

        target_days = self.target_cfg.churn_window_days
        active_statuses = tuple(self.cohort_cfg.active_status_codes)
        status_filter = str(active_statuses) if len(active_statuses) > 1 else f"('{active_statuses[0]}')"

        return f"""
            SELECT
                cm.customer_unique_id,
                COUNT(DISTINCT o.order_id) AS future_orders,
                SUM(p.payment_value) AS future_ltv
            FROM read_parquet('{orders_path}', hive_partitioning=true) o
            JOIN read_parquet('{customers_path}', hive_partitioning=true) cm 
                ON o.customer_id = cm.customer_id
            LEFT JOIN read_parquet('{payments_path}', hive_partitioning=true) p 
                ON o.order_id = p.order_id
            WHERE o.order_status IN {status_filter}
              AND CAST(o.year || '-' || o.month || '-' || o.day AS DATE) >= CAST('{end_date}' AS DATE)
              AND CAST(o.year || '-' || o.month || '-' || o.day AS DATE) <= CAST('{end_date}' AS DATE) + INTERVAL {target_days} DAY
              AND TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) >= TIMESTAMP '{end_date}'
              AND TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) < TIMESTAMP '{end_date}' + INTERVAL {target_days} DAY
            GROUP BY cm.customer_unique_id
        """

    def _build_full_query(self, start_date: str, end_date: str) -> str:
        """
        Merges shared historical features query with target labels query.

        Args:
            start_date (str): Start date of historical feature window.
            end_date (str): Cutoff snapshot date.

        Returns:
            str: Full DuckDB Lakehouse SQL query string.
        """
        base_features_sql = self.feature_generator.get_feature_query(start_date, end_date)
        future_target_sql = self._build_target_query(end_date)

        churn_col = self.target_cfg.churn_column_name
        ltv_col = self.target_cfg.ltv_column_name

        return f"""
            SELECT 
                hf.*,
                COALESCE(ft.future_ltv, 0.0) AS {ltv_col},
                CASE WHEN COALESCE(ft.future_orders, 0) > 0 THEN 0 ELSE 1 END AS {churn_col},
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC' AS ingested_at_utc
            FROM (
                {base_features_sql}
            ) hf
            LEFT JOIN (
                {future_target_sql}
            ) ft 
                ON hf.customer_unique_id = ft.customer_unique_id
        """

    def run(self) -> None:
        """
        Executes feature materialization and streams output directly to S3.

        Raises:
            CustomException: If execution or export fails.
        """
        logging.info(
            "Starting Feature Materialization for window [%s to %s)",
            self.context.start_date,
            self.context.end_date,
        )
        start_time = time.time()

        try:
            full_sql = self._build_full_query(
                self.context.start_date, 
                self.context.end_date
            )

            export_format = self.context.config.storage.feature_store.export_format.upper()
            export_compression = self.context.config.storage.feature_store.export_compression

            copy_query = f"""
                COPY ({full_sql})
                TO '{self.s3_output_uri}'
                (FORMAT {export_format}, COMPRESSION '{export_compression}');
            """

            logging.info("Executing out-of-core DuckDB materialization directly to S3 Feature Store...")
            self.con.execute(copy_query)

            elapsed_time = round(time.time() - start_time, 2)
            logging.info(
                "Feature Materialization completed successfully in %s seconds. Dataset written to %s",
                elapsed_time,
                self.s3_output_uri,
            )

        except Exception as exc:
            logging.exception("Failed during Feature Materialization stage.")
            raise CustomException(exc, sys) from exc