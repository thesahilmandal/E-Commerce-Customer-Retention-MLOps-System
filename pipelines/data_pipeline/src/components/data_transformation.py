import os
import sys
import time
from datetime import datetime, timezone

import duckdb

from pipelines.data_pipeline.src.entity.config_entity import DataTransformationConfig
from pipelines.data_pipeline.src.entity.artifact_entity import (
    DataExtractorArtifact,
    DataTransformationArtifact,
)
from shared_core.features.shared_feature import SharedFeatureGenerator
from shared_core.utils.main_utils import write_json_file
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class DataTransformation:
    """
    Transformer component for generating the Training Master Feature Panel.

    Responsibilities:
    - Utilize the injected SharedFeatureGenerator to ensure zero training-serving skew.
    - Dynamically compute forward-looking targets (churn) strictly isolated from features.
    - Loop through historical snapshots and compile a unified analytical base table.
    - Execute out-of-core via DuckDB directly to a single optimized Parquet file.
    - Generate observability metadata (class balance, null counts, lineage).
    """

    def __init__(
        self,
        config: DataTransformationConfig,
        extractor_artifact: DataExtractorArtifact,
        feature_generator: SharedFeatureGenerator,
    ) -> None:
        """
        Initializes the Transformer component.

        Args:
            config (DataTransformationConfig): Pipeline configuration details.
            extractor_artifact (DataExtractorArtifact): Details of the raw data from the extraction phase.
            feature_generator (SharedFeatureGenerator): Injected feature generator for business logic.
        """
        self.config = config
        self.extractor_artifact = extractor_artifact
        self.raw_data_dir = self.extractor_artifact.raw_data_dir_path

        self.transformed_data_file_path = os.path.join(
            self.config.transformer_root_dir, "master_panel.parquet"
        )
        
        self.feature_generator = feature_generator
        logging.info("Transformer initialized with dependency-injected SharedFeatureGenerator.")

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataTransformationArtifact:
        """
        Executes the out-of-core data transformation pipeline.

        Returns:
            DataTransformationArtifact: Path to the generated Parquet file and metadata.
        """
        logging.info("Starting Data Transformation pipeline via DuckDB.")
        start_time = time.time()

        con = self._initialize_duckdb()

        try:
            self._generate_master_panel(con)

            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(con, execution_time)

            artifact = DataTransformationArtifact(
                transformed_data_file_path=self.transformed_data_file_path,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("Transformer pipeline completed successfully in %ss.", execution_time)
            return artifact

        except duckdb.Error as exc:
            logging.error("DuckDB execution failed during transformation.")
            raise CustomException(exc, sys) from exc
        finally:
            con.close()
            logging.debug("DuckDB connection closed safely.")

    # ==========================================================
    # DATABASE & SQL GENERATION
    # ==========================================================
    def _initialize_duckdb(self) -> duckdb.DuckDBPyConnection:
        """
        Initializes an ephemeral, in-memory DuckDB connection with disk-spilling enabled.
        Prevents database file locks and concurrent execution corruption.
        """
        logging.info("Initializing in-memory DuckDB with temp directory: %s", self.config.duckdb_temp_dir)
        con = duckdb.connect(database=":memory:")
        con.execute(f"PRAGMA threads={self.config.threads}")
        con.execute("PRAGMA memory_limit='8GB'")
        con.execute(f"PRAGMA temp_directory='{self.config.duckdb_temp_dir}'")
        return con

    def _build_target_query(self, snapshot_date: str) -> str:
        """
        Generates the forward-looking target variables (Churn and LTV).
        Mathematically isolated from the shared feature logic to prevent target leakage.
        """
        # Resolving absolute paths for DuckDB Parquet readers
        orders_path = os.path.join(self.raw_data_dir, "olist_orders_dataset.parquet")
        customers_path = os.path.join(self.raw_data_dir, "olist_customers_dataset.parquet")
        payments_path = os.path.join(self.raw_data_dir, "olist_order_payments_dataset.parquet")
        
        return f"""
        SELECT
            cm.customer_unique_id,
            COUNT(DISTINCT o.order_id) AS future_orders,
            SUM(p.payment_value) AS future_ltv
        FROM read_parquet('{orders_path}') o
        JOIN read_parquet('{customers_path}') cm 
            ON o.customer_id = cm.customer_id
        LEFT JOIN read_parquet('{payments_path}') p 
            ON o.order_id = p.order_id
        WHERE o.order_status IN ('delivered', 'shipped')
          AND TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) >= TIMESTAMP '{snapshot_date}'
          AND TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) < TIMESTAMP '{snapshot_date}' + INTERVAL {self.config.target_days} DAY
        GROUP BY cm.customer_unique_id
        """

    def _build_full_snapshot_query(self, snapshot_date: str) -> str:
        """
        Merges the Shared Feature SQL (past) with the Target SQL (future).
        Safely composes dynamic SQL by using derived tables (subqueries) 
        to avoid CTE (WITH clause) nesting violations.
        """
        base_features_sql = self.feature_generator.get_feature_query(snapshot_date)
        future_target_sql = self._build_target_query(snapshot_date)

        return f"""
        SELECT 
            hf.*,
            COALESCE(ft.future_ltv, 0.0) AS target_180d_ltv,
            CASE WHEN COALESCE(ft.future_orders, 0) > 0 THEN 0 ELSE 1 END AS target_is_churn,
            CURRENT_TIMESTAMP AT TIME ZONE 'UTC' AS ingested_at_utc
        FROM (
            {base_features_sql}
        ) hf
        LEFT JOIN (
            {future_target_sql}
        ) ft 
            ON hf.customer_unique_id = ft.customer_unique_id
        """

    def _generate_master_panel(self, con: duckdb.DuckDBPyConnection) -> None:
        """
        Compiles all snapshots into a single UNION ALL query and streams directly to Parquet.
        """
        logging.info("Orchestrating Master Panel generation across %s snapshots.", len(self.config.snapshots))

        snapshot_queries = [
            self._build_full_snapshot_query(date) for date in self.config.snapshots
        ]
        union_query = " UNION ALL ".join(snapshot_queries)

        logging.info("Executing optimized DuckDB execution graph out-of-core...")

        # Zero-Pandas operation: Direct streaming to Snappy-compressed Parquet
        con.execute(f"""
            COPY ({union_query}) 
            TO '{self.transformed_data_file_path}' 
            (FORMAT PARQUET, COMPRESSION 'snappy');
        """)

        logging.info("Successfully exported Master Panel to Parquet without memory spillage.")

    # ==========================================================
    # OBSERVABILITY METADATA
    # ==========================================================
    def _generate_metadata(self, con: duckdb.DuckDBPyConnection, execution_time: float) -> None:
        """
        Generates lineage and telemetry using DuckDB to inspect the generated Parquet file.
        """
        logging.info("Calculating Transformer telemetry via SQL...")

        parquet_path = self.transformed_data_file_path

        total_rows_res = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')").fetchone()
        total_rows = total_rows_res[0] if total_rows_res else 0

        describe_res = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}') LIMIT 1").fetchall()
        columns = [col[0] for col in describe_res]

        churn_rate_res = con.execute(
            f"SELECT AVG(target_is_churn) * 100 FROM read_parquet('{parquet_path}')"
        ).fetchone()
        churn_rate = float(churn_rate_res[0]) if churn_rate_res and churn_rate_res[0] else 0.0

        metadata = {
            "pipeline_stage": "Data Transformation (Training Panel Generation)",
            "architecture_pattern": "Shared Feature Module Integration",
            "execution_time_seconds": execution_time,
            "data_profiles": {
                "total_rows_generated": total_rows,
                "total_columns": len(columns),
                "snapshots_processed": self.config.snapshots,
                "target_days_window": self.config.target_days,
            },
            "business_metrics": {
                "global_churn_rate_percentage": round(churn_rate, 2),
            },
            "schema": {
                "features_and_targets": columns,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_json_file(file_path=self.config.metadata_file_path, content=metadata)
        logging.info("Transformer metadata saved at: %s", self.config.metadata_file_path)