"""
Data Validation Component for the Continual Learning Data Pipeline.

This module performs out-of-core structural and data quality validations 
directly against the S3 Bronze Data Lake using DuckDB httpfs.
It enforces schema integrity and data sanity checks prior to feature 
materialization without downloading physical files to local disk.
"""

import sys
from typing import Dict, List

from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class DataValidation:
    """
    Data Validation Execution Stage.

    Responsibilities:
    - Execute out-of-core schema validation queries directly against S3 via DuckDB httpfs.
    - Verify record counts and temporal partition accessibility within the window.
    - Ensure primary identifiers and critical temporal columns contain no nulls.
    - Enforce configurable quality gates (fail-fast, strict mode).
    """

    # Critical columns per dataset that must pass strict non-null integrity checks
    CRITICAL_COLUMNS: Dict[str, List[str]] = {
        "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
        "customers": ["customer_id", "customer_unique_id"],
        "order_payments": ["order_id", "payment_value"],
    }

    def __init__(self, context: PipelineContext) -> None:
        """
        Initializes the Data Validation component.

        Args:
            context (PipelineContext): The injected pipeline execution context 
                                       containing DuckDB connection and state.
        """
        self.context = context
        self.con = self.context.db_con
        self.bronze_uri = self.context.config.storage.bronze_data_lake.base_uri.rstrip("/")
        self.validation_cfg = self.context.config.validation
        self.datasets = self.context.config.storage.bronze_data_lake.datasets

        logging.info("DataValidation component initialized.")

    def _validate_dataset_schema_and_integrity(self, dataset_name: str) -> None:
        """
        Validates schema structure, record count, and non-null constraints for a dataset.

        Args:
            dataset_name (str): Name of the dataset in the Bronze Data Lake.

        Raises:
            ValueError: If validation rules are violated.
        """
        s3_path = f"{self.bronze_uri}/{dataset_name}/**/*.parquet"
        start_date = self.context.start_date
        end_date = self.context.end_date

        logging.debug("Validating dataset '%s' via DuckDB httpfs at path: %s", dataset_name, s3_path)

        # 1. Schema Check: Verify column presence
        if self.validation_cfg.enable_schema_checks:
            describe_query = f"""
                DESCRIBE SELECT * FROM read_parquet('{s3_path}', hive_partitioning=true) LIMIT 1
            """
            describe_res = self.con.execute(describe_query).fetchall()
            existing_columns = {row[0]: row[1] for row in describe_res}

            logging.debug("Dataset '%s' schema columns: %s", dataset_name, list(existing_columns.keys()))

            required_cols = self.CRITICAL_COLUMNS.get(dataset_name, [])
            missing_cols = [col for col in required_cols if col not in existing_columns]
            if missing_cols:
                raise ValueError(
                    f"Dataset '{dataset_name}' is missing required critical columns: {missing_cols}"
                )

        # 2. Record Count & Temporal Filtering Validation
        if dataset_name == "orders":
            count_query = f"""
                SELECT COUNT(*) 
                FROM read_parquet('{s3_path}', hive_partitioning=true)
                WHERE CAST(year || '-' || month || '-' || day AS DATE) >= CAST('{start_date}' AS DATE)
                  AND CAST(year || '-' || month || '-' || day AS DATE) <= CAST('{end_date}' AS DATE)
                  AND TRY_CAST(order_purchase_timestamp AS TIMESTAMP) >= TIMESTAMP '{start_date}'
                  AND TRY_CAST(order_purchase_timestamp AS TIMESTAMP) < TIMESTAMP '{end_date}'
            """
        else:
            count_query = f"""
                SELECT COUNT(*) 
                FROM read_parquet('{s3_path}', hive_partitioning=true)
                WHERE CAST(year || '-' || month || '-' || day AS DATE) >= CAST('{start_date}' AS DATE)
                  AND CAST(year || '-' || month || '-' || day AS DATE) <= CAST('{end_date}' AS DATE)
            """

        row_count_res = self.con.execute(count_query).fetchone()
        row_count = row_count_res[0] if row_count_res else 0

        logging.info(
            "Validation check for dataset '%s' returned %d rows in temporal window [%s to %s).",
            dataset_name,
            row_count,
            start_date,
            end_date,
        )

        if row_count == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' contains zero records in temporal window [{start_date} to {end_date})."
            )

        # 3. Null Checks on Critical Columns
        if self.validation_cfg.enable_null_checks:
            required_cols = self.CRITICAL_COLUMNS.get(dataset_name, [])
            if required_cols:
                null_selects = [
                    f"SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END) AS null_{col}"
                    for col in required_cols
                ]

                if dataset_name == "orders":
                    where_clause = f"""
                        WHERE CAST(year || '-' || month || '-' || day AS DATE) >= CAST('{start_date}' AS DATE)
                          AND CAST(year || '-' || month || '-' || day AS DATE) <= CAST('{end_date}' AS DATE)
                          AND TRY_CAST(order_purchase_timestamp AS TIMESTAMP) >= TIMESTAMP '{start_date}'
                          AND TRY_CAST(order_purchase_timestamp AS TIMESTAMP) < TIMESTAMP '{end_date}'
                    """
                else:
                    where_clause = f"""
                        WHERE CAST(year || '-' || month || '-' || day AS DATE) >= CAST('{start_date}' AS DATE)
                          AND CAST(year || '-' || month || '-' || day AS DATE) <= CAST('{end_date}' AS DATE)
                    """

                null_query = f"""
                    SELECT {', '.join(null_selects)}
                    FROM read_parquet('{s3_path}', hive_partitioning=true)
                    {where_clause}
                """

                null_res = self.con.execute(null_query).fetchone()
                if null_res:
                    for idx, col in enumerate(required_cols):
                        null_count = null_res[idx] or 0
                        if null_count > 0:
                            msg = (
                                f"Critical column '{col}' in dataset '{dataset_name}' "
                                f"contains {null_count} null values in the target temporal window."
                            )
                            logging.error(msg)
                            if self.validation_cfg.strict_mode:
                                raise ValueError(msg)

    def run(self) -> None:
        """
        Executes the data validation pipeline across all configured datasets.

        Raises:
            CustomException: If data validation fails and fail_fast or strict_mode is enabled.
        """
        logging.info(
            "Starting Data Validation stage for temporal window: [%s to %s)",
            self.context.start_date,
            self.context.end_date,
        )

        validation_errors: List[str] = []

        try:
            for dataset in self.datasets:
                try:
                    self._validate_dataset_schema_and_integrity(dataset.name)
                except Exception as exc:
                    error_msg = f"Validation failed for dataset '{dataset.name}': {exc}"
                    logging.error(error_msg)
                    validation_errors.append(error_msg)

                    if self.validation_cfg.fail_fast:
                        raise ValueError(error_msg) from exc

            if validation_errors:
                combined_msg = "Data Validation stage failed with the following errors:\n" + "\n".join(validation_errors)
                raise ValueError(combined_msg)

            logging.info("Data Validation completed successfully. All quality gates passed.")

        except Exception as exc:
            logging.exception("Data Validation stage failed quality gates.")
            raise CustomException(exc, sys) from exc