"""
Pipeline Context Module.

This module defines the PipelineContext, which acts as the Dependency Injection 
container and state manager for a single execution of the Data Pipeline.
It encapsulates the runtime parameters, the parsed configuration, the global 
infrastructure clients (S3Sync), and the ephemeral DuckDB database connection.
"""

import os
import sys
from typing import Any, Optional

import duckdb

from pipelines.data_pipeline.src.core.config_parser import DataPipelineConfig
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class PipelineContext:
    """
    State manager and dependency injection container for the pipeline execution.

    Responsibilities:
    - Hold runtime parameters (`run_id`, `start_date`, `end_date`).
    - Hold the parsed pipeline configuration.
    - Initialize and manage the lifecycle of the DuckDB connection with `httpfs`.
    - Provide access to the global `S3Sync` client.
    - Ensure safe teardown of database connections and temporary resources.
    """

    def __init__(
        self,
        run_id: str,
        start_date: str,
        end_date: str,
        config: DataPipelineConfig,
        s3_sync: S3Sync,
    ) -> None:
        """
        Initializes the PipelineContext and establishes the database connection.

        Args:
            run_id (str): Unique identifier for this pipeline execution.
            start_date (str): The start date of the temporal window (YYYY-MM-DD).
            end_date (str): The end date of the temporal window (YYYY-MM-DD).
            config (DataPipelineConfig): The parsed YAML configuration.
            s3_sync (S3Sync): The global AWS S3 synchronization client.
        """
        self.run_id = run_id
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        self.s3_sync = s3_sync
        
        logging.info(
            "Initializing PipelineContext for Run ID: %s | Window: [%s to %s)",
            self.run_id,
            self.start_date,
            self.end_date
        )
        
        self.db_con: Optional[duckdb.DuckDBPyConnection] = None
        self._initialize_duckdb()

    def _initialize_duckdb(self) -> None:
        """
        Initializes an ephemeral, in-memory DuckDB connection with disk-spilling 
        enabled and AWS credentials loaded for direct S3 querying.

        Raises:
            CustomException: If database initialization or extension loading fails.
        """
        try:
            temp_dir = self.config.compute.runtime.temp_directory
            os.makedirs(temp_dir, exist_ok=True)

            logging.debug("Initializing in-memory DuckDB with temp directory: %s", temp_dir)
            
            self.db_con = duckdb.connect(database=":memory:")
            
            # Apply hardware constraints
            self.db_con.execute(f"PRAGMA threads={self.config.compute.hardware.threads};")
            self.db_con.execute(f"PRAGMA memory_limit='{self.config.compute.hardware.memory_limit}';")
            self.db_con.execute(f"PRAGMA temp_directory='{temp_dir}';")
            
            # Load runtime extensions (e.g., httpfs for S3 access)
            for ext in self.config.compute.runtime.extensions:
                logging.debug("Installing and loading DuckDB extension: %s", ext)
                self.db_con.execute(f"INSTALL {ext};")
                self.db_con.execute(f"LOAD {ext};")

            # Automatically resolve credentials via standard AWS environment variables or IAM roles
            if "httpfs" in self.config.compute.runtime.extensions:
                logging.debug("Loading AWS credentials into DuckDB httpfs context.")
                self.db_con.execute("CALL load_aws_credentials();")

            logging.info("DuckDB engine initialized successfully with S3 httpfs support.")

        except Exception as exc:
            logging.exception("Failed to initialize DuckDB connection.")
            raise CustomException(exc, sys) from exc

    def close(self) -> None:
        """
        Safely closes the DuckDB connection and releases associated resources.
        """
        if self.db_con is not None:
            try:
                self.db_con.close()
                logging.info("DuckDB connection closed safely.")
            except Exception as exc:
                logging.warning("Error occurred while closing DuckDB connection: %s", exc)
            finally:
                self.db_con = None

    def __enter__(self) -> "PipelineContext":
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point to guarantee resource cleanup."""
        self.close()