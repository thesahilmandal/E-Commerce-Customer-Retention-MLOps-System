import os
import sys
from typing import Optional
import duckdb

from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from pipelines.inference_pipeline.src.core.config_parser import ConfigParser

class InferencePipelineContext:
    """
    Centralized context manager for the Inference Pipeline.

    ```
    Responsibilities:
    - Acts as a single source of truth for runtime state (e.g., run_id).
    - Injects core dependencies and configurations (ConfigParser, S3Sync).
    - Manages the lifecycle of the out-of-core DuckDB engine, ensuring secure 
    AWS credential resolution upon initialization and safe teardown upon exit.

    This context is designed to be largely immutable during execution to prevent 
    hidden side effects and promote clean separation of concerns.
    """

    def __init__(self, run_id: str) -> None:
        """
        Initializes the Inference Pipeline Context.

        Args:
            run_id (str): A unique identifier for the current inference execution.
        """
        try:
            self._run_id: str = run_id
            self._config: ConfigParser = ConfigParser()
            self._s3_sync: S3Sync = S3Sync()
            self._duckdb_con: Optional[duckdb.DuckDBPyConnection] = None
            
            logging.debug(
                "InferencePipelineContext initialized for run_id: %s", self._run_id
            )
        except Exception as e:
            logging.exception("Failed to initialize InferencePipelineContext.")
            raise CustomException(e, sys) from e

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def config(self) -> ConfigParser:
        return self._config

    @property
    def s3_sync(self) -> S3Sync:
        return self._s3_sync

    @property
    def duckdb_con(self) -> duckdb.DuckDBPyConnection:
        if self._duckdb_con is None:
            raise RuntimeError(
                "DuckDB connection is not initialized. "
                "Ensure context manager (__enter__) is active."
            )
        return self._duckdb_con

    def __enter__(self) -> "InferencePipelineContext":
        """
        Establishes resources bound to the pipeline's lifecycle, specifically 
        the DuckDB connection configured with AWS and HTTPFS capabilities.
        """
        try:
            logging.info("Initializing context resources and DuckDB engine.")
            self._duckdb_con = duckdb.connect(database=":memory:")
            
            # Install and load extensions required for secure S3 file access
            self._duckdb_con.execute("INSTALL httpfs;")
            self._duckdb_con.execute("LOAD httpfs;")
            self._duckdb_con.execute("INSTALL aws;")
            self._duckdb_con.execute("LOAD aws;")

            # Instruct DuckDB to automatically evaluate environment keys, profiles, 
            # and local credential managers using the standard AWS provider chain.
            self._duckdb_con.execute("CALL load_aws_credentials();")
            logging.debug("DuckDB AWS credential chain auto-loaded successfully.")

            # Explicitly apply the region AFTER loading credentials to ensure
            # it is not overwritten by the AWS SDK defaulting mechanism.
            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            self._duckdb_con.execute(f"SET s3_region='{aws_region}';")

            # Optimize memory and thread utilization for inference batch processing
            self._duckdb_con.execute("PRAGMA threads=4;")
            self._duckdb_con.execute("PRAGMA memory_limit='4GB';")
            
            return self

        except Exception as e:
            logging.exception("Failed to initialize context resources.")
            self._safe_close_connection()
            raise CustomException(e, sys) from e

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Safely tears down context resources, ensuring database connections 
        are closed to prevent memory leaks, even if the pipeline crashes.
        """
        logging.info("Tearing down context resources.")
        self._safe_close_connection()

        if exc_type is not None:
            logging.error(
                "InferencePipelineContext exited with exception: %s", exc_val
            )

    def _safe_close_connection(self) -> None:
        """Helper method to safely close the DuckDB connection."""
        try:
            if self._duckdb_con is not None:
                self._duckdb_con.close()
                self._duckdb_con = None
                logging.debug("DuckDB S3 connection safely closed.")
        except Exception as e:
            logging.warning("Error encountered while closing DuckDB connection: %s", str(e))