import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import duckdb

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class InferenceContext:
    """
    Central execution context for the Inference Pipeline.

    Manages the lifecycle of shared resources including the local artifact scratchpad,
    the centralized metadata ledger, and the out-of-core DuckDB engine connection.
    Implements the Context Manager protocol for safe resource allocation and teardown.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the pipeline context and local workspace.

        Args:
            config (Dict[str, Any]): The parsed pipeline configuration.
        """
        try:
            self.config = config
            self.run_id: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.target_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            # Centralized observability ledger
            self.metadata_ledger: Dict[str, Any] = {
                "inference_run_id": self.run_id,
                "target_date": self.target_date,
                "start_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stages": {}
            }

            # Setup local scratchpad directory
            base_artifact_dir = self.config["system"].get("artifact_dir", "artifacts")
            self.run_dir: str = os.path.join(
                base_artifact_dir, 
                "inference_pipeline", 
                self.run_id
            )
            os.makedirs(self.run_dir, exist_ok=True)

            self.db_connection: Optional[duckdb.DuckDBPyConnection] = None

            logging.info("InferenceContext initialized. Run ID: %s", self.run_id)

        except Exception as e:
            logging.exception("Failed to initialize InferenceContext.")
            raise CustomException(e, sys) from e

    def __enter__(self) -> "InferenceContext":
        """
        Establishes the DuckDB connection with S3 capabilities upon entering the context.
        """
        try:
            logging.info("Establishing in-memory DuckDB connection with native AWS auto-discovery.")
            self.db_connection = duckdb.connect(database=":memory:")
            
            self.db_connection.execute("INSTALL httpfs;")
            self.db_connection.execute("LOAD httpfs;")
            self.db_connection.execute("INSTALL aws;")
            self.db_connection.execute("LOAD aws;")

            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            self.db_connection.execute(f"SET s3_region='{aws_region}';")
            self.db_connection.execute("CALL load_aws_credentials();")

            self.db_connection.execute("PRAGMA threads=4;")
            self.db_connection.execute("PRAGMA memory_limit='4GB';")
            
            logging.debug("DuckDB AWS credential chain auto-loaded successfully.")
            return self

        except Exception as e:
            logging.exception("Failed to initialize DuckDB engine in InferenceContext.")
            raise CustomException(e, sys) from e

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Safely shuts down the DuckDB connection to prevent memory leaks.
        """
        try:
            if self.db_connection is not None:
                self.db_connection.close()
                logging.debug("DuckDB connection safely closed.")
            
            # Record final pipeline outcome in the ledger
            self.metadata_ledger["end_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            self.metadata_ledger["status"] = "FAILURE" if exc_type else "SUCCESS"
            
        except Exception as e:
            logging.exception("Error during InferenceContext teardown.")
            raise CustomException(e, sys) from e

    def add_metadata(self, stage_name: str, stage_metadata: Dict[str, Any]) -> None:
        """
        Appends component-specific execution telemetry to the central metadata ledger.

        Args:
            stage_name (str): Identifier for the pipeline stage.
            stage_metadata (Dict[str, Any]): Telemetry dictionary for the stage.
        """
        self.metadata_ledger["stages"][stage_name] = stage_metadata