import os
import sys
import shutil
from typing import Optional
import duckdb

from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from pipelines.inference_pipeline.src.core.config_parser import ConfigParser


class InferencePipelineContext:
    """
    Centralized context manager for the Inference Pipeline.

    Responsibilities:
    - Acts as a single source of truth for runtime state (e.g., run_id).
    - Injects core dependencies and configurations (ConfigParser, S3Sync).
    - Manages the lifecycle of the out-of-core DuckDB engine, ensuring secure 
      AWS credential resolution upon initialization and safe teardown upon exit.
    - Automatically cleans up ephemeral local artifacts post-execution to prevent storage bloat.
    
    This context is designed to be largely immutable during execution to prevent 
    hidden side effects and promote clean separation of concerns.
    """

    def __init__(self, run_id: str, cleanup_on_exit: bool = True) -> None:
        """
        Initializes the Inference Pipeline Context.

        Args:
            run_id (str): A unique identifier for the current inference execution.
            cleanup_on_exit (bool, optional): Whether to delete local artifacts upon exit. Defaults to True.
        """
        try:
            self._run_id: str = run_id
            self._config: ConfigParser = ConfigParser()
            self._s3_sync: S3Sync = S3Sync()
            self._duckdb_con: Optional[duckdb.DuckDBPyConnection] = None
            self.cleanup_on_exit: bool = cleanup_on_exit
            
            # Pre-calculate the root local artifact directory for this run for cleanup purposes
            sys_cfg = self._config.get_system_config()
            self._run_artifact_dir = os.path.join(
                sys_cfg.get("artifact_dir", "artifacts"),
                sys_cfg.get("pipeline_name", "inference_pipeline"),
                self._run_id
            )
            
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

            # Query system configuration parameters
            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            self._duckdb_con.execute(f"SET s3_region='{aws_region}';")

            # Instruct DuckDB to automatically evaluate environment keys, profiles, 
            # and local credential managers using the standard AWS provider chain.
            self._duckdb_con.execute("CALL load_aws_credentials();")
            logging.debug("DuckDB AWS credential chain auto-loaded successfully.")

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
        are closed to prevent memory leaks, even if the pipeline crashes,
        and cleaning up temporary workspaces.
        """
        logging.info("Tearing down context resources.")
        self._safe_close_connection()
        self._cleanup_workspace()

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

    def _cleanup_workspace(self) -> None:
        """
        Deletes the local artifact directory for this specific run to prevent 
        storage bloat on the compute instance. Also cleans up the parent artifact
        directory if it becomes entirely empty after the run cleanup.
        """
        try:
            if not hasattr(self, "_run_artifact_dir"):
                return
                
            if self.cleanup_on_exit and os.path.exists(self._run_artifact_dir):
                # Using ignore_errors=True to forcefully bypass transient OS-level file locks 
                # (e.g., dangling file handles) that could interrupt the cleanup process.
                shutil.rmtree(self._run_artifact_dir, ignore_errors=True)
                
                # Verify deletion was successful
                if not os.path.exists(self._run_artifact_dir):
                    logging.info("Local workspace cleaned up successfully: %s", self._run_artifact_dir)
                else:
                    logging.warning("Could not completely remove local workspace: %s", self._run_artifact_dir)
                
                # Prevent abandoned root directories by attempting to remove the base directory
                # if it is now completely empty.
                base_dir = os.path.dirname(self._run_artifact_dir)
                try:
                    if os.path.exists(base_dir) and not os.listdir(base_dir):
                        os.rmdir(base_dir)
                        logging.debug("Empty base artifact directory removed successfully: %s", base_dir)
                except OSError:
                    # Harmless; the directory either isn't empty (other runs exist) or cannot be deleted.
                    pass
                    
            elif not self.cleanup_on_exit:
                logging.info("cleanup_on_exit is False. Retaining local workspace at: %s", self._run_artifact_dir)
                
        except Exception as e:
            logging.warning("Error encountered while cleaning up local workspace: %s", str(e))