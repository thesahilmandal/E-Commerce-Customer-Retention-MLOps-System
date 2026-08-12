import os
import sys
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import duckdb

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import read_yaml


class MonitoringPipelineContext:
    """
    Central context manager for the Monitoring Pipeline.

    Responsibilities:
    - Manage global configurations loaded from the YAML file.
    - Establish and validate run identifiers (run_id, execution_date) to ensure idempotency.
    - Manage the shared DuckDB in-memory database connection and its lifecycle.
    - Provide standardized local artifact paths for all downstream components.
    - Automatically clean up ephemeral local artifacts post-execution to prevent storage bloat.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        execution_date: Optional[str] = None,
        config_path: str = os.path.join("pipelines", "monitoring_pipeline", "configs", "global_config.yaml"),
        cleanup_on_exit: bool = True
    ) -> None:
        """
        Initializes the Monitoring Pipeline Context.

        Args:
            run_id (Optional[str]): Orchestrator-injected run identifier.
            execution_date (Optional[str]): Orchestrator-injected logical execution date (YYYY-MM-DD).
            config_path (str): Path to the pipeline's global configuration file.
            cleanup_on_exit (bool): Whether to delete local artifacts upon exit. Defaults to True.
        """
        try:
            self.config_path: str = config_path
            self.config: Dict[str, Any] = self._load_config()
            self.cleanup_on_exit: bool = cleanup_on_exit

            # Ensure strict data-level idempotency by utilizing passed logical time or current UTC time.
            now = datetime.now(timezone.utc)
            self.run_id: str = run_id if run_id else f"run_{now.strftime('%Y%m%d_%H%M%S')}"
            self.execution_date: str = execution_date if execution_date else now.strftime("%Y-%m-%d")

            # DEFENSIVE FIX: Safely resolve nested dictionary blocks to prevent NoneType errors 
            project_config = self.config.get("project_config") or {}
            
            artifact_dir = project_config.get("local_artifact_dir", "artifacts")
            pipeline_name = project_config.get("pipeline_name", "monitoring_pipeline")

            self.root_dir: str = os.path.join(artifact_dir, pipeline_name, self.run_id)
            os.makedirs(self.root_dir, exist_ok=True)

            self.duckdb_con: Optional[duckdb.DuckDBPyConnection] = None

            logging.info(
                "MonitoringPipelineContext initialized. Run ID: %s | Execution Date: %s",
                self.run_id,
                self.execution_date
            )

        except Exception as e:
            logging.exception("Failed to initialize MonitoringPipelineContext.")
            raise CustomException(e, sys) from e

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads and parses the YAML configuration file safely.

        Returns:
            Dict[str, Any]: Parsed configuration dictionary.
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
            
            config_data = read_yaml(self.config_path)
            
            # DEFENSIVE FIX: Ensure an empty file does not evaluate to None
            if config_data is None:
                logging.warning("Configuration file %s is empty. Using default empty dictionary.", self.config_path)
                return {}
                
            return config_data
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def _cleanup_workspace(self) -> None:
        """
        Deletes the local artifact directory for this specific run to prevent 
        storage bloat on the compute instance. Also cleans up the parent artifact
        directory if it becomes entirely empty after the run cleanup.
        """
        try:
            if not hasattr(self, "root_dir"):
                return
                
            if self.cleanup_on_exit and os.path.exists(self.root_dir):
                # Using ignore_errors=True to forcefully bypass transient OS-level file locks 
                # (e.g., dangling file handles) that could interrupt the cleanup process.
                shutil.rmtree(self.root_dir, ignore_errors=True)
                
                # Verify deletion was successful
                if not os.path.exists(self.root_dir):
                    logging.info("Local workspace cleaned up successfully: %s", self.root_dir)
                else:
                    logging.warning("Could not completely remove local workspace: %s", self.root_dir)
                
                # Prevent abandoned root directories by attempting to remove the base directory
                # if it is now completely empty.
                base_dir = os.path.dirname(self.root_dir)
                try:
                    if os.path.exists(base_dir) and not os.listdir(base_dir):
                        os.rmdir(base_dir)
                        logging.debug("Empty base artifact directory removed successfully: %s", base_dir)
                except OSError:
                    # Harmless; the directory either isn't empty (other runs exist) or cannot be deleted.
                    pass
                    
            elif not self.cleanup_on_exit:
                logging.info("cleanup_on_exit is False. Retaining local workspace at: %s", self.root_dir)
                
        except Exception as e:
            logging.warning("Error encountered while cleaning up local workspace: %s", str(e))

    def __enter__(self) -> "MonitoringPipelineContext":
        """
        Context manager entry point. Initializes shared heavy resources such as the DuckDB engine.
        """
        try:
            logging.info("Entering MonitoringPipelineContext. Initializing shared resources.")
            
            self.duckdb_con = duckdb.connect(database=":memory:")
            self.duckdb_con.execute("INSTALL httpfs;")
            self.duckdb_con.execute("LOAD httpfs;")
            self.duckdb_con.execute("INSTALL aws;")
            self.duckdb_con.execute("LOAD aws;")

            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            self.duckdb_con.execute(f"SET s3_region='{aws_region}';")
            self.duckdb_con.execute("CALL load_aws_credentials();")

            self.duckdb_con.execute("PRAGMA threads=4;")
            self.duckdb_con.execute("PRAGMA memory_limit='4GB';")

            logging.debug("Shared DuckDB connection with AWS extensions initialized successfully.")

            return self
            
        except Exception as e:
            logging.exception("Failed to initialize resources during context entry.")
            self._safe_close_connection()
            raise CustomException(e, sys) from e

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Context manager exit point. Ensures all shared resources are gracefully closed and cleaned up.
        """
        try:
            self._safe_close_connection()
            self._cleanup_workspace()

            if exc_type is not None:
                logging.error("MonitoringPipelineContext exited with an exception: %s", exc_value)
            else:
                logging.info("Exited MonitoringPipelineContext. Resources cleaned up successfully.")
                
        except Exception as e:
            logging.exception("Failed during MonitoringPipelineContext teardown.")
            raise CustomException(e, sys) from e

    def _safe_close_connection(self) -> None:
        """Helper method to safely close the DuckDB connection."""
        try:
            if self.duckdb_con:
                self.duckdb_con.close()
                self.duckdb_con = None
                logging.debug("Shared DuckDB connection safely closed.")
        except Exception as e:
            logging.warning("Error encountered while closing DuckDB connection: %s", str(e))