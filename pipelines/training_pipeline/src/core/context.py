import os
import sys
import shutil
import duckdb

from pipelines.training_pipeline.src.core.config_parser import ConfigParser
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.cloud.s3_operations import S3Sync


class PipelineContext:
    """
    Centralized execution context for the Training Pipeline.

    Responsibilities:
    - Hold the dynamic runtime parameters (run_id, training_dataset_s3_uri_path).
    - Provide centralized access to the parsed YAML configuration.
    - Establish and manage the lifecycle of shared resources (DuckDB connection, S3Sync).
    - Dynamically generate and create local artifact directories isolated by run_id.
    - Support context management protocol (`with` statement) for safe resource teardown.
    - Automatically clean up ephemeral local artifacts post-execution to prevent storage bloat.
    """

    def __init__(
        self,
        run_id: str,
        training_dataset_s3_uri_path: str,
        config_parser: ConfigParser = None,
        cleanup_on_exit: bool = True,
    ) -> None:
        """
        Initializes the Pipeline Context.

        Args:
            run_id (str): Unique identifier for the current pipeline execution.
            training_dataset_s3_uri_path (str): The S3 URI of the master panel dataset.
            config_parser (ConfigParser, optional): Pre-instantiated config parser.
            cleanup_on_exit (bool, optional): Whether to delete local artifacts upon exit. Defaults to True.
        """
        try:
            self.run_id = run_id
            self.training_dataset_s3_uri_path = training_dataset_s3_uri_path
            self.config = config_parser or ConfigParser()
            self.cleanup_on_exit = cleanup_on_exit

            logging.info("Initializing PipelineContext for Run ID: %s", self.run_id)

            # 1. Initialize Shared Cloud Utilities
            self.s3_sync = S3Sync()

            # 2. Setup Isolated Local Artifact Directories
            self._setup_artifact_directories()

            # 3. Initialize Shared Database Connection (DuckDB for Out-of-Core Processing)
            self.db_conn = self._initialize_duckdb()

        except Exception as e:
            logging.exception("Failed to initialize PipelineContext.")
            raise CustomException(e, sys) from e

    def _setup_artifact_directories(self) -> None:
        """
        Dynamically constructs and creates isolated local artifact directories 
        based on the run_id to prevent cross-run contamination.
        """
        try:
            global_config = self.config.get_global_config()
            base_dir = global_config["local_artifact_dir"]

            # Root directory for this specific run
            self.run_artifact_dir = os.path.join(base_dir, self.run_id)

            # Component-specific subdirectories
            self.data_processor_dir = os.path.join(self.run_artifact_dir, "01_data_processor")
            self.model_trainer_dir = os.path.join(self.run_artifact_dir, "02_model_trainer")
            self.model_evaluator_dir = os.path.join(self.run_artifact_dir, "03_model_evaluator")
            self.model_registry_dir = os.path.join(self.run_artifact_dir, "04_model_registry")

            # Create directories on disk
            directories = [
                self.run_artifact_dir,
                self.data_processor_dir,
                self.model_trainer_dir,
                self.model_evaluator_dir,
                self.model_registry_dir,
            ]

            for directory in directories:
                os.makedirs(directory, exist_ok=True)

            logging.debug("Local artifact directories created at: %s", self.run_artifact_dir)

        except Exception as e:
            raise CustomException(e, sys) from e

    def _initialize_duckdb(self) -> duckdb.DuckDBPyConnection:
        """
        Initializes an in-memory DuckDB connection loaded with the httpfs and aws 
        extensions. This allows the Data Processor to query massive Parquet files 
        directly from S3 without loading them entirely into RAM.

        Returns:
            duckdb.DuckDBPyConnection: Configured DuckDB connection.
        """
        try:
            logging.info("Initializing DuckDB in-memory connection with AWS extensions.")
            conn = duckdb.connect(database=":memory:")
            
            # Install and load extensions required for reading S3 Parquet files
            conn.execute("INSTALL httpfs;")
            conn.execute("LOAD httpfs;")
            conn.execute("INSTALL aws;")
            conn.execute("LOAD aws;")
            
            # Automatically load AWS credentials from the environment chain
            conn.execute("CALL load_aws_credentials();")
            
            return conn

        except Exception as e:
            logging.exception("Failed to initialize DuckDB connection.")
            raise CustomException(e, sys) from e

    def _cleanup_workspace(self) -> None:
        """
        Deletes the local artifact directory for this specific run to prevent 
        storage bloat on the compute instance.
        """
        try:
            if not hasattr(self, "run_artifact_dir"):
                return
                
            if self.cleanup_on_exit and os.path.exists(self.run_artifact_dir):
                shutil.rmtree(self.run_artifact_dir)
                logging.info("Local workspace cleaned up successfully: %s", self.run_artifact_dir)
            elif not self.cleanup_on_exit:
                logging.info("cleanup_on_exit is False. Retaining local workspace at: %s", self.run_artifact_dir)
                
        except Exception as e:
            logging.warning("Error encountered while cleaning up local workspace: %s", str(e))

    def close(self) -> None:
        """
        Gracefully closes shared resources, explicitly terminating the DuckDB connection 
        to free up memory and system handles, and cleans up the ephemeral workspace.
        """
        try:
            if hasattr(self, "db_conn") and self.db_conn:
                self.db_conn.close()
                logging.info("DuckDB connection closed successfully.")
        except Exception as e:
            logging.warning("Error encountered while closing DuckDB connection: %s", str(e))
        finally:
            self._cleanup_workspace()

    def __enter__(self) -> "PipelineContext":
        """
        Context manager entry point.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit point. Ensures resources are cleaned up regardless 
        of pipeline execution success or failure.
        """
        self.close()