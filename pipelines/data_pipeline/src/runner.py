"""
Data Pipeline Orchestration Runner.

This module serves as the primary entry point and orchestrator for the 
Continual Learning Data Pipeline. It loads the pipeline configuration, 
initializes the runtime execution context (PipelineContext), and sequentially 
executes the stateless pipeline components (Data Discovery, Data Validation, 
Feature Materialization, Metadata Registry) within a single, managed lifecycle.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from pipelines.data_pipeline.src.components.data_discovery import DataDiscovery
from pipelines.data_pipeline.src.components.data_validation import DataValidation
from pipelines.data_pipeline.src.components.feature_materializer import FeatureMaterializer
from pipelines.data_pipeline.src.components.metadata_registry import MetadataRegistry
from pipelines.data_pipeline.src.core.config_parser import PipelineConfigParser
from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


DEFAULT_CONFIG_PATH = os.path.join(
    "pipelines", "data_pipeline", "configs", "pipeline_config.yaml"
)


class DataPipeline:
    """
    Orchestrator class for the Continual Learning Data Pipeline.

    Responsibilities:
    - Parse YAML pipeline configuration.
    - Instantiate shared platform utilities (S3Sync) and pipeline context.
    - Execute pipeline components sequentially in a managed lifecycle.
    - Handle exceptions gracefully and ensure proper resource cleanup.
    """

    @classmethod
    def run(
        cls,
        run_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        """
        Executes the end-to-end Data Pipeline for a given temporal window.

        Args:
            run_id (Optional[str]): Unique run identifier. Auto-generated if None.
            start_date (Optional[str]): Temporal window start date (YYYY-MM-DD).
            end_date (Optional[str]): Temporal window end date (YYYY-MM-DD).
            config_path (str): Path to the pipeline_config.yaml file.

        Raises:
            CustomException: If any pipeline stage fails or inputs are invalid.
        """
        if not run_id:
            run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        if not start_date or not end_date:
            error_msg = "Both 'start_date' and 'end_date' must be provided to run the Data Pipeline."
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.info("=" * 70)
        logging.info("STARTING CONTINUAL LEARNING DATA PIPELINE EXECUTION")
        logging.info("Run ID: %s | Window: [%s to %s)", run_id, start_date, end_date)
        logging.info("Config Path: %s", config_path)
        logging.info("=" * 70)

        try:
            # 1. Parse configuration
            config_parser = PipelineConfigParser(config_file_path=config_path)
            pipeline_config = config_parser.parse()

            # 2. Instantiate global shared utilities
            s3_sync = S3Sync()

            # 3. Initialize PipelineContext within a context manager for safe teardown
            with PipelineContext(
                run_id=run_id,
                start_date=start_date,
                end_date=end_date,
                config=pipeline_config,
                s3_sync=s3_sync,
            ) as context:
                
                # Stage 1: Data Discovery
                logging.info(">>> Stage 1/4: Data Discovery")
                discovery = DataDiscovery(context=context)
                discovery.run()

                # Stage 2: Data Validation
                logging.info(">>> Stage 2/4: Data Validation")
                validation = DataValidation(context=context)
                validation.run()

                # Stage 3: Feature Materialization
                logging.info(">>> Stage 3/4: Feature Materialization")
                materializer = FeatureMaterializer(context=context)
                materializer.run()

                # Stage 4: Metadata Registry
                logging.info(">>> Stage 4/4: Metadata Registry")
                registry = MetadataRegistry(context=context)
                registry.run()

            logging.info("=" * 70)
            logging.info("CONTINUAL LEARNING DATA PIPELINE EXECUTED SUCCESSFULLY")
            logging.info("Run ID: %s", run_id)
            logging.info("=" * 70)

        except Exception as exc:
            logging.exception("Critical failure encountered during Data Pipeline execution.")
            raise CustomException(exc, sys) from exc


def parse_cli_args() -> argparse.Namespace:
    """
    Parses command-line arguments for manual or containerized pipeline execution.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the Continual Learning Production Data Pipeline."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Unique identifier for the pipeline run. Generated if omitted.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date for temporal extraction window (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date for temporal extraction window (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline_config.yaml.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_cli_args()
        DataPipeline.run(
            run_id=args.run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            config_path=args.config_path,
        )
    except Exception:
        logging.critical(
            "Data Pipeline execution terminated due to an unrecoverable failure.",
            exc_info=True,
        )
        sys.exit(1)