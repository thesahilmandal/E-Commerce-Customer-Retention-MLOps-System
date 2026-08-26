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

from dotenv import load_dotenv

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
    """

    @classmethod
    def run(
        cls,
        run_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        # Load environment variables from .env file if it exists (e.g., for local development)
        load_dotenv()

        if not run_id:
            run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        if not start_date or not end_date:
            error_msg = "Both 'start_date' and 'end_date' must be provided."
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.info("=" * 70)
        logging.info("STARTING DATA PIPELINE EXECUTION")
        logging.info("Run ID: %s | Window: [%s to %s)", run_id, start_date, end_date)
        logging.info("Config Path: %s", config_path)
        logging.info("=" * 70)

        try:
            config_parser = PipelineConfigParser(config_file_path=config_path)
            pipeline_config = config_parser.parse()

            s3_sync = S3Sync()

            with PipelineContext(
                run_id=run_id,
                start_date=start_date,
                end_date=end_date,
                config=pipeline_config,
                s3_sync=s3_sync,
            ) as context:
                
                logging.info(">>> Stage 1/4: Data Discovery")
                discovery = DataDiscovery(context=context)
                discovery.run()

                logging.info(">>> Stage 2/4: Data Validation")
                validation = DataValidation(context=context)
                validation.run()

                logging.info(">>> Stage 3/4: Feature Materialization")
                materializer = FeatureMaterializer(context=context)
                materializer.run()

                logging.info(">>> Stage 4/4: Metadata Registry")
                registry = MetadataRegistry(context=context)
                registry.run()

            logging.info("=" * 70)
            logging.info("DATA PIPELINE EXECUTED SUCCESSFULLY")
            logging.info("Run ID: %s", run_id)
            logging.info("=" * 70)

        except Exception as exc:
            logging.exception("Critical failure encountered during Data Pipeline execution.")
            raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Learning Data Pipeline")
    parser.add_argument("--run-id", type=str, required=False, help="Unique identifier for the run. Auto-generated if omitted.")
    parser.add_argument("--start-date", type=str, required=True, help="Temporal window start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="Temporal window end date (YYYY-MM-DD)")
    parser.add_argument("--config-path", type=str, required=False, default=DEFAULT_CONFIG_PATH, help="Path to pipeline_config.yaml")

    args = parser.parse_args()

    try:
        DataPipeline.run(
            run_id=args.run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            config_path=args.config_path
        )
    except Exception:
        logging.critical(
            "Data Pipeline execution terminated due to an unrecoverable failure.",
            exc_info=True,
        )
        sys.exit(1)


# if __name__ == "__main__":
#     try:
#         DataPipeline.run(
#             run_id="testing_01",
#             start_date="2016-09-01",
#             end_date="2018-03-01"
#         )
#     except Exception:
#         logging.critical(
#             "Data Pipeline execution terminated due to an unrecoverable failure.",
#             exc_info=True,
#         )
#         sys.exit(1)