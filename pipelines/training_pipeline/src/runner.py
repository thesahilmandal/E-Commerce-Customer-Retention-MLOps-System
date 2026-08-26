import sys
import argparse

from pipelines.training_pipeline.src.core.config_parser import ConfigParser
from pipelines.training_pipeline.src.core.context import PipelineContext

from pipelines.training_pipeline.src.entity.config_entity import (
DataProcessorConfig,
ModelTrainerConfig,
ModelEvaluatorConfig,
ModelRegistryConfig,
)

from pipelines.training_pipeline.src.components.data_processor import DataProcessor
from pipelines.training_pipeline.src.components.model_trainer import ModelTrainer
from pipelines.training_pipeline.src.components.model_evaluator import ModelEvaluator
from pipelines.training_pipeline.src.components.model_registry import ModelRegistry

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging

class TrainingPipeline:
    """
    Main orchestrator for the Training Pipeline.

    ```
    Responsibilities:
    - Expose the standardized entry point: `TrainingPipeline.run(run_id, dataset_uri)`.
    - Coordinate the execution of Data Processor, Model Trainer, Evaluator, and Registry.
    - Guarantee safe teardown of database connections and temporary files via context management.
    - Act as the central error boundary, catching component-level exceptions and failing gracefully.
    """

    @classmethod
    def run(cls, run_id: str, training_dataset_s3_uri_path: str) -> None:
        """
        Executes the Training Pipeline DAG end-to-end.

        Args:
            run_id (str): Unique identifier for this pipeline execution, ensuring traceable 
                        lineage and isolated S3/local artifact directories.
            training_dataset_s3_uri_path (str): The exact S3 URI of the Master Panel generated 
                                                by the upstream Data Pipeline.

        Raises:
            CustomException: If any component within the pipeline fails to execute.
        """
        try:
            logging.info("=" * 60)
            logging.info("Starting Training Pipeline Execution")
            logging.info("Run ID: %s", run_id)
            logging.info("Dataset URI: %s", training_dataset_s3_uri_path)
            logging.info("=" * 60)

            # 1. Initialize Configuration
            config_parser = ConfigParser()

            # 2. Context Management (Ensures resources like DuckDB are safely closed)
            with PipelineContext(
                run_id=run_id, 
                training_dataset_s3_uri_path=training_dataset_s3_uri_path, 
                config_parser=config_parser
            ) as context:

                # --- Stage 1: Data Processing ---
                logging.info("\n>>> STAGE 1: Data Processing")
                dp_config = DataProcessorConfig.from_context(context)
                data_processor = DataProcessor(config=dp_config, context=context)
                data_artifact = data_processor.run()

                # --- Stage 2: Model Training ---
                logging.info("\n>>> STAGE 2: Model Training")
                trainer_config = ModelTrainerConfig.from_context(context)
                model_trainer = ModelTrainer(
                    config=trainer_config, 
                    context=context, 
                    data_artifact=data_artifact
                )
                trainer_artifact = model_trainer.run()

                # --- Stage 3: Model Evaluation ---
                logging.info("\n>>> STAGE 3: Model Evaluation")
                evaluator_config = ModelEvaluatorConfig.from_context(context)
                model_evaluator = ModelEvaluator(
                    config=evaluator_config,
                    context=context,
                    data_artifact=data_artifact,
                    trainer_artifact=trainer_artifact
                )
                evaluator_artifact = model_evaluator.run()

                # --- Stage 4: Model Registration ---
                logging.info("\n>>> STAGE 4: Model Registration")
                registry_config = ModelRegistryConfig.from_context(context)
                model_registry = ModelRegistry(
                    config=registry_config,
                    context=context,
                    data_artifact=data_artifact,
                    trainer_artifact=trainer_artifact,
                    evaluator_artifact=evaluator_artifact
                )
                registry_artifact = model_registry.run()

                logging.info("=" * 60)
                if registry_artifact.deployment_status:
                    logging.info("Training Pipeline Completed: NEW CHAMPION DEPLOYED.")
                    logging.info("Vault URI: %s", registry_artifact.s3_model_uri)
                else:
                    logging.info("Training Pipeline Completed: CHALLENGER REJECTED.")
                logging.info("=" * 60)

        except Exception as e:
            logging.exception("Training Pipeline execution failed critically.")
            raise CustomException(e, sys) from e


def main() -> None:
    """
    Main CLI entry point for the Training Pipeline.
    Parses arguments and initiates the pipeline execution.
    """
    parser = argparse.ArgumentParser(
    description="Orchestrator for the ML Training Pipeline.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Unique identifier for this pipeline execution to ensure traceable lineage."
    )

    parser.add_argument(
        "--dataset-uri",
        type=str,
        required=True,
        help="The S3 URI of the Master Panel dataset (e.g., s3://bucket-name/path/to/dataset.parquet)."
    )

    args = parser.parse_args()

    run_id = args.run_id.strip()
    dataset_uri = args.dataset_uri.strip()

    if not run_id:
        logging.error("Validation Error: '--run-id' cannot be empty or whitespace.")
        sys.exit(2)

    if not dataset_uri.startswith("s3://") or not dataset_uri.endswith(".parquet"):
        logging.error("Validation Error: '--dataset-uri' must be a valid S3 URI pointing to a .parquet file.")
        sys.exit(2)

    try:
        TrainingPipeline.run(
            run_id=run_id,
            training_dataset_s3_uri_path=dataset_uri
        )
        sys.exit(0)
    except Exception:
        # Exception details are already logged by the TrainingPipeline,
        # but we exit cleanly with a failure code for Docker/orchestrators.
        sys.exit(1)


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     try:
#         TrainingPipeline.run(
#             run_id="testing_01",
#             training_dataset_s3_uri_path="s3://ml-platform-production/feature_store/testing_01/dataset.parquet"
#         )
#     except Exception as e:
#         raise CustomException(e)