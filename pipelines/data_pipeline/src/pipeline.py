import sys

from pipelines.data_pipeline.src.entity.config_entity import (
    DataPipelineConfig,
    DataPipelineExtractorConfig,
    DataPipelineValidatorConfig,
    DataPipelineTransformerConfig,
    DataPipelineLoaderConfig,
)

from pipelines.data_pipeline.src.components.data_extractor import Extractor
from pipelines.data_pipeline.src.components.data_validation import Validator
from pipelines.data_pipeline.src.components.data_transformation import Transformer
from pipelines.data_pipeline.src.components.data_loading import Loader

from shared_core.cloud.s3_operations import S3Sync
from shared_core import global_constants
from pipelines.data_pipeline.src import constants
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class DataPipeline:
    """
    Orchestrates the complete Data Pipeline.
    """

    def __init__(self) -> None:
        try:
            logging.info("Initializing Data Pipeline orchestration.")
            self.data_pipeline_config = DataPipelineConfig()
        except Exception as e:
            logging.exception("Failed to initialize Data Pipeline.")
            raise CustomException(e, sys) from e

    def _run_extractor(self):
        try:
            logging.info(">>> Starting Phase 1: Data Extraction")
            config = DataPipelineExtractorConfig(self.data_pipeline_config)
            extractor = Extractor(config=config)
            artifact = extractor.run()
            logging.info("<<< Phase 1: Data Extraction completed successfully.\n")
            return artifact
        except Exception as e:
            logging.exception("Data Extraction phase failed.")
            raise CustomException(e, sys) from e

    def _run_validator(self, extractor_artifact):
        try:
            logging.info(">>> Starting Phase 2: Data Validation")
            config = DataPipelineValidatorConfig(self.data_pipeline_config)
            validator = Validator(config=config, extractor_artifact=extractor_artifact)
            artifact = validator.run()
            logging.info("<<< Phase 2: Data Validation completed successfully.\n")
            return artifact
        except Exception as e:
            logging.exception("Data Validation phase failed.")
            raise CustomException(e, sys) from e

    def _run_transformer(self, extractor_artifact):
        try:
            logging.info(">>> Starting Phase 3: Data Transformation")
            config = DataPipelineTransformerConfig(self.data_pipeline_config)
            transformer = Transformer(
                config=config, extractor_artifact=extractor_artifact
            )
            artifact = transformer.run()
            logging.info("<<< Phase 3: Data Transformation completed successfully.\n")
            return artifact
        except Exception as e:
            logging.exception("Data Transformation phase failed.")
            raise CustomException(e, sys) from e

    def _run_loader(self, transformer_artifact):
        try:
            logging.info(">>> Starting Phase 4: Data Loading")
            config = DataPipelineLoaderConfig(self.data_pipeline_config)
            loader = Loader(config=config, transformer_artifact=transformer_artifact)
            artifact = loader.run()
            logging.info("<<< Phase 4: Data Loading completed successfully.\n")
            return artifact
        except Exception as e:
            logging.exception("Data Loading phase failed.")
            raise CustomException(e, sys) from e

    def _sync_artifacts(self) -> None:
        try:
            logging.info(">>> Starting Phase 6: Artifact Sync (S3)")

            S3Sync().sync_folder_to_s3(
                folder=global_constants.ARTIFACT_DIR_NAME,
                aws_bucket_url=(
                    f"s3://{global_constants.S3_BUCKET_NAME}/"
                    f"{global_constants.ARTIFACT_DIR_NAME}/"
                    f"{constants.DATA_PIPELINE_ROOT_DIR_NAME}"
                ),
            )

            logging.info("<<< Phase 6: Artifact Sync completed successfully.\n")

        except Exception as e:
            logging.exception("Artifact Sync phase failed.")
            raise CustomException(e, sys) from e

    def run(self) -> None:
        try:
            logging.info("===================================================")
            logging.info("STARTING DATA PIPELINE EXECUTION")
            logging.info("===================================================")

            extractor_artifact = self._run_extractor()

            validator_artifact = self._run_validator(extractor_artifact)

            if getattr(validator_artifact, "is_valid", False):
                logging.info("Gate Check Passed: Proceeding with downstream stages.")

                transformer_artifact = self._run_transformer(extractor_artifact)

                loader_artifact = self._run_loader(transformer_artifact)

                self._sync_artifacts()
            else:
                logging.warning(
                    "Gate Check Failed: Data validation unsuccessful. "
                    "Skipping transformation, loading, and syncing."
                )

            logging.info("===================================================")
            logging.info("DATA PIPELINE EXECUTION COMPLETED")
            logging.info("===================================================")

        except Exception as e:
            logging.exception("Critical Failure in Data Pipeline execution.")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        pipeline = DataPipeline()
        pipeline.run()
    except Exception:
        logging.critical(
            "Pipeline execution terminated due to an error.", exc_info=True
        )
        sys.exit(1)