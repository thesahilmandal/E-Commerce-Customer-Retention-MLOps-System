import os
import sys

from pipelines.inference_pipeline.src.components.feature_matrix_builder import FeatureMatrixBuilder
from pipelines.inference_pipeline.src.components.inference_validator import InferenceValidator
from pipelines.inference_pipeline.src.components.model_loader import ModelLoader
from pipelines.inference_pipeline.src.components.report_generator import ReportGenerator
from pipelines.inference_pipeline.src.components.report_publisher import ReportPublisher
from pipelines.inference_pipeline.src.core.config_parser import InferenceConfigParser
from pipelines.inference_pipeline.src.core.context import InferenceContext
from pipelines.inference_pipeline.src.entity.artifact_entity import ReportPublishingArtifact
from pipelines.inference_pipeline.src.entity.config_entity import (
    FeatureMatrixBuilderConfig,
    InferenceValidatorConfig,
    ModelLoaderConfig,
    ReportGeneratorConfig,
    ReportPublisherConfig,
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class InferencePipeline:
    """
    Inference Pipeline Orchestrator.

    Coordinates the end-to-end execution of the batch inference workflow. 
    Manages component dependencies, state resolution, schema validation, 
    batch scoring, and secure artifact publishing via a unified InferenceContext.
    """

    def __init__(self) -> None:
        """
        Initializes the InferencePipeline by parsing central configurations.
        """
        try:
            self.config_parser = InferenceConfigParser()
            self.config = self.config_parser.get_config()
            logging.info("InferencePipeline orchestrator initialized.")
        except Exception as e:
            logging.exception("Failed to initialize InferencePipeline.")
            raise CustomException(e, sys) from e

    def run(self) -> ReportPublishingArtifact:
        """
        Executes the Inference Pipeline DAG.

        Returns:
            ReportPublishingArtifact: S3 URIs of the published artifacts and manifest.
        """
        try:
            logging.info("Starting Inference Pipeline execution DAG.")
            
            # Utilize context manager for safe resource initialization and teardown
            with InferenceContext(self.config) as context:
                s3_bucket = self.config["system"]["s3_bucket_name"]
                
                # 1. Model Loading Phase
                model_loader_config = ModelLoaderConfig(
                    s3_pointer_uri=f"s3://{s3_bucket}/{self.config['registry']['registry_dir']}/{self.config['registry']['state_dir']}/{self.config['registry']['pointer_file']}",
                    local_model_file_path=os.path.join(context.run_dir, "model.joblib"),
                    local_schema_file_path=os.path.join(context.run_dir, "expected_schema.json")
                )
                model_loader = ModelLoader(model_loader_config, context)
                model_artifact = model_loader.run()

                # 2. Feature Matrix Construction Phase
                feature_builder_config = FeatureMatrixBuilderConfig(
                    s3_data_lake_uri=f"s3://{s3_bucket}/{self.config['data_lake']['bronze_dir']}",
                    snapshot_date=context.target_date,
                    local_feature_matrix_file_path=os.path.join(context.run_dir, "feature_matrix.parquet"),
                    local_schema_file_path=os.path.join(context.run_dir, "runtime_schema.json")
                )
                feature_builder = FeatureMatrixBuilder(feature_builder_config, context)
                feature_artifact = feature_builder.run()

                # 3. Inference Validation Phase
                validator_config = InferenceValidatorConfig(
                    local_validation_report_path=os.path.join(context.run_dir, "validation_report.json")
                )
                validator = InferenceValidator(validator_config, context)
                validation_artifact = validator.run(model_artifact, feature_artifact)

                # Fail-deadly execution block: Abort on contract breach
                if not validation_artifact.is_valid:
                    raise RuntimeError(
                        "Inference Pipeline aborted: Structural data contract breach detected "
                        "between the Model schema and the Feature Matrix schema."
                    )

                # 4. Report Generation Phase
                report_gen_config = ReportGeneratorConfig(
                    probability_threshold=self.config["business_logic"]["probability_threshold"],
                    system_columns_to_drop=self.config["business_logic"]["system_columns_to_drop"],
                    local_business_report_path=os.path.join(context.run_dir, "churn_predictions.csv"),
                    local_telemetry_log_path=os.path.join(context.run_dir, "telemetry.parquet")
                )
                report_generator = ReportGenerator(report_gen_config, context)
                report_artifact = report_generator.run(model_artifact, feature_artifact)

                # 5. Report Publishing Phase
                report_pub_config = ReportPublisherConfig(
                    run_id=context.run_id,
                    target_date=context.target_date,
                    local_metadata_manifest_path=os.path.join(context.run_dir, f"{context.run_id}_metadata.json"),
                    s3_business_reports_base_uri=f"s3://{s3_bucket}/{self.config['destinations']['business_reports_dir']}",
                    s3_telemetry_logs_base_uri=f"s3://{s3_bucket}/{self.config['destinations']['mlops_telemetry_dir']}",
                    s3_metadata_manifest_base_uri=f"s3://{s3_bucket}/{self.config['destinations']['metadata_manifest_dir']}"
                )
                publisher = ReportPublisher(report_pub_config, context)
                publishing_artifact = publisher.run(report_artifact)

                logging.info("Inference Pipeline DAG executed successfully.")
                return publishing_artifact

        except Exception as e:
            logging.exception("Critical Failure in Inference Pipeline DAG.")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        artifact = InferencePipeline().run()
        print(artifact)
    except Exception as e:
        raise CustomException(e, sys)