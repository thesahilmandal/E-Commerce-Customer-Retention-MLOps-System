import json
import os
import sys
import time
from typing import Any, Dict, List

from pipelines.inference_pipeline.src.core.context import InferenceContext
from pipelines.inference_pipeline.src.entity.artifact_entity import FeatureMatrixBuilderArtifact
from pipelines.inference_pipeline.src.entity.config_entity import FeatureMatrixBuilderConfig
from shared_core.exceptions.custom_exception import CustomException
from shared_core.features.shared_feature import SharedFeatureGenerator
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class FeatureMatrixBuilder:
    """
    Feature Matrix Builder Component.

    Executes out-of-core feature engineering via DuckDB against the S3 customer data lake,
    materializing the point-in-time feature matrix for batch inference while generating
    the runtime schema contract artifact.
    """

    def __init__(self, config: FeatureMatrixBuilderConfig, context: InferenceContext) -> None:
        """
        Initializes the FeatureMatrixBuilder component.

        Args:
            config (FeatureMatrixBuilderConfig): Configuration specifying lake URIs and target paths.
            context (InferenceContext): Centralized pipeline execution context.
        """
        try:
            self.config = config
            self.context = context

            os.makedirs(os.path.dirname(self.config.local_feature_matrix_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.local_schema_file_path), exist_ok=True)

            logging.info("FeatureMatrixBuilder component initialized.")
        except Exception as e:
            logging.exception("Failed to initialize FeatureMatrixBuilder component.")
            raise CustomException(e, sys) from e

    def run(self) -> FeatureMatrixBuilderArtifact:
        """
        Executes out-of-core feature generation, materializes feature matrix Parquet,
        and produces the structural schema contract JSON file.

        Returns:
            FeatureMatrixBuilderArtifact: Output artifact containing paths to matrix and schema files.
        """
        try:
            logging.info("Starting feature matrix generation for snapshot date: %s", self.config.snapshot_date)
            start_time = time.time()

            # 1. Instantiate shared feature generator to eliminate training-serving skew
            feature_generator = SharedFeatureGenerator(
                bronze_base_uri=self.config.s3_data_lake_uri
            )

            # 2. Build exact point-in-time feature query
            # We use a broad historical start_date to capture all customer history up to the execution snapshot
            feature_query = feature_generator.get_feature_query(
                start_date="2000-01-01",
                end_date=self.config.snapshot_date,
            )

            # 3. Materialize feature matrix directly to local Parquet via DuckDB
            logging.info("Executing DuckDB feature extraction query and exporting to Parquet.")
            copy_sql = f"""
                COPY (
                    {feature_query}
                ) TO '{self.config.local_feature_matrix_file_path}' (FORMAT PARQUET);
            """
            self.context.db_connection.execute(copy_sql)

            # 4. Extract physical schema and total row count from materialized Parquet
            describe_results = self.context.db_connection.execute(
                f"DESCRIBE SELECT * FROM '{self.config.local_feature_matrix_file_path}'"
            ).fetchall()

            total_rows = self.context.db_connection.execute(
                f"SELECT COUNT(*) FROM '{self.config.local_feature_matrix_file_path}'"
            ).fetchone()[0]

            features_list: List[Dict[str, Any]] = []
            for idx, row in enumerate(describe_results):
                col_name = str(row[0])
                col_type = str(row[1]).lower()
                is_nullable = str(row[2]).upper() == "YES" if len(row) > 2 else True

                features_list.append(
                    {
                        "name": col_name,
                        "index": idx,
                        "physical_type": col_type,
                        "is_nullable": is_nullable,
                    }
                )

            # 5. Construct and save schema contract JSON
            schema_contract = {
                "features": features_list,
                "scoring_population": {
                    "total_eligible_customers": total_rows,
                    "snapshot_date": self.config.snapshot_date,
                },
            }
            write_json_file(file_path=self.config.local_schema_file_path, content=schema_contract)
            logging.info("Runtime schema contract saved to: %s", self.config.local_schema_file_path)

            # 6. Record execution metadata in central context ledger
            execution_time = round(time.time() - start_time, 2)
            telemetry: Dict[str, Any] = {
                "snapshot_date": self.config.snapshot_date,
                "total_rows_generated": total_rows,
                "total_features_generated": len(features_list),
                "local_matrix_path": self.config.local_feature_matrix_file_path,
                "local_schema_path": self.config.local_schema_file_path,
                "execution_time_seconds": execution_time,
            }
            self.context.add_metadata("FeatureMatrixBuilder", telemetry)

            # 7. Package and return artifact
            artifact = FeatureMatrixBuilderArtifact(
                feature_matrix_file_path=self.config.local_feature_matrix_file_path,
                schema_file_path=self.config.local_schema_file_path,
                snapshot_date=self.config.snapshot_date,
            )

            logging.info("FeatureMatrixBuilder execution completed successfully.")
            return artifact

        except Exception as e:
            logging.exception("Critical Failure in FeatureMatrixBuilder component.")
            raise CustomException(e, sys) from e