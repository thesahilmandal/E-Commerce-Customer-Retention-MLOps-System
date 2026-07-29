import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.features.shared_feature import SharedFeatureGenerator
from pipelines.inference_pipeline.src.core.context import InferencePipelineContext
from pipelines.inference_pipeline.src.entity.config_entity import FeatureMatrixBuilderConfig
from pipelines.inference_pipeline.src.entity.artifact_entity import FeatureMatrixBuilderArtifact


class FeatureMatrixBuilder:
    """
    Feature Matrix Builder Component.

    Responsibilities:
    - Guarantees Zero Training-Serving Skew by invoking the exact same 
      `SharedFeatureGenerator` used during the Training Pipeline.
    - Generates the deterministic point-in-time feature SQL query.
    - Executes out-of-core SQL transformations via the injected DuckDB engine.
    - Materializes the inference feature matrix locally as a Parquet file.
    - Extracts the structural data schema for downstream validation.
    """

    def __init__(self, context: InferencePipelineContext) -> None:
        """
        Initializes the Feature Matrix Builder component.

        Args:
            context (InferencePipelineContext): The centralized execution context providing
                                                configurations and the DuckDB engine.
        """
        try:
            self.context = context
            self.config = FeatureMatrixBuilderConfig.from_context(context)
            logging.info("Inference Pipeline: Feature Matrix Builder component initialized.")
        except Exception as e:
            logging.exception("Failed to initialize Feature Matrix Builder component.")
            raise CustomException(e, sys) from e

    def run(self) -> FeatureMatrixBuilderArtifact:
        """
        Executes the out-of-core feature engineering workflow.

        Returns:
            FeatureMatrixBuilderArtifact: Dataclass containing paths to the materialized 
                                          feature matrix, schema, and metadata.
        """
        try:
            logging.info("Starting Feature Matrix Construction Sequence.")
            start_time = time.time()

            # 1. Build Feature Matrix using Shared Core Engine
            self._generate_feature_matrix()

            # 2. Extract Structural Schema
            self._extract_schema()

            # 3. Generate Operational Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(execution_time=execution_time)

            # 4. Package Artifact
            artifact = FeatureMatrixBuilderArtifact(
                feature_matrix_file_path=self.config.feature_matrix_file_path,
                schema_file_path=self.config.schema_file_path,
                metadata_file_path=self.config.metadata_file_path,
                snapshot_date=self.config.snapshot_date
            )

            logging.info("Feature Matrix Construction completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Critical Failure inside Feature Matrix Builder execution routine.")
            raise CustomException(e, sys) from e

    def _generate_feature_matrix(self) -> None:
        """
        Invokes the domain-driven SharedFeatureGenerator to obtain the zero-skew feature query, 
        and executes it via DuckDB to materialize the results directly to Parquet.
        """
        try:
            logging.info("Executing SharedFeatureGenerator for temporal snapshot: %s", self.config.snapshot_date)
            
            # The SharedFeatureGenerator manages S3 logical paths internally using the Bronze Data Lake URI
            feature_generator = SharedFeatureGenerator(bronze_base_uri=self.config.s3_data_lake_uri)
            
            # Define the historical window. A static early date ensures we capture the lifetime behavior 
            # of all active customers up to the snapshot date. DuckDB's partition pushdown handles pruning.
            historical_start_date = "2015-01-01"
            
            # Retrieve the deterministic SQL string
            query = feature_generator.get_feature_query(
                start_date=historical_start_date,
                end_date=self.config.snapshot_date
            )
            
            # Execute and materialize out-of-core
            copy_query = f"COPY ({query}) TO '{self.config.feature_matrix_file_path}' (FORMAT PARQUET);"
            self.context.duckdb_con.execute(copy_query)
            
            logging.debug("Feature matrix successfully materialized at: %s", self.config.feature_matrix_file_path)

        except Exception as e:
            logging.exception("Failed to generate out-of-core feature matrix.")
            raise CustomException(e, sys) from e

    def _extract_schema(self) -> None:
        """
        Queries the materialized Parquet file via DuckDB to extract its exact 
        structural schema, ensuring precise compatibility validation downstream.
        """
        try:
            logging.info("Extracting structural schema from materialized feature matrix.")
            
            query = f"DESCRIBE SELECT * FROM '{self.config.feature_matrix_file_path}'"
            result = self.context.duckdb_con.execute(query).fetchall()

            # result format: [(column_name, column_type, null, key, default, extra), ...]
            schema = [
                {"name": row[0], "physical_type": row[1]}
                for row in result
            ]

            with open(self.config.schema_file_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=4)
                
            logging.debug("Feature matrix schema saved to: %s", self.config.schema_file_path)

        except Exception as e:
            logging.exception("Failed to extract or save feature matrix schema.")
            raise CustomException(e, sys) from e

    def _generate_metadata(self, execution_time: float) -> None:
        """
        Generates comprehensive operational metadata, including volumetric counts 
        from the DuckDB engine.
        """
        try:
            # Efficient out-of-core row counting
            count_query = f"SELECT COUNT(*) FROM '{self.config.feature_matrix_file_path}'"
            total_customers = self.context.duckdb_con.execute(count_query).fetchone()[0]

            file_size = os.path.getsize(self.config.feature_matrix_file_path)

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Feature Matrix Builder",
                "inference_run_id": self.context.run_id,
                "execution_time_seconds": execution_time,
                "data_provenance": {
                    "snapshot_date": self.config.snapshot_date,
                    "total_customers_scored": total_customers,
                    "feature_matrix_size_bytes": file_size
                },
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }

            with open(self.config.metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
                
            logging.debug("Feature Matrix Builder metadata saved to: %s", self.config.metadata_file_path)

        except Exception as e:
            logging.exception("Failed to generate Feature Matrix Builder metadata.")
            raise CustomException(e, sys) from e