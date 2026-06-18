import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import duckdb

from pipelines.inference_pipeline.src.entity.config_entity import FeatureMatrixGenerationConfig
from pipelines.inference_pipeline.src.entity.artifact_entity import InferenceInputFeatureMatrixBuilderArtifact
from shared_core.features.shared_feature import SharedFeatureGenerator
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file
from shared_core.constants import SYSTEM_COLUMNS_TO_DROP


class FeatureMatrixGeneration:
    """
    Input Feature Matrix Builder for the Inference Pipeline.

    Responsibilities:
    - Establish an out-of-core DuckDB connection directly to the S3 Data Lake.
    - Utilize the `SharedFeatureGenerator` to mathematically enforce zero training-serving skew.
    - Execute a federated query across Hive-partitioned S3 directories.
    - Materialize the active customer scoring population up to the execution snapshot bound.
    - Export the final feature matrix directly to a local Parquet file.
    - Generate strict schema definitions that explicitly isolate Entity IDs from Predictive Features
      to support downstream Validator and Predictor requirements.
    """

    def __init__(self, config: FeatureMatrixGenerationConfig) -> None:
        """
        Initializes the Input Feature Matrix Builder component.
        """
        try:
            self.config = config
            
            # Initialize the Shared Feature Generator configured for Hive-Partitioned S3 Data
            self.feature_generator = SharedFeatureGenerator(
                data_dir=self.config.s3_data_lake_uri,
                is_partitioned=True
            )

            logging.info("Inference Pipeline: Input Feature Matrix Builder initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Input Feature Matrix Builder.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> InferenceInputFeatureMatrixBuilderArtifact:
        """
        Executes the feature engineering pipeline directly against the S3 Data Lake.

        Returns:
            InferenceInputFeatureMatrixBuilderArtifact: Local paths to the generated matrix, schema, and metadata.
        """
        try:
            logging.info("Starting Input Feature Matrix Builder execution.")
            start_time = time.time()

            # 1. Establish DuckDB Connection with S3 Capabilities
            con = self._initialize_duckdb_s3_connection()

            try:
                # 2. Build and export the Feature Matrix (Payload includes Entities + Features)
                self._build_and_export_feature_matrix(con)

                # 3. Generate Schema Definition (Strictly isolates Entities from Features)
                schema_info = self._generate_schema(con)

                # 4. Generate Telemetry & Metadata
                execution_time = round(time.time() - start_time, 2)
                self._generate_metadata(con, schema_info, execution_time)

            finally:
                # Ensure graceful connection closure to prevent memory leaks
                con.close()
                logging.debug("DuckDB S3 connection safely closed.")

            # 5. Package Artifact
            artifact = InferenceInputFeatureMatrixBuilderArtifact(
                feature_matrix_file_path=self.config.feature_matrix_file_path,
                schema_file_path=self.config.schema_file_path,
                metadata_file_path=self.config.metadata_file_path,
                snapshot_date=self.config.snapshot_date
            )

            logging.info("Input Feature Matrix generation completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Critical Failure: Input Feature Matrix Builder run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DATA ENGINE CONFIGURATION
    # ==========================================================
    def _initialize_duckdb_s3_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Initializes an ephemeral DuckDB connection configured with the `aws` extension
        to securely resolve authentication chains natively from the operating environment.
        """
        try:
            logging.info("Initializing DuckDB with native AWS credential auto-discovery.")
            con = duckdb.connect(database=":memory:")
            
            # Install and load extensions required for secure S3 file access
            con.execute("INSTALL httpfs;")
            con.execute("LOAD httpfs;")
            con.execute("INSTALL aws;")
            con.execute("LOAD aws;")

            # Query system configuration parameters
            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            con.execute(f"SET s3_region='{aws_region}';")

            # Instruct DuckDB to automatically evaluate environment keys, configuration profiles, 
            # and local credential managers using the standard AWS provider chain.
            con.execute("CALL load_aws_credentials();")
            logging.info("DuckDB AWS credential chain auto-loaded successfully.")

            # Optimize memory and thread utilization for inference
            con.execute("PRAGMA threads=4;")
            con.execute("PRAGMA memory_limit='4GB';")

            return con

        except Exception as e:
            logging.exception("Failed to initialize DuckDB S3 connection using native chain auto-discovery.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # FEATURE MATRIX GENERATION
    # ==========================================================
    def _build_and_export_feature_matrix(self, con: duckdb.DuckDBPyConnection) -> None:
        """
        Retrieves the exact unified SQL feature logic from the SharedFeatureGenerator,
        executes it against the Hive-partitioned S3 Lake, and streams the output 
        directly into a local Snappy-compressed Parquet file.
        
        Note: The resulting Parquet file intentionally retains system columns (like 
        customer_unique_id) to serve as Entity Keys for downstream prediction mapping.
        """
        try:
            logging.info("Acquiring point-in-time SQL logic from SharedFeatureGenerator.")
            sql_query = self.feature_generator.get_feature_query(self.config.snapshot_date)

            logging.info(
                "Streaming remote feature aggregations directly to local Parquet: %s",
                self.config.feature_matrix_file_path
            )

            # Zero-copy stream to local disk (Bypassing Pandas Memory)
            copy_query = f"""
                COPY ({sql_query}) 
                TO '{self.config.feature_matrix_file_path}' 
                (FORMAT PARQUET, COMPRESSION 'snappy');
            """
            
            con.execute(copy_query)
            logging.info("Feature Matrix materialized and saved successfully.")

        except Exception as e:
            logging.exception("Failed to compile or execute the feature generation query.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # SCHEMA & METADATA GENERATION
    # ==========================================================
    def _generate_schema(self, con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
        """
        Dynamically infers the exact physical schema of the generated inference matrix.
        Crucially, this method implements the Payload Split pattern by explicitly
        isolating Entity IDs (System Columns) from Predictive Features to ensure
        the downstream Validator and Predictor evaluate the correct mathematical contract.
        """
        try:
            logging.info("Inferring structural schema and categorizing Entities vs Features.")
            
            describe_query = f"DESCRIBE SELECT * FROM read_parquet('{self.config.feature_matrix_file_path}')"
            describe_res = con.execute(describe_query).fetchall()

            features_list = []
            entities_list = []
            feature_index = 0
            
            for row in describe_res:
                col_name = str(row[0])
                physical_type = str(row[1])
                is_nullable = str(row[2]).strip().upper() == "YES"
                
                definition = {
                    "name": col_name,
                    "physical_type": physical_type,
                    "is_nullable": is_nullable
                }
                
                # Categorize columns based on the global pipeline constants
                if col_name in SYSTEM_COLUMNS_TO_DROP:
                    definition["description"] = f"Entity Key / System Column: {col_name}"
                    entities_list.append(definition)
                else:
                    definition["index"] = feature_index
                    definition["description"] = f"Predictive Feature: {col_name}"
                    features_list.append(definition)
                    feature_index += 1

            schema_blueprint = {
                "metadata": {
                    "schema_version": "1.0",
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "snapshot_date": self.config.snapshot_date,
                    "environment": "inference"
                },
                "entities": entities_list,
                "features": features_list
            }

            write_json_file(file_path=self.config.schema_file_path, content=schema_blueprint)
            logging.info("Schema definition saved to %s", self.config.schema_file_path)

            return schema_blueprint

        except Exception as e:
            logging.exception("Failed to dynamically generate schema definition.")
            raise CustomException(e, sys) from e

    def _generate_metadata(
        self, 
        con: duckdb.DuckDBPyConnection, 
        schema_blueprint: Dict[str, Any], 
        execution_time: float
    ) -> None:
        """
        Generates rich operational telemetry including source lineage, row counts, 
        and timing metrics for downstream observability and drift detection.
        """
        try:
            logging.info("Compiling telemetry and metadata for the Inference Builder.")

            # Compute actual scoring population size
            count_query = f"SELECT COUNT(*) FROM read_parquet('{self.config.feature_matrix_file_path}')"
            total_rows_res = con.execute(count_query).fetchone()
            total_rows = total_rows_res[0] if total_rows_res else 0
            
            # Extract names for logging
            features_list = schema_blueprint.get("features", [])
            feature_names = [feature.get("name") for feature in features_list]
            
            entities_list = schema_blueprint.get("entities", [])
            entity_names = [entity.get("name") for entity in entities_list]

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Inference Feature Matrix Builder",
                "execution_time_seconds": execution_time,
                "data_provenance": {
                    "s3_data_lake_source": self.config.s3_data_lake_uri,
                    "storage_layout": "Hive Partitioned (year/month/day)",
                    "snapshot_date_anchor": self.config.snapshot_date,
                },
                "scoring_population": {
                    "total_eligible_customers": total_rows,
                    "eligibility_rule": "At least one successful historical order prior to snapshot date.",
                },
                "schema_enforcement": {
                    "strategy": "Payload Split (Entities separated from Features)",
                    "entity_keys_retained": entity_names,
                    "total_features_generated": len(features_list),
                    "features_list": feature_names,
                },
                "artifacts": {
                    "feature_matrix_path": self.config.feature_matrix_file_path,
                    "schema_path": self.config.schema_file_path
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            logging.info(
                "Input Feature Matrix Builder metadata securely saved at: %s", 
                self.config.metadata_file_path
            )

        except Exception as e:
            logging.exception("Failed to generate and save component metadata.")
            raise CustomException(e, sys) from e