"""
Data Processor Module for the Training Pipeline.

This module is responsible for loading the master dataset from S3 out-of-core using DuckDB,
performing a memory-efficient random split (Train/Validation/Test), isolating features 
and targets, and fitting a stateful Categorical Schema Enforcer. This enforcer guarantees 
that data types during inference perfectly match the types expected by the XGBoost model.
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Tuple

import pandas as pd
import joblib

from shared_core.features.custom_transformers import CategoricalSchemaEnforcer
from pipelines.training_pipeline.src.core.context import PipelineContext
from pipelines.training_pipeline.src.entity.config_entity import DataProcessorConfig
from pipelines.training_pipeline.src.entity.artifact_entity import DataProcessorArtifact
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class DataProcessor:
    """
    Data Processor component for the Training Pipeline.

    Responsibilities:
    - Ingest the Master Panel from S3 using DuckDB for memory efficiency.
    - Perform a robust out-of-core Random Split (Train, Val, Test).
    - Strip system metadata columns.
    - Identify schema definitions (Categorical vs Numerical).
    - Fit and serialize the CategoricalSchemaEnforcer.
    - Save processed datasets and schemas as immutable artifacts for downstream components.
    """

    def __init__(self, config: DataProcessorConfig, context: PipelineContext) -> None:
        """
        Initializes the DataProcessor component.
        """
        try:
            self.config = config
            self.context = context
            logging.info("Training Pipeline: Data Processor component initialized.")
        except Exception as e:
            logging.exception("Failed to initialize Data Processor component.")
            raise CustomException(e, sys) from e

    def run(self) -> DataProcessorArtifact:
        """
        Executes the data processing pipeline stage.
        """
        try:
            logging.info("Starting Data Processor execution.")
            start_time = time.time()

            # 1. Execute Out-of-Core Data Split via DuckDB
            self._split_data_out_of_core()

            # 2. Load splits into Pandas for schema enforcement
            train_df = self._load_local_parquet("tmp_train.parquet")
            val_df = self._load_local_parquet("tmp_val.parquet")
            test_df = self._load_local_parquet("tmp_test.parquet")

            # 3. Clean system columns and isolate target
            X_train, y_train = self._isolate_features_and_target(train_df)
            X_val, y_val = self._isolate_features_and_target(val_df)
            X_test, y_test = self._isolate_features_and_target(test_df)

            # 4. Infer Schema and Fit Enforcer
            categorical_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

            logging.info(
                "Inferred Schema - Categorical: %d features | Numerical: %d features", 
                len(categorical_cols), len(numerical_cols)
            )

            schema_enforcer = CategoricalSchemaEnforcer(
                categorical_features=categorical_cols, 
                numerical_features=numerical_cols
            )
            
            logging.info("Fitting CategoricalSchemaEnforcer on training data.")
            schema_enforcer.fit(X_train)

            # 5. Transform all datasets to enforce rigid types
            X_train = schema_enforcer.transform(X_train)
            X_val = schema_enforcer.transform(X_val)
            X_test = schema_enforcer.transform(X_test)

            # 6. Save Artifacts to Disk
            self._save_datasets(X_train, y_train, X_val, y_val, X_test, y_test)
            
            joblib.dump(schema_enforcer, self.config.preprocessor_file_path)
            logging.info("CategoricalSchemaEnforcer serialized successfully.")

            # 7. Generate Observability and Lineage Metadata
            self._generate_schema_blueprint(schema_enforcer)
            
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(execution_time, X_train, X_val, X_test)

            # 8. Clean up temporary split files
            self._cleanup_temp_files()

            artifact = DataProcessorArtifact(
                preprocessor_file_path=self.config.preprocessor_file_path,
                schema_file_path=self.config.schema_file_path,
                metadata_file_path=self.config.metadata_file_path,
                x_train_file_path=self.config.x_train_file_path,
                y_train_file_path=self.config.y_train_file_path,
                x_val_file_path=self.config.x_val_file_path,
                y_val_file_path=self.config.y_val_file_path,
                x_test_file_path=self.config.x_test_file_path,
                y_test_file_path=self.config.y_test_file_path,
            )

            logging.info("Data Processor execution completed successfully.")
            return artifact

        except Exception as e:
            logging.exception("Data Processor run failed.")
            raise CustomException(e, sys) from e

    def _split_data_out_of_core(self) -> None:
        """
        Executes a memory-efficient random split directly from S3 using DuckDB, 
        saving the fragmented datasets to local disk to prevent OOM errors.
        """
        try:
            s3_uri = self.config.training_dataset_s3_uri_path
            logging.info("Streaming and splitting data out-of-core from S3: %s", s3_uri)

            # Seed DuckDB's random generator for reproducibility
            seed_val = self.config.random_state / 1000.0
            self.context.db_conn.execute(f"SELECT setseed({seed_val});")

            # Create a temporary table assigning a persistent random float to every row
            table_query = f"""
                CREATE OR REPLACE TEMP TABLE source_table AS 
                SELECT *, random() as _split_val 
                FROM read_parquet('{s3_uri}');
            """
            self.context.db_conn.execute(table_query)

            # Calculate bounds
            train_bound = 1.0 - self.config.val_size - self.config.test_size
            val_bound = train_bound + self.config.val_size

            # Paths for temporary out-of-core splits
            tmp_train = os.path.join(self.config.data_processor_dir, "tmp_train.parquet")
            tmp_val = os.path.join(self.config.data_processor_dir, "tmp_val.parquet")
            tmp_test = os.path.join(self.config.data_processor_dir, "tmp_test.parquet")

            logging.info("Executing Train split (<= %.2f)", train_bound)
            self.context.db_conn.execute(f"""
                COPY (SELECT * EXCLUDE(_split_val) FROM source_table WHERE _split_val <= {train_bound}) 
                TO '{tmp_train}' (FORMAT PARQUET);
            """)

            logging.info("Executing Validation split (> %.2f AND <= %.2f)", train_bound, val_bound)
            self.context.db_conn.execute(f"""
                COPY (SELECT * EXCLUDE(_split_val) FROM source_table WHERE _split_val > {train_bound} AND _split_val <= {val_bound}) 
                TO '{tmp_val}' (FORMAT PARQUET);
            """)

            logging.info("Executing Test split (> %.2f)", val_bound)
            self.context.db_conn.execute(f"""
                COPY (SELECT * EXCLUDE(_split_val) FROM source_table WHERE _split_val > {val_bound}) 
                TO '{tmp_test}' (FORMAT PARQUET);
            """)

        except Exception as e:
            logging.exception("Failed to split data out-of-core using DuckDB.")
            raise CustomException(e, sys) from e

    def _load_local_parquet(self, filename: str) -> pd.DataFrame:
        """Loads a locally written Parquet file into a Pandas DataFrame."""
        try:
            filepath = os.path.join(self.config.data_processor_dir, filename)
            return pd.read_parquet(filepath, engine="pyarrow")
        except Exception as e:
            logging.exception("Failed to load local parquet file: %s", filename)
            raise CustomException(e, sys) from e

    def _isolate_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Removes system columns (which must not be used for modeling) and separates 
        the target variable.
        """
        try:
            target_col = self.config.target_column
            
            if target_col not in df.columns:
                raise KeyError(f"Target column '{target_col}' not found in dataset.")

            y = df[target_col]
            
            # Identify columns to drop (system columns + target)
            cols_to_drop = [c for c in self.config.system_columns_to_drop if c in df.columns]
            cols_to_drop.append(target_col)
            
            X = df.drop(columns=cols_to_drop)
            return X, y
            
        except Exception as e:
            logging.exception("Failed to isolate features and target.")
            raise CustomException(e, sys) from e

    def _save_datasets(
        self, 
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        """Writes the final, strictly typed datasets to Parquet."""
        try:
            X_train.to_parquet(self.config.x_train_file_path, engine="pyarrow", index=False)
            y_train.to_frame().to_parquet(self.config.y_train_file_path, engine="pyarrow", index=False)
            
            X_val.to_parquet(self.config.x_val_file_path, engine="pyarrow", index=False)
            y_val.to_frame().to_parquet(self.config.y_val_file_path, engine="pyarrow", index=False)
            
            X_test.to_parquet(self.config.x_test_file_path, engine="pyarrow", index=False)
            y_test.to_frame().to_parquet(self.config.y_test_file_path, engine="pyarrow", index=False)
            
            logging.info("Final feature matrices and target vectors saved successfully.")
        except Exception as e:
            logging.exception("Failed to save final datasets.")
            raise CustomException(e, sys) from e

    def _generate_schema_blueprint(self, enforcer: CategoricalSchemaEnforcer) -> None:
        """
        Creates a JSON representation of the physical data contract. 
        This is used by the Inference Pipeline to validate incoming payloads.
        """
        try:
            schema_blueprint = {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": self.context.run_id
                },
                "features": {
                    "numerical": enforcer.numerical_features,
                    "categorical": enforcer.categories_
                }
            }
            write_json_file(self.config.schema_file_path, schema_blueprint)
            logging.info("Schema blueprint JSON generated successfully.")
        except Exception as e:
            logging.exception("Failed to generate schema blueprint.")
            raise CustomException(e, sys) from e

    def _generate_metadata(
        self, execution_time: float, 
        X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> None:
        """Generates observability telemetry for this pipeline stage."""
        try:
            metadata = {
                "pipeline_stage": "Data Processing & Splitting",
                "execution_time_seconds": execution_time,
                "data_provenance": {
                    "source_dataset_uri": self.config.training_dataset_s3_uri_path,
                    "target_column": self.config.target_column
                },
                "dataset_shapes": {
                    "train_rows": len(X_train),
                    "val_rows": len(X_val),
                    "test_rows": len(X_test),
                    "feature_count": len(X_train.columns)
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(self.config.metadata_file_path, metadata)
            logging.info("Data Processor metadata generated successfully.")
        except Exception as e:
            logging.exception("Failed to generate metadata.")
            raise CustomException(e, sys) from e

    def _cleanup_temp_files(self) -> None:
        """Deletes the temporary un-typed parquet files to save disk space."""
        try:
            for file_name in ["tmp_train.parquet", "tmp_val.parquet", "tmp_test.parquet"]:
                file_path = os.path.join(self.config.data_processor_dir, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
        except Exception as e:
            logging.warning("Failed to clean up temporary split files: %s", str(e))