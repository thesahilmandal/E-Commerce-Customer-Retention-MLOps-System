"""
Feature Transformation Module for the Training Pipeline.

This module enforces strict categorical schemas to prevent integer mapping misalignment 
in XGBoost and dynamically infers an immutable JSON schema blueprint for downstream inference.
It is highly memory-optimized, utilizing the PyArrow backend to prevent Out-Of-Memory (OOM) 
bottlenecks during large-scale Parquet-to-Pandas data loading.
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from pipelines.training_pipeline.src.entity.config_entity import FeatureTransformationConfig
from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataIngestionArtifact,
    FeatureTransformationArtifact,
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


# ==========================================================
# CUSTOM STATEFUL TRANSFORMERS
# ==========================================================
class CategoricalSchemaEnforcer(BaseEstimator, TransformerMixin):
    """
    Stateful Scikit-Learn transformer to strictly enforce Pandas categorical types.
    
    Why this is required over a stateless function:
    XGBoost relies on the underlying integer codes of Pandas categories. If a validation
    set is missing a category present in the train set, a stateless conversion will misalign
    the integer codes, causing the model to learn and predict incorrect patterns silently.
    
    This transformer learns the exact categorical schema (including unique classes) during 
    the `.fit()` stage on the training data, and applies those exact mappings during `.transform()`.
    Unknown categories in validation/test sets are safely coerced to NaN, which XGBoost 
    handles natively.
    """

    def __init__(self) -> None:
        self.schema_: Dict[str, pd.CategoricalDtype] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CategoricalSchemaEnforcer":
        """
        Learns the exact categorical mappings from the training feature matrix.
        Safely identifies standard and PyArrow-backed string/object columns.
        """
        for col in X.columns:
            if pd.api.types.is_string_dtype(X[col]) or pd.api.types.is_object_dtype(X[col]):
                unique_categories = X[col].dropna().unique()
                self.schema_[col] = pd.CategoricalDtype(
                    categories=unique_categories, ordered=False
                )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """
        Applies the learned categorical mappings to the incoming feature matrix.
        """
        X_out = X.copy()
        for col, cat_dtype in self.schema_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].astype(cat_dtype)
        return X_out


class FeatureTransformation:
    """
    Data Transformation component for the Training Pipeline.

    Responsibilities:
    - Act as a Strict Schema Enforcer rather than a heavy math processor.
    - Resolve DataFrame memory bottlenecks by explicitly utilizing the PyArrow backend.
    - Isolate features (X) from the target variable (y) and system metadata for ALL data splits.
    - Build and fit a stateful categorical preprocessor on the training data.
    - Apply the transformation safely to the Validation and Test sets to ensure schema alignment.
    - Serialize the Scikit-Learn Pipeline as `preprocessor.pkl`.
    - Dynamically infer and generate an immutable, production-ready system `schema.json` blueprint.
    - Save the fully transformed X and y arrays as Parquet files for the Model Training stage.
    """

    def __init__(
        self,
        config: FeatureTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
    ) -> None:
        """
        Initializes the Data Transformation component.
        """
        try:
            self.config = config
            self.ingestion_artifact = ingestion_artifact

            os.makedirs(self.config.data_transformation_root_dir, exist_ok=True)
            logging.info("Training Pipeline: Feature Transformation component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Feature Transformation component.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> FeatureTransformationArtifact:
        """
        Executes the data transformation pipeline across all temporal splits.
        """
        try:
            logging.info("Starting Feature Transformation (Stateful Schema Enforcement).")
            start_time = time.time()

            # 1. Load data from the ingestion artifacts using PyArrow
            train_df = self._load_data(self.ingestion_artifact.train_data_path)
            val_df = self._load_data(self.ingestion_artifact.val_data_path)
            test_df = self._load_data(self.ingestion_artifact.test_data_path)

            # 2. Isolate Features (X) and Target (y)
            X_train, y_train = self._isolate_features_and_target(train_df, "Train")
            X_val, y_val = self._isolate_features_and_target(val_df, "Validation")
            X_test, y_test = self._isolate_features_and_target(test_df, "Test")

            # 3. Build and Fit Pipeline exclusively on Training data
            logging.info("Fitting stateful categorical preprocessor on Train split.")
            preprocessor_pipeline = Pipeline(
                steps=[("schema_enforcer", CategoricalSchemaEnforcer())]
            )
            X_train_transformed = preprocessor_pipeline.fit_transform(X_train)

            # 4. Transform Validation and Test data safely
            logging.info("Applying learned transformations to Validation and Test splits.")
            X_val_transformed = preprocessor_pipeline.transform(X_val)
            X_test_transformed = preprocessor_pipeline.transform(X_test)

            # 5. Serialize Artifacts to Disk
            self._serialize_preprocessor(preprocessor_pipeline)
            self._export_dynamic_production_schema(X_train_transformed)
            self._save_transformed_datasets(
                X_train_transformed, y_train,
                X_val_transformed, y_val,
                X_test_transformed, y_test
            )

            # 6. Generate Observability Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(X_train_transformed, execution_time)

            # 7. Package and Return Artifact
            artifact = FeatureTransformationArtifact(
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

            logging.info("Feature Transformation completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Feature Transformation run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DATA PROCESSING OPERATIONS
    # ==========================================================
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads a Parquet file into a Pandas DataFrame utilizing the PyArrow backend.
        This provides zero-copy memory transfers, resolving the RAM spike bottleneck
        previously caused by Polars-to-Pandas default numpy conversions.
        """
        try:
            return pd.read_parquet(
                file_path, 
                engine="pyarrow", 
                dtype_backend="pyarrow"
            )
        except Exception as e:
            logging.exception("Failed to load data from %s using PyArrow backend.", file_path)
            raise CustomException(e, sys) from e

    def _isolate_features_and_target(
        self, df: pd.DataFrame, split_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separates features (X) from the target (y) and drops system metadata.
        """
        try:
            logging.debug("Isolating features and target for %s split.", split_name)
            
            if self.config.target_column not in df.columns:
                raise ValueError(f"Target column '{self.config.target_column}' missing in {split_name} split.")
            y = df[[self.config.target_column]].copy()

            cols_to_drop = self.config.columns_to_drop + [self.config.target_column]
            cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
            
            X = df.drop(columns=cols_to_drop_existing)
            return X, y

        except Exception as e:
            logging.exception("Failed to isolate features for %s split.", split_name)
            raise CustomException(e, sys) from e

    # ==========================================================
    # ARTIFACT SERIALIZATION
    # ==========================================================
    def _serialize_preprocessor(self, pipeline: Pipeline) -> None:
        """
        Saves the fitted Scikit-Learn pipeline to disk via Joblib.
        """
        try:
            logging.info("Serializing preprocessor pipeline to %s", self.config.preprocessor_file_path)
            joblib.dump(pipeline, self.config.preprocessor_file_path)
        except Exception as e:
            logging.exception("Failed to serialize the preprocessor.")
            raise CustomException(e, sys) from e

    def _export_dynamic_production_schema(self, X: pd.DataFrame) -> None:
        """
        Dynamically extracts structural metadata, types, and constraints from the 
        training dataframe to build an exact, immutable JSON schema blueprint.
        The schema is structured to ensure absolute compatibility with downstream
        Inference Validators.
        """
        try:
            logging.info("Dynamically generating production schema blueprint.")
            
            features_list = []
            
            for index, col_name in enumerate(X.columns):
                physical_type = str(X[col_name].dtype)
                is_nullable = bool(X[col_name].isnull().any())
                
                feature_definition = {
                    "name": col_name,
                    "index": index,
                    "physical_type": physical_type,
                    "is_nullable": is_nullable,
                    "description": f"Feature representing {col_name}."
                }
                
                # Identify Categorical Features
                if physical_type in ["category"] or pd.api.types.is_string_dtype(X[col_name]) or pd.api.types.is_object_dtype(X[col_name]):
                    feature_definition["logical_type"] = "categorical"
                    allowed_vals = sorted([str(val) for val in X[col_name].dropna().unique()])
                    feature_definition["domain"] = {"allowed_values": allowed_vals}
                else:
                    # Treat as Numerical Feature
                    min_val = float(X[col_name].min()) if pd.notnull(X[col_name].min()) else None
                    max_val = float(X[col_name].max()) if pd.notnull(X[col_name].max()) else None
                    
                    # Clean up integers for JSON serialization
                    if min_val is not None and min_val.is_integer():
                        min_val = int(min_val)
                    if max_val is not None and max_val.is_integer():
                        max_val = int(max_val)
                        
                    unique_vals = X[col_name].dropna().unique()
                    
                    # Heuristic for Binary / Flag Numerical Indicators
                    if len(unique_vals) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in unique_vals):
                        feature_definition["logical_type"] = "boolean_indicator"
                        allowed = sorted([int(v) for v in unique_vals])
                        feature_definition["domain"] = {"allowed_values": allowed}
                    else:
                        feature_definition["logical_type"] = "numerical"
                        domain_constraints = {}
                        if min_val is not None:
                            domain_constraints["min"] = min_val
                        if max_val is not None:
                            domain_constraints["max"] = max_val
                        feature_definition["domain"] = domain_constraints

                features_list.append(feature_definition)

            # Assemble Final Schema Structure
            schema_blueprint = {
                "metadata": {
                    "schema_version": "1.0",
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "model_compatibility": "XGBClassifier_Native_Categorical"
                },
                "target": {
                    "name": self.config.target_column,
                    "description": "Forward-looking target variable.",
                    "logical_type": "boolean_indicator"
                },
                "features": features_list
            }
            
            write_json_file(file_path=self.config.schema_file_path, content=schema_blueprint)
            logging.info("Successfully exported dynamic schema blueprint to %s", self.config.schema_file_path)

        except Exception as e:
            logging.exception("Failed to write dynamic schema definition artifact.")
            raise CustomException(e, sys) from e

    def _save_transformed_datasets(
        self,
        X_train: pd.DataFrame, y_train: pd.DataFrame,
        X_val: pd.DataFrame, y_val: pd.DataFrame,
        X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> None:
        """
        Saves the processed X and y DataFrames as Parquet files.
        """
        try:
            logging.info("Saving transformed feature matrices and target vectors to disk.")
            
            X_train.to_parquet(self.config.x_train_file_path, index=False)
            y_train.to_parquet(self.config.y_train_file_path, index=False)
            
            X_val.to_parquet(self.config.x_val_file_path, index=False)
            y_val.to_parquet(self.config.y_val_file_path, index=False)
            
            X_test.to_parquet(self.config.x_test_file_path, index=False)
            y_test.to_parquet(self.config.y_test_file_path, index=False)
            
        except Exception as e:
            logging.exception("Failed to save transformed datasets.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY
    # ==========================================================
    def _generate_metadata(self, X_train: pd.DataFrame, execution_time: float) -> None:
        """
        Generates observability telemetry detailing the strict API contract.
        """
        try:
            logging.info("Generating Data Transformation observability metadata.")

            input_features = X_train.columns.tolist()
            categorical_columns = [
                col for col in input_features 
                if pd.api.types.is_categorical_dtype(X_train[col]) or str(X_train[col].dtype) == "category"
            ]
            numerical_columns = [col for col in input_features if col not in categorical_columns]

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Training Feature Transformation",
                "execution_time_seconds": execution_time,
                "strategy": "Stateful Schema Enforcement (XGBoost Native Support, PyArrow Backend)",
                "api_contract": {
                    "total_features": len(input_features),
                    "input_features": input_features,
                    "categorical_columns": categorical_columns,
                    "numerical_columns": numerical_columns,
                },
                "dropped_system_columns": self.config.columns_to_drop,
                "target_column": self.config.target_column,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            
        except Exception as e:
            logging.exception("Failed to generate metadata.")
            raise CustomException(e, sys) from e