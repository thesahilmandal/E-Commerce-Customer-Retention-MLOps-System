import json
import os
import sys
import time
from typing import Any, Dict, List, Set

from pipelines.inference_pipeline.src.core.context import InferenceContext
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    FeatureMatrixBuilderArtifact,
    InferenceValidationArtifact,
    ModelLoadingArtifact,
)
from pipelines.inference_pipeline.src.entity.config_entity import InferenceValidatorConfig
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class InferenceValidator:
    """
    Inference Validator Component.

    Strictly enforces data contracts between the deployment-ready Model schema 
    (expected) and the just-in-time materialized Feature Matrix schema (runtime).
    Implements a Schema Evolution Adapter to bridge low-level engine type differences 
    (e.g., DuckDB BIGINT to Scikit-Learn Float64) to prevent silent execution bugs.
    """

    def __init__(self, config: InferenceValidatorConfig, context: InferenceContext) -> None:
        """
        Initializes the InferenceValidator component.

        Args:
            config (InferenceValidatorConfig): Configuration specifying output paths.
            context (InferenceContext): Centralized pipeline execution context.
        """
        try:
            self.config = config
            self.context = context
            os.makedirs(os.path.dirname(self.config.local_validation_report_path), exist_ok=True)
            logging.info("InferenceValidator component initialized.")
        except Exception as e:
            logging.exception("Failed to initialize InferenceValidator component.")
            raise CustomException(e, sys) from e

    def run(
        self,
        model_artifact: ModelLoadingArtifact,
        feature_artifact: FeatureMatrixBuilderArtifact,
    ) -> InferenceValidationArtifact:
        """
        Executes strict schema validation and generates an audit report.

        Args:
            model_artifact (ModelLoadingArtifact): Output from the Model Loader.
            feature_artifact (FeatureMatrixBuilderArtifact): Output from the Feature Builder.

        Returns:
            InferenceValidationArtifact: Contains validation status and report path.
        """
        try:
            logging.info("Starting inference schema validation.")
            start_time = time.time()

            # 1. Load Schemas
            with open(model_artifact.schema_file_path, "r", encoding="utf-8") as f:
                expected_schema = json.load(f)

            with open(feature_artifact.schema_file_path, "r", encoding="utf-8") as f:
                runtime_schema = json.load(f)

            # 2. Extract Features
            expected_features: Dict[str, str] = {
                feat["name"]: feat["physical_type"] for feat in expected_schema.get("features", [])
            }
            runtime_features: Dict[str, str] = {
                feat["name"]: feat["physical_type"] for feat in runtime_schema.get("features", [])
            }

            # 3. Perform Validation
            missing_features: List[str] = []
            type_mismatches: List[Dict[str, str]] = []

            for feature_name, expected_type in expected_features.items():
                if feature_name not in runtime_features:
                    missing_features.append(feature_name)
                    continue

                runtime_type = runtime_features[feature_name]
                if not self._is_type_compatible(expected_type, runtime_type):
                    type_mismatches.append(
                        {
                            "feature": feature_name,
                            "expected_type": expected_type,
                            "runtime_type": runtime_type,
                        }
                    )

            # 4. Determine Validation Status
            is_valid = len(missing_features) == 0 and len(type_mismatches) == 0

            # 5. Generate Audit Report
            validation_report: Dict[str, Any] = {
                "validation_status": "PASSED" if is_valid else "FAILED",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "champion_run_id": model_artifact.champion_run_id,
                "snapshot_date": feature_artifact.snapshot_date,
                "anomalies": {
                    "missing_features": missing_features,
                    "type_mismatches": type_mismatches,
                },
            }

            write_json_file(self.config.local_validation_report_path, validation_report)

            if not is_valid:
                logging.error(
                    "Schema Validation Failed. Missing: %s, Mismatches: %s",
                    len(missing_features),
                    len(type_mismatches),
                )
            else:
                logging.info("Schema Validation Passed. No anomalies detected.")

            # 6. Record execution metadata in central context ledger
            execution_time = round(time.time() - start_time, 2)
            telemetry: Dict[str, Any] = {
                "is_valid": is_valid,
                "missing_features_count": len(missing_features),
                "type_mismatches_count": len(type_mismatches),
                "local_report_path": self.config.local_validation_report_path,
                "execution_time_seconds": execution_time,
            }
            self.context.add_metadata("InferenceValidator", telemetry)

            # 7. Package and return artifact
            artifact = InferenceValidationArtifact(
                is_valid=is_valid,
                validation_report_file_path=self.config.local_validation_report_path,
            )

            logging.info("InferenceValidator execution completed.")
            return artifact

        except Exception as e:
            logging.exception("Critical Failure in InferenceValidator component.")
            raise CustomException(e, sys) from e

    def _is_type_compatible(self, expected_type: str, runtime_type: str) -> bool:
        """
        Implements a Schema Evolution Adapter logic to safely bridge equivalent
        primitive types across different computation engines (DuckDB vs Scikit-Learn/Pandas).

        Args:
            expected_type (str): The physical type required by the model.
            runtime_type (str): The physical type materialized by DuckDB.

        Returns:
            bool: True if the types are functionally compatible, False otherwise.
        """
        expected = expected_type.lower()
        runtime = runtime_type.lower()

        if expected == runtime:
            return True

        # Numeric bridging: Pandas float64/int64 vs DuckDB DOUBLE/BIGINT
        numeric_types: Set[str] = {
            "int", "int32", "int64", "integer", "bigint",
            "float", "float32", "float64", "double", "numeric", "decimal"
        }
        if expected in numeric_types and runtime in numeric_types:
            return True

        # String bridging: Pandas object/string vs DuckDB VARCHAR/TEXT
        string_types: Set[str] = {
            "str", "string", "object", "varchar", "text", "char", "o"
        }
        if expected in string_types and runtime in string_types:
            return True

        # Boolean bridging: Pandas bool vs DuckDB BOOLEAN
        bool_types: Set[str] = {
            "bool", "boolean", "b"
        }
        if expected in bool_types and runtime in bool_types:
            return True

        return False