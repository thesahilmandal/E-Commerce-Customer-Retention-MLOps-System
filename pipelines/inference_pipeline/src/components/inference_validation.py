import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

from pipelines.inference_pipeline.src.entity.config_entity import InferenceValidationConfig
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    InferenceModelLoaderArtifact,
    InferenceInputFeatureMatrixBuilderArtifact,
    InferenceValidatorArtifact
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file


class InferenceValidation:
    """
    Inference Validator Component.

    Acts as a synchronous data contract gatekeeper for the Inference Pipeline.
    It inspects the structural schema produced by the Inference Builder against 
    the gold-standard schema retrieved from the Model Loader to prevent silent 
    failures during batch prediction.
    
    Implements a Schema Evolution Adapter pattern to support zero-downtime 
    inference scoring using legacy model artifacts, and handles safe cross-engine
    type bridges (e.g., Pandas vs DuckDB).
    """

    def __init__(self, config: InferenceValidationConfig) -> None:
        """
        Initializes the Inference Validator component.

        Args:
            config (InferenceValidatorConfig): Configuration parameters for the component.
        """
        try:
            self.config = config
            
            # Ensure the component's root directory exists
            os.makedirs(self.config.validator_root_dir, exist_ok=True)
            
            logging.info("Inference Pipeline: Validator component initialized.")
            
        except Exception as e:
            logging.exception("Failed to initialize Validator component.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(
        self, 
        loader_artifact: InferenceModelLoaderArtifact, 
        builder_artifact: InferenceInputFeatureMatrixBuilderArtifact
    ) -> InferenceValidatorArtifact:
        """
        Executes zero-dependency structural data contract validation with 
        backward compatibility handling for legacy models.

        Args:
            loader_artifact (InferenceModelLoaderArtifact): Expected model schema.
            builder_artifact (InferenceInputFeatureMatrixBuilderArtifact): Actual runtime schema.

        Returns:
            InferenceValidatorArtifact: Output artifact containing the boolean validation state.
        """
        try:
            logging.info("Starting structural data contract verification.")
            start_time = time.time()
            errors: List[str] = []

            # 1. Load Contract Schemas and Upstream Metadata
            raw_expected_schema = self._read_json(loader_artifact.schema_file_path)
            actual_schema = self._read_json(builder_artifact.schema_file_path)
            builder_metadata = self._read_json(builder_artifact.metadata_file_path)

            # 2. Schema Evolution Adapter (Normalize Legacy Schemas)
            expected_features = self._normalize_expected_schema(raw_expected_schema)
            actual_features = actual_schema.get("features", [])
            
            # 3. Guardrail 1: Volumetric Row Count Verification
            total_rows = builder_metadata.get("scoring_population", {}).get("total_eligible_customers", 0)
            if total_rows == 0:
                errors.append("Empty Population Error: Input feature matrix contains zero rows. Scoring aborted.")
                logging.warning("Validation failed: Zero rows detected in the feature matrix.")

            # 4. Guardrails 2, 3, 4: Structural Integrity, Sequence Alignment, and Type Mapping
            if total_rows > 0:
                errors.extend(self._validate_schema_structure(expected_features, actual_features))

            # 5. Evaluate final state and compile audit report
            is_valid = len(errors) == 0
            execution_time = round(time.time() - start_time, 4)
            
            self._generate_validation_report(
                is_valid=is_valid,
                errors=errors,
                total_rows=total_rows,
                expected_count=len(expected_features),
                actual_count=len(actual_features),
                execution_time=execution_time
            )
            
            self._generate_component_metadata(is_valid, execution_time)

            # 6. Package and return the validation artifact
            artifact = InferenceValidatorArtifact(
                is_valid=is_valid,
                report_file_path=self.config.report_file_path
            )

            logging.info("Validation phase completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Critical Failure inside Validator execution routine.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # SCHEMA EVOLUTION ADAPTER
    # ==========================================================
    def _normalize_expected_schema(self, raw_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detects schema format version and normalizes legacy (V0) schemas into the 
        standardized V1 List-of-Dictionaries format in memory to maintain backward compatibility.
        """
        # Detection Heuristic: V1 schemas have a explicit metadata block with a schema_version.
        # Legacy V0 schemas relied on root-level 'features' dict and 'column_ordering' list.
        is_legacy = "metadata" not in raw_schema or "schema_version" not in raw_schema.get("metadata", {})

        if not is_legacy:
            logging.debug("Detected V1 standard schema contract. Bypassing adapter.")
            return raw_schema.get("features", [])

        logging.warning("Legacy Model Schema detected. Invoking in-memory schema evolution adapter.")
        
        normalized_features = []
        legacy_ordering = raw_schema.get("column_ordering", [])
        legacy_features_dict = raw_schema.get("features", {})

        for index, col_name in enumerate(legacy_ordering):
            # Extract metadata for the specific column, fallback to empty dict if missing
            col_metadata = legacy_features_dict.get(col_name, {})
            
            # Map legacy 'pandas_dtype' to new 'physical_type' terminology
            legacy_dtype = col_metadata.get("pandas_dtype", "unknown")
            
            normalized_features.append({
                "name": col_name,
                "index": index,
                "physical_type": legacy_dtype,
                "is_nullable": col_metadata.get("nullable", True)
            })

        return normalized_features

    # ==========================================================
    # VALIDATION LOGIC ENGINE
    # ==========================================================
    def _validate_schema_structure(
        self, 
        expected_features: List[Dict[str, Any]], 
        actual_features: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Performs strict structural checks including presence, sequence, and data types.
        """
        errors = []
        actual_features_list = [feat.get("name", "") for feat in actual_features]
        actual_features_dict = {feat.get("name", ""): feat for feat in actual_features}

        for idx, expected_feat in enumerate(expected_features):
            exp_name = expected_feat.get("name", "")
            exp_type = expected_feat.get("physical_type", "")

            # Check: Feature Presence
            if exp_name not in actual_features_dict:
                errors.append(f"Missing Feature Error: Expected feature '{exp_name}' was not generated.")
                continue

            # Check: Strict Positional Sequence Alignment
            # The model requires the feature to be exactly at the same index position as during training.
            if idx >= len(actual_features_list) or actual_features_list[idx] != exp_name:
                act_name_at_idx = actual_features_list[idx] if idx < len(actual_features_list) else "None"
                errors.append(
                    f"Sequence Alignment Error: Feature mismatch at index {idx}. "
                    f"Expected '{exp_name}', but found '{act_name_at_idx}'."
                )

            # Check: Primitive Type-Bridge Auditing
            act_type = actual_features_dict[exp_name].get("physical_type", "")
            if not self._is_type_compatible(exp_type, act_type):
                errors.append(
                    f"Type Mismatch Error: Column '{exp_name}' expects '{exp_type}', "
                    f"but received raw type '{act_type}'."
                )

        return errors

    def _is_type_compatible(self, expected: str, actual: str) -> bool:
        """
        Maps primitive system types to prevent false negatives caused by discrepancies 
        between Pandas training types and DuckDB inference engine types.
        
        This layer explicitly authorizes safe logical casts, such as passing an exact 
        integer (DuckDB BIGINT) into a slot trained on Pandas floats (float64) to support
        model robustness.
        """
        expected = str(expected).lower()
        actual = str(actual).lower()
        
        if expected == actual:
            return True
            
        # Equivalence mappings (Model Logical Type vs Query Engine Physical Type)
        type_bridges = {
            # Standard integer alignments
            "int64": ["int64", "bigint", "int", "long", "int32", "integer"],
            "int32": ["int32", "integer", "int", "int4", "int64", "bigint"],
            
            # Safe upcast alignments: XGBoost natively converts all numerics to float32 internally.
            # Passing a strict integer into a float-trained slot is mathematically lossless and safe.
            "float64": ["float64", "double", "float", "numeric", "float32", "bigint", "integer", "int64", "int32", "int", "long"],
            "float32": ["float32", "double", "float", "numeric", "float64", "bigint", "integer", "int64", "int32", "int", "long"],
            
            # Categorical string alignments
            "object": ["object", "varchar", "text", "string", "category"],
            "category": ["category", "object", "varchar", "text", "string"]
        }
        
        for base_type, aliases in type_bridges.items():
            if expected == base_type or expected in aliases:
                return actual in aliases
                
        return False

    # ==========================================================
    # FILE I/O & ARTIFACT GENERATION
    # ==========================================================
    def _read_json(self, file_path: str) -> Dict[str, Any]:
        """Safely reads a JSON file from disk."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.exception(f"Failed to read JSON artifact at {file_path}")
            raise CustomException(e, sys) from e

    def _generate_validation_report(
        self, 
        is_valid: bool, 
        errors: List[str], 
        total_rows: int, 
        expected_count: int, 
        actual_count: int, 
        execution_time: float
    ) -> None:
        """
        Generates the definitive validation audit report artifact.
        """
        try:
            logging.info("Generating structural data contract validation report.")
            
            report = {
                "pipeline_stage": "Inference Structural Contract Validator",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_time_seconds": execution_time,
                "is_valid": is_valid,
                "metrics": {
                    "total_rows_inspected": total_rows,
                    "expected_features_count": expected_count,
                    "actual_features_count": actual_count
                },
                "errors": errors
            }
            
            write_json_file(file_path=self.config.report_file_path, content=report)
            logging.info(f"Validation report saved to: {self.config.report_file_path}")

        except Exception as e:
            logging.exception("Failed to write validation report.")
            raise CustomException(e, sys) from e

    def _generate_component_metadata(self, is_valid: bool, execution_time: float) -> None:
        """
        Generates standard telemetry metadata for the pipeline component.
        """
        try:
            metadata: Dict[str, Any] = {
                "pipeline_stage": "Inference Validator",
                "execution_time_seconds": execution_time,
                "data_contract_enforced": is_valid,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(file_path=self.config.metadata_file_path, content=metadata)

        except Exception as e:
            logging.exception("Failed to generate component metadata.")
            raise CustomException(e, sys) from e