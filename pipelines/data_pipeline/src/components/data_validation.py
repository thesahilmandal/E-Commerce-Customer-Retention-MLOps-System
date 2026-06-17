import os
import sys
from typing import Dict, Any

from pipelines.data_pipeline.src.entity.config_entity import DataPipelineValidatorConfig
from pipelines.data_pipeline.src.entity.artifact_entity import (
    DataPipelineExtractorArtifact,
    DataPipelineValidatorArtifact,
)
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import (
    write_json_file,
    read_json_file,
    read_csv_file,
    read_parquet_file,
)


class Validator:
    """
    Validator component for validating raw data against predefined schema.

    Responsibilities:
    - Compare predefined schema with generated schema
    - Perform structural + data quality validation
    - Generate validation report
    - Output validation status
    """

    def __init__(
        self,
        config: DataPipelineValidatorConfig,
        extractor_artifact: DataPipelineExtractorArtifact,
    ):
        try:
            self.config = config
            self.extractor_artifact = extractor_artifact

            self.raw_data_dir_path = extractor_artifact.raw_data_dir_path
            self.raw_schema_path = extractor_artifact.raw_data_schema_file_path
            self.predefined_schema_path = (
                self.config.reference_schema_file_path
            )

            os.makedirs(self.config.validator_root_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataPipelineValidatorArtifact:
        """
        Executes validation pipeline.

        Returns:
            DataPipelineValidatorArtifact
        """
        try:
            logging.info("Starting data validation pipeline")

            predefined_schema = read_json_file(self.predefined_schema_path)
            raw_schema = read_json_file(self.raw_schema_path)

            report = {
                "tables": {},
                "summary": {
                    "total_tables": 0,
                    "passed_tables": 0,
                    "failed_tables": 0,
                    "is_valid": False,
                },
            }

            validation_rules = predefined_schema.get("validation_rules", {})
            tables = predefined_schema.get("tables", {})

            for table_name, table_rules in tables.items():
                table_report = self._validate_table(
                    table_name,
                    table_rules,
                    raw_schema,
                    validation_rules,
                )

                report["tables"][table_name] = table_report
                report["summary"]["total_tables"] += 1

                if table_report["is_valid"]:
                    report["summary"]["passed_tables"] += 1
                else:
                    report["summary"]["failed_tables"] += 1

            is_valid = report["summary"]["failed_tables"] == 0
            report["summary"]["is_valid"] = is_valid

            write_json_file(self.config.report_file_path, report)

            logging.info(
                f"Validation completed. is_valid={is_valid}"
            )
            logging.info(
                f"Validation report saved at: {self.config.report_file_path}"
            )

            artifact = DataPipelineValidatorArtifact(
                report_file_path=self.config.report_file_path,
                is_valid=is_valid,
            )

            logging.info(f"Validator artifact created: {artifact}")

            return artifact

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # LOAD JSON
    # ==========================================================
    def _load_json(self, path: str) -> Dict[str, Any]:
        try:
            return read_json_file(path)
        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # TABLE VALIDATION
    # ==========================================================
    def _validate_table(
        self,
        table_name: str,
        table_rules: Dict[str, Any],
        raw_schema: Dict[str, Any],
        validation_rules: Dict[str, Any],
    ) -> Dict[str, Any]:

        table_report = {
            "exists": True,
            "num_columns_match": True,
            "column_validation": {},
            "is_valid": True,
            "errors": [],
        }

        # Check table existence
        if table_name not in raw_schema:
            table_report["exists"] = False
            table_report["is_valid"] = False
            table_report["errors"].append("Table missing")

            if not validation_rules.get("allow_missing_tables", False):
                return table_report

        raw_table = raw_schema.get(table_name, {})
        expected_columns = table_rules.get("columns", {})
        raw_columns = raw_table.get("columns", [])

        # Column count validation
        if "num_columns" in table_rules:
            if len(raw_columns) != table_rules["num_columns"]:
                table_report["num_columns_match"] = False
                table_report["is_valid"] = False
                table_report["errors"].append(
                    "Column count mismatch"
                )

        # Column validation
        for col_name, col_rules in expected_columns.items():
            col_report = self._validate_column(
                table_name,
                col_name,
                col_rules,
                raw_table,
            )

            table_report["column_validation"][col_name] = col_report

            if not col_report["is_valid"]:
                table_report["is_valid"] = False

        return table_report

    # ==========================================================
    # COLUMN VALIDATION
    # ==========================================================
    def _validate_column(
        self,
        table_name: str,
        col_name: str,
        col_rules: Dict[str, Any],
        raw_table: Dict[str, Any],
    ) -> Dict[str, Any]:

        col_report = {
            "exists": True,
            "dtype_match": True,
            "missing_within_threshold": True,
            "allowed_values_check": True,
            "unique_check": True,
            "is_valid": True,
            "errors": [],
        }

        raw_columns = raw_table.get("columns", [])
        raw_dtypes = raw_table.get("dtypes", {})
        missing_values = raw_table.get("missing_values", {})
        total_rows = raw_table.get("num_rows", 1)

        # Column existence
        if col_name not in raw_columns:
            col_report["exists"] = False
            col_report["is_valid"] = False
            col_report["errors"].append("Column missing")
            return col_report

        # Dtype validation
        expected_dtype = col_rules.get("dtype")
        actual_dtype = raw_dtypes.get(col_name)

        if expected_dtype and actual_dtype != expected_dtype:
            col_report["dtype_match"] = False
            col_report["is_valid"] = False
            col_report["errors"].append(
                f"dtype mismatch (expected={expected_dtype}, actual={actual_dtype})"
            )

        # Missing value validation (Column-Specific Threshold)
        max_missing_pct = col_rules.get("max_missing_percentage", 0.0)

        missing_count = missing_values.get(col_name, 0)
        missing_pct = missing_count / max(total_rows, 1)

        if missing_pct > max_missing_pct:
            col_report["missing_within_threshold"] = False
            col_report["is_valid"] = False
            col_report["errors"].append(
                f"Missing percentage exceeded ({missing_pct:.2f} > {max_missing_pct})"
            )

        # Load dataframe only when needed (lazy loading)
        df = None
        csv_path = os.path.join(self.raw_data_dir_path, f"{table_name}.csv")
        parquet_path = os.path.join(self.raw_data_dir_path, f"{table_name}.parquet")

        # Allowed values validation
        if "allowed_values" in col_rules:
            try:
                if df is None:
                    if os.path.exists(parquet_path):
                        df = read_parquet_file(parquet_path)
                    else:
                        df = read_csv_file(csv_path)

                if not df[col_name].dropna().isin(
                    col_rules["allowed_values"]
                ).all():
                    col_report["allowed_values_check"] = False
                    col_report["is_valid"] = False
                    col_report["errors"].append(
                        "Invalid categorical values"
                    )

            except Exception:
                col_report["allowed_values_check"] = False
                col_report["is_valid"] = False
                col_report["errors"].append(
                    "Failed allowed_values validation"
                )

        # Uniqueness validation
        if col_rules.get("unique", False):
            try:
                if df is None:
                    if os.path.exists(parquet_path):
                        df = read_parquet_file(parquet_path)
                    else:
                        df = read_csv_file(csv_path)

                if df[col_name].duplicated().any():
                    col_report["unique_check"] = False
                    col_report["is_valid"] = False
                    col_report["errors"].append(
                        "Duplicate values found"
                    )

            except Exception:
                col_report["unique_check"] = False
                col_report["is_valid"] = False
                col_report["errors"].append(
                    "Failed uniqueness validation"
                )

        return col_report