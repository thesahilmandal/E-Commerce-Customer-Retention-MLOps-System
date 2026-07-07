import os
from typing import Dict, Any

import duckdb

from pipelines.data_pipeline.src import constants
from pipelines.data_pipeline.src.entity.config_entity import DataValidationConfig
from pipelines.data_pipeline.src.entity.artifact_entity import (
DataExtractorArtifact,
DataValidationArtifact,
)
from shared_core.logging.custom_logging import logging
from shared_core.utils.main_utils import write_json_file, read_json_file

class DataValidation:
    """
    Validator component for validating raw data against predefined schema.

    ```
    Responsibilities:
    - Compare predefined schema with generated schema out-of-core.
    - Perform structural validation (column counts, existence).
    - Perform data quality validation (dtype mapping, categorical bounds, uniqueness) natively via DuckDB.
    - Generate comprehensive validation report and boolean status.
    - Strictly enforce database memory limits and disk-spilling to prevent Out-Of-Memory (OOM) errors.
    """

    # Logical type mappings to prevent brittle strict string comparisons
    TYPE_FAMILIES = {
        "NUMERIC": ["BIGINT", "INTEGER", "DOUBLE", "FLOAT", "HUGEINT", "UBIGINT", "UINTEGER", "TINYINT", "SMALLINT", "DECIMAL"],
        "STRING": ["VARCHAR", "TEXT", "CHAR", "BLOB"],
        "DATETIME": ["TIMESTAMP", "DATE", "TIME"],
        "BOOLEAN": ["BOOLEAN", "BOOL"]
    }

    def __init__(
        self,
        config: DataValidationConfig,
        extractor_artifact: DataExtractorArtifact,
    ) -> None:
        self.config = config
        self.extractor_artifact = extractor_artifact

        self.raw_data_dir_path = extractor_artifact.raw_data_dir_path
        self.raw_schema_path = extractor_artifact.raw_data_schema_file_path
        self.predefined_schema_path = self.config.reference_schema_file_path

        # Dedicated temporary directory for DuckDB disk-spilling
        self.duckdb_temp_dir = os.path.join(self.config.validator_root_dir, "tmp")
        os.makedirs(self.duckdb_temp_dir, exist_ok=True)

        os.makedirs(self.config.validator_root_dir, exist_ok=True)
        logging.info("DataValidation initialized successfully.")

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataValidationArtifact:
        """
        Executes validation pipeline.

        Returns:
            DataValidationArtifact: Report path and boolean validation status.
        """
        logging.info("Starting out-of-core data validation pipeline")

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

        logging.info("Validation completed. is_valid=%s", is_valid)
        logging.info("Validation report saved at: %s", self.config.report_file_path)

        artifact = DataValidationArtifact(
            report_file_path=self.config.report_file_path,
            is_valid=is_valid,
        )

        return artifact

    # ==========================================================
    # UTILITIES
    # ==========================================================
    def _initialize_duckdb(self) -> duckdb.DuckDBPyConnection:
        """
        Initializes an ephemeral, in-memory DuckDB connection with disk-spilling enabled.
        Prevents Out-Of-Memory (OOM) errors during high-cardinality aggregations.
        """
        logging.debug("Initializing DuckDB with temp directory: %s", self.duckdb_temp_dir)
        con = duckdb.connect(database=":memory:")
        con.execute(f"PRAGMA threads={constants.COMPUTE_THREADS}")
        con.execute("PRAGMA memory_limit='8GB'")
        con.execute(f"PRAGMA temp_directory='{self.duckdb_temp_dir}'")
        return con

    def _get_type_family(self, dtype: str) -> str:
        """
        Maps a concrete DuckDB dtype to its broader logical family.
        """
        dtype_upper = str(dtype).upper()
        for family, types in self.TYPE_FAMILIES.items():
            if dtype_upper in types or any(t in dtype_upper for t in types):
                return family
        return "UNKNOWN"

    def _get_table_file_path(self, table_name: str) -> str:
        """
        Resolves the local file path for a given table name.
        """
        parquet_path = os.path.join(self.raw_data_dir_path, f"{table_name}.parquet")
        if os.path.exists(parquet_path):
            return parquet_path
            
        csv_path = os.path.join(self.raw_data_dir_path, f"{table_name}.csv")
        if os.path.exists(csv_path):
            return csv_path
            
        return ""

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
                    f"Column count mismatch (expected={table_rules['num_columns']}, actual={len(raw_columns)})"
                )

        file_path = self._get_table_file_path(table_name)
        
        # Connect to DuckDB once per table with strictly constrained memory and disk-spill paths
        con = self._initialize_duckdb()
        try:
            for col_name, col_rules in expected_columns.items():
                col_report = self._validate_column(
                    con,
                    table_name,
                    col_name,
                    col_rules,
                    raw_table,
                    file_path
                )

                table_report["column_validation"][col_name] = col_report

                if not col_report["is_valid"]:
                    table_report["is_valid"] = False
        finally:
            con.close()
            logging.debug("DuckDB connection for table %s closed safely.", table_name)

        return table_report

    # ==========================================================
    # COLUMN VALIDATION
    # ==========================================================
    def _validate_column(
        self,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        col_name: str,
        col_rules: Dict[str, Any],
        raw_table: Dict[str, Any],
        file_path: str
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

        # 1. Column existence
        if col_name not in raw_columns:
            col_report["exists"] = False
            col_report["is_valid"] = False
            col_report["errors"].append("Column missing")
            return col_report

        # 2. Flexible Dtype validation
        expected_dtype = col_rules.get("dtype")
        actual_dtype = raw_dtypes.get(col_name)

        if expected_dtype and actual_dtype:
            expected_family = self._get_type_family(expected_dtype)
            actual_family = self._get_type_family(actual_dtype)
            
            if expected_family != actual_family and expected_family != "UNKNOWN":
                col_report["dtype_match"] = False
                col_report["is_valid"] = False
                col_report["errors"].append(
                    f"dtype mismatch (expected family={expected_family}, actual family={actual_family})"
                )

        # 3. Missing value validation (Column-Specific Threshold)
        max_missing_pct = col_rules.get("max_missing_percentage", 0.0)
        missing_count = missing_values.get(col_name, 0)
        missing_pct = missing_count / max(total_rows, 1)

        if missing_pct > max_missing_pct:
            col_report["missing_within_threshold"] = False
            col_report["is_valid"] = False
            col_report["errors"].append(
                f"Missing percentage exceeded ({missing_pct:.4f} > {max_missing_pct})"
            )

        # Return early if file_path is missing to avoid SQL errors
        if not file_path:
            return col_report
            
        reader_func = "read_parquet" if file_path.endswith(".parquet") else "read_csv_auto"
        query_base = f"{reader_func}('{file_path}')"

        # 4. Out-of-core Allowed values validation via DuckDB
        if "allowed_values" in col_rules:
            try:
                allowed_tuple = tuple(col_rules["allowed_values"])
                # If tuple has 1 element, ensure correct SQL syntax
                if len(allowed_tuple) == 1:
                    allowed_sql = f"('{allowed_tuple[0]}')"
                else:
                    allowed_sql = str(allowed_tuple)
                    
                query = f"""
                    SELECT COUNT(*) FROM {query_base}
                    WHERE "{col_name}" NOT IN {allowed_sql} 
                    AND "{col_name}" IS NOT NULL
                """
                invalid_count = con.execute(query).fetchone()[0]

                if invalid_count > 0:
                    col_report["allowed_values_check"] = False
                    col_report["is_valid"] = False
                    col_report["errors"].append(
                        f"Found {invalid_count} invalid categorical values"
                    )

            except Exception as exc:
                logging.error("Failed allowed_values validation for %s: %s", col_name, exc)
                col_report["allowed_values_check"] = False
                col_report["is_valid"] = False
                col_report["errors"].append("SQL Execution failed for allowed_values validation")

        # 5. Out-of-core Uniqueness validation via DuckDB
        if col_rules.get("unique", False):
            try:
                query = f"""
                    SELECT COUNT("{col_name}") - COUNT(DISTINCT "{col_name}") 
                    FROM {query_base}
                """
                duplicates = con.execute(query).fetchone()[0]

                if duplicates > 0:
                    col_report["unique_check"] = False
                    col_report["is_valid"] = False
                    col_report["errors"].append(
                        f"Found {duplicates} duplicate values"
                    )

            except Exception as exc:
                logging.error("Failed uniqueness validation for %s: %s", col_name, exc)
                col_report["unique_check"] = False
                col_report["is_valid"] = False
                col_report["errors"].append("SQL Execution failed for uniqueness validation")

        return col_report