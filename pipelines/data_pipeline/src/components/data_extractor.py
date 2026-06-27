import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import duckdb

from pipelines.data_pipeline.src.entity.config_entity import DataExtractorConfig
from pipelines.data_pipeline.src.entity.artifact_entity import DataExtractorArtifact
from shared_core.utils.main_utils import write_json_file, read_json_file
from shared_core.logging.custom_logging import logging
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException

class DataExtractor:
    """
    Extractor component for selectively downloading and preparing required datasets.

    Responsibilities:
    - Read predefined_schema.json to dynamically determine the Single Source of Truth (SSOT) for required datasets.
    - Sync ONLY the required dataset files from AWS S3 Data Lake via injected S3 utility to prevent unnecessary I/O.
    - Store raw files in the standardized local artifact directory.
    - Generate execution schema for the downloaded datasets dynamically using out-of-core DuckDB processing.
    - Generate metadata for pipeline observability and lineage.
    """

    def __init__(self, config: DataExtractorConfig, s3_sync: S3Sync) -> None:
        """
        Initializes Extractor with configuration and dependency-injected S3 utility.
        
        Args:
            config (DataExtractorConfig): Configuration settings for extraction paths.
            s3_sync (S3Sync): Injected S3 synchronization client for cloud operations.
        """
        self.config = config
        self.s3_sync = s3_sync

        os.makedirs(self.config.raw_data_dir_path, exist_ok=True)
        logging.info("Extractor initialized successfully. Configured for selective extraction.")

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataExtractorArtifact:
        """
        Executes the main extraction pipeline.

        Returns:
            DataExtractorArtifact: Details of the extracted data paths and metadata.
        """
        logging.info("Starting targeted data extraction pipeline (S3 to Local)")

        try:
            # Step 1: Read predefined schema to identify required datasets (SSOT)
            required_datasets = self._get_required_datasets()

            # Step 2: Validate upstream existence in S3 before downloading
            self._validate_upstream_datasets(required_datasets)

            # Step 3: Selectively download ONLY required datasets
            self._download_required_datasets(required_datasets)

            # Step 4: Collect downloaded files
            data_files = self._collect_downloaded_files()

            # Step 5: Map file paths to dataset names
            stored_files_map = self._map_stored_files(data_files)

            # Step 6: Generate structural schema for validation (Optimized with DuckDB)
            schema_info = self._generate_schema(stored_files_map)

            # Step 7: Persist schema definition
            write_json_file(
                file_path=self.config.raw_data_schema_file_path, content=schema_info
            )

            # Step 8: Generate and persist execution metadata
            self._generate_metadata(stored_files_map, schema_info)

            # Package and return Artifact
            artifact = DataExtractorArtifact(
                raw_data_dir_path=self.config.raw_data_dir_path,
                raw_data_schema_file_path=self.config.raw_data_schema_file_path,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("Extraction completed successfully: %s", artifact)
            return artifact
            
        except Exception as exc:
            logging.error("Failed during Data Extraction pipeline execution.")
            raise CustomException(exc, sys) from exc

    # ==========================================================
    # DEPENDENCY RESOLUTION (SSOT)
    # ==========================================================
    def _get_required_datasets(self) -> List[str]:
        """
        Reads the predefined schema to dynamically determine which datasets 
        must be extracted.
        
        Returns:
            List[str]: A list of dataset table names defined in the schema.
            
        Raises:
            CustomException: If the predefined schema is missing or unreadable.
        """
        logging.info("Reading predefined schema to determine required datasets: %s", self.config.predefined_schema_file_path)
        
        try:
            predefined_schema = read_json_file(self.config.predefined_schema_file_path)
            tables = predefined_schema.get("tables", {})
            
            required_datasets = list(tables.keys())
            
            if not required_datasets:
                raise ValueError("Predefined schema contains no tables.")
                
            logging.info("Identified %d required dataset(s): %s", len(required_datasets), required_datasets)
            return required_datasets
            
        except Exception as exc:
            logging.error("Failed to parse predefined schema for dataset resolution.")
            raise CustomException(exc, sys) from exc

    # ==========================================================
    # AWS S3 CLOUD OPERATIONS
    # ==========================================================
    def _validate_upstream_datasets(self, required_datasets: List[str]) -> None:
        """
        Fail-fast check: Verifies that all required datasets exist in the upstream 
        S3 bucket before attempting any downloads.
        """
        logging.info("Validating existence of required datasets in S3 bucket.")
        s3_client = self.s3_sync._get_client()
        bucket, prefix = self.s3_sync._parse_s3_uri(self.config.s3_raw_data_uri)
        
        # Ensure prefix format for precise matching
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            available_keys = set()
            
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        # Store just the filename, not the full prefix path
                        filename = os.path.basename(obj["Key"])
                        available_keys.add(filename)

            missing_datasets = []
            for dataset in required_datasets:
                # Check for both parquet and csv extensions
                if not (f"{dataset}.parquet" in available_keys or f"{dataset}.csv" in available_keys):
                    missing_datasets.append(dataset)
                    
            if missing_datasets:
                raise FileNotFoundError(
                    f"Required datasets missing from S3 source ({self.config.s3_raw_data_uri}): {missing_datasets}"
                )
                
            logging.info("Upstream validation successful. All required datasets are available.")
            
        except Exception as exc:
            logging.error("S3 upstream validation failed.")
            raise CustomException(exc, sys) from exc

    def _download_required_datasets(self, required_datasets: List[str]) -> None:
        """
        Selectively downloads only the required datasets from S3.
        Assumes .parquet preference, falling back to .csv implicitly via 
        the validation phase logic if implemented.
        """
        logging.info("Initiating targeted download of required datasets.")
        
        bucket, prefix = self.s3_sync._parse_s3_uri(self.config.s3_raw_data_uri)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        download_tasks = []
        for dataset in required_datasets:
            # Note: We enforce attempting to download .parquet as per FAANG best practices.
            # If the upstream data relies on CSVs, the filename construction here must be adapted.
            file_name = f"{dataset}.parquet"
            s3_key = f"{prefix}{file_name}"
            s3_uri = f"s3://{bucket}/{s3_key}"
            local_path = os.path.join(self.config.raw_data_dir_path, file_name)
            
            download_tasks.append((s3_uri, local_path))

        def _download_task(args: tuple) -> None:
            self.s3_sync.download_file(s3_uri=args[0], local_path=args[1])

        try:
            with ThreadPoolExecutor(max_workers=self.s3_sync.max_workers) as executor:
                list(executor.map(_download_task, download_tasks))
                
            logging.info("Successfully downloaded %d required datasets.", len(download_tasks))
        except Exception as exc:
            logging.error("Concurrent download of targeted datasets failed.")
            raise CustomException(exc, sys) from exc

    # ==========================================================
    # FILE DISCOVERY & MAPPING
    # ==========================================================
    def _collect_downloaded_files(self) -> List[str]:
        """
        Traverses the local raw data directory to find downloaded dataset files.

        Returns:
            List[str]: A list of absolute file paths.
            
        Raises:
            FileNotFoundError: If no valid data files are found in the target directory.
        """
        logging.info("Collecting downloaded files from local raw data directory")
        data_files = []

        for root, _, files in os.walk(self.config.raw_data_dir_path):
            for file in files:
                if file.endswith(".parquet") or file.endswith(".csv"):
                    data_files.append(os.path.join(root, file))

        if not data_files:
            logging.error("No valid data files found in %s", self.config.raw_data_dir_path)
            raise FileNotFoundError(
                f"No valid data files found in {self.config.raw_data_dir_path}"
            )

        logging.info("Found %d data files locally.", len(data_files))
        return data_files

    def _map_stored_files(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Maps clean dataset names (keys) to their local file paths (values).

        Args:
            file_paths (List[str]): List of absolute file paths.

        Returns:
            Dict[str, str]: Mapping of dataset names to file paths.
        """
        logging.info("Mapping stored raw data files")
        stored_files = {}

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            # Remove known extensions to derive the base dataset name
            dataset_name = file_name.replace(".parquet", "").replace(".csv", "")

            stored_files[dataset_name] = file_path
            logging.debug("Mapped dataset: %s -> %s", dataset_name, file_name)

        return stored_files

    # ==========================================================
    # SCHEMA GENERATION
    # ==========================================================
    def _generate_schema(self, stored_files: Dict[str, str]) -> Dict[str, Any]:
        """
        Dynamically infers schema using DuckDB to prevent Out-Of-Memory (OOM) 
        errors and improve computational efficiency.

        Args:
            stored_files (Dict[str, str]): Mapping of dataset names to file paths.

        Returns:
            Dict[str, Any]: Nested dictionary defining the schema for all downloaded datasets.
        """
        logging.info("Generating schema for downloaded datasets dynamically via DuckDB")
        schema = {}

        with duckdb.connect(":memory:") as con:
            for name, path in stored_files.items():
                reader_func = (
                    "read_parquet" if path.endswith(".parquet") else "read_csv_auto"
                )
                query_base = f"{reader_func}('{path}')"

                # 1. Infer columns and data types lazily
                describe_query = f"DESCRIBE SELECT * FROM {query_base}"
                describe_res = con.execute(describe_query).fetchall()

                columns = [row[0] for row in describe_res]
                dtypes = {row[0]: row[1] for row in describe_res}

                # 2. Dynamically build aggregation query for exact row and null counts
                agg_selects = ["COUNT(*)"]
                for col in columns:
                    agg_selects.append(
                        f"CAST(SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END) AS INTEGER)"
                    )

                agg_query = f"SELECT {', '.join(agg_selects)} FROM {query_base}"
                agg_res = con.execute(agg_query).fetchone()

                num_rows = agg_res[0]
                missing_values = {
                    col: agg_res[i + 1] for i, col in enumerate(columns)
                }

                schema[name] = {
                    "columns": columns,
                    "dtypes": dtypes,
                    "num_rows": num_rows,
                    "num_columns": len(columns),
                    "missing_values": missing_values,
                }

                logging.info("Schema successfully generated for table: %s", name)

        return schema

    # ==========================================================
    # METADATA GENERATION
    # ==========================================================
    def _generate_metadata(
        self, stored_files: Dict[str, str], schema: Dict[str, Any]
    ) -> None:
        """
        Generates and persists execution metadata (observability data).

        Args:
            stored_files (Dict[str, str]): Mapping of dataset names to file paths.
            schema (Dict[str, Any]): The generated schema payload.
        """
        logging.info("Generating extractor metadata and telemetry")

        total_rows = sum(s["num_rows"] for s in schema.values())

        metadata = {
            "dataset_name": "Brazilian Olist E-commerce",
            "extraction_mode": "selective",
            "num_tables": len(stored_files),
            "total_rows": total_rows,
            "tables": list(stored_files.keys()),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": "aws_s3",
            "data_version": "latest",
        }

        write_json_file(file_path=self.config.metadata_file_path, content=metadata)

        logging.info(
            "Extractor metadata securely saved at: %s",
            self.config.metadata_file_path,
        )