import os
from datetime import datetime, timezone
from typing import Dict, Any, List

import duckdb

from pipelines.data_pipeline.src.entity.config_entity import DataExtractorConfig
from pipelines.data_pipeline.src.entity.artifact_entity import DataExtractorArtifact
from shared_core.utils.main_utils import write_json_file
from shared_core.logging.custom_logging import logging
from shared_core.cloud.s3_operations import S3Sync


class DataExtractor:
    """
    Extractor component for downloading and preparing raw datasets.

    Responsibilities:
    - Sync raw dataset files directly from AWS S3 Data Lake via injected S3 utility.
    - Store raw files in the standardized local artifact directory.
    - Generate schema for each dataset dynamically using out-of-core DuckDB processing.
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

        # Ensure local raw data directory exists
        os.makedirs(self.config.raw_data_dir_path, exist_ok=True)
        logging.info("Extractor initialized successfully.")

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataExtractorArtifact:
        """
        Executes the main extraction pipeline.

        Returns:
            DataExtractorArtifact: Details of the extracted data paths and metadata.
        """
        logging.info("Starting data extraction pipeline (S3 to Local)")

        # Step 1: Sync raw files from S3 to local directory
        self._sync_data_from_s3()

        # Step 2: Collect downloaded files
        data_files = self._collect_downloaded_files()

        # Step 3: Map file paths to dataset names
        stored_files_map = self._map_stored_files(data_files)

        # Step 4: Generate structural schema for validation (Optimized with DuckDB)
        schema_info = self._generate_schema(stored_files_map)

        # Step 5: Persist schema definition
        write_json_file(
            file_path=self.config.raw_data_schema_file_path, content=schema_info
        )

        # Step 6: Generate and persist execution metadata
        self._generate_metadata(stored_files_map, schema_info)

        # Package and return Artifact
        artifact = DataExtractorArtifact(
            raw_data_dir_path=self.config.raw_data_dir_path,
            raw_data_schema_file_path=self.config.raw_data_schema_file_path,
            metadata_file_path=self.config.metadata_file_path,
        )

        logging.info("Extraction completed successfully: %s", artifact)
        return artifact

    # ==========================================================
    # AWS S3 DATA SYNC
    # ==========================================================
    def _sync_data_from_s3(self) -> None:
        """
        Synchronizes the remote S3 raw data bucket with the local directory.
        """
        logging.info(
            "Syncing raw data from %s to local directory %s",
            self.config.s3_raw_data_uri,
            self.config.raw_data_dir_path,
        )
        self.s3_sync.sync_folder_from_s3(
            folder=self.config.raw_data_dir_path,
            aws_bucket_url=self.config.s3_raw_data_uri,
        )

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

        logging.info("Found %d data files", len(data_files))
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
            Dict[str, Any]: Nested dictionary defining the schema for all datasets.
        """
        logging.info("Generating schema for datasets dynamically via DuckDB")
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