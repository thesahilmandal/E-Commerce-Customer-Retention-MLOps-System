"""
Data Discovery Component for the Continual Learning Data Pipeline.

This module validates the availability of required upstream datasets in the 
Bronze Data Lake before allowing compute-heavy pipeline stages to proceed.
It adheres to a fail-fast design philosophy by translating the pipeline's 
temporal window into expected Hive partition paths and verifying their existence 
via lightweight S3 listing operations.
"""

import sys
from datetime import datetime
from typing import Set, Tuple

from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class DataDiscovery:
    """
    Data Discovery Execution Stage.

    Responsibilities:
    - Validate the existence of required datasets in the S3 Bronze Data Lake.
    - Translate the pipeline's temporal window (start_date, end_date) into 
      expected Hive partition paths (year=YYYY/month=MM).
    - Execute compute-efficient S3 listing operations (via injected S3Sync) to 
      ensure data exists for the required timeframe.
    - Fail fast if critical upstream data is missing, preventing silent omissions 
      and wasted compute resources down the execution graph.
    """

    def __init__(self, context: PipelineContext) -> None:
        """
        Initializes the Data Discovery component.

        Args:
            context (PipelineContext): The injected pipeline execution context 
                                       containing global clients and state.
        """
        self.context = context
        self.s3_client = self.context.s3_sync._get_client()
        self.bronze_uri = self.context.config.storage.bronze_data_lake.base_uri.rstrip("/")
        self.datasets = self.context.config.storage.bronze_data_lake.datasets

        logging.info("DataDiscovery component initialized.")

    def _get_required_months(self) -> Set[Tuple[str, str]]:
        """
        Generates the unique (year, month) combinations spanning the temporal window.
        This provides a highly efficient mechanism for partition pruning validation.

        Returns:
            Set[Tuple[str, str]]: A set of (YYYY, MM) tuples.
        """
        # Safely handle both 'YYYY-MM-DD' and 'YYYY-MM-DD HH:MM:SS' formats
        start_str = self.context.start_date.split(" ")[0]
        end_str = self.context.end_date.split(" ")[0]

        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")

        months = set()
        current = start.replace(day=1)
        
        while current <= end:
            months.add((str(current.year), f"{current.month:02d}"))
            
            # Advance to the next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return months

    def _prefix_exists_in_s3(self, s3_uri: str) -> bool:
        """
        Checks if a specific S3 URI prefix contains any objects.
        Uses MaxKeys=1 to ensure the network operation is O(1) and compute-efficient.

        Args:
            s3_uri (str): The S3 URI to check.

        Returns:
            bool: True if objects exist under the prefix, False otherwise.
        """
        bucket, prefix = self.context.s3_sync._parse_s3_uri(s3_uri)
        
        # Ensure the prefix acts as a directory boundary
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        response = self.s3_client.list_objects_v2(
            Bucket=bucket, 
            Prefix=prefix, 
            MaxKeys=1
        )
        
        return "Contents" in response or "CommonPrefixes" in response

    def run(self) -> None:
        """
        Executes the discovery process to validate upstream data existence.

        Raises:
            CustomException: If required datasets or temporal partitions are missing.
        """
        logging.info(
            "Starting Data Discovery for temporal window: [%s to %s)", 
            self.context.start_date, 
            self.context.end_date
        )

        try:
            required_months = self._get_required_months()
            missing_datasets = []

            for dataset in self.datasets:
                dataset_base_uri = f"{self.bronze_uri}/{dataset.name}"
                logging.debug("Verifying availability for dataset: %s", dataset.name)

                # 1. Check if the dataset root directory exists
                if not self._prefix_exists_in_s3(dataset_base_uri):
                    missing_datasets.append(dataset.name)
                    logging.error("Dataset root missing in Bronze Data Lake: %s", dataset_base_uri)
                    continue

                # 2. Validate temporal partitions (Month-level resolution for optimal I/O)
                # We ensure at least one partition in the required window exists to prevent
                # pipeline execution against a completely empty temporal window.
                window_has_data = False
                for year, month in required_months:
                    partition_uri = f"{dataset_base_uri}/year={year}/month={month}"
                    if self._prefix_exists_in_s3(partition_uri):
                        window_has_data = True
                        break  # Fail-fast success for this dataset; no need to check other months

                if not window_has_data:
                    logging.warning(
                        "No data found for dataset '%s' in the requested temporal window.",
                        dataset.name
                    )
                    # Strict enforcement: Fail if the temporal window is entirely empty for a core dataset
                    missing_datasets.append(f"{dataset.name} (No temporal overlap)")

            if missing_datasets:
                raise FileNotFoundError(
                    f"Missing required upstream data in Bronze Data Lake: {missing_datasets}"
                )

            logging.info("Data Discovery successful. Upstream Bronze Lake partitions validated.")

        except Exception as exc:
            logging.exception("Data Discovery validation failed.")
            raise CustomException(exc, sys) from exc