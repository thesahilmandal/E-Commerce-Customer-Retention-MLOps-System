import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Tuple, Union

import boto3
from botocore.exceptions import ClientError

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging

PathLike = Union[str, Path]

class S3Sync:
    """
    Production-grade AWS S3 sync utility using native boto3.

    Responsibilities:
    - Handle single file uploads/downloads with thread-safe client reuse.
    - Handle directory synchronization (recursive upload/download) via concurrent execution.
    - Utilize Thread-Local Storage (TLS) to manage boto3 clients safely across concurrent threads.
    - Rely on native botocore credential chains to ensure automatic STS token rotation.
    - Provide structured logging and standardized exception handling.
    """

    def __init__(self, max_workers: int = 10) -> None:
        """
        Initializes the S3Sync utility.

        Args:
            max_workers (int): Maximum number of concurrent threads for bulk transfers.
        """
        self.max_workers = max_workers
        self._thread_local = threading.local()
        logging.info(
            "S3Sync initialized with %s max workers. Native Boto3 credential management enabled.",
            self.max_workers
        )

    # --------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------

    def _get_client(self) -> Any:
        """
        Retrieves or creates a thread-local boto3 S3 client.
        Relying on boto3.client() without frozen credentials ensures that 
        STS tokens automatically rotate in production cloud environments.
        """
        if not hasattr(self._thread_local, "s3_client"):
            self._thread_local.s3_client = boto3.client("s3")
        return self._thread_local.s3_client

    def _parse_s3_uri(self, s3_uri: str) -> Tuple[str, str]:
        """
        Parses an S3 URI into bucket and key/prefix components.

        Args:
            s3_uri (str): S3 URI (e.g., s3://bucket-name/path/to/key)

        Returns:
            Tuple[str, str]: (bucket_name, key_or_prefix)
        """
        if not str(s3_uri).startswith("s3://"):
            raise ValueError(f"Invalid S3 URI. Must start with 's3://': {s3_uri}")

        parts = str(s3_uri).replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    # --------------------------------------------------
    # DOWNLOAD SINGLE FILE
    # --------------------------------------------------

    def download_file(self, s3_uri: str, local_path: PathLike) -> None:
        """
        Download a single file from S3.

        Args:
            s3_uri (str): Source S3 URI (e.g., s3://bucket/key)
            local_path (PathLike): Local file destination
        """
        try:
            bucket, key = self._parse_s3_uri(s3_uri)
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            s3_client = self._get_client()
            s3_client.download_file(bucket, key, str(local_path))

        except ClientError as exc:
            logging.error("AWS Boto3 ClientError during file download: %s", exc)
            raise CustomException(exc, sys) from exc
        except Exception as exc:
            logging.error("Unexpected error during file download: %s", exc)
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # UPLOAD SINGLE FILE
    # --------------------------------------------------

    def upload_file(self, local_path: PathLike, s3_uri: str) -> None:
        """
        Upload a single file to S3.

        Args:
            local_path (PathLike): Local file path
            s3_uri (str): Destination S3 URI (e.g., s3://bucket/key)
        """
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                raise FileNotFoundError(f"File not found: {local_path}")

            bucket, key = self._parse_s3_uri(s3_uri)

            s3_client = self._get_client()
            s3_client.upload_file(str(local_path), bucket, key)

        except ClientError as exc:
            logging.error("AWS Boto3 ClientError during file upload: %s", exc)
            raise CustomException(exc, sys) from exc
        except Exception as exc:
            logging.error("Unexpected error during file upload: %s", exc)
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # SYNC LOCAL → S3 (CONCURRENT)
    # --------------------------------------------------

    def sync_folder_to_s3(self, folder: PathLike, aws_bucket_url: str) -> None:
        """
        Sync local folder to S3 bucket recursively using concurrent execution.

        Args:
            folder (PathLike): Local folder path
            aws_bucket_url (str): Destination S3 URI prefix
        """
        folder_path = Path(folder)
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {folder_path}")

        bucket, prefix = self._parse_s3_uri(aws_bucket_url)
        logging.info("Syncing local folder %s to S3 %s", str(folder_path), aws_bucket_url)

        upload_tasks = []
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                local_file_path = Path(root) / file_name
                relative_path = local_file_path.relative_to(folder_path)
                s3_key = f"{prefix}/{relative_path}".replace("\\", "/")
                s3_key = s3_key.lstrip("/")

                upload_tasks.append((local_file_path, f"s3://{bucket}/{s3_key}"))

        if not upload_tasks:
            logging.warning("No files found to sync in %s", str(folder_path))
            return

        def _upload_task(args: Tuple[PathLike, str]) -> None:
            self.upload_file(args[0], args[1])

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                list(executor.map(_upload_task, upload_tasks))
            logging.info("Successfully synced %s files to %s", len(upload_tasks), aws_bucket_url)
        except Exception as exc:
            logging.error("Failed during concurrent folder upload.")
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # SYNC S3 → LOCAL (CONCURRENT)
    # --------------------------------------------------

    def sync_folder_from_s3(self, folder: PathLike, aws_bucket_url: str) -> None:
        """
        Sync S3 folder to local directory recursively using concurrent execution.

        Args:
            folder (PathLike): Local folder destination
            aws_bucket_url (str): Source S3 URI prefix
        """
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        bucket, prefix = self._parse_s3_uri(aws_bucket_url)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        logging.info("Syncing S3 %s to local folder %s", aws_bucket_url, str(folder_path))

        s3_client = self._get_client()
        paginator = s3_client.get_paginator("list_objects_v2")
        download_tasks = []

        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith("/"):
                        continue

                    relative_key = key[len(prefix):] if key.startswith(prefix) else key
                    local_file_path = folder_path / relative_key

                    download_tasks.append((f"s3://{bucket}/{key}", local_file_path))

            if not download_tasks:
                logging.warning("No files found to sync from %s", aws_bucket_url)
                return

            def _download_task(args: Tuple[str, PathLike]) -> None:
                self.download_file(args[0], args[1])

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                list(executor.map(_download_task, download_tasks))

            logging.info("Successfully synced %s files from %s", len(download_tasks), aws_bucket_url)

        except ClientError as exc:
            logging.error("AWS Boto3 ClientError during folder sync from S3: %s", exc)
            raise CustomException(exc, sys) from exc
        except Exception as exc:
            logging.error("Failed during concurrent folder download.")
            raise CustomException(exc, sys) from exc