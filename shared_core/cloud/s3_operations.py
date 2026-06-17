import os
import sys
from pathlib import Path
from typing import Tuple, Union

import boto3
from botocore.exceptions import ClientError

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging

PathLike = Union[str, Path]


class S3Sync:
    """
    Production-grade AWS S3 sync utility using native boto3.

    Responsibilities:
    - Replace fragile OS-level subprocess calls with robust AWS SDK (boto3).
    - Handle single file uploads/downloads.
    - Handle directory synchronization (recursive upload/download).
    - Provide structured logging and standardized exception handling.
    """

    def __init__(self) -> None:
        """Initializes the boto3 S3 client."""
        try:
            self.s3_client = boto3.client("s3")
        except Exception as exc:
            logging.exception("Failed to initialize boto3 S3 client.")
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------

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
        Download a single file from S3 using boto3.

        Args:
            s3_uri (str): Source S3 URI (e.g., s3://bucket/key)
            local_path (PathLike): Local file destination
        """
        try:
            bucket, key = self._parse_s3_uri(s3_uri)
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            logging.info("Downloading file from %s to %s", s3_uri, str(local_path))
            self.s3_client.download_file(bucket, key, str(local_path))
            logging.info("Successfully downloaded file to %s", str(local_path))

        except ClientError as exc:
            logging.error("AWS Boto3 ClientError during file download: %s", exc)
            raise CustomException(exc, sys) from exc
        except Exception as exc:
            logging.exception("Failed to download file from S3.")
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # UPLOAD SINGLE FILE
    # --------------------------------------------------

    def upload_file(self, local_path: PathLike, s3_uri: str) -> None:
        """
        Upload a single file to S3 using boto3.

        Args:
            local_path (PathLike): Local file path
            s3_uri (str): Destination S3 URI (e.g., s3://bucket/key)
        """
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                raise FileNotFoundError(f"File not found: {local_path}")

            bucket, key = self._parse_s3_uri(s3_uri)

            logging.info("Uploading file from %s to %s", str(local_path), s3_uri)
            self.s3_client.upload_file(str(local_path), bucket, key)
            logging.info("Successfully uploaded file to %s", s3_uri)

        except ClientError as exc:
            logging.error("AWS Boto3 ClientError during file upload: %s", exc)
            raise CustomException(exc, sys) from exc
        except Exception as exc:
            logging.exception("Failed to upload file to S3.")
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # SYNC LOCAL → S3
    # --------------------------------------------------

    def sync_folder_to_s3(self, folder: PathLike, aws_bucket_url: str) -> None:
        """
        Sync local folder to S3 bucket recursively using boto3.

        Args:
            folder (PathLike): Local folder path
            aws_bucket_url (str): Destination S3 URI prefix
        """
        try:
            folder_path = Path(folder)
            if not folder_path.exists() or not folder_path.is_dir():
                raise FileNotFoundError(f"Directory not found: {folder_path}")

            bucket, prefix = self._parse_s3_uri(aws_bucket_url)
            logging.info("Syncing local folder %s to S3 %s", str(folder_path), aws_bucket_url)

            upload_count = 0
            for root, _, files in os.walk(folder_path):
                for file_name in files:
                    local_file_path = Path(root) / file_name
                    
                    # Calculate the relative path to construct the correct S3 key
                    relative_path = local_file_path.relative_to(folder_path)
                    s3_key = f"{prefix}/{relative_path}".replace("\\", "/")  # Ensure posix paths
                    s3_key = s3_key.lstrip("/") # Remove leading slash if prefix was empty

                    self.s3_client.upload_file(str(local_file_path), bucket, s3_key)
                    upload_count += 1

            logging.info("Successfully synced %s files to %s", upload_count, aws_bucket_url)

        except ClientError as exc:
            logging.error("AWS Boto3 ClientError during folder sync to S3: %s", exc)
            raise CustomException(exc, sys) from exc
        except Exception as exc:
            logging.exception("Failed to sync folder to S3.")
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # SYNC S3 → LOCAL
    # --------------------------------------------------

    def sync_folder_from_s3(self, folder: PathLike, aws_bucket_url: str) -> None:
        """
        Sync S3 folder to local directory recursively using boto3 paginator.

        Args:
            folder (PathLike): Local folder destination
            aws_bucket_url (str): Source S3 URI prefix
        """
        try:
            folder_path = Path(folder)
            folder_path.mkdir(parents=True, exist_ok=True)

            bucket, prefix = self._parse_s3_uri(aws_bucket_url)
            # Ensure prefix ends with a slash for proper directory scoping
            if prefix and not prefix.endswith("/"):
                prefix += "/"

            logging.info("Syncing S3 %s to local folder %s", aws_bucket_url, str(folder_path))

            paginator = self.s3_client.get_paginator("list_objects_v2")
            download_count = 0

            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    
                    # Skip 'directory' markers (keys ending in '/')
                    if key.endswith("/"):
                        continue

                    # Strip the prefix to get the relative file path locally
                    relative_key = key[len(prefix):] if key.startswith(prefix) else key
                    local_file_path = folder_path / relative_key

                    # Ensure subdirectories exist
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)

                    self.s3_client.download_file(bucket, key, str(local_file_path))
                    download_count += 1

            logging.info("Successfully synced %s files from %s", download_count, aws_bucket_url)

        except ClientError as exc:
            logging.error("AWS Boto3 ClientError during folder sync from S3: %s", exc)
            raise CustomException(exc, sys) from exc
        except Exception as exc:
            logging.exception("Failed to sync folder from S3.")
            raise CustomException(exc, sys) from exc