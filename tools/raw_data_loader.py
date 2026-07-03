import os
import tempfile
from typing import Iterator
from dotenv import load_dotenv

load_dotenv()

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import kagglehub
from tqdm import tqdm

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class S3StreamingUploader:
    """Handles streaming parquet writes + S3 upload."""

    def __init__(self, logger):
        self.logger = logger
        self.s3_client = boto3.client("s3")
        self.bucket = os.getenv("ML_S3_BUCKET_NAME")
        

    def upload_parquet_stream(
        self,
        df_iterator: Iterator[pd.DataFrame],
        s3_key: str
    ) -> None:
        """
        Stream DataFrame chunks into a single parquet file using ParquetWriter.
        """
        try:
            self.logger.info("Uploading to S3: %s", s3_key)

            with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
                writer = None

                for df in df_iterator:
                    table = pa.Table.from_pandas(df)

                    if writer is None:
                        writer = pq.ParquetWriter(
                            tmp_file.name,
                            table.schema
                        )

                    writer.write_table(table)

                if writer:
                    writer.close()

                self.s3_client.upload_file(
                    tmp_file.name,
                    self.bucket,
                    s3_key
                )

            self.logger.info("Upload successful: %s", s3_key)

        except Exception as exc:
            self.logger.exception("S3 upload failed: %s", s3_key)
            raise CustomException(exc, str(exc)) from exc


class StreamingCSVProcessor:
    """Chunked CSV reader."""

    def __init__(self, logger, chunk_size: int = 100_000):
        self.logger = logger
        self.chunk_size = chunk_size

    def read_in_chunks(self, file_path: str) -> Iterator[pd.DataFrame]:
        try:
            return pd.read_csv(
                file_path,
                chunksize=self.chunk_size,
                low_memory=False
            )
        except Exception as exc:
            self.logger.exception("CSV read failed: %s", file_path)
            raise CustomException(exc, str(exc)) from exc


class RawDataPipeline:
    """Main pipeline."""

    def __init__(self):
        self.logger = logging
        self.dataset_name = "olistbr/brazilian-ecommerce"

        self.processor = StreamingCSVProcessor(self.logger)
        self.uploader = S3StreamingUploader(self.logger)

    def _discover_csv_files(self, dataset_path: str):
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".csv"):
                    yield os.path.join(root, file)

    def run(self):
        try:
            self.logger.info("Starting pipeline...")

            dataset_path = kagglehub.dataset_download(self.dataset_name)

            for csv_file in tqdm(
                list(self._discover_csv_files(dataset_path)),
                desc="Processing files"
            ):
                try:
                    df_iter = self.processor.read_in_chunks(csv_file)

                    file_name = os.path.basename(csv_file).replace(
                        ".csv", ".parquet"
                    )

                    s3_key = f"{os.getenv("S3_RAW_DATA_DIR")}/{file_name}"

                    self.uploader.upload_parquet_stream(
                        df_iter,
                        s3_key
                    )

                except Exception as file_exc:
                    self.logger.error(
                        "Failed processing %s: %s",
                        csv_file,
                        str(file_exc)
                    )

            self.logger.info("Pipeline completed successfully")

        except Exception as exc:
            self.logger.exception("Pipeline failed")
            raise CustomException(exc, str(exc)) from exc


def main():
    pipeline = RawDataPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()