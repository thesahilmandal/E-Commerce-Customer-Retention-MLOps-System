import os
import sys
import json
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


# ============================================================
# Internal Helpers
# ============================================================


def _prepare_file_path(file_path: str, replace: bool = True) -> None:
    """
    Prepare file path by optionally removing existing file and
    ensuring parent directories exist.
    """
    try:
        if replace and os.path.exists(file_path):
            logging.info(f"Removing existing file: {file_path}")
            os.remove(file_path)

        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    except Exception as e:
        raise CustomException(e, sys)


def _validate_file_exists(file_path: str, file_type: str) -> None:
    """
    Validate if a file exists.

    Args:
        file_path (str): Path to file.
        file_type (str): Type of file for logging.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_type} file not found: {file_path}")


# ============================================================
# YAML Utilities
# ============================================================


def read_yaml(file_path: str) -> Dict[str, Any]:
    """Read YAML file."""
    try:
        _validate_file_exists(file_path, "YAML")
        logging.info(f"Reading YAML file: {file_path}")

        with open(file_path, "rb") as file:
            return yaml.safe_load(file)

    except Exception as e:
        raise CustomException(e, sys)


def write_yaml(
    file_path: str,
    content: Dict[str, Any],
    replace: bool = True,
) -> None:
    """Write YAML file."""
    try:
        logging.info(f"Writing YAML file: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "w") as file:
            yaml.dump(content, file, sort_keys=False)

    except Exception as e:
        raise CustomException(e, sys)


# ============================================================
# JSON Utilities
# ============================================================


def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read JSON file."""
    try:
        _validate_file_exists(file_path, "JSON")
        logging.info(f"Reading JSON file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as e:
        raise CustomException(e, sys)


def write_json_file(
    file_path: str,
    content: Dict[str, Any],
    replace: bool = True,
) -> None:
    """Write JSON file."""
    try:
        logging.info(f"Writing JSON file: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(content, file, indent=4)

    except Exception as e:
        raise CustomException(e, sys)


# ============================================================
# NumPy Utilities
# ============================================================


def save_numpy(
    file_path: str,
    array: np.ndarray,
    replace: bool = True,
) -> None:
    """Save NumPy array."""
    try:
        logging.info(f"Saving NumPy array: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "wb") as file:
            np.save(file, array)

    except Exception as e:
        raise CustomException(e, sys)


def load_numpy(file_path: str) -> np.ndarray:
    """Load NumPy array."""
    try:
        _validate_file_exists(file_path, "NumPy")
        logging.info(f"Loading NumPy array: {file_path}")

        with open(file_path, "rb") as file:
            return np.load(file)

    except Exception as e:
        raise CustomException(e, sys)


# ============================================================
# Pickle Utilities
# ============================================================


def save_object(
    file_path: str,
    obj: Any,
    replace: bool = True,
) -> None:
    """Save Python object using pickle."""
    try:
        logging.info(f"Saving object: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """Load Python object from pickle."""
    try:
        _validate_file_exists(file_path, "Object")
        logging.info(f"Loading object: {file_path}")

        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        raise CustomException(e, sys)


# ============================================================
# CSV Utilities
# ============================================================


def read_csv_file(file_path: str, **kwargs) -> pd.DataFrame:
    """Read CSV file into DataFrame."""
    try:
        _validate_file_exists(file_path, "CSV")
        logging.info(f"Reading CSV file: {file_path}")

        df = pd.read_csv(file_path, **kwargs)
        logging.info(f"CSV loaded successfully | Shape: {df.shape}")

        return df

    except Exception as e:
        raise CustomException(e, sys)


def write_csv(
    file_path: str,
    df: pd.DataFrame,
    replace: bool = True,
    index: bool = False,
    **kwargs,
) -> None:
    """Write DataFrame to CSV."""
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        logging.info(f"Saving CSV file: {file_path}")
        _prepare_file_path(file_path, replace)

        df.to_csv(file_path, index=index, **kwargs)
        logging.info(f"CSV saved successfully | Path: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


# ============================================================
# Parquet Utilities
# ============================================================


def csv_to_parquet(
    csv_path: str,
    parquet_path: str,
    replace: bool = True,
    **read_kwargs,
) -> None:
    """Convert CSV to Parquet."""
    try:
        _validate_file_exists(csv_path, "CSV")
        logging.info(f"Converting CSV to Parquet: {csv_path}")

        _prepare_file_path(parquet_path, replace)

        df = pd.read_csv(csv_path, **read_kwargs)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path)

        logging.info(f"Parquet created at: {parquet_path}")

    except Exception as e:
        raise CustomException(e, sys)


def parquet_to_csv(
    parquet_path: str,
    csv_path: str,
    replace: bool = True,
    index: bool = False,
    **kwargs,
) -> None:
    """Convert Parquet to CSV."""
    try:
        _validate_file_exists(parquet_path, "Parquet")
        logging.info(f"Converting Parquet to CSV: {parquet_path}")

        _prepare_file_path(csv_path, replace)

        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        df.to_csv(csv_path, index=index, **kwargs)

        logging.info(f"CSV created at: {csv_path}")

    except Exception as e:
        raise CustomException(e, sys)


def read_parquet_file(file_path: str) -> pd.DataFrame:
    """Read Parquet file into DataFrame."""
    try:
        _validate_file_exists(file_path, "Parquet")
        logging.info(f"Reading Parquet file: {file_path}")

        table = pq.read_table(file_path)
        df = table.to_pandas()

        logging.info(f"Parquet loaded successfully | Shape: {df.shape}")
        return df

    except Exception as e:
        raise CustomException(e, sys)