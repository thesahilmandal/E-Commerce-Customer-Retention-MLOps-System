import os
from pathlib import Path
from typing import List

REFERENCE_SCHEMA_FILE_PATH: Path = Path("pipelines/data_pipeline/src/predefined_schema/v1/schema.json")

# ==========================================================
# BUSINESS LOGIC & HYPERPARAMETERS
# ==========================================================
# Centralized configuration for bitemporal engineering
TARGET_DAYS: int = 180
COMPUTE_THREADS: int = 4

# Temporal Snapshot Definitions
SNAPSHOT_DATES: List[str] = [
    "2017-09-01",
    "2017-11-01",
    "2018-01-01",
    "2018-03-01",
]

# Out-of-Time (OOT) Splitting Definitions
TRAIN_SNAPSHOTS: List[str] = ["2017-09-01", "2017-11-01"]
VAL_SNAPSHOT: str = "2018-01-01"
TEST_SNAPSHOT: str = "2018-03-01"

# Target Variable and Metadata Columns (Used to isolate X and y)
TARGET_COLUMN: str = "target_is_churn"
SYSTEM_COLUMNS_TO_DROP: List[str] = [
    "customer_unique_id",
    "snapshot_date",
    "ingested_at_utc",
    "target_180d_ltv"
]

# ==========================================================
# DATA PIPELINE COMPONENT CONSTANTS
# ==========================================================
DATA_PIPELINE_ROOT_DIR_NAME: str = "data_pipeline"

# 01 - Extractor
EXTRACTOR_ROOT_DIR_NAME: str = "01_extractor"
EXTRACTOR_RAW_DATA_DIR_NAME: str = "raw_data"
EXTRACTOR_RAW_DATA_SCHEMA_FILE_NAME: str = "raw_data_schema.json"
EXTRACTOR_METADATA_FILE_NAME: str = "metadata.json"

# 02 - Validator
VALIDATOR_ROOT_DIR_NAME: str = "02_validator"
VALIDATOR_REPORT_FILE_NAME: str = "report.json"

# 03 - Transformer
TRANSFORMER_ROOT_DIR_NAME: str = "03_transformer"
TRANSFORMER_METADATA_FILE_NAME: str = "metadata.json"
TRANSFORMER_CACHE_FILE_NAME: str = "transformer_cache.db"

# 04 - Loader
LOADER_ROOT_DIR_NAME: str = "04_loader"
LOADER_METADATA_FILE_NAME: str = "metadata.json"
LOADER_MASTER_PANEL_LOCAL_FILE_NAME: str = "master_panel.parquet"
