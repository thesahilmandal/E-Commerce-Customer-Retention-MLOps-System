import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# INFRASTRUCTURE & CLOUD CONFIGURATIONS
# ==========================================================
S3_BUCKET_NAME: str = os.getenv("ML_S3_BUCKET")
S3_RAW_DATA_DIR_NAME: str = os.getenv("S3_RAW_DATA_DIR")
S3_FEATURE_STORE_DIR_NAME: str = os.getenv("S3_FEATURE_STORE_DIR")

# ==========================================================
# PIPELINE ARTIFACT DIRECTORIES
# ==========================================================
ARTIFACT_DIR_NAME: str = "artifacts"
DATA_PIPELINE_ROOT_DIR_NAME: str = "data_pipeline"

# 1. Extractor
EXTRACTOR_ROOT_DIR_NAME: str = "01_extractor"
EXTRACTOR_RAW_DATA_DIR_NAME: str = "raw_data"
EXTRACTOR_RAW_DATA_SCHEMA_FILE_NAME: str = "raw_schema.json"
EXTRACTOR_METADATA_FILE_NAME: str = "extractor_metadata.json"

# 2. Validator
VALIDATOR_ROOT_DIR_NAME: str = "02_validator"
VALIDATOR_REPORT_FILE_NAME: str = "validation_report.json"
REFERENCE_SCHEMA_FILE_PATH: str = str(Path("pipelines/data_pipeline/src/predefined_schema/v1/schema.json"))

# 3. Transformer
TRANSFORMER_ROOT_DIR_NAME: str = "03_transformer"
TRANSFORMER_METADATA_FILE_NAME: str = "transformer_metadata.json"
TRANSFORMER_CACHE_FILE_NAME: str = "transformer_cache.db"

# 4. Loader
LOADER_ROOT_DIR_NAME: str = "04_loader"
LOADER_METADATA_FILE_NAME: str = "loader_metadata.json"
LOADER_MASTER_PANEL_LOCAL_FILE_NAME: str = "master_panel.parquet"

# ==========================================================
# BUSINESS LOGIC & COMPUTE PARAMETERS
# ==========================================================
# Target window for predictive modeling
TARGET_DAYS: int = 180

# Threads for out-of-core and parallel processing
COMPUTE_THREADS: int = 4

# Historical snapshot dates for bitemporal feature generation
SNAPSHOT_DATES: List[str] = [
    "2017-09-01",
    "2017-11-01",
    "2018-01-01",
    "2018-03-01",
]