"""
Global Constants Configuration for the Inference Pipeline.

This module centralizes all static configurations, artifact directory names,
cloud storage paths, and business logic thresholds required by the Inference
Pipeline. Centralizing these values eliminates hardcoded strings, promotes
consistency across pipeline components, and allows shared infrastructure
configuration to be managed through environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# GLOBAL SYSTEM & CLOUD CONSTANTS (AWS S3)
# ==========================================================
ARTIFACT_DIR_NAME: str = os.getenv("ARTIFACT_DIR_NAME")

S3_BUCKET_NAME: str = os.getenv("ML_S3_BUCKET_NAME")
S3_FEATURE_STORE_DIR_NAME: str = os.getenv("S3_FEATURE_STORE_DIR")

# ML Model Registry S3 Paths
S3_MODEL_REGISTRY_DIR_NAME: str = os.getenv("S3_MODEL_REGISTRY_DIR_NAME")
S3_MODEL_REGISTRY_MODELS_DIR: str = os.getenv("S3_MODEL_REGISTRY_MODELS_DIR")
S3_MODEL_REGISTRY_STATE_DIR: str = os.getenv("S3_MODEL_REGISTRY_STATE_DIR")
S3_MODEL_REGISTRY_POINTER_FILE_NAME: str = os.getenv(
    "S3_MODEL_REGISTRY_POINTER_FILE_NAME"
)

# Data Lake S3 Paths
S3_CUSTOMER_DATABASE_NAME: str = "company-central-data-lake"
S3_DATA_LAKE_BRONZE_DIR_NAME: str = "bronze"


# Shared file name reference from the Data Pipeline Feature Store
LOADER_MASTER_PANEL_LOCAL_FILE_NAME: str = "master_panel.parquet"

# Inference Output S3 Paths
S3_INFERENCE_BUSINESS_REPORTS_DIR: str = (
    "business_reports/customer_churn"
)
S3_INFERENCE_MLOPS_TELEMETRY_DIR: str = (
    "mlops_telemetry/inference_logs/pipeline=churn_prediction"
)


# ==========================================================
# BUSINESS LOGIC
# ==========================================================
INFERENCE_REPORT_GENERATOR_PROBABILITY_THRESHOLD: float = 0.5


# ==========================================================
# INFERENCE PIPELINE COMPONENT CONSTANTS
# ==========================================================
INFERENCE_PIPELINE_ROOT_DIR_NAME: str = "inference_pipeline"

# ----------------------------------------------------------
# 01 - Production Model Loader
# ----------------------------------------------------------
INFERENCE_MODEL_LOADER_ROOT_DIR_NAME: str = "01_model_loader"
INFERENCE_MODEL_LOADER_MODEL_FILE_NAME: str = "model.pkl"
INFERENCE_MODEL_LOADER_SCHEMA_FILE_NAME: str = "schema.json"
INFERENCE_MODEL_LOADER_METADATA_FILE_NAME: str = "metadata.json"

# ----------------------------------------------------------
# 02 - Input Feature Matrix Builder
# ----------------------------------------------------------
INFERENCE_FEATURE_MATRIX_BUILDER_ROOT_DIR_NAME: str = (
    "02_input_feature_matrix_builder"
)
INFERENCE_FEATURE_MATRIX_FILE_NAME: str = "input_feature_matrix.parquet"
INFERENCE_FEATURE_MATRIX_SCHEMA_FILE_NAME: str = "schema.json"
INFERENCE_FEATURE_MATRIX_METADATA_FILE_NAME: str = "metadata.json"

# ----------------------------------------------------------
# 03 - Input Feature Matrix Validator
# ----------------------------------------------------------
INFERENCE_VALIDATOR_ROOT_DIR_NAME: str = "03_validator"
INFERENCE_VALIDATOR_REPORT_FILE_NAME: str = "report.json"
INFERENCE_VALIDATOR_METADATA_FILE_NAME: str = "metadata.json"

# ----------------------------------------------------------
# 04 - Report Generator
# ----------------------------------------------------------
INFERENCE_REPORT_GENERATOR_ROOT_DIR_NAME: str = "04_report_generator"
INFERENCE_REPORT_GENERATOR_CSV_FILE_NAME: str = (
    "customer_churn_report.csv"
)
INFERENCE_REPORT_GENERATOR_TELEMETRY_FILE_NAME: str = (
    "telemetry_log.parquet"
)
INFERENCE_REPORT_GENERATOR_METADATA_FILE_NAME: str = "metadata.json"

# ----------------------------------------------------------
# 05 - Report Publisher
# ----------------------------------------------------------
INFERENCE_REPORT_PUBLISHER_ROOT_DIR_NAME: str = "05_report_publisher"
INFERENCE_REPORT_PUBLISHER_METADATA_FILE_NAME: str = "metadata.json"