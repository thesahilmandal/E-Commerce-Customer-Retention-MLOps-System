"""
Global Constants Configuration for the Training Pipeline.

This module centralizes all static configurations, path names, cloud URIs, 
and business logic thresholds required by the Training Pipeline. Centralizing 
these values prevents hardcoding, ensures consistency across components, 
and allows for dynamic overrides via environment variables where applicable.
"""

import os
from typing import List
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
S3_MODEL_REGISTRY_POINTER_FILE_NAME: str = os.getenv("S3_MODEL_REGISTRY_POINTER_FILE_NAME")

# Shared file name reference from the Data Pipeline Feature Store
LOADER_MASTER_PANEL_LOCAL_FILE_NAME: str = "master_panel.parquet"


# ==========================================================
# BUSINESS LOGIC, ML HYPERPARAMETERS & SPLIT DEFINITIONS
# ==========================================================
COMPUTE_THREADS: int = 4

# Out-of-Time (OOT) Temporal Splitting Definitions
TRAIN_SNAPSHOT: List[str] = ["2017-09-01", "2017-11-01"]
VAL_SNAPSHOT: str = "2018-01-01"
TEST_SNAPSHOT: str = "2018-03-01"

# Target Variable and Feature Isolation Rules
TARGET_COLUMN: str = "target_is_churn"
SYSTEM_COLUMNS_TO_DROP: List[str] = [
    "customer_unique_id",
    "snapshot_date",
    "ingested_at_utc",
    "target_180d_ltv",
]


# ==========================================================
# TRAINING PIPELINE COMPONENT CONSTANTS
# ==========================================================
TRAINING_PIPELINE_ROOT_DIR_NAME: str = "training_pipeline"

# ----------------------------------------------------------
# 01 - Data Ingestion
# ----------------------------------------------------------
DATA_INGESTION_ROOT_DIR_NAME: str = "01_data_ingestion"
DATA_INGESTION_TRAIN_FILE_NAME: str = "train.parquet"
DATA_INGESTION_VAL_FILE_NAME: str = "val.parquet"
DATA_INGESTION_TEST_FILE_NAME: str = "test.parquet"
DATA_INGESTION_METADATA_FILE_NAME: str = "metadata.json"

# ----------------------------------------------------------
# 02 - Feature Transformation
# ----------------------------------------------------------
DATA_TRANSFORMATION_ROOT_DIR_NAME: str = "02_data_transformation"
DATA_TRANSFORMATION_PREPROCESSOR_FILE_NAME: str = "preprocessor.pkl"
DATA_TRANSFORMATION_SCHEMA_FILE_NAME: str = "schema.json"
DATA_TRANSFORMATION_METADATA_FILE_NAME: str = "metadata.json"
DATA_TRANSFORMATION_X_TRAIN_FILE_NAME: str = "x_train.parquet"
DATA_TRANSFORMATION_Y_TRAIN_FILE_NAME: str = "y_train.parquet"
DATA_TRANSFORMATION_X_VAL_FILE_NAME: str = "x_val.parquet"
DATA_TRANSFORMATION_Y_VAL_FILE_NAME: str = "y_val.parquet"
DATA_TRANSFORMATION_X_TEST_FILE_NAME: str = "x_test.parquet"
DATA_TRANSFORMATION_Y_TEST_FILE_NAME: str = "y_test.parquet"

# ----------------------------------------------------------
# 03 - Model Training
# ----------------------------------------------------------
MODEL_TRAINER_ROOT_DIR_NAME: str = "03_model_trainer"
MODEL_TRAINER_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_SHAP_SUMMARY_FILE_NAME: str = "shap_summary.png"
MODEL_TRAINER_METADATA_FILE_NAME: str = "metadata.json"
MODEL_TRAINER_MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "Customer_Retention_Optimization"
)

# ----------------------------------------------------------
# 04 - Model Evaluation
# ----------------------------------------------------------
MODEL_EVALUATION_ROOT_DIR_NAME: str = "04_model_evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME: str = "evaluation_report.json"
MODEL_EVALUATION_METADATA_FILE_NAME: str = "metadata.json"

# Evaluation Thresholds (The Gatekeeper Rules)
MODEL_EVALUATION_MIN_EROI_THRESHOLD: float = 0.05     # 5% minimum acceptable EROI
MODEL_EVALUATION_EROI_HYSTERESIS_MARGIN: float = 0.02 # 2% required to beat Champion

# Decoupled Financial Business Logic for Expected ROI (EROI) calculations
MODEL_EVALUATION_CAMPAIGN_COST: float = 10.0
MODEL_EVALUATION_CUSTOMER_LTV: float = 500.0
MODEL_EVALUATION_INTERVENTION_SAVE_RATE: float = 0.10

# ----------------------------------------------------------
# 05 - Model Registration
# ----------------------------------------------------------
MODEL_REGISTRY_ROOT_DIR_NAME: str = "05_model_registry"
MODEL_REGISTRY_METADATA_FILE_NAME: str = "metadata.json"