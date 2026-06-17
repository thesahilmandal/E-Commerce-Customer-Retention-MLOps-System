import os
from pathlib import Path
from typing import List

# ==========================================================
# GLOBAL SYSTEM CONSTANTS
# ==========================================================
ARTIFACT_DIR_NAME: str = "artifacts"

# ==========================================================
# CLOUD & INFRASTRUCTURE CONSTANTS (AWS S3)
# ==========================================================
S3_BUCKET_NAME: str = "ml-platform-production"
S3_RAW_DATA_DIR_NAME: str = "raw_data"
S3_FEATURE_STORE_DIR_NAME: str = "feature_store"
S3_ARTIFACT_DIR_NAME: str = "artifacts"
S3_LOGS_DIR_NAME: str = "logs"

# ML Model Registry S3 Paths
S3_MODEL_REGISTRY_DIR_NAME: str = "model_registry"
S3_MODEL_REGISTRY_MODELS_DIR: str = "models"
S3_MODEL_REGISTRY_STATE_DIR: str = "state"
S3_MODEL_REGISTRY_POINTER_FILE_NAME: str = "production_champion.json"

# Data Lake S3 Paths
S3_CUSTOMER_DATABASE_NAME: str = "company-central-data-lake"
S3_DATA_LAKE_BRONZE_DIR_NAME: str = "bronze"

# Inference Publisher S3 Paths
S3_INFERENCE_BUSINESS_REPORTS_DIR: str = "business_reports/customer_churn"
S3_INFERENCE_MLOPS_TELEMETRY_DIR: str = "mlops_telemetry/inference_logs/pipeline=churn_prediction"
