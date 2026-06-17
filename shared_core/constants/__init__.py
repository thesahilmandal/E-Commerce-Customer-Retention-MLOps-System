from pathlib import Path
from typing import List

# ==========================================================
# GLOBAL SYSTEM CONSTANTS
# ==========================================================
ARTIFACT_DIR_NAME: str = "artifacts"
REFERENCE_SCHEMA_FILE_PATH: Path = Path("pipelines/data_pipeline/src/predefined_schema/v1/schema.json")

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
TRAIN_SNAPSHOT: List[str] = ["2017-09-01", "2017-11-01"]
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

# ==========================================================
# TRAINING PIPELINE COMPONENT CONSTANTS
# ==========================================================
TRAINING_PIPELINE_ROOT_DIR_NAME: str = "training_pipeline"

# 01 - Data Ingestion
DATA_INGESTION_ROOT_DIR_NAME: str = "01_data_ingestion"
DATA_INGESTION_TRAIN_FILE_NAME: str = "train.parquet"
DATA_INGESTION_VAL_FILE_NAME: str = "val.parquet"
DATA_INGESTION_TEST_FILE_NAME: str = "test.parquet"
DATA_INGESTION_METADATA_FILE_NAME: str = "metadata.json"

# 02 - Data Transformation
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

# 03 - Model Trainer
MODEL_TRAINER_ROOT_DIR_NAME: str = "03_model_trainer"
MODEL_TRAINER_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_SHAP_SUMMARY_FILE_NAME: str = "shap_summary.png"
MODEL_TRAINER_METADATA_FILE_NAME: str = "metadata.json"
MODEL_TRAINER_MLFLOW_EXPERIMENT_NAME: str = "Customer_Retention_Optimization"

# 04 - Model Evaluation
MODEL_EVALUATION_ROOT_DIR_NAME: str = "04_model_evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME: str = "evaluation_report.json"
MODEL_EVALUATION_METADATA_FILE_NAME: str = "metadata.json"

# Evaluation Business Thresholds (The Gatekeeper Rules)
# A model must generate an Expected ROI > 0% to be considered.
MODEL_EVALUATION_MIN_EROI_THRESHOLD: float = 0.05  # 5% minimum acceptable EROI
# A Challenger must beat the Champion by at least 2% EROI to justify promotion risk.
MODEL_EVALUATION_EROI_HYSTERESIS_MARGIN: float = 0.02

# 05 - Model Registry
MODEL_REGISTRY_ROOT_DIR_NAME: str = "05_model_registry"
MODEL_REGISTRY_METADATA_FILE_NAME: str = "metadata.json"


# ==========================================================
# INFERENCE PIPELINE COMPONENT CONSTANTS
# ==========================================================
INFERENCE_PIPELINE_ROOT_DIR_NAME: str = "inference_pipeline"

# 01 - Current Production Model Loader
INFERENCE_MODEL_LOADER_ROOT_DIR_NAME: str = "01_model_loader"
INFERENCE_MODEL_LOADER_MODEL_FILE_NAME: str = "model.pkl"
INFERENCE_MODEL_LOADER_SCHEMA_FILE_NAME: str = "schema.json"
INFERENCE_MODEL_LOADER_METADATA_FILE_NAME: str = "metadata.json"

# 02 - Input Feature Matrix Builder
INFERENCE_FEATURE_MATRIX_BUILDER_ROOT_DIR_NAME: str = "02_input_feature_matrix_builder"
INFERENCE_FEATURE_MATRIX_FILE_NAME: str = "input_feature_matrix.parquet"
INFERENCE_FEATURE_MATRIX_SCHEMA_FILE_NAME: str = "schema.json"
INFERENCE_FEATURE_MATRIX_METADATA_FILE_NAME: str = "metadata.json"

# 03 - Input Feature Matrix Validator
INFERENCE_VALIDATOR_ROOT_DIR_NAME: str = "03_validator"
INFERENCE_VALIDATOR_REPORT_FILE_NAME: str = "report.json"
INFERENCE_VALIDATOR_METADATA_FILE_NAME: str = "metadata.json"

# 04 - Report Generator (Inference Publisher)
INFERENCE_REPORT_GENERATOR_ROOT_DIR_NAME: str = "04_report_generator"
INFERENCE_REPORT_GENERATOR_CSV_FILE_NAME: str = "customer_churn_report.csv"
INFERENCE_REPORT_GENERATOR_TELEMETRY_FILE_NAME: str = "telemetry_log.parquet"
INFERENCE_REPORT_GENERATOR_METADATA_FILE_NAME: str = "metadata.json"
INFERENCE_REPORT_GENERATOR_PROBABILITY_THRESHOLD: float = 0.5

# 05 - Report Publisher
INFERENCE_REPORT_PUBLISHER_ROOT_DIR_NAME: str = "05_report_publisher"
INFERENCE_REPORT_PUBLISHER_METADATA_FILE_NAME: str = "metadata.json"