"""
Global Constants Configuration for the Monitoring Pipeline.

This module centralizes all static configurations, artifact directory names,
cloud storage paths, and deterministic business rules required by the 
Monitoring Pipeline. Centralizing these values ensures consistency across 
pipeline components and prevents hardcoded magic numbers in statistical logic.
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# GLOBAL SYSTEM & CLOUD CONSTANTS (AWS S3)
# ==========================================================
ARTIFACT_DIR_NAME: str = os.getenv("ARTIFACT_DIR_NAME", "artifacts")

S3_BUCKET_NAME: str = os.getenv("ML_S3_BUCKET_NAME")

# ML Model Registry S3 Paths (To fetch Champion baselines)
S3_MODEL_REGISTRY_DIR_NAME: str = os.getenv("S3_MODEL_REGISTRY_DIR_NAME", "model_registry")
S3_MODEL_REGISTRY_STATE_DIR: str = os.getenv("S3_MODEL_REGISTRY_STATE_DIR", "state")
S3_MODEL_REGISTRY_POINTER_FILE_NAME: str = os.getenv("S3_MODEL_REGISTRY_POINTER_FILE_NAME", "model_state.json")

# Data Lake S3 Paths (To fetch matured ground-truth labels)
S3_CUSTOMER_DATABASE_NAME: str = "company-central-data-lake"
S3_DATA_LAKE_BRONZE_DIR_NAME: str = "bronze"

# Inference Telemetry S3 Paths (To fetch daily operational logs)
S3_INFERENCE_MLOPS_TELEMETRY_DIR: str = "mlops_telemetry/inference_logs/pipeline=churn_prediction"

# ==========================================================
# BUSINESS LOGIC & STATISTICAL THRESHOLDS
# ==========================================================
TARGET_COLUMN: str = "target_is_churn"
CUSTOMER_ID_COLUMN: str = "customer_unique_id"

# The number of days required for churn labels to fully mature
MONITORING_LOOKBACK_DAYS: int = 30 

# Retraining Logic Thresholds (Deterministic Rule Engine Constraints)
MONITORING_TOP_SHAP_FEATURES_COUNT: int = 5
MONITORING_PREDICTION_DRIFT_THRESHOLD_PSI: float = 0.20
MONITORING_FEATURE_DRIFT_THRESHOLD_PSI: float = 0.20
MONITORING_MIN_DRIFTED_FEATURES_FOR_RETRAIN: int = 2

# Brier Score Degradation Threshold (e.g., 1.05 = 5% relative degradation)
MONITORING_BRIER_DEGRADATION_THRESHOLD: float = 1.05

# ROI calculation constants (mirroring Training/Inference config)
MODEL_EVALUATION_CAMPAIGN_COST: float = 10.0
MODEL_EVALUATION_CUSTOMER_LTV: float = 150.0
MODEL_EVALUATION_INTERVENTION_SAVE_RATE: float = 0.20

# ==========================================================
# MONITORING PIPELINE COMPONENT CONSTANTS
# ==========================================================
MONITORING_PIPELINE_ROOT_DIR_NAME: str = "monitoring_pipeline"

# ----------------------------------------------------------
# 01 - Baseline & Telemetry Resolver
# ----------------------------------------------------------
MONITORING_RESOLVER_ROOT_DIR_NAME: str = "01_baseline_and_telemetry_resolver"
MONITORING_RESOLVER_CURRENT_TELEMETRY_FILE_NAME: str = "current_telemetry.parquet"
MONITORING_RESOLVER_LOOKBACK_TELEMETRY_FILE_NAME: str = "lookback_telemetry.parquet"
MONITORING_RESOLVER_LOOKBACK_LABELS_FILE_NAME: str = "lookback_matured_labels.parquet"
MONITORING_RESOLVER_METADATA_FILE_NAME: str = "metadata.json"

# ----------------------------------------------------------
# 02 - Statistical Drift Calculator
# ----------------------------------------------------------
MONITORING_DRIFT_ROOT_DIR_NAME: str = "02_statistical_drift_calculator"
MONITORING_DRIFT_REPORT_FILE_NAME: str = "drift_report.json"
MONITORING_DRIFT_METADATA_FILE_NAME: str = "metadata.json"

# ----------------------------------------------------------
# 03 - Performance Evaluator
# ----------------------------------------------------------
MONITORING_EVALUATOR_ROOT_DIR_NAME: str = "03_performance_evaluator"
MONITORING_EVALUATOR_REPORT_FILE_NAME: str = "performance_report.json"
MONITORING_EVALUATOR_METADATA_FILE_NAME: str = "metadata.json"

# ----------------------------------------------------------
# 04 - Rule Engine
# ----------------------------------------------------------
MONITORING_RULE_ENGINE_ROOT_DIR_NAME: str = "04_rule_engine"
MONITORING_REPORT_FILE_NAME: str = "monitoring_report.json"
MONITORING_NEED_UPDATE_FILE_NAME: str = "need_update.json"
MONITORING_RULE_ENGINE_METADATA_FILE_NAME: str = "metadata.json"