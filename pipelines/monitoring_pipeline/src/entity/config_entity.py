import os
import sys
from datetime import datetime, timezone, timedelta

from pipelines.monitoring_pipeline.src import constants
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class MonitoringPipelineConfig:
    """
    Base configuration for the Monitoring Pipeline.
    Responsible for establishing the unique execution run ID and the 
    isolated root artifact directory for the current execution.
    """

    def __init__(self) -> None:
        try:
            self.run_id: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            self.root_dir: str = os.path.join(
                constants.ARTIFACT_DIR_NAME,
                constants.MONITORING_PIPELINE_ROOT_DIR_NAME,
                self.run_id,
            )
            os.makedirs(self.root_dir, exist_ok=True)

            logging.info(
                "MonitoringPipelineConfig initialized. Run ID: %s",
                self.run_id,
            )

        except Exception as e:
            logging.exception("Error initializing MonitoringPipelineConfig.")
            raise CustomException(e, sys) from e


class BaselineAndTelemetryResolverConfig:
    """
    Configuration for the Baseline & Telemetry Resolver component.
    Defines S3 URIs for fetching champion baselines, proactive telemetry, 
    historical reactive telemetry, and matured ground-truth labels.
    Calculates dynamic temporal bounds (current and lookback dates).
    """

    def __init__(self, pipeline_config: MonitoringPipelineConfig) -> None:
        try:
            self.resolver_root_dir: str = os.path.join(
                pipeline_config.root_dir,
                constants.MONITORING_RESOLVER_ROOT_DIR_NAME,
            )
            os.makedirs(self.resolver_root_dir, exist_ok=True)

            # S3 URIs
            self.s3_bucket_name: str = constants.S3_BUCKET_NAME
            self.s3_registry_pointer_uri: str = (
                f"s3://{self.s3_bucket_name}/"
                f"{constants.S3_MODEL_REGISTRY_DIR_NAME}/"
                f"{constants.S3_MODEL_REGISTRY_STATE_DIR}/"
                f"{constants.S3_MODEL_REGISTRY_POINTER_FILE_NAME}"
            )
            self.s3_telemetry_base_uri: str = (
                f"s3://{self.s3_bucket_name}/{constants.S3_INFERENCE_MLOPS_TELEMETRY_DIR}"
            )
            self.s3_data_lake_bronze_uri: str = (
                f"s3://{constants.S3_CUSTOMER_DATABASE_NAME}/{constants.S3_DATA_LAKE_BRONZE_DIR_NAME}"
            )

            # Temporal Bounds (UTC)
            now = datetime.now(timezone.utc)
            self.current_date: str = now.strftime("%Y-%m-%d")
            
            lookback_obj = now - timedelta(days=constants.MONITORING_LOOKBACK_DAYS)
            self.lookback_date: str = lookback_obj.strftime("%Y-%m-%d")

            # Hive Partition Keys for S3 Paths
            self.current_partition_suffix: str = (
                f"year={now.year}/month={now.month:02d}/day={now.day:02d}"
            )
            self.lookback_partition_suffix: str = (
                f"year={lookback_obj.year}/month={lookback_obj.month:02d}/day={lookback_obj.day:02d}"
            )

            # Local Artifact Paths (Stashing downloaded assets)
            self.baseline_metrics_file_path: str = os.path.join(
                self.resolver_root_dir, "baseline_performance_metrics.json"
            )
            self.reference_distributions_file_path: str = os.path.join(
                self.resolver_root_dir, "reference_feature_distributions.json"
            )
            self.shap_importance_file_path: str = os.path.join(
                self.resolver_root_dir, "shap_feature_importance_summary.json"
            )
            
            self.current_telemetry_file_path: str = os.path.join(
                self.resolver_root_dir, constants.MONITORING_RESOLVER_CURRENT_TELEMETRY_FILE_NAME
            )
            self.lookback_telemetry_file_path: str = os.path.join(
                self.resolver_root_dir, constants.MONITORING_RESOLVER_LOOKBACK_TELEMETRY_FILE_NAME
            )
            self.lookback_labels_file_path: str = os.path.join(
                self.resolver_root_dir, constants.MONITORING_RESOLVER_LOOKBACK_LABELS_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.resolver_root_dir, constants.MONITORING_RESOLVER_METADATA_FILE_NAME
            )

            logging.info(
                "BaselineAndTelemetryResolverConfig initialized. "
                "Current Date: %s | Lookback Date: %s",
                self.current_date, self.lookback_date
            )

        except Exception as e:
            logging.exception("Error initializing BaselineAndTelemetryResolverConfig.")
            raise CustomException(e, sys) from e


class StatisticalDriftCalculatorConfig:
    """
    Configuration for the label-independent Statistical Drift Calculator component.
    """

    def __init__(self, pipeline_config: MonitoringPipelineConfig) -> None:
        try:
            self.drift_root_dir: str = os.path.join(
                pipeline_config.root_dir,
                constants.MONITORING_DRIFT_ROOT_DIR_NAME,
            )
            os.makedirs(self.drift_root_dir, exist_ok=True)

            self.drift_report_file_path: str = os.path.join(
                self.drift_root_dir, constants.MONITORING_DRIFT_REPORT_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.drift_root_dir, constants.MONITORING_DRIFT_METADATA_FILE_NAME
            )

            # Constraints
            self.top_shap_features_count: int = constants.MONITORING_TOP_SHAP_FEATURES_COUNT

            logging.info("StatisticalDriftCalculatorConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing StatisticalDriftCalculatorConfig.")
            raise CustomException(e, sys) from e


class PerformanceEvaluatorConfig:
    """
    Configuration for the label-dependent Performance Evaluator component.
    """

    def __init__(self, pipeline_config: MonitoringPipelineConfig) -> None:
        try:
            self.evaluator_root_dir: str = os.path.join(
                pipeline_config.root_dir,
                constants.MONITORING_EVALUATOR_ROOT_DIR_NAME,
            )
            os.makedirs(self.evaluator_root_dir, exist_ok=True)

            self.performance_report_file_path: str = os.path.join(
                self.evaluator_root_dir, constants.MONITORING_EVALUATOR_REPORT_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.evaluator_root_dir, constants.MONITORING_EVALUATOR_METADATA_FILE_NAME
            )
            
            # Business Logic Config (to compute Realized ROI)
            self.campaign_cost: float = constants.MODEL_EVALUATION_CAMPAIGN_COST
            self.customer_ltv: float = constants.MODEL_EVALUATION_CUSTOMER_LTV
            self.intervention_save_rate: float = constants.MODEL_EVALUATION_INTERVENTION_SAVE_RATE

            logging.info("PerformanceEvaluatorConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing PerformanceEvaluatorConfig.")
            raise CustomException(e, sys) from e


class RuleEngineConfig:
    """
    Configuration for the Rule Engine component.
    Consolidates drift and performance artifacts to compute the deterministic
    retraining decision (`need_update`).
    """

    def __init__(self, pipeline_config: MonitoringPipelineConfig) -> None:
        try:
            self.rule_engine_root_dir: str = os.path.join(
                pipeline_config.root_dir,
                constants.MONITORING_RULE_ENGINE_ROOT_DIR_NAME,
            )
            os.makedirs(self.rule_engine_root_dir, exist_ok=True)

            self.monitoring_report_file_path: str = os.path.join(
                self.rule_engine_root_dir, constants.MONITORING_REPORT_FILE_NAME
            )
            self.need_update_file_path: str = os.path.join(
                self.rule_engine_root_dir, constants.MONITORING_NEED_UPDATE_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.rule_engine_root_dir, constants.MONITORING_RULE_ENGINE_METADATA_FILE_NAME
            )

            # Retraining Thresholds
            self.prediction_drift_threshold_psi: float = constants.MONITORING_PREDICTION_DRIFT_THRESHOLD_PSI
            self.feature_drift_threshold_psi: float = constants.MONITORING_FEATURE_DRIFT_THRESHOLD_PSI
            self.min_drifted_features_for_retrain: int = constants.MONITORING_MIN_DRIFTED_FEATURES_FOR_RETRAIN
            self.brier_degradation_threshold: float = constants.MONITORING_BRIER_DEGRADATION_THRESHOLD

            logging.info("RuleEngineConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing RuleEngineConfig.")
            raise CustomException(e, sys) from e