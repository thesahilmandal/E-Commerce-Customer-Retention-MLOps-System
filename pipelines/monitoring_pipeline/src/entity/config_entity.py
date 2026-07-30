import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging

# Use TYPE_CHECKING to prevent circular imports at runtime while allowing type hints
if TYPE_CHECKING:
    from pipelines.monitoring_pipeline.src.core.context import MonitoringPipelineContext


@dataclass(frozen=True)
class BaselineAndTelemetryResolverConfig:
    """
    Configuration for the Baseline & Telemetry Resolver component.
    Defines S3 URIs for fetching champion baselines, proactive telemetry, 
    historical reactive telemetry, and matured ground-truth labels.
    """
    resolver_root_dir: str
    s3_registry_pointer_uri: str
    s3_telemetry_base_uri: str
    s3_data_lake_bronze_uri: str
    current_date: str
    lookback_date: str
    current_partition_suffix: str
    lookback_partition_suffix: str
    baseline_metrics_file_path: str
    reference_distributions_file_path: str
    shap_importance_file_path: str
    current_telemetry_file_path: str
    lookback_telemetry_file_path: str
    lookback_labels_file_path: str
    metadata_file_path: str
    lookback_period_days: int

    @classmethod
    def get_config(cls, context: "MonitoringPipelineContext") -> "BaselineAndTelemetryResolverConfig":
        try:
            config = context.config or {}
            
            # DEFENSIVE PATTERN: Use `or {}` to prevent NoneType errors if YAML keys exist but are empty (null)
            cloud_storage = config.get("cloud_storage") or {}
            component_config = config.get("component_config") or {}
            comp_cfg = component_config.get("baseline_and_telemetry_resolver") or {}
            monitoring_parameters = config.get("monitoring_parameters") or {}

            resolver_root_dir = os.path.join(context.root_dir, comp_cfg.get("dir_name", "01_baseline_and_telemetry_resolver"))
            os.makedirs(resolver_root_dir, exist_ok=True)

            lookback_days = monitoring_parameters.get("lookback_period_days", 30)
            exec_date_obj = datetime.strptime(context.execution_date, "%Y-%m-%d")
            lookback_date_obj = exec_date_obj - timedelta(days=lookback_days)
            
            lookback_date = lookback_date_obj.strftime("%Y-%m-%d")
            current_partition_suffix = f"year={exec_date_obj.year}/month={exec_date_obj.month:02d}/day={exec_date_obj.day:02d}"
            lookback_partition_suffix = f"year={lookback_date_obj.year}/month={lookback_date_obj.month:02d}/day={lookback_date_obj.day:02d}"

            bucket = cloud_storage.get("s3_data_lake_bucket", "company-central-data-lake")
            s3_registry_pointer = (
                f"s3://{bucket}/{cloud_storage.get('s3_model_registry_prefix', 'model_registry')}/"
                f"{cloud_storage.get('s3_model_registry_state_dir', 'state')}/"
                f"{cloud_storage.get('s3_model_registry_pointer_file', 'model_state.json')}"
            )
            s3_telemetry_base = f"s3://{bucket}/{cloud_storage.get('s3_inference_telemetry_prefix', 'inference_pipeline_artifacts/mlops_telemetry/inference_logs')}"
            s3_bronze_layer = f"s3://{bucket}/{cloud_storage.get('s3_bronze_layer_prefix', 'bronze')}"

            instance = cls(
                resolver_root_dir=resolver_root_dir,
                s3_registry_pointer_uri=s3_registry_pointer,
                s3_telemetry_base_uri=s3_telemetry_base,
                s3_data_lake_bronze_uri=s3_bronze_layer,
                current_date=context.execution_date,
                lookback_date=lookback_date,
                current_partition_suffix=current_partition_suffix,
                lookback_partition_suffix=lookback_partition_suffix,
                baseline_metrics_file_path=os.path.join(resolver_root_dir, comp_cfg.get("baseline_metrics_file", "baseline_performance_metrics.json")),
                reference_distributions_file_path=os.path.join(resolver_root_dir, comp_cfg.get("reference_distributions_file", "reference_feature_distributions.json")),
                shap_importance_file_path=os.path.join(resolver_root_dir, comp_cfg.get("shap_importance_file", "shap_feature_importance_summary.json")),
                current_telemetry_file_path=os.path.join(resolver_root_dir, comp_cfg.get("current_telemetry_file", "current_telemetry.parquet")),
                lookback_telemetry_file_path=os.path.join(resolver_root_dir, comp_cfg.get("lookback_telemetry_file", "lookback_telemetry.parquet")),
                lookback_labels_file_path=os.path.join(resolver_root_dir, comp_cfg.get("lookback_labels_file", "lookback_matured_labels.parquet")),
                metadata_file_path=os.path.join(resolver_root_dir, "metadata.json"),
                lookback_period_days=lookback_days
            )
            logging.info("BaselineAndTelemetryResolverConfig successfully constructed.")
            return instance
        except Exception as e:
            logging.exception("Error constructing BaselineAndTelemetryResolverConfig.")
            raise CustomException(e, sys) from e


@dataclass(frozen=True)
class StatisticalDriftCalculatorConfig:
    """
    Configuration for the label-independent Statistical Drift Calculator component.
    """
    drift_root_dir: str
    drift_report_file_path: str
    metadata_file_path: str
    top_shap_features_count: int
    zero_bin_epsilon_psi: float

    @classmethod
    def get_config(cls, context: "MonitoringPipelineContext") -> "StatisticalDriftCalculatorConfig":
        try:
            config = context.config or {}
            
            component_config = config.get("component_config") or {}
            comp_cfg = component_config.get("statistical_drift_calculator") or {}
            params = config.get("monitoring_parameters") or {}

            drift_root_dir = os.path.join(context.root_dir, comp_cfg.get("dir_name", "02_statistical_drift_calculator"))
            os.makedirs(drift_root_dir, exist_ok=True)

            instance = cls(
                drift_root_dir=drift_root_dir,
                drift_report_file_path=os.path.join(drift_root_dir, comp_cfg.get("drift_report_file", "drift_report.json")),
                metadata_file_path=os.path.join(drift_root_dir, "metadata.json"),
                top_shap_features_count=params.get("top_shap_features_count", 5),
                zero_bin_epsilon_psi=params.get("zero_bin_epsilon_psi", 0.0001)
            )
            logging.info("StatisticalDriftCalculatorConfig successfully constructed.")
            return instance
        except Exception as e:
            logging.exception("Error constructing StatisticalDriftCalculatorConfig.")
            raise CustomException(e, sys) from e


@dataclass(frozen=True)
class PerformanceEvaluatorConfig:
    """
    Configuration for the label-dependent Performance Evaluator component.
    """
    evaluator_root_dir: str
    performance_report_file_path: str
    metadata_file_path: str
    campaign_cost: float
    customer_ltv: float
    intervention_save_rate: float
    log_loss_epsilon: float

    @classmethod
    def get_config(cls, context: "MonitoringPipelineContext") -> "PerformanceEvaluatorConfig":
        try:
            config = context.config or {}
            
            component_config = config.get("component_config") or {}
            comp_cfg = component_config.get("performance_evaluator") or {}
            fin_params = config.get("financial_parameters") or {}
            mon_params = config.get("monitoring_parameters") or {}

            evaluator_root_dir = os.path.join(context.root_dir, comp_cfg.get("dir_name", "03_performance_evaluator"))
            os.makedirs(evaluator_root_dir, exist_ok=True)

            instance = cls(
                evaluator_root_dir=evaluator_root_dir,
                performance_report_file_path=os.path.join(evaluator_root_dir, comp_cfg.get("performance_report_file", "performance_report.json")),
                metadata_file_path=os.path.join(evaluator_root_dir, "metadata.json"),
                campaign_cost=fin_params.get("campaign_cost", 10.0),
                customer_ltv=fin_params.get("customer_ltv", 150.0),
                intervention_save_rate=fin_params.get("intervention_save_rate", 0.20),
                log_loss_epsilon=mon_params.get("log_loss_epsilon", 1.0e-15)
            )
            logging.info("PerformanceEvaluatorConfig successfully constructed.")
            return instance
        except Exception as e:
            logging.exception("Error constructing PerformanceEvaluatorConfig.")
            raise CustomException(e, sys) from e


@dataclass(frozen=True)
class RuleEngineConfig:
    """
    Configuration for the Rule Engine component.
    Consolidates drift and performance artifacts to compute the deterministic
    retraining decision.
    """
    rule_engine_root_dir: str
    monitoring_report_file_path: str
    need_update_file_path: str
    metadata_file_path: str
    prediction_drift_threshold_psi: float
    feature_drift_threshold_psi: float
    min_drifted_features_for_retrain: int
    brier_degradation_threshold_factor: float

    @classmethod
    def get_config(cls, context: "MonitoringPipelineContext") -> "RuleEngineConfig":
        try:
            config = context.config or {}
            
            component_config = config.get("component_config") or {}
            comp_cfg = component_config.get("rule_engine") or {}
            thresholds = config.get("rule_engine_thresholds") or {}

            rule_engine_root_dir = os.path.join(context.root_dir, comp_cfg.get("dir_name", "04_rule_engine"))
            os.makedirs(rule_engine_root_dir, exist_ok=True)

            instance = cls(
                rule_engine_root_dir=rule_engine_root_dir,
                monitoring_report_file_path=os.path.join(rule_engine_root_dir, comp_cfg.get("monitoring_report_file", "monitoring_report.json")),
                need_update_file_path=os.path.join(rule_engine_root_dir, comp_cfg.get("need_update_file", "need_update.json")),
                metadata_file_path=os.path.join(rule_engine_root_dir, "metadata.json"),
                prediction_drift_threshold_psi=thresholds.get("prediction_drift_threshold_psi", 0.20),
                feature_drift_threshold_psi=thresholds.get("feature_drift_threshold_psi", 0.20),
                min_drifted_features_for_retrain=thresholds.get("min_drifted_features_for_retrain", 2),
                brier_degradation_threshold_factor=thresholds.get("brier_degradation_threshold_factor", 1.05)
            )
            logging.info("RuleEngineConfig successfully constructed.")
            return instance
        except Exception as e:
            logging.exception("Error constructing RuleEngineConfig.")
            raise CustomException(e, sys) from e


@dataclass(frozen=True)
class ArtifactPublisherConfig:
    """
    Configuration for the Artifact Publisher component.
    Manages the upload of highly curated production artifacts to S3.
    """
    publisher_root_dir: str
    s3_bucket_name: str
    s3_monitoring_output_prefix: str
    s3_audit_reports_dir: str
    s3_action_tokens_dir: str
    s3_matured_evaluations_dir: str
    s3_metadata_dir: str
    run_id: str
    execution_date: str

    @classmethod
    def get_config(cls, context: "MonitoringPipelineContext") -> "ArtifactPublisherConfig":
        try:
            config = context.config or {}
            
            component_config = config.get("component_config") or {}
            comp_cfg = component_config.get("artifact_publisher") or {}
            cloud_storage = config.get("cloud_storage") or {}

            publisher_root_dir = os.path.join(context.root_dir, comp_cfg.get("dir_name", "05_artifact_publisher"))
            os.makedirs(publisher_root_dir, exist_ok=True)

            instance = cls(
                publisher_root_dir=publisher_root_dir,
                s3_bucket_name=cloud_storage.get("s3_data_lake_bucket", "company-central-data-lake"),
                s3_monitoring_output_prefix=cloud_storage.get("s3_monitoring_output_prefix", "monitoring_pipeline_artifacts"),
                s3_audit_reports_dir=cloud_storage.get("s3_monitoring_audit_reports_dir", "audit_reports"),
                s3_action_tokens_dir=cloud_storage.get("s3_monitoring_action_tokens_dir", "action_tokens"),
                s3_matured_evaluations_dir=cloud_storage.get("s3_monitoring_matured_evaluations_dir", "matured_evaluations"),
                s3_metadata_dir=cloud_storage.get("s3_monitoring_metadata_dir", "monitoring_metadata"),
                run_id=context.run_id,
                execution_date=context.execution_date
            )
            logging.info("ArtifactPublisherConfig successfully constructed.")
            return instance
        except Exception as e:
            logging.exception("Error constructing ArtifactPublisherConfig.")
            raise CustomException(e, sys) from e