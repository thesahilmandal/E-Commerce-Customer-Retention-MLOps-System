import sys
from typing import Any

from dotenv import load_dotenv

from pipelines.monitoring_pipeline.src import constants
from shared_core.cloud.s3_operations import S3Sync
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging

from pipelines.monitoring_pipeline.src.entity.config_entity import (
    MonitoringPipelineConfig,
    BaselineAndTelemetryResolverConfig,
    StatisticalDriftCalculatorConfig,
    PerformanceEvaluatorConfig,
    RuleEngineConfig,
)
from pipelines.monitoring_pipeline.src.components.baseline_and_telemetry_resolver import (
    BaselineAndTelemetryResolver,
)
from pipelines.monitoring_pipeline.src.components.statistical_drift_calculator import (
    StatisticalDriftCalculator,
)
from pipelines.monitoring_pipeline.src.components.performance_evaluator import (
    PerformanceEvaluator,
)
from pipelines.monitoring_pipeline.src.components.rule_engine import RuleEngine

load_dotenv()


class MonitoringPipeline:
    """
    Orchestrates the end-to-end monitoring diagnostic workflow.

    Pipeline Stages:
        1. Baseline & Telemetry Resolution (Proactive & Reactive Data Loading)
        2. Statistical Drift Calculation (Label-Independent Evaluation)
        3. Performance Evaluation (Label-Dependent Evaluation via Lookback)
        4. Rule Engine (Deterministic Retraining Decision)
        5. Artifact Synchronization to AWS S3
    """

    def __init__(self) -> None:
        """
        Initialize the monitoring pipeline configuration.
        """
        try:
            logging.info("Initializing monitoring pipeline.")
            self.pipeline_config = MonitoringPipelineConfig()
        except Exception as exc:
            logging.exception(
                "Failed to initialize monitoring pipeline configuration."
            )
            raise CustomException(exc, sys) from exc

    @staticmethod
    def _log_artifact(artifact: Any) -> None:
        """
        Log pipeline artifact details.

        Args:
            artifact: Pipeline stage artifact.
        """
        logging.debug("Generated artifact: %s", artifact)

    def _run_baseline_and_telemetry_resolver(self) -> Any:
        """
        Execute the baseline and telemetry resolution stage.

        Returns:
            Any: Baseline and telemetry resolver artifact.
        """
        try:
            logging.info("Phase 1/5 - Baseline & Telemetry Resolution started.")

            config = BaselineAndTelemetryResolverConfig(self.pipeline_config)
            artifact = BaselineAndTelemetryResolver(config).run()

            self._log_artifact(artifact)

            logging.info("Phase 1/5 - Baseline & Telemetry Resolution completed successfully.")
            return artifact

        except Exception as exc:
            logging.exception("Baseline & Telemetry Resolution stage failed.")
            raise CustomException(exc, sys) from exc

    def _run_statistical_drift_calculator(
        self,
        resolver_artifact: Any,
    ) -> Any:
        """
        Execute the statistical drift calculation stage.

        Args:
            resolver_artifact: Output from the baseline & telemetry resolution stage.

        Returns:
            Any: Statistical drift calculator artifact.
        """
        try:
            logging.info("Phase 2/5 - Statistical Drift Calculation started.")

            config = StatisticalDriftCalculatorConfig(self.pipeline_config)
            artifact = StatisticalDriftCalculator(
                config=config, 
                resolver_artifact=resolver_artifact
            ).run()

            self._log_artifact(artifact)

            logging.info(
                "Phase 2/5 - Statistical Drift Calculation completed successfully."
            )
            return artifact

        except Exception as exc:
            logging.exception("Statistical Drift Calculation stage failed.")
            raise CustomException(exc, sys) from exc

    def _run_performance_evaluator(
        self,
        resolver_artifact: Any,
    ) -> Any:
        """
        Execute the performance evaluation stage.

        Args:
            resolver_artifact: Output from the baseline & telemetry resolution stage.

        Returns:
            Any: Performance evaluator artifact.
        """
        try:
            logging.info("Phase 3/5 - Performance Evaluation started.")

            config = PerformanceEvaluatorConfig(self.pipeline_config)
            artifact = PerformanceEvaluator(
                config=config, 
                resolver_artifact=resolver_artifact
            ).run()

            self._log_artifact(artifact)

            logging.info(
                "Phase 3/5 - Performance Evaluation completed successfully."
            )
            return artifact

        except Exception as exc:
            logging.exception("Performance Evaluation stage failed.")
            raise CustomException(exc, sys) from exc

    def _run_rule_engine(
        self,
        resolver_artifact: Any,
        drift_artifact: Any,
        performance_artifact: Any,
    ) -> Any:
        """
        Execute the rule engine stage to determine the retraining decision.

        Args:
            resolver_artifact: Output from the resolution stage.
            drift_artifact: Output from the drift calculation stage.
            performance_artifact: Output from the performance evaluation stage.

        Returns:
            Any: Rule engine artifact containing the deterministic trigger.
        """
        try:
            logging.info("Phase 4/5 - Rule Engine started.")

            config = RuleEngineConfig(self.pipeline_config)
            artifact = RuleEngine(
                config=config,
                resolver_artifact=resolver_artifact,
                drift_artifact=drift_artifact,
                performance_artifact=performance_artifact,
            ).run()

            self._log_artifact(artifact)

            logging.info(
                "Phase 4/5 - Rule Engine completed successfully."
            )
            return artifact

        except Exception as exc:
            logging.exception("Rule Engine stage failed.")
            raise CustomException(exc, sys) from exc

    def _sync_artifacts_to_s3(self) -> None:
        """
        Synchronize generated monitoring artifacts to AWS S3 for auditability.
        """
        try:
            logging.info("Phase 5/5 - Artifact Synchronization started.")

            s3_path = (
                f"s3://{constants.S3_BUCKET_NAME}/"
                f"{constants.ARTIFACT_DIR_NAME}/"
                f"{constants.MONITORING_PIPELINE_ROOT_DIR_NAME}/"
                f"{self.pipeline_config.run_id}"
            )

            S3Sync().sync_folder_to_s3(
                folder=self.pipeline_config.root_dir,
                aws_bucket_url=s3_path,
            )

            logging.info(
                "Phase 5/5 - Artifact Synchronization completed successfully."
            )

        except Exception as exc:
            logging.exception("Artifact Synchronization stage failed.")
            raise CustomException(exc, sys) from exc

    def run(self) -> None:
        """
        Execute the complete monitoring pipeline workflow.
        """
        try:
            logging.info("=" * 80)
            logging.info("MONITORING PIPELINE EXECUTION STARTED")
            logging.info("=" * 80)

            resolver_artifact = self._run_baseline_and_telemetry_resolver()

            drift_artifact = self._run_statistical_drift_calculator(
                resolver_artifact=resolver_artifact
            )

            performance_artifact = self._run_performance_evaluator(
                resolver_artifact=resolver_artifact
            )

            rule_engine_artifact = self._run_rule_engine(
                resolver_artifact=resolver_artifact,
                drift_artifact=drift_artifact,
                performance_artifact=performance_artifact,
            )

            self._sync_artifacts_to_s3()

            logging.info("=" * 80)
            if rule_engine_artifact.need_update:
                logging.warning("RETRAINING TRIGGERED: System requires Continual Learning execution.")
            else:
                logging.info("SYSTEM HEALTHY: No retraining required.")
            logging.info("MONITORING PIPELINE EXECUTION COMPLETED")
            logging.info("=" * 80)

        except Exception as exc:
            logging.exception(
                "Critical failure during monitoring pipeline execution."
            )
            raise CustomException(exc, sys) from exc


def main() -> int:
    """
    Application entry point for the Monitoring Pipeline.

    Returns:
        int: Process exit code.
    """
    try:
        pipeline = MonitoringPipeline()
        pipeline.run()
        return 0

    except Exception:
        logging.critical(
            "Monitoring pipeline execution terminated unexpectedly.",
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())