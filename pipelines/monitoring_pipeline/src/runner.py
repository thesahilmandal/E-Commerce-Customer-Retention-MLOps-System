import sys
import argparse
from datetime import datetime
from typing import Optional

from pipelines.monitoring_pipeline.src.core.context import MonitoringPipelineContext
from pipelines.monitoring_pipeline.src.entity.config_entity import (
    BaselineAndTelemetryResolverConfig,
    StatisticalDriftCalculatorConfig,
    PerformanceEvaluatorConfig,
    RuleEngineConfig,
    ArtifactPublisherConfig
)
from pipelines.monitoring_pipeline.src.components.baseline_and_telemetry_resolver import BaselineAndTelemetryResolver
from pipelines.monitoring_pipeline.src.components.statistical_drift_calculator import StatisticalDriftCalculator
from pipelines.monitoring_pipeline.src.components.performance_evaluator import PerformanceEvaluator
from pipelines.monitoring_pipeline.src.components.rule_engine import RuleEngine
from pipelines.monitoring_pipeline.src.components.artifact_publisher import ArtifactPublisher
from shared_core.exceptions.custom_exception import CustomException
from shared_core.logging.custom_logging import logging


class MonitoringPipelineRunner:
    """
    Orchestration layer for the Monitoring Pipeline.

    Responsibilities:
    - Act as the central orchestrator coordinating the execution of all monitoring components.
    - Initialize and manage the global pipeline context (`MonitoringPipelineContext`), ensuring
      safe setup and teardown of shared resources (e.g., DuckDB connections).
    - Maintain strict, reproducible execution order and artifact lineage.
    - Integrate seamlessly with the `master_orchestrator.py` via `run_id` and `execution_date` injection.
    """

    def __init__(self, run_id: Optional[str] = None, execution_date: Optional[str] = None) -> None:
        """
        Initializes the Monitoring Pipeline Runner.

        Args:
            run_id (Optional[str]): Orchestrator-injected run identifier for traceability.
            execution_date (Optional[str]): Orchestrator-injected logical date (YYYY-MM-DD) for idempotency.
        """
        self.run_id = run_id
        self.execution_date = execution_date

    def run(self) -> None:
        """
        Executes the end-to-end monitoring pipeline workflow.
        """
        try:
            logging.info("=" * 80)
            logging.info("MONITORING PIPELINE EXECUTION STARTED")
            logging.info("=" * 80)

            # Enter the context manager to safely provision and tear down shared resources
            with MonitoringPipelineContext(
                run_id=self.run_id,
                execution_date=self.execution_date
            ) as context:

                # ---------------------------------------------------------
                # Phase 1: Baseline & Telemetry Resolution
                # ---------------------------------------------------------
                logging.info("--- Phase 1/5: Baseline & Telemetry Resolution ---")
                resolver_config = BaselineAndTelemetryResolverConfig.get_config(context)
                resolver = BaselineAndTelemetryResolver(
                    config=resolver_config,
                    context=context
                )
                resolver_artifact = resolver.run()

                # ---------------------------------------------------------
                # Phase 2: Statistical Drift Calculation
                # ---------------------------------------------------------
                logging.info("--- Phase 2/5: Statistical Drift Calculation ---")
                drift_config = StatisticalDriftCalculatorConfig.get_config(context)
                drift_calculator = StatisticalDriftCalculator(
                    config=drift_config,
                    context=context,
                    resolver_artifact=resolver_artifact
                )
                drift_artifact = drift_calculator.run()

                # ---------------------------------------------------------
                # Phase 3: Performance Evaluation
                # ---------------------------------------------------------
                logging.info("--- Phase 3/5: Performance Evaluation ---")
                performance_config = PerformanceEvaluatorConfig.get_config(context)
                performance_evaluator = PerformanceEvaluator(
                    config=performance_config,
                    context=context,
                    resolver_artifact=resolver_artifact
                )
                performance_artifact = performance_evaluator.run()

                # ---------------------------------------------------------
                # Phase 4: Rule Engine (Decision Logic)
                # ---------------------------------------------------------
                logging.info("--- Phase 4/5: Rule Engine ---")
                rule_engine_config = RuleEngineConfig.get_config(context)
                rule_engine = RuleEngine(
                    config=rule_engine_config,
                    context=context,
                    resolver_artifact=resolver_artifact,
                    drift_artifact=drift_artifact,
                    performance_artifact=performance_artifact
                )
                rule_engine_artifact = rule_engine.run()

                # ---------------------------------------------------------
                # Phase 5: Artifact Publication
                # ---------------------------------------------------------
                logging.info("--- Phase 5/5: Artifact Publication ---")
                publisher_config = ArtifactPublisherConfig.get_config(context)
                publisher = ArtifactPublisher(
                    config=publisher_config,
                    context=context,
                    resolver_artifact=resolver_artifact,
                    drift_artifact=drift_artifact,
                    performance_artifact=performance_artifact,
                    rule_engine_artifact=rule_engine_artifact
                )
                publisher_artifact = publisher.run()

                # ---------------------------------------------------------
                # Final Summary
                # ---------------------------------------------------------
                logging.info("=" * 80)
                if rule_engine_artifact.need_update:
                    logging.warning("MONITORING VERDICT: RETRAINING TRIGGERED")
                else:
                    logging.info("MONITORING VERDICT: SYSTEM HEALTHY")

                logging.info("Production Artifacts Published:")
                logging.info("  Audit Report: %s", publisher_artifact.s3_audit_report_uri)
                logging.info("  Action Token: %s", publisher_artifact.s3_action_token_uri)
                logging.info("=" * 80)
                logging.info("MONITORING PIPELINE EXECUTION COMPLETED")
                logging.info("=" * 80)

        except Exception as e:
            logging.exception("Critical failure during Monitoring Pipeline execution.")
            raise CustomException(e, sys) from e


def main() -> None:
    """
    Main CLI entry point for the Monitoring Pipeline.
    Parses arguments and initiates the pipeline execution.
    """
    parser = argparse.ArgumentParser(
        description="Orchestrator for the ML Monitoring Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Unique identifier for this pipeline execution to ensure traceable lineage."
    )

    parser.add_argument(
        "--execution-date",
        type=str,
        required=True,
        help="Logical execution date in YYYY-MM-DD format for idempotency."
    )

    args = parser.parse_args()

    run_id = args.run_id.strip()
    execution_date = args.execution_date.strip()

    if not run_id:
        logging.error("Validation Error: '--run-id' cannot be empty or whitespace.")
        sys.exit(2)

    try:
        datetime.strptime(execution_date, "%Y-%m-%d")
    except ValueError:
        logging.error("Validation Error: 'f' must be a valid date in YYYY-MM-DD format.")
        sys.exit(2)

    try:
        MonitoringPipelineRunner(
            run_id=run_id,
            execution_date=execution_date
        ).run()
        sys.exit(0)
    except Exception:
        # Exception details are already logged by the MonitoringPipelineRunner
        sys.exit(1)


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     try:
#         MonitoringPipelineRunner(run_id="testing_01", execution_date="2026-08-17").run()
#     except Exception as e:
#         raise CustomException(e, sys)