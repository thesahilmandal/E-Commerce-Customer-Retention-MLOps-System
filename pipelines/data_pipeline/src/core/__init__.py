"""
Core Application Layer for the Continual Learning Data Pipeline.

This package contains the orchestration logic specific to the Data Pipeline's execution.
It strictly adheres to the Config-Driven Directed Acyclic Graph (DAG) pattern, isolating 
pipeline-specific state, configuration parsing, and dependency injection from the global 
ML Platform infrastructure (which resides in the `shared_core` namespace).

Responsibilities:
- State Management: Maintain the pipeline execution context (run_id, temporal bounds) 
  for a single execution run via the PipelineContext.
- Configuration Management: Parse, validate, and expose configuration parameters 
  from the centralized YAML configuration file.
- Dependency Injection: Bind global infrastructure clients (e.g., DuckDB httpfs engines, 
  boto3 S3 wrappers) to the local execution context to enable stateless component execution.

Note: To strictly adhere to proper dependency sequencing and prevent cyclic imports 
during initialization, internal modules (e.g., `config_parser`, `context`) are designed 
to be imported explicitly by the pipeline runner and execution components.
"""

__version__ = "2.0.0"
__author__ = "ML Platform Engineering"