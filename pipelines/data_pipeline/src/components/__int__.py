"""
Execution Components for the Continual Learning Data Pipeline.

This package contains the stateless execution stages (Verbs) of the Data Pipeline.
Each component is designed to perform a highly specific task within the pipeline's 
execution graph, adhering to the Single Responsibility Principle (SRP).

Components do not maintain internal pipeline state or communicate directly with 
one another. Instead, they receive the `PipelineContext` (which acts as a 
Dependency Injection container) and interact purely through shared infrastructure 
clients (e.g., DuckDB, S3Sync) and standard S3 URIs.

Pipeline Stages:
1. Data Discovery: Validates the existence of required Hive partitions in the 
   Bronze Data Lake for the requested temporal window.
2. Data Validation: Executes out-of-core schema and data quality validations 
   directly against S3 via DuckDB httpfs.
3. Feature Materializer: Generates the point-in-time analytical base table (Master Panel) 
   and exports it directly to the S3 Feature Store without local disk spillage.
4. Metadata Registry: Extracts remote file metrics and persists the strict 
   Continual Learning JSON metadata contract.

Note: To prevent cyclic dependencies and maintain strict execution sequencing, 
these components are designed to be explicitly imported and executed by the 
central pipeline runner.
"""

__version__ = "2.0.0"
__author__ = "ML Platform Engineering"