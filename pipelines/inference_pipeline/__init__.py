"""
Inference Pipeline Package.

This package implements the scheduled batch inference workflow. It resolves the active
Champion model from the Model Registry, materializes out-of-core feature matrices via
DuckDB, strictly enforces data contracts, and securely publishes partitioned business
reports and operational telemetry to AWS S3.
"""