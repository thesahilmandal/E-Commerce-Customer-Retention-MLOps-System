"""
Components package for the Inference Pipeline.

This package encapsulates the individual, independent stages of the inference 
workflow. Each component is designed to be highly cohesive, performing a single 
architectural responsibility (e.g., loading models, building features, validating 
contracts, generating predictions, publishing artifacts).

Components consume strictly typed Configuration Entities and produce immutable 
Artifact Entities, ensuring clear data lineage and testability across the pipeline.
"""

from .model_loader import ModelLoader
from .feature_matrix_builder import FeatureMatrixBuilder
from .inference_validator import InferenceValidator
from .report_generator import ReportGenerator
from .report_publisher import ReportPublisher

__all__ = [
    "ModelLoader",
    "FeatureMatrixBuilder",
    "InferenceValidator",
    "ReportGenerator",
    "ReportPublisher",
]