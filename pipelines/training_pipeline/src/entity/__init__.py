"""
Entity package for the Training Pipeline.

Contains lightweight Dataclasses representing strict data contracts:
- Configuration Entities: Type-safe configurations for pipeline components.
- Artifact Entities: Immutable data passed between pipeline stages.
"""

from pipelines.training_pipeline.src.entity.artifact_entity import (
    DataProcessorArtifact,
    ModelEvaluatorArtifact,
    ModelRegistryArtifact,
    ModelTrainerArtifact,
)
from pipelines.training_pipeline.src.entity.config_entity import (
    DataProcessorConfig,
    ModelEvaluatorConfig,
    ModelRegistryConfig,
    ModelTrainerConfig,
)

__all__ = [
    "DataProcessorArtifact",
    "ModelEvaluatorArtifact",
    "ModelRegistryArtifact",
    "ModelTrainerArtifact",
    "DataProcessorConfig",
    "ModelEvaluatorConfig",
    "ModelRegistryConfig",
    "ModelTrainerConfig",
]