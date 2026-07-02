from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Artifact containing paths for the Out-Of-Time (OOT) temporal data splits.
    """
    train_data_path: str
    val_data_path: str
    test_data_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nTrainingPipelineDataIngestionArtifact(\n"
            f"  train_data_path = {self.train_data_path}\n"
            f"  val_data_path = {self.val_data_path}\n"
            f"  test_data_path = {self.test_data_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class FeatureTransformationArtifact:
    """
    Artifact containing paths for the serialized preprocessor, schema metadata,
    and the fully transformed Feature Matrices (X) and Target Vectors (y) ready for model training.
    """
    preprocessor_file_path: str
    schema_file_path: str
    metadata_file_path: str
    x_train_file_path: str
    y_train_file_path: str
    x_val_file_path: str
    y_val_file_path: str
    x_test_file_path: str
    y_test_file_path: str

    def __str__(self) -> str:
        return (
            "\nTrainingPipelineDataTransformationArtifact(\n"
            f"  preprocessor_file_path = {self.preprocessor_file_path}\n"
            f"  schema_file_path = {self.schema_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  x_train_file_path = {self.x_train_file_path}\n"
            f"  y_train_file_path = {self.y_train_file_path}\n"
            f"  x_val_file_path = {self.x_val_file_path}\n"
            f"  y_val_file_path = {self.y_val_file_path}\n"
            f"  x_test_file_path = {self.x_test_file_path}\n"
            f"  y_test_file_path = {self.y_test_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ModelTrainingArtifact:
    """
    Artifact containing the path to the definitive Scikit-Learn Mega-Pipeline (model.pkl),
    the SHAP global summary plot, MLflow run metadata, and the JSON contracts required 
    for downstream monitoring (feature distributions and SHAP feature importance).
    """
    model_file_path: str
    shap_summary_file_path: str
    metadata_file_path: str
    reference_feature_distributions_file_path: str
    shap_feature_importance_summary_file_path: str

    def __str__(self) -> str:
        return (
            "\nTrainingPipelineModelTrainerArtifact(\n"
            f"  model_file_path = {self.model_file_path}\n"
            f"  shap_summary_file_path = {self.shap_summary_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  reference_feature_distributions_file_path = {self.reference_feature_distributions_file_path}\n"
            f"  shap_feature_importance_summary_file_path = {self.shap_feature_importance_summary_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    """
    Artifact containing the evaluation report (Champion vs. Challenger metrics),
    component metadata, the critical deployment gating boolean, and the JSON contract
    for baseline performance metrics required by the downstream monitoring pipeline.
    """
    report_file_path: str
    metadata_file_path: str
    approval_status: bool
    baseline_performance_metrics_file_path: str

    def __str__(self) -> str:
        return (
            "\nTrainingPipelineModelEvaluationArtifact(\n"
            f"  report_file_path = {self.report_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  approval_status = {self.approval_status}\n"
            f"  baseline_performance_metrics_file_path = {self.baseline_performance_metrics_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ModelRegistrationArtifact:
    """
    Artifact containing the remote S3 URI of the newly registered model bundle,
    local component metadata, and the final state of the deployment action.
    """
    s3_model_uri: str
    metadata_file_path: str
    deployment_status: bool

    def __str__(self) -> str:
        return (
            "\nTrainingPipelineModelRegistryArtifact(\n"
            f"  s3_model_uri = {self.s3_model_uri}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  deployment_status = {self.deployment_status}\n"
            ")"
        )