from dataclasses import dataclass


@dataclass(frozen=True)
class DataProcessorArtifact:
    """
    Artifact containing paths for the split, transformed Feature Matrices (X) 
    and Target Vectors (y), along with the serialized categorical schema enforcer 
    and the dynamically generated JSON schema blueprint.
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
            "\nDataProcessorArtifact(\n"
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
class ModelTrainerArtifact:
    """
    Artifact containing the path to the definitive Scikit-Learn Mega-Pipeline (model.pkl),
    the SHAP global summary plots/JSONs, MLflow run metadata, and the JSON contracts 
    required for downstream monitoring (reference feature distributions).
    """
    model_file_path: str
    shap_summary_file_path: str
    shap_feature_importance_file_path: str
    reference_feature_distributions_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nModelTrainerArtifact(\n"
            f"  model_file_path = {self.model_file_path}\n"
            f"  shap_summary_file_path = {self.shap_summary_file_path}\n"
            f"  shap_feature_importance_file_path = {self.shap_feature_importance_file_path}\n"
            f"  reference_feature_distributions_file_path = {self.reference_feature_distributions_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ModelEvaluatorArtifact:
    """
    Artifact containing the evaluation report (Champion vs. Challenger EROI duel),
    the critical deployment gating boolean, and the JSON contract for baseline 
    performance metrics required by the downstream monitoring pipeline.
    """
    approval_status: bool
    report_file_path: str
    baseline_performance_metrics_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nModelEvaluatorArtifact(\n"
            f"  approval_status = {self.approval_status}\n"
            f"  report_file_path = {self.report_file_path}\n"
            f"  baseline_performance_metrics_file_path = {self.baseline_performance_metrics_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ModelRegistryArtifact:
    """
    Artifact containing the deployment status, the remote S3 URI of the newly 
    vaulted immutable model bundle, and the unified deployment metadata.
    """
    deployment_status: bool
    s3_model_uri: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nModelRegistryArtifact(\n"
            f"  deployment_status = {self.deployment_status}\n"
            f"  s3_model_uri = {self.s3_model_uri}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )