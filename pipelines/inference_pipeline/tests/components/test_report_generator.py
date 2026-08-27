import json

import pytest
import numpy as np
import pandas as pd

from unittest.mock import MagicMock, patch, mock_open

from pipelines.inference_pipeline.src.components.report_generator import (
    ReportGenerator,
)
from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    ModelLoaderArtifact,
    FeatureMatrixBuilderArtifact,
)


@patch(
    "pipelines.inference_pipeline.src.components.report_generator."
    "ReportGeneratorConfig.from_context"
)
def test_report_generator_initialization_failure(
    mock_from_context: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_from_context.side_effect = Exception(
        "Config initialization failed"
    )

    with pytest.raises(CustomException) as exc_info:
        ReportGenerator(context=mock_pipeline_context)

    assert "Config initialization failed" in str(exc_info.value)


@patch.object(ReportGenerator, "_generate_metadata")
@patch.object(ReportGenerator, "_generate_telemetry_log")
@patch.object(ReportGenerator, "_generate_business_report")
@patch.object(ReportGenerator, "_generate_predictions")
@patch.object(ReportGenerator, "_load_data")
@patch.object(ReportGenerator, "_load_model")
def test_run_success(
    mock_load_model: MagicMock,
    mock_load_data: MagicMock,
    mock_gen_preds: MagicMock,
    mock_gen_biz: MagicMock,
    mock_gen_tel: MagicMock,
    mock_gen_meta: MagicMock,
    mock_pipeline_context: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    mock_model = MagicMock()
    mock_df = pd.DataFrame()
    mock_probs = np.array([0.1, 0.9])

    mock_load_model.return_value = mock_model
    mock_load_data.return_value = mock_df
    mock_gen_preds.return_value = mock_probs

    artifact = generator.run(
        model_loader_artifact=dummy_model_loader_artifact,
        feature_matrix_artifact=dummy_feature_matrix_artifact,
    )

    assert (
        artifact.csv_report_path
        == generator.config.csv_report_path
    )
    assert (
        artifact.telemetry_log_path
        == generator.config.telemetry_log_path
    )
    assert (
        artifact.metadata_file_path
        == generator.config.metadata_file_path
    )

    mock_load_model.assert_called_once_with(
        dummy_model_loader_artifact.model_file_path
    )
    mock_load_data.assert_called_once_with(
        dummy_feature_matrix_artifact.feature_matrix_file_path
    )
    mock_gen_preds.assert_called_once_with(
        model=mock_model,
        df=mock_df,
    )
    mock_gen_biz.assert_called_once_with(
        df=mock_df,
        probabilities=mock_probs,
    )
    mock_gen_tel.assert_called_once_with(
        df=mock_df,
        probabilities=mock_probs,
    )
    mock_gen_meta.assert_called_once()


@patch.object(ReportGenerator, "_load_model")
def test_run_failure(
    mock_load_model: MagicMock,
    mock_pipeline_context: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
) -> None:
    mock_load_model.side_effect = Exception("Model loading failed")

    generator = ReportGenerator(context=mock_pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        generator.run(
            model_loader_artifact=dummy_model_loader_artifact,
            feature_matrix_artifact=dummy_feature_matrix_artifact,
        )

    assert "Model loading failed" in str(exc_info.value)


@patch(
    "pipelines.inference_pipeline.src.components.report_generator."
    "joblib.load"
)
def test_load_model_success(
    mock_joblib_load: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    mock_joblib_load.return_value = "dummy_model"

    model = generator._load_model("/path/to/model.pkl")

    assert model == "dummy_model"
    mock_joblib_load.assert_called_once_with(
        "/path/to/model.pkl"
    )


@patch(
    "pipelines.inference_pipeline.src.components.report_generator."
    "pd.read_parquet"
)
def test_load_data_success(
    mock_read_parquet: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    mock_df = pd.DataFrame({"col": [1, 2]})
    mock_read_parquet.return_value = mock_df

    df = generator._load_data("/path/to/data.parquet")

    pd.testing.assert_frame_equal(df, mock_df)
    mock_read_parquet.assert_called_once_with(
        "/path/to/data.parquet"
    )


@patch(
    "pipelines.inference_pipeline.src.components.report_generator."
    "pd.read_parquet"
)
def test_load_data_empty_raises(
    mock_read_parquet: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    mock_read_parquet.return_value = pd.DataFrame()

    with pytest.raises(CustomException) as exc_info:
        generator._load_data("/path/to/data.parquet")

    assert "empty" in str(exc_info.value).lower()


def test_generate_predictions_success(
    mock_pipeline_context: MagicMock,
    sample_feature_matrix_df: pd.DataFrame,
    mock_predictive_model: MagicMock,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    probabilities = generator._generate_predictions(
        model=mock_predictive_model,
        df=sample_feature_matrix_df,
    )

    expected_probs = np.array([0.15, 0.80, 0.05])

    assert np.array_equal(
        probabilities,
        expected_probs,
    )

    called_df = mock_predictive_model.predict_proba.call_args[0][0]

    assert "customer_unique_id" not in called_df.columns
    assert "snapshot_date" not in called_df.columns
    assert "recency_days" in called_df.columns


def test_generate_predictions_failure(
    mock_pipeline_context: MagicMock,
    sample_feature_matrix_df: pd.DataFrame,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = Exception(
        "Prediction engine failed"
    )

    with pytest.raises(CustomException) as exc_info:
        generator._generate_predictions(
            model=mock_model,
            df=sample_feature_matrix_df,
        )

    assert "Prediction engine failed" in str(exc_info.value)


@patch("pandas.DataFrame.to_csv")
def test_generate_business_report_with_monetary(
    mock_to_csv: MagicMock,
    mock_pipeline_context: MagicMock,
    sample_feature_matrix_df: pd.DataFrame,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    probabilities = np.array([0.85, 0.20, 0.95])

    generator._generate_business_report(
        df=sample_feature_matrix_df,
        probabilities=probabilities,
    )

    mock_to_csv.assert_called_once_with(
        generator.config.csv_report_path,
        index=False,
    )

    metrics = generator._business_impact_metrics

    assert metrics["total_customers_scored"] == 3
    assert metrics["total_churners_flagged"] == 2

    expected_risk = (
        (0.85 * 150.0)
        + (0.20 * 20.0)
        + (0.95 * 1200.0)
    )

    assert metrics["total_revenue_at_risk_flagged"] == pytest.approx(
        expected_risk,
        0.01,
    )


@patch("pandas.DataFrame.to_csv")
def test_generate_business_report_without_monetary_and_index_fallback(
    mock_to_csv: MagicMock,
    mock_pipeline_context: MagicMock,
    sample_feature_matrix_df: pd.DataFrame,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    # Drop identifiers and monetary to trigger fallback paths.
    df = sample_feature_matrix_df.drop(
        columns=[
            "customer_unique_id",
            "monetary_total",
        ]
    )

    probabilities = np.array([0.85, 0.20, 0.95])

    generator._generate_business_report(
        df=df,
        probabilities=probabilities,
    )

    mock_to_csv.assert_called_once_with(
        generator.config.csv_report_path,
        index=False,
    )

    metrics = generator._business_impact_metrics

    assert metrics["total_customers_scored"] == 3
    assert metrics["total_churners_flagged"] == 2
    assert (
        metrics["total_revenue_at_risk_flagged"]
        is None
    )


@patch("pandas.DataFrame.to_parquet")
def test_generate_telemetry_log(
    mock_to_parquet: MagicMock,
    mock_pipeline_context: MagicMock,
    sample_feature_matrix_df: pd.DataFrame,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    probabilities = np.array([0.85, 0.20, 0.95])

    generator._generate_telemetry_log(
        df=sample_feature_matrix_df,
        probabilities=probabilities,
    )

    mock_to_parquet.assert_called_once_with(
        generator.config.telemetry_log_path,
        index=False,
        compression="snappy",
    )

    # Extract the DataFrame passed to to_parquet
    # to verify its structure.
    called_df = (
        mock_to_parquet.call_args[0][0]
        if mock_to_parquet.call_args.args
        else None
    )

    if called_df is None:
        raise ValueError(
            "DataFrame to_parquet mock interception failed structure."
        )


@patch(
    "pipelines.inference_pipeline.src.components.report_generator."
    "pd.DataFrame.to_parquet"
)
def test_generate_telemetry_log_dataframe_method(
    mock_to_parquet: MagicMock,
    mock_pipeline_context: MagicMock,
    sample_feature_matrix_df: pd.DataFrame,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    probabilities = np.array([0.85, 0.20, 0.95])

    generator._generate_telemetry_log(
        df=sample_feature_matrix_df,
        probabilities=probabilities,
    )

    mock_to_parquet.assert_called_once_with(
        generator.config.telemetry_log_path,
        index=False,
        compression="snappy",
    )


def test_generate_metadata(
    mock_pipeline_context: MagicMock,
) -> None:
    generator = ReportGenerator(context=mock_pipeline_context)

    generator._business_impact_metrics = {
        "test_metric": 123
    }

    execution_time = 2.5

    m_open = mock_open()

    with patch("builtins.open", m_open):
        generator._generate_metadata(
            execution_time=execution_time
        )

    m_open.assert_called_once_with(
        generator.config.metadata_file_path,
        "w",
        encoding="utf-8",
    )

    handle = m_open()

    written_data = "".join(
        call.args[0]
        for call in handle.write.call_args_list
    )

    parsed_metadata = json.loads(written_data)

    assert (
        parsed_metadata["pipeline_stage"]
        == "Report Generator"
    )
    assert (
        parsed_metadata["inference_run_id"]
        == mock_pipeline_context.run_id
    )
    assert (
        parsed_metadata["execution_time_seconds"]
        == execution_time
    )
    assert (
        parsed_metadata["business_impact"]
        == {"test_metric": 123}
    )
    assert (
        parsed_metadata["artifacts_generated"][
            "business_report"
        ]
        == generator.config.csv_report_path
    )
    assert (
        parsed_metadata["artifacts_generated"][
            "telemetry_log"
        ]
        == generator.config.telemetry_log_path
    )