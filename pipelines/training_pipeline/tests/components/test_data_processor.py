import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from pipelines.training_pipeline.src.components.data_processor import DataProcessor
from pipelines.training_pipeline.src.entity.config_entity import DataProcessorConfig
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def dp_config(
    mock_pipeline_context: MagicMock,
) -> DataProcessorConfig:
    return DataProcessorConfig.from_context(mock_pipeline_context)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_is_churn": [1, 0, 1, 0, 1],
            "customer_unique_id": ["c1", "c2", "c3", "c4", "c5"],
            "snapshot_date": ["2023-01-01"] * 5,
            "ingested_at_utc": ["2023"] * 5,
            "target_180d_ltv": [10.0, 20.0, 30.0, 40.0, 50.0],
            "numerical_feat": [1.1, 2.2, 3.3, 4.4, 5.5],
            "categorical_feat": ["A", "B", "A", "C", "B"],
        }
    )


@patch(
    "pipelines.training_pipeline.src.components.data_processor.pd.read_parquet"
)
@patch(
    "pipelines.training_pipeline.src.components.data_processor.joblib.dump"
)
@patch(
    "pipelines.training_pipeline.src.components.data_processor.write_json_file"
)
@patch(
    "pipelines.training_pipeline.src.components.data_processor.os.remove"
)
@patch(
    "pipelines.training_pipeline.src.components.data_processor.os.path.exists",
    return_value=True,
)
@patch("pandas.DataFrame.to_parquet")
def test_run_success(
    mock_to_parquet: MagicMock,
    mock_exists: MagicMock,
    mock_remove: MagicMock,
    mock_write_json: MagicMock,
    mock_joblib: MagicMock,
    mock_read_parquet: MagicMock,
    dp_config: DataProcessorConfig,
    mock_pipeline_context: MagicMock,
    sample_df: pd.DataFrame,
) -> None:
    # Ensure reading parquet returns our mocked DataFrame
    mock_read_parquet.return_value = sample_df.copy()

    processor = DataProcessor(
        config=dp_config,
        context=mock_pipeline_context,
    )

    # Mock the DuckDB out-of-core splitting to avoid actual DB and S3 interaction
    with patch.object(processor, "_split_data_out_of_core"):
        artifact = processor.run()

    # Validate Artifact output matches configuration paths
    assert artifact.preprocessor_file_path == dp_config.preprocessor_file_path
    assert artifact.schema_file_path == dp_config.schema_file_path
    assert artifact.metadata_file_path == dp_config.metadata_file_path
    assert artifact.x_train_file_path == dp_config.x_train_file_path

    # Schema blueprint and execution metadata should be generated
    assert mock_write_json.call_count == 2

    # DataFrames saved (X_train, y_train, X_val, y_val, X_test, y_test)
    assert mock_to_parquet.call_count == 6

    # CategoricalSchemaEnforcer serialized
    mock_joblib.assert_called_once()

    # Cleanup of 3 temporary split files
    assert mock_remove.call_count == 3


def test_split_data_out_of_core_queries(
    dp_config: DataProcessorConfig,
    mock_pipeline_context: MagicMock,
) -> None:
    processor = DataProcessor(
        config=dp_config,
        context=mock_pipeline_context,
    )

    processor._split_data_out_of_core()

    execute_mock = mock_pipeline_context.db_conn.execute
    assert execute_mock.call_count == 5

    calls = execute_mock.call_args_list

    # 1. Random seed
    assert "SELECT setseed(0.042);" in calls[0][0][0]

    # 2. Source table creation
    assert "CREATE OR REPLACE TEMP TABLE source_table" in calls[1][0][0]
    assert dp_config.training_dataset_s3_uri_path in calls[1][0][0]

    # 3. Train Split (1 - 0.15 - 0.15 = 0.70)
    assert "<= 0.7" in calls[2][0][0]

    # 4. Validation Split (> 0.70 AND <= 0.85)
    assert "> 0.7" in calls[3][0][0]
    assert "<= 0.85" in calls[3][0][0]

    # 5. Test Split (> 0.85)
    assert "> 0.85" in calls[4][0][0]


def test_isolate_features_and_target_success(
    dp_config: DataProcessorConfig,
    mock_pipeline_context: MagicMock,
    sample_df: pd.DataFrame,
) -> None:
    processor = DataProcessor(
        config=dp_config,
        context=mock_pipeline_context,
    )

    X, y = processor._isolate_features_and_target(sample_df)

    assert y.name == "target_is_churn"
    assert len(y) == 5

    # System columns and target must be removed from the feature matrix
    for col in dp_config.system_columns_to_drop + ["target_is_churn"]:
        assert col not in X.columns

    # Valid predictive features remain
    assert "numerical_feat" in X.columns
    assert "categorical_feat" in X.columns


def test_isolate_features_and_target_missing_target(
    dp_config: DataProcessorConfig,
    mock_pipeline_context: MagicMock,
    sample_df: pd.DataFrame,
) -> None:
    processor = DataProcessor(
        config=dp_config,
        context=mock_pipeline_context,
    )

    bad_df = sample_df.drop(columns=["target_is_churn"])

    with pytest.raises(CustomException) as exc_info:
        processor._isolate_features_and_target(bad_df)

    assert "not found in dataset" in str(exc_info.value)
    assert "target_is_churn" in str(exc_info.value)


@patch(
    "pipelines.training_pipeline.src.components.data_processor.pd.read_parquet"
)
def test_load_local_parquet_failure(
    mock_read_parquet: MagicMock,
    dp_config: DataProcessorConfig,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_read_parquet.side_effect = Exception("PyArrow stream error")

    processor = DataProcessor(
        config=dp_config,
        context=mock_pipeline_context,
    )

    with pytest.raises(CustomException) as exc_info:
        processor._load_local_parquet("fake_file.parquet")

    assert "PyArrow stream error" in str(exc_info.value)


@patch(
    "pipelines.training_pipeline.src.components.data_processor.write_json_file"
)
def test_generate_schema_blueprint(
    mock_write_json: MagicMock,
    dp_config: DataProcessorConfig,
    mock_pipeline_context: MagicMock,
) -> None:
    processor = DataProcessor(
        config=dp_config,
        context=mock_pipeline_context,
    )

    enforcer_mock = MagicMock()
    enforcer_mock.numerical_features = ["num_feat"]
    enforcer_mock.categories_ = {"cat_feat": ["A", "B"]}

    processor._generate_schema_blueprint(enforcer_mock)

    mock_write_json.assert_called_once()

    args = mock_write_json.call_args[0]
    filepath, payload = args[0], args[1]

    assert filepath == dp_config.schema_file_path
    assert payload["features"]["numerical"] == ["num_feat"]
    assert payload["features"]["categorical"] == {
        "cat_feat": ["A", "B"]
    }
    assert payload["metadata"]["run_id"] == "test_run_12345"
    assert "generated_at_utc" in payload["metadata"]


def test_run_top_level_exception_handling(
    dp_config: DataProcessorConfig,
    mock_pipeline_context: MagicMock,
) -> None:
    processor = DataProcessor(
        config=dp_config,
        context=mock_pipeline_context,
    )

    # Force a failure at the first step
    with patch.object(
        processor,
        "_split_data_out_of_core",
        side_effect=Exception("Database crash"),
    ):
        with pytest.raises(CustomException) as exc_info:
            processor.run()

        assert "Database crash" in str(exc_info.value)