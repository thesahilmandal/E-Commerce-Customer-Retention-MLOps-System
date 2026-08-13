import os
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from pipelines.training_pipeline.src.components.data_processor import (
    CategoricalSchemaEnforcer,
    DataProcessor,
)
from pipelines.training_pipeline.src.entity.config_entity import DataProcessorConfig
from shared_core.exceptions.custom_exception import CustomException


def test_categorical_schema_enforcer_fit_transform():
    df = pd.DataFrame(
        {
            "num1": [1, 2, np.nan, 4],
            "num2": ["10", "20", "30", "40"],
            "cat1": ["A", "B", "A", np.nan],
            "cat2": ["X", "Y", "X", "Y"],
        }
    )

    enforcer = CategoricalSchemaEnforcer(
        categorical_features=["cat1", "cat2"],
        numerical_features=["num1", "num2"],
    )

    enforcer.fit(df)

    assert set(enforcer.categories_["cat1"]) == {"A", "B"}
    assert set(enforcer.categories_["cat2"]) == {"X", "Y"}

    df_transformed = enforcer.transform(df)

    assert pd.api.types.is_numeric_dtype(df_transformed["num1"])
    assert pd.api.types.is_numeric_dtype(df_transformed["num2"])

    assert isinstance(df_transformed["cat1"].dtype, pd.CategoricalDtype)
    assert set(df_transformed["cat1"].cat.categories) == {"A", "B"}

    expected_cols = ["num1", "num2", "cat1", "cat2"]
    assert list(df_transformed.columns) == expected_cols


def test_categorical_schema_enforcer_unseen_categories_and_bad_numeric():
    df_train = pd.DataFrame(
        {
            "num": [1, 2],
            "cat": ["A", "B"],
        }
    )

    enforcer = CategoricalSchemaEnforcer(
        categorical_features=["cat"],
        numerical_features=["num"],
    )
    enforcer.fit(df_train)

    df_test = pd.DataFrame(
        {
            "num": [3, "not_a_number"],
            "cat": ["A", "C"],
            "extra_col": ["drop", "me"],
        }
    )

    df_transformed = enforcer.transform(df_test)

    assert pd.isna(df_transformed.loc[1, "cat"])
    assert pd.isna(df_transformed.loc[1, "num"])
    assert "extra_col" not in df_transformed.columns


def test_categorical_schema_enforcer_missing_columns_during_transform():
    df_train = pd.DataFrame({"num": [1], "cat": ["A"]})
    enforcer = CategoricalSchemaEnforcer(
        categorical_features=["cat"],
        numerical_features=["num"],
    ).fit(df_train)

    df_test = pd.DataFrame({"num": [2]})
    df_transformed = enforcer.transform(df_test)

    assert "cat" in df_transformed.columns
    assert df_transformed["cat"].isna().all()
    assert isinstance(df_transformed["cat"].dtype, pd.CategoricalDtype)


@pytest.fixture
def dp_config(pipeline_context):
    return DataProcessorConfig.from_context(pipeline_context)


def test_data_processor_isolate_features_and_target(
    pipeline_context, dp_config, synthetic_dataframe
):
    processor = DataProcessor(config=dp_config, context=pipeline_context)

    X, y = processor._isolate_features_and_target(synthetic_dataframe)

    assert y.name == dp_config.target_column
    assert len(y) == len(synthetic_dataframe)

    system_cols = dp_config.system_columns_to_drop
    for col in system_cols:
        assert col not in X.columns

    assert dp_config.target_column not in X.columns
    assert "num_feature_1" in X.columns


def test_data_processor_isolate_missing_target(
    pipeline_context, dp_config, synthetic_dataframe
):
    processor = DataProcessor(config=dp_config, context=pipeline_context)

    df_no_target = synthetic_dataframe.drop(columns=[dp_config.target_column])

    with pytest.raises(CustomException) as excinfo:
        processor._isolate_features_and_target(df_no_target)

    assert f"Target column '{dp_config.target_column}' not found" in str(
        excinfo.value
    )


def test_data_processor_split_data_out_of_core(pipeline_context, dp_config):
    processor = DataProcessor(config=dp_config, context=pipeline_context)

    processor._split_data_out_of_core()

    execute_calls = pipeline_context.db_conn.execute.call_args_list
    assert len(execute_calls) >= 4

    queries = [call[0][0] for call in execute_calls]

    assert any("setseed" in q for q in queries)
    assert any(dp_config.training_dataset_s3_uri_path in q for q in queries)

    train_bound = 1.0 - dp_config.val_size - dp_config.test_size
    val_bound = train_bound + dp_config.val_size

    assert any(f"<= {train_bound}" in q for q in queries)
    assert any(
        f"> {train_bound} AND _split_val <= {val_bound}" in q
        for q in queries
    )
    assert any(f"> {val_bound}" in q for q in queries)


def test_data_processor_run_success(
    pipeline_context, dp_config, synthetic_dataframe
):
    processor = DataProcessor(config=dp_config, context=pipeline_context)

    def mock_split():
        train, val, test = np.split(
            synthetic_dataframe.sample(frac=1, random_state=42),
            [
                int(0.6 * len(synthetic_dataframe)),
                int(0.8 * len(synthetic_dataframe)),
            ],
        )

        train.to_parquet(
            os.path.join(
                dp_config.data_processor_dir, "tmp_train.parquet"
            ),
            index=False,
        )
        val.to_parquet(
            os.path.join(
                dp_config.data_processor_dir, "tmp_val.parquet"
            ),
            index=False,
        )
        test.to_parquet(
            os.path.join(
                dp_config.data_processor_dir, "tmp_test.parquet"
            ),
            index=False,
        )

    with patch.object(
        processor, "_split_data_out_of_core", side_effect=mock_split
    ):
        artifact = processor.run()

    assert artifact.preprocessor_file_path == dp_config.preprocessor_file_path
    assert artifact.x_train_file_path == dp_config.x_train_file_path

    assert os.path.exists(artifact.preprocessor_file_path)
    assert os.path.exists(artifact.schema_file_path)
    assert os.path.exists(artifact.metadata_file_path)
    assert os.path.exists(artifact.x_train_file_path)
    assert os.path.exists(artifact.y_test_file_path)

    assert not os.path.exists(
        os.path.join(dp_config.data_processor_dir, "tmp_train.parquet")
    )
    assert not os.path.exists(
        os.path.join(dp_config.data_processor_dir, "tmp_val.parquet")
    )
    assert not os.path.exists(
        os.path.join(dp_config.data_processor_dir, "tmp_test.parquet")
    )

    with open(artifact.schema_file_path, "r") as f:
        schema = json.load(f)

    assert "numerical" in schema["features"]
    assert "categorical" in schema["features"]
    assert "cat_feature_1" in schema["features"]["categorical"]


def test_data_processor_run_failure_handling(
    pipeline_context, dp_config
):
    processor = DataProcessor(config=dp_config, context=pipeline_context)

    with patch.object(
        processor,
        "_split_data_out_of_core",
        side_effect=Exception("Simulated DuckDB OOM"),
    ):
        with pytest.raises(CustomException) as excinfo:
            processor.run()

        assert "Simulated DuckDB OOM" in str(excinfo.value)