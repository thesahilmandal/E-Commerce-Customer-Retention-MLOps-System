import json

import pytest
from unittest.mock import MagicMock, patch, mock_open

from pipelines.inference_pipeline.src.components.feature_matrix_builder import (
    FeatureMatrixBuilder,
)
from shared_core.exceptions.custom_exception import CustomException


@patch(
    "pipelines.inference_pipeline.src.components.feature_matrix_builder."
    "FeatureMatrixBuilderConfig.from_context"
)
def test_feature_matrix_builder_initialization_failure(
    mock_from_context: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_from_context.side_effect = Exception(
        "Config initialization failed"
    )

    with pytest.raises(CustomException) as exc_info:
        FeatureMatrixBuilder(context=mock_pipeline_context)

    assert "Config initialization failed" in str(exc_info.value)


@patch.object(FeatureMatrixBuilder, "_generate_metadata")
@patch.object(FeatureMatrixBuilder, "_extract_schema")
@patch.object(FeatureMatrixBuilder, "_generate_feature_matrix")
def test_run_success(
    mock_gen_fm: MagicMock,
    mock_extract_schema: MagicMock,
    mock_gen_metadata: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)
    artifact = builder.run()

    assert (
        artifact.feature_matrix_file_path
        == builder.config.feature_matrix_file_path
    )
    assert (
        artifact.schema_file_path
        == builder.config.schema_file_path
    )
    assert (
        artifact.metadata_file_path
        == builder.config.metadata_file_path
    )
    assert (
        artifact.snapshot_date
        == builder.config.snapshot_date
    )

    mock_gen_fm.assert_called_once()
    mock_extract_schema.assert_called_once()
    mock_gen_metadata.assert_called_once()


@patch.object(FeatureMatrixBuilder, "_generate_feature_matrix")
def test_run_failure(
    mock_gen_fm: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_gen_fm.side_effect = Exception("Feature generation failed")

    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        builder.run()

    assert "Feature generation failed" in str(exc_info.value)


@patch(
    "pipelines.inference_pipeline.src.components.feature_matrix_builder."
    "SharedFeatureGenerator"
)
def test_generate_feature_matrix_success(
    mock_shared_feature_gen_class: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    mock_generator_instance = mock_shared_feature_gen_class.return_value
    mock_generator_instance.get_feature_query.return_value = (
        "SELECT * FROM valid_orders"
    )

    builder._generate_feature_matrix()

    mock_shared_feature_gen_class.assert_called_once_with(
        bronze_base_uri=builder.config.s3_data_lake_uri
    )

    mock_generator_instance.get_feature_query.assert_called_once_with(
        start_date="2015-01-01",
        end_date=builder.config.snapshot_date,
    )

    expected_copy_query = (
        "COPY (SELECT * FROM valid_orders) "
        f"TO '{builder.config.feature_matrix_file_path}' "
        "(FORMAT PARQUET);"
    )

    mock_pipeline_context.duckdb_con.execute.assert_called_once_with(
        expected_copy_query
    )


def test_generate_feature_matrix_duckdb_failure(
    mock_pipeline_context: MagicMock,
) -> None:
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    mock_pipeline_context.duckdb_con.execute.side_effect = Exception(
        "DuckDB query execution failed"
    )

    with pytest.raises(CustomException) as exc_info:
        builder._generate_feature_matrix()

    assert "DuckDB query execution failed" in str(exc_info.value)


def test_extract_schema_success(
    mock_pipeline_context: MagicMock,
) -> None:
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    mock_pipeline_context.duckdb_con.execute.return_value.fetchall.return_value = [
        (
            "customer_unique_id",
            "VARCHAR",
            "YES",
            None,
            None,
            None,
        ),
        (
            "recency_days",
            "BIGINT",
            "YES",
            None,
            None,
            None,
        ),
        (
            "payment_value",
            "DOUBLE",
            "YES",
            None,
            None,
            None,
        ),
    ]

    m_open = mock_open()

    with patch("builtins.open", m_open):
        builder._extract_schema()

    expected_query = (
        "DESCRIBE SELECT * FROM "
        f"'{builder.config.feature_matrix_file_path}'"
    )

    mock_pipeline_context.duckdb_con.execute.assert_called_once_with(
        expected_query
    )

    handle = m_open()
    written_data = "".join(
        call.args[0]
        for call in handle.write.call_args_list
    )

    parsed_schema = json.loads(written_data)

    assert len(parsed_schema) == 3

    assert parsed_schema[0] == {
        "name": "customer_unique_id",
        "physical_type": "VARCHAR",
    }

    assert parsed_schema[1] == {
        "name": "recency_days",
        "physical_type": "BIGINT",
    }

    assert parsed_schema[2] == {
        "name": "payment_value",
        "physical_type": "DOUBLE",
    }


def test_extract_schema_failure(
    mock_pipeline_context: MagicMock,
) -> None:
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    mock_pipeline_context.duckdb_con.execute.side_effect = Exception(
        "Schema extraction failed"
    )

    with pytest.raises(CustomException) as exc_info:
        builder._extract_schema()

    assert "Schema extraction failed" in str(exc_info.value)


@patch(
    "pipelines.inference_pipeline.src.components.feature_matrix_builder."
    "os.path.getsize"
)
def test_generate_metadata_success(
    mock_getsize: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    mock_getsize.return_value = 4096

    (
        mock_pipeline_context
        .duckdb_con
        .execute
        .return_value
        .fetchone
        .return_value
    ) = [2500]

    execution_time = 3.45

    m_open = mock_open()

    with patch("builtins.open", m_open):
        builder._generate_metadata(
            execution_time=execution_time
        )

    expected_count_query = (
        "SELECT COUNT(*) FROM "
        f"'{builder.config.feature_matrix_file_path}'"
    )

    mock_pipeline_context.duckdb_con.execute.assert_called_once_with(
        expected_count_query
    )

    handle = m_open()
    written_data = "".join(
        call.args[0]
        for call in handle.write.call_args_list
    )

    parsed_metadata = json.loads(written_data)

    assert (
        parsed_metadata["pipeline_stage"]
        == "Feature Matrix Builder"
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
        parsed_metadata["data_provenance"][
            "total_customers_scored"
        ]
        == 2500
    )
    assert (
        parsed_metadata["data_provenance"][
            "feature_matrix_size_bytes"
        ]
        == 4096
    )
    assert (
        parsed_metadata["data_provenance"]["snapshot_date"]
        == builder.config.snapshot_date
    )