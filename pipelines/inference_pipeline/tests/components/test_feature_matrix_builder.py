import os
import json
import pytest
from unittest.mock import patch, MagicMock

from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.components.feature_matrix_builder import FeatureMatrixBuilder
from pipelines.inference_pipeline.src.entity.artifact_entity import FeatureMatrixBuilderArtifact


def test_feature_matrix_builder_initialization_success(mock_pipeline_context):
    """
    Tests successful initialization of the FeatureMatrixBuilder component.
    """
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    assert builder.context is mock_pipeline_context
    assert builder.config is not None
    assert "02_input_feature_matrix_builder" in builder.config.builder_root_dir
    assert builder.config.s3_data_lake_uri == "s3://test-data-lake/bronze"


def test_feature_matrix_builder_initialization_failure(mock_pipeline_context):
    """
    Tests that FeatureMatrixBuilder raises a CustomException if configuration extraction fails.
    """
    with patch(
        "pipelines.inference_pipeline.src.entity.config_entity.FeatureMatrixBuilderConfig.from_context"
    ) as mock_from_context:
        mock_from_context.side_effect = Exception("Config context mapping error")
        with pytest.raises(CustomException) as exc_info:
            FeatureMatrixBuilder(context=mock_pipeline_context)

        assert "Config context mapping error" in str(exc_info.value)


def test_feature_matrix_builder_run_success(mock_pipeline_context):
    """
    Tests the complete happy path of the Feature Matrix Builder workflow.
    Validates DuckDB out-of-core execution handling, schema extraction, and metadata recording.
    """
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    def execute_side_effect(query):
        mock_result = MagicMock()
        if "COPY" in query.upper():
            # Create a dummy Parquet file to satisfy os.path.getsize in metadata generation
            with open(
                builder.config.feature_matrix_file_path, "w", encoding="utf-8"
            ) as f:
                f.write("mock_parquet_data")
            return mock_result
        elif "DESCRIBE" in query.upper():
            mock_result.fetchall.return_value = [
                ("customer_unique_id", "VARCHAR", "YES", "PRI", None, ""),
                ("snapshot_date", "VARCHAR", "YES", "", None, ""),
                ("f1", "DOUBLE", "YES", "", None, ""),
                ("c1", "VARCHAR", "YES", "", None, "")
            ]
            return mock_result
        elif "SELECT COUNT" in query.upper():
            mock_result.fetchone.return_value = (5000,)
            return mock_result
        raise ValueError(
            f"Unexpected query received by mock DuckDB engine: {query}"
        )

    mock_pipeline_context.duckdb_con.execute.side_effect = execute_side_effect

    with patch(
        "pipelines.inference_pipeline.src.components.feature_matrix_builder.SharedFeatureGenerator"
    ) as mock_sfg:
        mock_sfg_instance = MagicMock()
        mock_sfg.return_value = mock_sfg_instance
        mock_sfg_instance.get_feature_query.return_value = (
            "SELECT * FROM bronze.customers"
        )

        artifact = builder.run()

        # Validate output artifact
        assert isinstance(artifact, FeatureMatrixBuilderArtifact)
        assert os.path.exists(artifact.feature_matrix_file_path)
        assert os.path.exists(artifact.schema_file_path)
        assert os.path.exists(artifact.metadata_file_path)

        # Validate SharedFeatureGenerator was initialized correctly to enforce zero-skew
        mock_sfg.assert_called_once_with(
            bronze_base_uri="s3://test-data-lake/bronze"
        )
        mock_sfg_instance.get_feature_query.assert_called_once_with(
            start_date="2015-01-01",
            end_date=builder.config.snapshot_date
        )

        # Validate schema extraction contents
        with open(artifact.schema_file_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
            assert len(schema) == 4
            assert schema[0]["name"] == "customer_unique_id"
            assert schema[0]["physical_type"] == "VARCHAR"
            assert schema[2]["name"] == "f1"
            assert schema[2]["physical_type"] == "DOUBLE"

        # Validate generated metadata
        with open(artifact.metadata_file_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            assert metadata["pipeline_stage"] == "Feature Matrix Builder"
            assert metadata["data_provenance"]["total_customers_scored"] == 5000
            assert metadata["data_provenance"]["feature_matrix_size_bytes"] > 0
            assert metadata["data_provenance"]["snapshot_date"] == (
                builder.config.snapshot_date
            )


def test_feature_matrix_builder_feature_generation_failure(mock_pipeline_context):
    """
    Tests that a failure in the SharedFeatureGenerator SQL construction halts execution.
    """
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    with patch(
        "pipelines.inference_pipeline.src.components.feature_matrix_builder.SharedFeatureGenerator"
    ) as mock_sfg:
        mock_sfg_instance = MagicMock()
        mock_sfg.return_value = mock_sfg_instance
        mock_sfg_instance.get_feature_query.side_effect = Exception(
            "SQL Compilation Error"
        )

        with pytest.raises(CustomException) as exc_info:
            builder.run()

        assert "SQL Compilation Error" in str(exc_info.value)
        # Verify DuckDB was not called
        assert mock_pipeline_context.duckdb_con.execute.call_count == 0


def test_feature_matrix_builder_duckdb_copy_failure(mock_pipeline_context):
    """
    Tests that an out-of-core execution failure (e.g., DuckDB OOM, IO error, or S3 permission)
    is caught and wrapped properly.
    """
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    def execute_side_effect(query):
        if "COPY" in query.upper():
            raise Exception("DuckDB Parquet Write IOError")
        return MagicMock()

    mock_pipeline_context.duckdb_con.execute.side_effect = execute_side_effect

    with patch(
        "pipelines.inference_pipeline.src.components.feature_matrix_builder.SharedFeatureGenerator"
    ):
        with pytest.raises(CustomException) as exc_info:
            builder.run()

        assert "DuckDB Parquet Write IOError" in str(exc_info.value)


def test_feature_matrix_builder_extract_schema_failure(mock_pipeline_context):
    """
    Tests that if the Parquet file materializes successfully, but schema extraction fails
    (e.g., file corruption, DuckDB query error), execution is halted securely.
    """
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    def execute_side_effect(query):
        mock_result = MagicMock()
        if "COPY" in query.upper():
            with open(
                builder.config.feature_matrix_file_path, "w", encoding="utf-8"
            ) as f:
                f.write("mock_parquet_data")
            return mock_result
        if "DESCRIBE" in query.upper():
            raise Exception("Schema parsing failed")
        return mock_result

    mock_pipeline_context.duckdb_con.execute.side_effect = execute_side_effect

    with patch(
        "pipelines.inference_pipeline.src.components.feature_matrix_builder.SharedFeatureGenerator"
    ):
        with pytest.raises(CustomException) as exc_info:
            builder.run()

        assert "Schema parsing failed" in str(exc_info.value)


def test_feature_matrix_builder_metadata_generation_failure(mock_pipeline_context):
    """
    Tests that a failure in DuckDB aggregation queries for metadata generation halts the pipeline.
    """
    builder = FeatureMatrixBuilder(context=mock_pipeline_context)

    def execute_side_effect(query):
        mock_result = MagicMock()
        if "COPY" in query.upper():
            with open(
                builder.config.feature_matrix_file_path, "w", encoding="utf-8"
            ) as f:
                f.write("mock_parquet_data")
            return mock_result
        if "DESCRIBE" in query.upper():
            mock_result.fetchall.return_value = [("f1", "DOUBLE")]
            return mock_result
        if "SELECT COUNT" in query.upper():
            raise Exception("Table count aggregation failed")
        return mock_result

    mock_pipeline_context.duckdb_con.execute.side_effect = execute_side_effect

    with patch(
        "pipelines.inference_pipeline.src.components.feature_matrix_builder.SharedFeatureGenerator"
    ):
        with pytest.raises(CustomException) as exc_info:
            builder.run()

        assert "Table count aggregation failed" in str(exc_info.value)