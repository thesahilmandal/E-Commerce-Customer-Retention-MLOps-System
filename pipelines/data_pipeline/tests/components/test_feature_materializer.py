import dataclasses
import pytest
from unittest.mock import MagicMock

from pipelines.data_pipeline.src.components.feature_materializer import FeatureMaterializer
from shared_core.exceptions.custom_exception import CustomException


def test_feature_materializer_initialization(mock_pipeline_context: MagicMock) -> None:
    materializer = FeatureMaterializer(mock_pipeline_context)
    
    expected_uri = "s3://test-feature-store-bucket/feature_store/test_run_12345/test_dataset.parquet"
    assert materializer.s3_output_uri == expected_uri
    assert materializer.bronze_uri == "s3://test-bronze-bucket/bronze"
    assert materializer.target_cfg == mock_pipeline_context.config.business_logic.target_definition
    assert materializer.cohort_cfg == mock_pipeline_context.config.business_logic.cohort_definition


def test_build_target_query_multiple_statuses(mock_pipeline_context: MagicMock) -> None:
    new_cohort = dataclasses.replace(
        mock_pipeline_context.config.business_logic.cohort_definition,
        active_status_codes=["delivered", "shipped"],
    )
    new_business_logic = dataclasses.replace(
        mock_pipeline_context.config.business_logic,
        cohort_definition=new_cohort,
    )
    mock_pipeline_context.config = dataclasses.replace(
        mock_pipeline_context.config,
        business_logic=new_business_logic,
    )

    materializer = FeatureMaterializer(mock_pipeline_context)
    
    target_sql = materializer._build_target_query("2023-06-01")
    
    assert "IN ('delivered', 'shipped')" in target_sql
    assert "INTERVAL 180 DAY" in target_sql
    assert "TIMESTAMP '2023-06-01'" in target_sql


def test_build_target_query_single_status(mock_pipeline_context: MagicMock) -> None:
    new_cohort = dataclasses.replace(
        mock_pipeline_context.config.business_logic.cohort_definition,
        active_status_codes=["delivered"],
    )
    new_business_logic = dataclasses.replace(
        mock_pipeline_context.config.business_logic,
        cohort_definition=new_cohort,
    )
    mock_pipeline_context.config = dataclasses.replace(
        mock_pipeline_context.config,
        business_logic=new_business_logic,
    )

    materializer = FeatureMaterializer(mock_pipeline_context)
    
    target_sql = materializer._build_target_query("2023-06-01")
    
    assert "IN ('delivered')" in target_sql
    assert "INTERVAL 180 DAY" in target_sql


def test_run_success(mock_pipeline_context: MagicMock) -> None:
    materializer = FeatureMaterializer(mock_pipeline_context)
    
    materializer.run()
    
    mock_pipeline_context.db_con.execute.assert_called_once()
    executed_query = mock_pipeline_context.db_con.execute.call_args[0][0]
    
    assert "COPY (" in executed_query
    assert "TO 's3://test-feature-store-bucket/feature_store/test_run_12345/test_dataset.parquet'" in executed_query
    assert "FORMAT PARQUET" in executed_query
    assert "COMPRESSION 'snappy'" in executed_query
    assert "target_180d_ltv" in executed_query
    assert "target_is_churn" in executed_query
    assert "ingested_at_utc" in executed_query


def test_run_execution_failure(mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = Exception("DuckDB out of memory error")
    materializer = FeatureMaterializer(mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        materializer.run()
        
    assert "DuckDB out of memory error" in str(exc_info.value)