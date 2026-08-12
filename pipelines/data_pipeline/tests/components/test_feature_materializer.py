import pytest
from unittest.mock import MagicMock

from pipelines.data_pipeline.src.components.feature_materializer import FeatureMaterializer
from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException


def test_materialization_success(pipeline_context: PipelineContext):
    mock_execute = MagicMock()
    pipeline_context.db_con = MagicMock()
    pipeline_context.db_con.execute = mock_execute

    materializer = FeatureMaterializer(pipeline_context)
    materializer.run()

    mock_execute.assert_called()

    executed_queries = [call.args[0] for call in mock_execute.call_args_list]
    copy_query = next((q for q in executed_queries if "COPY" in q.upper()), None)

    assert copy_query is not None, "A COPY query should be executed for materialization."
    assert pipeline_context.config.storage.feature_store.base_uri in copy_query
    assert pipeline_context.config.storage.feature_store.artifact_name in copy_query
    assert "FORMAT PARQUET" in copy_query.upper()


def test_materialization_query_contains_temporal_bounds(pipeline_context: PipelineContext):
    mock_execute = MagicMock()
    pipeline_context.db_con = MagicMock()
    pipeline_context.db_con.execute = mock_execute

    materializer = FeatureMaterializer(pipeline_context)
    materializer.run()

    executed_queries = [call.args[0] for call in mock_execute.call_args_list]
    materialization_query = "\n".join(executed_queries)

    assert pipeline_context.start_date in materialization_query
    assert pipeline_context.end_date in materialization_query


def test_materialization_query_applies_business_logic(pipeline_context: PipelineContext):
    mock_execute = MagicMock()
    pipeline_context.db_con = MagicMock()
    pipeline_context.db_con.execute = mock_execute

    materializer = FeatureMaterializer(pipeline_context)
    materializer.run()

    executed_queries = [call.args[0] for call in mock_execute.call_args_list]
    materialization_query = "\n".join(executed_queries)

    business_logic_cfg = pipeline_context.config.business_logic

    assert str(business_logic_cfg.target_definition.churn_window_days) in materialization_query
    assert str(business_logic_cfg.cohort_definition.minimum_orders) in materialization_query


def test_materialization_exception_propagation(pipeline_context: PipelineContext):
    pipeline_context.db_con = MagicMock()
    pipeline_context.db_con.execute.side_effect = Exception(
        "Simulated DuckDB out-of-core memory error"
    )

    materializer = FeatureMaterializer(pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        materializer.run()

    assert "Simulated DuckDB out-of-core memory error" in str(exc_info.value)