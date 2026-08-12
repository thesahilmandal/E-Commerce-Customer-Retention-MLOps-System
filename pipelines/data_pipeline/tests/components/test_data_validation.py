import pytest
from dataclasses import replace
from unittest.mock import MagicMock

from pipelines.data_pipeline.src.components.data_validation import DataValidation
from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def mock_db_execute(pipeline_context: PipelineContext):
    def _setup_mock(
        orders_schema=None,
        customers_schema=None,
        payments_schema=None,
        orders_count=100,
        customers_count=100,
        payments_count=100,
        orders_nulls=(0, 0, 0, 0),
        customers_nulls=(0, 0),
        payments_nulls=(0, 0)
    ):
        if orders_schema is None:
            orders_schema = [
                ("order_id", "VARCHAR"), ("customer_id", "VARCHAR"),
                ("order_status", "VARCHAR"), ("order_purchase_timestamp", "TIMESTAMP")
            ]
        if customers_schema is None:
            customers_schema = [("customer_id", "VARCHAR"), ("customer_unique_id", "VARCHAR")]
        if payments_schema is None:
            payments_schema = [("order_id", "VARCHAR"), ("payment_value", "DOUBLE")]

        def side_effect(query: str):
            mock_res = MagicMock()
            if "DESCRIBE" in query:
                if "orders" in query:
                    mock_res.fetchall.return_value = orders_schema
                elif "customers" in query:
                    mock_res.fetchall.return_value = customers_schema
                elif "order_payments" in query:
                    mock_res.fetchall.return_value = payments_schema
            elif "COUNT(*)" in query:
                if "orders" in query:
                    mock_res.fetchone.return_value = (orders_count,)
                elif "customers" in query:
                    mock_res.fetchone.return_value = (customers_count,)
                elif "order_payments" in query:
                    mock_res.fetchone.return_value = (payments_count,)
            elif "SUM(CASE" in query:
                if "orders" in query:
                    mock_res.fetchone.return_value = orders_nulls
                elif "customers" in query:
                    mock_res.fetchone.return_value = customers_nulls
                elif "order_payments" in query:
                    mock_res.fetchone.return_value = payments_nulls
            return mock_res

        mock_con = MagicMock()
        mock_con.execute.side_effect = side_effect
        pipeline_context.db_con = mock_con
        return mock_con

    return _setup_mock


def test_validation_success(pipeline_context: PipelineContext, mock_db_execute):
    mock_db_execute()
    validator = DataValidation(pipeline_context)
    validator.run()


def test_schema_validation_missing_column(pipeline_context: PipelineContext, mock_db_execute):
    mock_db_execute(orders_schema=[("order_id", "VARCHAR"), ("customer_id", "VARCHAR")])
    validator = DataValidation(pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        validator.run()

    assert "missing required critical columns" in str(exc_info.value)
    assert "order_status" in str(exc_info.value)


def test_record_count_zero(pipeline_context: PipelineContext, mock_db_execute):
    mock_db_execute(customers_count=0)
    validator = DataValidation(pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        validator.run()

    assert "Dataset 'customers' contains zero records" in str(exc_info.value)


def test_null_checks_strict_mode(pipeline_context: PipelineContext, mock_db_execute):
    mock_db_execute(payments_nulls=(5, 0))
    validator = DataValidation(pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        validator.run()

    assert "Critical column 'order_id' in dataset 'order_payments' contains 5 null values" in str(exc_info.value)


def test_null_checks_non_strict_mode(pipeline_context: PipelineContext, mock_db_execute):
    new_val_cfg = replace(pipeline_context.config.validation, strict_mode=False)
    pipeline_context.config = replace(pipeline_context.config, validation=new_val_cfg)

    mock_db_execute(payments_nulls=(5, 0))
    validator = DataValidation(pipeline_context)

    validator.run()


def test_fail_fast_behavior(pipeline_context: PipelineContext, mock_db_execute):
    mock_db_execute(orders_count=0, customers_count=0)
    validator = DataValidation(pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        validator.run()

    assert "Dataset 'orders' contains zero records" in str(exc_info.value)
    assert "customers" not in str(exc_info.value)


def test_aggregated_errors_fail_fast_false(pipeline_context: PipelineContext, mock_db_execute):
    new_val_cfg = replace(pipeline_context.config.validation, fail_fast=False)
    pipeline_context.config = replace(pipeline_context.config, validation=new_val_cfg)

    mock_db_execute(orders_count=0, customers_count=0)
    validator = DataValidation(pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        validator.run()

    assert "Data Validation stage failed with the following errors" in str(exc_info.value)
    assert "Dataset 'orders' contains zero records" in str(exc_info.value)
    assert "Dataset 'customers' contains zero records" in str(exc_info.value)


def test_disabled_checks(pipeline_context: PipelineContext, mock_db_execute):
    new_val_cfg = replace(
        pipeline_context.config.validation,
        enable_schema_checks=False,
        enable_null_checks=False
    )
    pipeline_context.config = replace(pipeline_context.config, validation=new_val_cfg)

    mock_con = mock_db_execute()
    validator = DataValidation(pipeline_context)
    validator.run()

    executed_queries = [call.args[0] for call in mock_con.execute.call_args_list]

    assert any("COUNT(*)" in query for query in executed_queries)
    assert not any("DESCRIBE" in query for query in executed_queries)
    assert not any("SUM(CASE" in query for query in executed_queries)