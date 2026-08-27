import dataclasses
import pytest
from unittest.mock import MagicMock

from pipelines.data_pipeline.src.components.data_validation import DataValidation
from shared_core.exceptions.custom_exception import CustomException


def create_mock_execute(
    describe_cols: list | None = None,
    count_val: int = 50,
    null_vals: tuple = (0, 0, 0, 0),
) -> callable:
    """Helper to generate a side_effect function for DuckDB connection execute method."""
    if describe_cols is None:
        describe_cols = [
            ("order_id", "VARCHAR"),
            ("customer_id", "VARCHAR"),
            ("order_status", "VARCHAR"),
            ("order_purchase_timestamp", "TIMESTAMP"),
            ("customer_unique_id", "VARCHAR"),
            ("payment_value", "DOUBLE"),
        ]

    def side_effect(query: str) -> MagicMock:
        mock_res = MagicMock()
        upper_query = query.upper()
        if "DESCRIBE" in upper_query:
            mock_res.fetchall.return_value = describe_cols
        elif "COUNT(*)" in upper_query:
            mock_res.fetchone.return_value = (count_val,)
        elif "SUM(CASE WHEN" in upper_query:
            mock_res.fetchone.return_value = null_vals
        else:
            mock_res.fetchone.return_value = None
            mock_res.fetchall.return_value = []
        return mock_res

    return side_effect


def test_data_validation_success(mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = create_mock_execute()
    validator = DataValidation(mock_pipeline_context)
    
    validator.run()
    
    assert mock_pipeline_context.db_con.execute.call_count == 9


def test_missing_critical_columns_schema_check(mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = create_mock_execute(
        describe_cols=[("unrelated_column", "VARCHAR")]
    )
    validator = DataValidation(mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        validator.run()
    
    assert "missing required critical columns" in str(exc_info.value)


def test_zero_records_validation_failure(mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = create_mock_execute(count_val=0)
    validator = DataValidation(mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        validator.run()
    
    assert "contains zero records in temporal window" in str(exc_info.value)


def test_null_values_with_strict_mode_failure(mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = create_mock_execute(
        null_vals=(1, 0, 0, 0)
    )
    validator = DataValidation(mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        validator.run()
        
    assert "contains 1 null values in the target temporal window" in str(exc_info.value)


def test_null_values_without_strict_mode_success(mock_pipeline_context: MagicMock) -> None:
    new_validation = dataclasses.replace(mock_pipeline_context.config.validation, strict_mode=False)
    mock_pipeline_context.config = dataclasses.replace(mock_pipeline_context.config, validation=new_validation)
    
    mock_pipeline_context.db_con.execute.side_effect = create_mock_execute(
        null_vals=(1, 0, 0, 0)
    )
    validator = DataValidation(mock_pipeline_context)
    
    validator.run()


def test_fail_fast_behavior(mock_pipeline_context: MagicMock) -> None:
    new_validation = dataclasses.replace(mock_pipeline_context.config.validation, fail_fast=True)
    mock_pipeline_context.config = dataclasses.replace(mock_pipeline_context.config, validation=new_validation)
    
    mock_pipeline_context.db_con.execute.side_effect = create_mock_execute(count_val=0)
    validator = DataValidation(mock_pipeline_context)
    
    with pytest.raises(CustomException):
        validator.run()
        
    assert mock_pipeline_context.db_con.execute.call_count == 2


def test_aggregate_failures_behavior(mock_pipeline_context: MagicMock) -> None:
    new_validation = dataclasses.replace(mock_pipeline_context.config.validation, fail_fast=False)
    mock_pipeline_context.config = dataclasses.replace(mock_pipeline_context.config, validation=new_validation)
    
    mock_pipeline_context.db_con.execute.side_effect = create_mock_execute(count_val=0)
    validator = DataValidation(mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        validator.run()
        
    assert mock_pipeline_context.db_con.execute.call_count == 6
    assert "orders" in str(exc_info.value)
    assert "customers" in str(exc_info.value)
    assert "order_payments" in str(exc_info.value)


def test_unexpected_database_exception(mock_pipeline_context: MagicMock) -> None:
    mock_pipeline_context.db_con.execute.side_effect = Exception("DuckDB memory allocation failed")
    validator = DataValidation(mock_pipeline_context)
    
    with pytest.raises(CustomException) as exc_info:
        validator.run()
        
    assert "DuckDB memory allocation failed" in str(exc_info.value)