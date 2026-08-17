import os
from unittest.mock import MagicMock, patch

import pytest

from pipelines.monitoring_pipeline.src.core.context import MonitoringPipelineContext
from shared_core.exceptions.custom_exception import CustomException


@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists", return_value=True)
def test_context_initialization_with_explicit_args(
    _mock_exists: MagicMock, mock_read_yaml: MagicMock, mock_makedirs: MagicMock
) -> None:
    """
    Validates that providing explicit run_id and execution_date assigns them correctly
    and successfully creates the context root directory.
    """
    mock_read_yaml.return_value = {
        "project_config": {
            "local_artifact_dir": "test_artifacts",
            "pipeline_name": "test_pipeline"
        }
    }
    
    context = MonitoringPipelineContext(
        run_id="test_run_001",
        execution_date="2026-08-14",
        config_path="dummy/path/config.yaml"
    )
    
    assert context.run_id == "test_run_001"
    assert context.execution_date == "2026-08-14"
    
    expected_root_dir = os.path.join("test_artifacts", "test_pipeline", "test_run_001")
    assert context.root_dir == expected_root_dir
    mock_makedirs.assert_called_once_with(expected_root_dir, exist_ok=True)
    assert context.duckdb_con is None


@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists", return_value=True)
def test_context_initialization_with_implicit_args(
    _mock_exists: MagicMock, mock_read_yaml: MagicMock, _mock_makedirs: MagicMock
) -> None:
    """
    Validates that omitting run_id and execution_date triggers automatic 
    generation of safe, formatted fallback identifiers.
    """
    mock_read_yaml.return_value = {}
    
    context = MonitoringPipelineContext()
    
    assert context.run_id.startswith("run_")
    assert len(context.run_id) > 4
    # Basic structural check for ISO-like date YYYY-MM-DD
    assert len(context.execution_date) == 10
    assert "-" in context.execution_date


@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists", return_value=False)
def test_context_missing_config_raises_custom_exception(_mock_exists: MagicMock) -> None:
    """
    Validates that a missing configuration file properly throws a CustomException
    rather than silently failing or raising a raw FileNotFoundError.
    """
    with pytest.raises(CustomException):
        MonitoringPipelineContext(config_path="invalid/path/config.yaml")


@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml", return_value=None)
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists", return_value=True)
def test_context_empty_config_fallback(
    _mock_exists: MagicMock, _mock_read_yaml: MagicMock, _mock_makedirs: MagicMock
) -> None:
    """
    Validates the defensive fix: an empty/null YAML file should evaluate 
    to an empty dictionary, not None.
    """
    context = MonitoringPipelineContext()
    assert isinstance(context.config, dict)
    assert len(context.config) == 0


@patch("pipelines.monitoring_pipeline.src.core.context.duckdb.connect")
@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml", return_value={})
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists", return_value=True)
def test_context_manager_lifecycle(
    _mock_exists: MagicMock,
    _mock_read_yaml: MagicMock,
    _mock_makedirs: MagicMock,
    mock_duckdb_connect: MagicMock
) -> None:
    """
    Validates the __enter__ and __exit__ lifecycle of the context manager,
    ensuring DuckDB is initialized with correct extensions and safely closed.
    """
    mock_conn = MagicMock()
    mock_duckdb_connect.return_value = mock_conn
    
    context = MonitoringPipelineContext()
    
    with context as ctx:
        assert ctx.duckdb_con == mock_conn
        
        # Verify that AWS and httpfs extensions and pragmas were executed
        execute_calls = mock_conn.execute.call_args_list
        assert len(execute_calls) >= 5, "DuckDB initialization queries were not executed."
        
        # Check for key setup commands
        commands = [call[0][0] for call in execute_calls]
        assert "INSTALL httpfs;" in commands
        assert "LOAD aws;" in commands
        assert "CALL load_aws_credentials();" in commands
        
        # Ensure connection isn't prematurely closed
        mock_conn.close.assert_not_called()
        
    # Outside the context manager block, the connection must be safely closed
    mock_conn.close.assert_called_once()


@patch("pipelines.monitoring_pipeline.src.core.context.duckdb.connect")
@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml", return_value={})
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists", return_value=True)
def test_context_manager_exception_teardown(
    _mock_exists: MagicMock,
    _mock_read_yaml: MagicMock,
    _mock_makedirs: MagicMock,
    mock_duckdb_connect: MagicMock
) -> None:
    """
    Validates that the DuckDB connection is safely closed even if an exception
    occurs inside the context block.
    """
    mock_conn = MagicMock()
    mock_duckdb_connect.return_value = mock_conn
    
    context = MonitoringPipelineContext()
    
    try:
        with context:
            raise ValueError("Simulated pipeline failure")
    except ValueError:
        pass
        
    # The teardown sequence MUST still execute
    mock_conn.close.assert_called_once()