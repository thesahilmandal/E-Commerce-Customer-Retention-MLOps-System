import os
from unittest.mock import MagicMock

import duckdb
import pytest

from pipelines.data_pipeline.src.core.config_parser import DataPipelineConfig
from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException


def test_context_initialization_success(parsed_config: DataPipelineConfig, mock_s3_sync: MagicMock):
    context = PipelineContext(
        run_id="test_run_init",
        start_date="2016-09-01",
        end_date="2016-10-01",
        config=parsed_config,
        s3_sync=mock_s3_sync
    )

    assert context.run_id == "test_run_init"
    assert context.start_date == "2016-09-01"
    assert context.end_date == "2016-10-01"
    assert context.config == parsed_config
    assert context.s3_sync == mock_s3_sync
    assert context.db_con is not None

    assert os.path.exists(parsed_config.compute.runtime.temp_directory)

    context.close()
    assert context.db_con is None


def test_context_manager_lifecycle(parsed_config: DataPipelineConfig, mock_s3_sync: MagicMock):
    with PipelineContext(
        run_id="test_run_lifecycle",
        start_date="2016-09-01",
        end_date="2016-10-01",
        config=parsed_config,
        s3_sync=mock_s3_sync
    ) as context:
        assert context.db_con is not None
        db_con_reference = context.db_con

        res = db_con_reference.execute("SELECT 1").fetchone()
        assert res == (1,)

    assert context.db_con is None

    with pytest.raises(Exception):
        db_con_reference.execute("SELECT 1")


def test_duckdb_initialization_failure(
    parsed_config: DataPipelineConfig, mock_s3_sync: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    def mock_connect(*args, **kwargs):
        raise RuntimeError("Simulated DuckDB connection failure")

    monkeypatch.setattr(duckdb, "connect", mock_connect)

    with pytest.raises(CustomException) as exc_info:
        PipelineContext(
            run_id="test_run_fail",
            start_date="2016-09-01",
            end_date="2016-10-01",
            config=parsed_config,
            s3_sync=mock_s3_sync
        )

    assert "Simulated DuckDB connection failure" in str(exc_info.value)


def test_duckdb_pragmas_and_extensions_loaded(
    parsed_config: DataPipelineConfig, mock_s3_sync: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    mock_con = MagicMock()
    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: mock_con)

    PipelineContext(
        run_id="test_run_pragmas",
        start_date="2016-09-01",
        end_date="2016-10-01",
        config=parsed_config,
        s3_sync=mock_s3_sync
    )

    executed_queries = [call.args[0] for call in mock_con.execute.call_args_list]

    assert any(f"PRAGMA threads={parsed_config.compute.hardware.threads}" in q for q in executed_queries)
    assert any(f"PRAGMA memory_limit='{parsed_config.compute.hardware.memory_limit}'" in q for q in executed_queries)
    assert any(f"PRAGMA temp_directory='{parsed_config.compute.runtime.temp_directory}'" in q for q in executed_queries)
    assert any("INSTALL httpfs" in q for q in executed_queries)
    assert any("LOAD httpfs" in q for q in executed_queries)
    assert any("CALL load_aws_credentials()" in q for q in executed_queries)


def test_close_handles_exceptions_gracefully(
    parsed_config: DataPipelineConfig, mock_s3_sync: MagicMock
):
    context = PipelineContext(
        run_id="test_run_close",
        start_date="2016-09-01",
        end_date="2016-10-01",
        config=parsed_config,
        s3_sync=mock_s3_sync
    )
    
    # Overwrite the real C-extension object entirely with a Mock
    mock_db_con = MagicMock()
    mock_db_con.close.side_effect = RuntimeError("Simulated close failure")
    context.db_con = mock_db_con
    
    context.close()
    
    mock_db_con.close.assert_called_once()
    assert context.db_con is None
    

def test_close_idempotency(parsed_config: DataPipelineConfig, mock_s3_sync: MagicMock):
    context = PipelineContext(
        run_id="test_run_idempotency",
        start_date="2016-09-01",
        end_date="2016-10-01",
        config=parsed_config,
        s3_sync=mock_s3_sync
    )

    context.close()
    assert context.db_con is None

    context.close()
    assert context.db_con is None