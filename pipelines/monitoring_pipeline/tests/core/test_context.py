
import pytest
from unittest.mock import MagicMock, patch

from pipelines.monitoring_pipeline.src.core.context import MonitoringPipelineContext
from shared_core.exceptions.custom_exception import CustomException


@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists")
def test_context_initialization_explicit(
    mock_exists: MagicMock,
    mock_read_yaml: MagicMock,
    mock_makedirs: MagicMock,
    mock_global_config_dict: dict,
) -> None:
    mock_exists.return_value = True
    mock_read_yaml.return_value = mock_global_config_dict

    context = MonitoringPipelineContext(
        run_id="test_run_999",
        execution_date="2026-09-01",
        config_path="dummy_config.yaml",
    )

    assert context.run_id == "test_run_999"
    assert context.execution_date == "2026-09-01"
    assert context.config == mock_global_config_dict
    assert context.cleanup_on_exit is True
    assert "test_run_999" in context.root_dir

    mock_makedirs.assert_called_once_with(
        context.root_dir,
        exist_ok=True,
    )
    mock_read_yaml.assert_called_once_with("dummy_config.yaml")


@patch("pipelines.monitoring_pipeline.src.core.context.datetime")
@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists")
def test_context_initialization_defaults(
    mock_exists: MagicMock,
    mock_read_yaml: MagicMock,
    mock_makedirs: MagicMock,
    mock_datetime: MagicMock,
) -> None:
    mock_exists.return_value = True
    mock_read_yaml.return_value = {}

    mock_now = MagicMock()
    mock_now.strftime.side_effect = [
        "20260827_123456",
        "2026-08-27",
    ]
    mock_datetime.now.return_value = mock_now

    context = MonitoringPipelineContext(
        config_path="dummy.yaml",
    )

    assert context.run_id == "run_20260827_123456"
    assert context.execution_date == "2026-08-27"
    assert context.config == {}


@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists")
def test_context_initialization_file_not_found(
    mock_exists: MagicMock,
) -> None:
    mock_exists.return_value = False

    with pytest.raises(CustomException) as exc_info:
        MonitoringPipelineContext(
            config_path="missing.yaml",
        )

    assert "Configuration file not found" in str(exc_info.value)


@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists")
def test_context_initialization_empty_yaml(
    mock_exists: MagicMock,
    mock_read_yaml: MagicMock,
    mock_makedirs: MagicMock,
) -> None:
    mock_exists.return_value = True
    mock_read_yaml.return_value = None

    context = MonitoringPipelineContext(
        config_path="dummy.yaml",
    )

    assert context.config == {}


@patch("pipelines.monitoring_pipeline.src.core.context.duckdb")
@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists")
def test_context_manager_lifecycle(
    mock_exists: MagicMock,
    mock_read_yaml: MagicMock,
    mock_makedirs: MagicMock,
    mock_duckdb: MagicMock,
) -> None:
    mock_exists.return_value = True
    mock_read_yaml.return_value = {}

    mock_con = MagicMock()
    mock_duckdb.connect.return_value = mock_con

    context = MonitoringPipelineContext(
        run_id="lifecycle_test",
    )

    with patch.object(
        context,
        "_cleanup_workspace",
    ) as mock_cleanup:
        with context as ctx:
            assert ctx.duckdb_con == mock_con

            mock_duckdb.connect.assert_called_once_with(
                database=":memory:",
            )

            mock_con.execute.assert_any_call("INSTALL httpfs;")
            mock_con.execute.assert_any_call("LOAD httpfs;")
            mock_con.execute.assert_any_call("INSTALL aws;")
            mock_con.execute.assert_any_call("LOAD aws;")
            mock_con.execute.assert_any_call(
                "CALL load_aws_credentials();"
            )
            mock_con.execute.assert_any_call(
                "SET s3_region='us-east-1';"
            )
            mock_con.execute.assert_any_call(
                "PRAGMA threads=4;"
            )
            mock_con.execute.assert_any_call(
                "PRAGMA memory_limit='4GB';"
            )

        mock_con.close.assert_called_once()
        mock_cleanup.assert_called_once()


@patch("pipelines.monitoring_pipeline.src.core.context.duckdb")
@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists")
def test_context_manager_custom_aws_region(
    mock_exists: MagicMock,
    mock_read_yaml: MagicMock,
    mock_makedirs: MagicMock,
    mock_duckdb: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_exists.return_value = True
    mock_read_yaml.return_value = {}

    monkeypatch.setenv(
        "AWS_DEFAULT_REGION",
        "eu-west-1",
    )

    mock_con = MagicMock()
    mock_duckdb.connect.return_value = mock_con

    with MonitoringPipelineContext():
        mock_con.execute.assert_any_call(
            "SET s3_region='eu-west-1';"
        )


@patch("pipelines.monitoring_pipeline.src.core.context.duckdb")
@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists")
def test_context_manager_initialization_failure(
    mock_exists: MagicMock,
    mock_read_yaml: MagicMock,
    mock_makedirs: MagicMock,
    mock_duckdb: MagicMock,
) -> None:
    mock_exists.return_value = True
    mock_read_yaml.return_value = {}

    mock_duckdb.connect.side_effect = Exception(
        "DuckDB startup crash"
    )

    context = MonitoringPipelineContext()

    with patch.object(
        context,
        "_safe_close_connection",
    ) as mock_safe_close:
        with pytest.raises(CustomException) as exc_info:
            with context:
                pass

        assert "DuckDB startup crash" in str(exc_info.value)
        mock_safe_close.assert_called_once()


@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists")
def test_safe_close_connection_swallows_exception(
    mock_exists: MagicMock,
    mock_read_yaml: MagicMock,
    mock_makedirs: MagicMock,
) -> None:
    mock_exists.return_value = True
    mock_read_yaml.return_value = {}

    context = MonitoringPipelineContext()

    mock_con = MagicMock()
    mock_con.close.side_effect = Exception(
        "Forced close failure"
    )

    context.duckdb_con = mock_con

    context._safe_close_connection()

    mock_con.close.assert_called_once()


@patch("pipelines.monitoring_pipeline.src.core.context.os.rmdir")
@patch("pipelines.monitoring_pipeline.src.core.context.shutil.rmtree")
@patch("pipelines.monitoring_pipeline.src.core.context.os.listdir")
@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists")
def test_cleanup_workspace_success(
    mock_exists: MagicMock,
    mock_read_yaml: MagicMock,
    mock_makedirs: MagicMock,
    mock_listdir: MagicMock,
    mock_rmtree: MagicMock,
    mock_rmdir: MagicMock,
) -> None:
    def exists_side_effect(path: str) -> bool:
        return True

    mock_exists.side_effect = exists_side_effect
    mock_read_yaml.return_value = {}
    mock_listdir.return_value = []

    context = MonitoringPipelineContext(
        cleanup_on_exit=True,
    )

    context._cleanup_workspace()

    mock_rmtree.assert_called_once_with(
        context.root_dir,
        ignore_errors=True,
    )
    mock_rmdir.assert_called_once()


@patch("pipelines.monitoring_pipeline.src.core.context.shutil.rmtree")
@patch("pipelines.monitoring_pipeline.src.core.context.os.makedirs")
@patch("pipelines.monitoring_pipeline.src.core.context.read_yaml")
@patch("pipelines.monitoring_pipeline.src.core.context.os.path.exists")
def test_cleanup_workspace_disabled(
    mock_exists: MagicMock,
    mock_read_yaml: MagicMock,
    mock_makedirs: MagicMock,
    mock_rmtree: MagicMock,
) -> None:
    mock_exists.return_value = True
    mock_read_yaml.return_value = {}

    context = MonitoringPipelineContext(
        cleanup_on_exit=False,
    )

    context._cleanup_workspace()

    mock_rmtree.assert_not_called()