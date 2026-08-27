import os
import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from pipelines.monitoring_pipeline.src.components.baseline_and_telemetry_resolver import (
    BaselineAndTelemetryResolver,
)
from pipelines.monitoring_pipeline.src.entity.config_entity import (
    BaselineAndTelemetryResolverConfig,
)
from pipelines.monitoring_pipeline.src.entity.artifact_entity import (
    BaselineAndTelemetryResolverArtifact,
)
from shared_core.exceptions.custom_exception import CustomException


@pytest.fixture
def resolver_config(
    tmp_path: Path,
) -> BaselineAndTelemetryResolverConfig:
    """Provides a mocked configuration for the Baseline & Telemetry Resolver."""
    return BaselineAndTelemetryResolverConfig(
        resolver_root_dir=str(tmp_path),
        s3_registry_pointer_uri="s3://test-bucket/model_state.json",
        s3_telemetry_base_uri="s3://test-bucket/telemetry",
        s3_data_lake_bronze_uri="s3://test-bucket/bronze",
        current_date="2026-08-26",
        lookback_date="2026-07-27",
        current_partition_suffix="year=2026/month=08/day=26",
        lookback_partition_suffix="year=2026/month=07/day=27",
        baseline_metrics_file_path=os.path.join(tmp_path, "metrics.json"),
        reference_distributions_file_path=os.path.join(tmp_path, "dist.json"),
        shap_importance_file_path=os.path.join(tmp_path, "shap.json"),
        current_telemetry_file_path=os.path.join(tmp_path, "curr_tel.parquet"),
        lookback_telemetry_file_path=os.path.join(tmp_path, "lb_tel.parquet"),
        lookback_labels_file_path=os.path.join(tmp_path, "lb_lbl.parquet"),
        metadata_file_path=os.path.join(tmp_path, "meta.json"),
        lookback_period_days=30,
    )


@pytest.fixture
def resolver(
    resolver_config: BaselineAndTelemetryResolverConfig,
    mock_pipeline_context: MagicMock,
) -> BaselineAndTelemetryResolver:
    """Yields a fully initialized BaselineAndTelemetryResolver with isolated paths."""
    with patch(
        "pipelines.monitoring_pipeline.src.components.baseline_and_telemetry_resolver.S3Sync"
    ):
        resolver_inst = BaselineAndTelemetryResolver(
            config=resolver_config,
            context=mock_pipeline_context,
        )
        resolver_inst.s3_sync = MagicMock()
        return resolver_inst


def test_resolver_initialization_failure(
    mock_pipeline_context: MagicMock,
) -> None:
    with pytest.raises(CustomException):
        BaselineAndTelemetryResolver(
            config=None,
            context=mock_pipeline_context,
        )


@patch.object(BaselineAndTelemetryResolver, "_resolve_champion_baselines")
@patch.object(BaselineAndTelemetryResolver, "_fetch_current_telemetry")
@patch.object(
    BaselineAndTelemetryResolver,
    "_fetch_lookback_telemetry_and_labels",
)
@patch.object(BaselineAndTelemetryResolver, "_generate_metadata")
def test_run_success(
    mock_generate_metadata: MagicMock,
    mock_fetch_lookback: MagicMock,
    mock_fetch_current: MagicMock,
    mock_resolve: MagicMock,
    resolver: BaselineAndTelemetryResolver,
) -> None:
    mock_resolve.return_value = "champion_123"
    mock_fetch_current.return_value = "CURRENT_TELEMETRY_RESOLVED"
    mock_fetch_lookback.return_value = "MATURED_EVALUATION_READY"

    artifact = resolver.run()

    assert isinstance(artifact, BaselineAndTelemetryResolverArtifact)
    assert artifact.champion_run_id == "champion_123"
    assert (
        artifact.baseline_metrics_file_path
        == resolver.config.baseline_metrics_file_path
    )
    assert artifact.metadata_file_path == resolver.config.metadata_file_path

    mock_resolve.assert_called_once()
    mock_fetch_current.assert_called_once_with(resolver.context.duckdb_con)
    mock_fetch_lookback.assert_called_once_with(resolver.context.duckdb_con)
    mock_generate_metadata.assert_called_once()


def test_run_missing_duckdb_con(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    resolver.context.duckdb_con = None

    with patch.object(
        resolver,
        "_resolve_champion_baselines",
        return_value="champion_123",
    ):
        with pytest.raises(CustomException) as exc_info:
            resolver.run()

        assert "Shared DuckDB connection is not initialized" in str(
            exc_info.value
        )


def test_resolve_champion_baselines_success(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    def mock_download_file(s3_uri: str, local_path: str) -> None:
        if "model_state" in s3_uri:
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "champion_run_id": "test_champ_99",
                        "s3_baseline_metrics_path": "s3://b/m.json",
                        "s3_reference_distributions_path": "s3://b/d.json",
                        "s3_shap_importance_path": "s3://b/s.json",
                    },
                    f,
                )

    resolver.s3_sync.download_file.side_effect = mock_download_file

    champ_id = resolver._resolve_champion_baselines()

    assert champ_id == "test_champ_99"
    assert resolver.s3_sync.download_file.call_count == 4


def test_resolve_champion_baselines_missing_shap_generates_synthetic(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    def mock_download_file(s3_uri: str, local_path: str) -> None:
        if "model_state" in s3_uri:
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_id": "test_champ_88",
                        "s3_baseline_metrics_path": "s3://b/m.json",
                        "s3_reference_distributions_path": "s3://b/d.json",
                    },
                    f,
                )
        elif "d.json" in local_path or "dist.json" in local_path:
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "distributions": {
                            "feature_a": {"type": "numerical"},
                            "feature_b": {"type": "categorical"},
                        }
                    },
                    f,
                )

    resolver.s3_sync.download_file.side_effect = mock_download_file

    champ_id = resolver._resolve_champion_baselines()

    assert champ_id == "test_champ_88"
    assert os.path.exists(resolver.config.shap_importance_file_path)

    with open(
        resolver.config.shap_importance_file_path,
        "r",
        encoding="utf-8",
    ) as f:
        synthetic_shap = json.load(f)

    assert "feature_importance" in synthetic_shap
    assert len(synthetic_shap["feature_importance"]) == 2


def test_resolve_champion_baselines_corrupted_pointer(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    def mock_download_file(s3_uri: str, local_path: str) -> None:
        if "model_state" in s3_uri:
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump({"random_key": "value"}, f)

    resolver.s3_sync.download_file.side_effect = mock_download_file

    with pytest.raises(CustomException) as exc_info:
        resolver._resolve_champion_baselines()

    assert "Corrupted Registry Pointer" in str(exc_info.value)


def test_resolve_champion_baselines_missing_uris(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    def mock_download_file(s3_uri: str, local_path: str) -> None:
        if "model_state" in s3_uri:
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"champion_run_id": "test_champ"},
                    f,
                )

    resolver.s3_sync.download_file.side_effect = mock_download_file

    with pytest.raises(CustomException) as exc_info:
        resolver._resolve_champion_baselines()

    assert "Missing required baseline URIs" in str(exc_info.value)


def test_generate_synthetic_shap(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    with open(
        resolver.config.reference_distributions_file_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "distributions": {
                    resolver.target_col: {},
                    resolver.prediction_col: {},
                    resolver.customer_id_col: {},
                    "feature_1": {},
                    "feature_2": {},
                }
            },
            f,
        )

    resolver._generate_synthetic_shap()

    with open(
        resolver.config.shap_importance_file_path,
        "r",
        encoding="utf-8",
    ) as f:
        shap_data = json.load(f)

    features = [
        feature["feature_name"]
        for feature in shap_data["feature_importance"]
    ]

    assert "feature_1" in features
    assert "feature_2" in features
    assert resolver.target_col not in features
    assert resolver.prediction_col not in features


def test_fetch_current_telemetry_success(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    mock_con = MagicMock()

    status = resolver._fetch_current_telemetry(mock_con)

    assert status == "CURRENT_TELEMETRY_RESOLVED"
    mock_con.execute.assert_called_once()
    assert (
        "COPY (SELECT * FROM read_parquet"
        in mock_con.execute.call_args[0][0]
    )


def test_fetch_current_telemetry_zero_traffic_fallback(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    mock_con = MagicMock()
    mock_con.execute.side_effect = [
        Exception("HTTP 404 Not Found"),
        None,
    ]

    status = resolver._fetch_current_telemetry(mock_con)

    assert status == "SKIPPED_ZERO_TRAFFIC_DAY"
    assert mock_con.execute.call_count == 2
    assert (
        "WHERE 1=0"
        in mock_con.execute.call_args_list[1][0][0]
    )


def test_fetch_current_telemetry_other_exception(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    mock_con = MagicMock()
    mock_con.execute.side_effect = Exception(
        "AWS Credentials Expired"
    )

    with pytest.raises(CustomException) as exc_info:
        resolver._fetch_current_telemetry(mock_con)

    assert "AWS Credentials Expired" in str(exc_info.value)


@patch(
    "pipelines.monitoring_pipeline.src.components.baseline_and_telemetry_resolver.SharedFeatureGenerator"
)
@patch(
    "pipelines.monitoring_pipeline.src.components.baseline_and_telemetry_resolver.os.rename"
)
def test_fetch_lookback_telemetry_success(
    mock_rename: MagicMock,
    mock_feature_gen: MagicMock,
    resolver: BaselineAndTelemetryResolver,
) -> None:
    mock_con = MagicMock()

    mock_gen_instance = MagicMock()
    mock_gen_instance.get_feature_query.return_value = (
        "SELECT * FROM dummy"
    )
    mock_feature_gen.return_value = mock_gen_instance

    status = resolver._fetch_lookback_telemetry_and_labels(mock_con)

    assert status == "MATURED_EVALUATION_READY"
    assert mock_con.execute.call_count == 2
    mock_rename.assert_called_once()


def test_fetch_lookback_telemetry_immature_fallback(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    mock_con = MagicMock()
    mock_con.execute.side_effect = [
        Exception("HTTP 404 Error"),
        None,
        None,
    ]

    status = resolver._fetch_lookback_telemetry_and_labels(mock_con)

    assert status == "SKIPPED_SYSTEM_IMMATURE"
    assert mock_con.execute.call_count == 3
    assert (
        "WHERE 1=0"
        in mock_con.execute.call_args_list[1][0][0]
    )
    assert (
        "WHERE 1=0"
        in mock_con.execute.call_args_list[2][0][0]
    )


def test_generate_metadata(
    resolver: BaselineAndTelemetryResolver,
) -> None:
    resolver._generate_metadata(
        champion_run_id="champ_run_123",
        current_status="CURRENT_TELEMETRY_RESOLVED",
        lookback_status="MATURED_EVALUATION_READY",
        execution_time=1.25,
    )

    assert os.path.exists(resolver.config.metadata_file_path)

    with open(
        resolver.config.metadata_file_path,
        "r",
        encoding="utf-8",
    ) as f:
        meta = json.load(f)

    assert (
        meta["pipeline_stage"]
        == "Monitoring Baseline & Telemetry Resolver"
    )
    assert meta["execution_time_seconds"] == 1.25
    assert (
        meta["resolved_state"]["champion_run_id_loaded"]
        == "champ_run_123"
    )
    assert (
        meta["temporal_bounds"]["current_telemetry_status"]
        == "CURRENT_TELEMETRY_RESOLVED"
    )
    assert (
        meta["temporal_bounds"]["lookback_evaluation_status"]
        == "MATURED_EVALUATION_READY"
    )
    assert meta["orchestration"]["run_id"] == resolver.context.run_id