import pytest
from unittest.mock import MagicMock

from pipelines.data_pipeline.src.components.data_discovery import DataDiscovery
from pipelines.data_pipeline.src.core.context import PipelineContext
from shared_core.exceptions.custom_exception import CustomException


def test_get_required_months_date_only(pipeline_context: PipelineContext):
    pipeline_context.start_date = "2016-09-15"
    pipeline_context.end_date = "2016-11-05"

    discovery = DataDiscovery(pipeline_context)
    months = discovery._get_required_months()

    assert months == {("2016", "09"), ("2016", "10"), ("2016", "11")}


def test_get_required_months_timestamp(pipeline_context: PipelineContext):
    pipeline_context.start_date = "2016-12-31 10:00:00"
    pipeline_context.end_date = "2017-02-01 12:00:00"

    discovery = DataDiscovery(pipeline_context)
    months = discovery._get_required_months()

    assert months == {("2016", "12"), ("2017", "01"), ("2017", "02")}


def test_get_required_months_single_month(pipeline_context: PipelineContext):
    pipeline_context.start_date = "2018-01-01"
    pipeline_context.end_date = "2018-01-31"

    discovery = DataDiscovery(pipeline_context)
    months = discovery._get_required_months()

    assert months == {("2018", "01")}


def test_prefix_exists_in_s3(pipeline_context: PipelineContext):
    discovery = DataDiscovery(pipeline_context)

    discovery.s3_client.list_objects_v2.return_value = {"Contents": [{"Key": "test_obj"}]}
    assert discovery._prefix_exists_in_s3("s3://test-bucket/path") is True
    discovery.s3_client.list_objects_v2.assert_called_with(
        Bucket="test-bucket", Prefix="path/", MaxKeys=1
    )

    discovery.s3_client.list_objects_v2.return_value = {"CommonPrefixes": [{"Prefix": "test_dir/"}]}
    assert discovery._prefix_exists_in_s3("s3://test-bucket/path/") is True

    discovery.s3_client.list_objects_v2.return_value = {}
    assert discovery._prefix_exists_in_s3("s3://test-bucket/empty_path") is False


def test_run_success(pipeline_context: PipelineContext, monkeypatch: pytest.MonkeyPatch):
    pipeline_context.start_date = "2016-09-01"
    pipeline_context.end_date = "2016-09-30"

    discovery = DataDiscovery(pipeline_context)

    def mock_prefix_exists(s3_uri: str) -> bool:
        return True

    monkeypatch.setattr(discovery, "_prefix_exists_in_s3", mock_prefix_exists)

    discovery.run()


def test_run_missing_dataset_root(pipeline_context: PipelineContext, monkeypatch: pytest.MonkeyPatch):
    pipeline_context.start_date = "2016-09-01"
    pipeline_context.end_date = "2016-09-30"

    discovery = DataDiscovery(pipeline_context)

    def mock_prefix_exists(s3_uri: str) -> bool:
        if "orders" in s3_uri and "year=" not in s3_uri:
            return False
        return True

    monkeypatch.setattr(discovery, "_prefix_exists_in_s3", mock_prefix_exists)

    with pytest.raises(CustomException) as exc_info:
        discovery.run()

    assert "Missing required upstream data" in str(exc_info.value)
    assert "orders" in str(exc_info.value)


def test_run_missing_temporal_partitions(
    pipeline_context: PipelineContext, monkeypatch: pytest.MonkeyPatch
):
    pipeline_context.start_date = "2016-09-01"
    pipeline_context.end_date = "2016-09-30"

    discovery = DataDiscovery(pipeline_context)

    def mock_prefix_exists(s3_uri: str) -> bool:
        if "year=" in s3_uri:
            return False
        return True

    monkeypatch.setattr(discovery, "_prefix_exists_in_s3", mock_prefix_exists)

    with pytest.raises(CustomException) as exc_info:
        discovery.run()

    assert "Missing required upstream data" in str(exc_info.value)
    assert "No temporal overlap" in str(exc_info.value)
    assert "orders" in str(exc_info.value)
    assert "customers" in str(exc_info.value)
    assert "order_payments" in str(exc_info.value)


def test_run_partial_temporal_overlap_success(
    pipeline_context: PipelineContext, monkeypatch: pytest.MonkeyPatch
):
    pipeline_context.start_date = "2016-08-01"
    pipeline_context.end_date = "2016-10-31"

    discovery = DataDiscovery(pipeline_context)

    def mock_prefix_exists(s3_uri: str) -> bool:
        if "year=2016/month=09" in s3_uri:
            return True
        if "year=" in s3_uri:
            return False
        return True

    monkeypatch.setattr(discovery, "_prefix_exists_in_s3", mock_prefix_exists)

    discovery.run()


def test_run_exception_propagation(
    pipeline_context: PipelineContext, monkeypatch: pytest.MonkeyPatch
):
    discovery = DataDiscovery(pipeline_context)

    def mock_prefix_exists_raise(s3_uri: str) -> bool:
        raise ValueError("Simulated network failure")

    monkeypatch.setattr(discovery, "_prefix_exists_in_s3", mock_prefix_exists_raise)

    with pytest.raises(CustomException) as exc_info:
        discovery.run()

    assert "Simulated network failure" in str(exc_info.value)