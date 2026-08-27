import pytest

from typing import Any, Dict, Set
from unittest.mock import MagicMock

from pipelines.data_pipeline.src.components.data_discovery import DataDiscovery
from shared_core.exceptions.custom_exception import CustomException


@pytest.mark.parametrize(
    "start_date, end_date, expected",
    [
        (
            "2023-01-15",
            "2023-03-10",
            {("2023", "01"), ("2023", "02"), ("2023", "03")},
        ),
        (
            "2023-12-25",
            "2024-02-05",
            {("2023", "12"), ("2024", "01"), ("2024", "02")},
        ),
        (
            "2023-06-01 10:00:00",
            "2023-06-01 12:00:00",
            {("2023", "06")},
        ),
        (
            "2023-01-31",
            "2023-01-31",
            {("2023", "01")},
        ),
    ],
)
def test_get_required_months(
    start_date: str,
    end_date: str,
    expected: Set[tuple],
    mock_pipeline_context: MagicMock,
) -> None:
    mock_pipeline_context.start_date = start_date
    mock_pipeline_context.end_date = end_date

    discovery = DataDiscovery(mock_pipeline_context)

    assert discovery._get_required_months() == expected


@pytest.mark.parametrize(
    "mock_response, expected_result",
    [
        ({"Contents": [{"Key": "test.parquet"}]}, True),
        ({"CommonPrefixes": [{"Prefix": "year=2023/"}]}, True),
        ({}, False),
        ({"ResponseMetadata": {"HTTPStatusCode": 200}}, False),
    ],
)
def test_prefix_exists_in_s3(
    mock_response: Dict[str, Any],
    expected_result: bool,
    mock_pipeline_context: MagicMock,
) -> None:
    s3_client = mock_pipeline_context.s3_sync._get_client.return_value
    s3_client.list_objects_v2.return_value = mock_response

    discovery = DataDiscovery(mock_pipeline_context)

    result = discovery._prefix_exists_in_s3(
        "s3://test-bucket/test-prefix"
    )

    assert result is expected_result
    s3_client.list_objects_v2.assert_called_once_with(
        Bucket="test-bucket",
        Prefix="test-prefix/",
        MaxKeys=1,
    )


def test_run_success(
    mock_pipeline_context: MagicMock,
) -> None:
    s3_client = mock_pipeline_context.s3_sync._get_client.return_value

    # Always return that content exists to simulate a valid directory.
    s3_client.list_objects_v2.return_value = {
        "Contents": [{"Key": "dummy"}]
    }

    discovery = DataDiscovery(mock_pipeline_context)
    discovery.run()

    # 3 datasets configured. Each needs 1 root check + at least 1
    # month check (returns True immediately).
    assert s3_client.list_objects_v2.call_count == 6


def test_run_missing_dataset_root(
    mock_pipeline_context: MagicMock,
) -> None:
    s3_client = mock_pipeline_context.s3_sync._get_client.return_value

    def list_objects_v2_side_effect(
        Bucket: str,
        Prefix: str,
        MaxKeys: int,
    ) -> dict:
        # Simulate 'orders' dataset missing completely.
        if "orders/" in Prefix:
            return {}

        return {"Contents": [{"Key": "dummy"}]}

    s3_client.list_objects_v2.side_effect = list_objects_v2_side_effect

    discovery = DataDiscovery(mock_pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        discovery.run()

    assert "Missing required upstream data" in str(exc_info.value)
    assert "'orders'" in str(exc_info.value)


def test_run_missing_temporal_partitions(
    mock_pipeline_context: MagicMock,
) -> None:
    s3_client = mock_pipeline_context.s3_sync._get_client.return_value

    def list_objects_v2_side_effect(
        Bucket: str,
        Prefix: str,
        MaxKeys: int,
    ) -> dict:
        # Simulate root exists, but no temporal partitions match
        # the required timeframe.
        if "year=" in Prefix or "month=" in Prefix:
            return {}

        return {"Contents": [{"Key": "dummy"}]}

    s3_client.list_objects_v2.side_effect = list_objects_v2_side_effect

    discovery = DataDiscovery(mock_pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        discovery.run()

    assert "No temporal overlap" in str(exc_info.value)
    assert "orders" in str(exc_info.value)
    assert "customers" in str(exc_info.value)
    assert "order_payments" in str(exc_info.value)


def test_run_s3_client_exception(
    mock_pipeline_context: MagicMock,
) -> None:
    s3_client = mock_pipeline_context.s3_sync._get_client.return_value

    s3_client.list_objects_v2.side_effect = Exception(
        "AWS IAM Access Denied"
    )

    discovery = DataDiscovery(mock_pipeline_context)

    with pytest.raises(CustomException) as exc_info:
        discovery.run()

    assert "AWS IAM Access Denied" in str(exc_info.value)