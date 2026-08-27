import json

import pytest
from typing import Any
from unittest.mock import MagicMock, patch, mock_open

from pipelines.inference_pipeline.src.components.inference_validator import (
    InferenceValidator,
)
from shared_core.exceptions.custom_exception import CustomException
from pipelines.inference_pipeline.src.entity.artifact_entity import (
    ModelLoaderArtifact,
    FeatureMatrixBuilderArtifact,
)


@patch(
    "pipelines.inference_pipeline.src.components.inference_validator."
    "InferenceValidatorConfig.from_context"
)
def test_inference_validator_initialization_failure(
    mock_from_context: MagicMock,
    mock_pipeline_context: MagicMock,
) -> None:
    mock_from_context.side_effect = Exception(
        "Config initialization failed"
    )

    with pytest.raises(CustomException) as exc_info:
        InferenceValidator(context=mock_pipeline_context)

    assert "Config initialization failed" in str(exc_info.value)


@patch.object(InferenceValidator, "_generate_metadata")
@patch.object(InferenceValidator, "_generate_validation_report")
@patch.object(InferenceValidator, "_validate_data_contract")
@patch.object(InferenceValidator, "_load_schemas")
def test_run_success_valid_contract(
    mock_load_schemas: MagicMock,
    mock_validate: MagicMock,
    mock_generate_report: MagicMock,
    mock_generate_metadata: MagicMock,
    mock_pipeline_context: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
) -> None:
    validator = InferenceValidator(context=mock_pipeline_context)

    mock_load_schemas.return_value = ({}, [])
    mock_validate.return_value = (True, [])

    artifact = validator.run(
        model_loader_artifact=dummy_model_loader_artifact,
        feature_matrix_artifact=dummy_feature_matrix_artifact,
    )

    assert artifact.is_valid is True
    assert (
        artifact.report_file_path
        == validator.config.report_file_path
    )

    mock_load_schemas.assert_called_once_with(
        model_schema_path=dummy_model_loader_artifact.schema_file_path,
        builder_schema_path=dummy_feature_matrix_artifact.schema_file_path,
    )
    mock_validate.assert_called_once()
    mock_generate_report.assert_called_once_with(
        is_valid=True,
        validation_errors=[],
    )
    mock_generate_metadata.assert_called_once()


@patch.object(InferenceValidator, "_generate_metadata")
@patch.object(InferenceValidator, "_generate_validation_report")
@patch.object(InferenceValidator, "_validate_data_contract")
@patch.object(InferenceValidator, "_load_schemas")
def test_run_failure_invalid_contract(
    mock_load_schemas: MagicMock,
    mock_validate: MagicMock,
    mock_generate_report: MagicMock,
    mock_generate_metadata: MagicMock,
    mock_pipeline_context: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
) -> None:
    validator = InferenceValidator(context=mock_pipeline_context)

    mock_load_schemas.return_value = ({}, [])
    mock_validate.return_value = (
        False,
        ["Missing features"],
    )

    artifact = validator.run(
        model_loader_artifact=dummy_model_loader_artifact,
        feature_matrix_artifact=dummy_feature_matrix_artifact,
    )

    assert artifact.is_valid is False
    assert (
        artifact.report_file_path
        == validator.config.report_file_path
    )

    mock_generate_report.assert_called_once_with(
        is_valid=False,
        validation_errors=["Missing features"],
    )


@patch.object(InferenceValidator, "_load_schemas")
def test_run_exception_propagation(
    mock_load_schemas: MagicMock,
    mock_pipeline_context: MagicMock,
    dummy_model_loader_artifact: ModelLoaderArtifact,
    dummy_feature_matrix_artifact: FeatureMatrixBuilderArtifact,
) -> None:
    validator = InferenceValidator(context=mock_pipeline_context)

    mock_load_schemas.side_effect = Exception("Schema load error")

    with pytest.raises(CustomException) as exc_info:
        validator.run(
            model_loader_artifact=dummy_model_loader_artifact,
            feature_matrix_artifact=dummy_feature_matrix_artifact,
        )

    assert "Schema load error" in str(exc_info.value)


def test_load_schemas_success(
    mock_pipeline_context: MagicMock,
) -> None:
    validator = InferenceValidator(context=mock_pipeline_context)

    model_schema_content = (
        '{"features": {"numerical": ["f1"]}}'
    )
    builder_schema_content = (
        '[{"name": "f1", "physical_type": "INT"}]'
    )

    def side_effect_open(
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "model_schema" in path:
            return mock_open(
                read_data=model_schema_content
            )()

        return mock_open(
            read_data=builder_schema_content
        )()

    with patch("builtins.open") as mock_file:
        mock_file.side_effect = side_effect_open

        model_schema, builder_schema = validator._load_schemas(
            model_schema_path="model_schema.json",
            builder_schema_path="builder_schema.json",
        )

    assert model_schema == {
        "features": {
            "numerical": ["f1"]
        }
    }

    assert builder_schema == [
        {
            "name": "f1",
            "physical_type": "INT",
        }
    ]


def test_load_schemas_failure(
    mock_pipeline_context: MagicMock,
) -> None:
    validator = InferenceValidator(context=mock_pipeline_context)

    with patch(
        "builtins.open",
        side_effect=FileNotFoundError("File missing"),
    ):
        with pytest.raises(CustomException) as exc_info:
            validator._load_schemas(
                "fake_1",
                "fake_2",
            )

    assert "File missing" in str(exc_info.value)


def test_validate_data_contract_success(
    mock_pipeline_context: MagicMock,
) -> None:
    validator = InferenceValidator(context=mock_pipeline_context)

    model_schema = {
        "features": {
            "numerical": [
                "recency_days",
                "frequency",
            ],
            "categorical": {
                "customer_state": [
                    "NY",
                    "CA",
                ]
            },
        }
    }

    builder_schema = [
        {
            "name": "customer_unique_id",
            "physical_type": "VARCHAR",
        },
        {
            "name": "snapshot_date",
            "physical_type": "VARCHAR",
        },
        {
            "name": "recency_days",
            "physical_type": "BIGINT",
        },
        {
            "name": "frequency",
            "physical_type": "BIGINT",
        },
        {
            "name": "customer_state",
            "physical_type": "VARCHAR",
        },
    ]

    is_valid, errors = validator._validate_data_contract(
        model_schema=model_schema,
        builder_schema=builder_schema,
    )

    assert is_valid is True
    assert len(errors) == 0


def test_validate_data_contract_missing_features(
    mock_pipeline_context: MagicMock,
) -> None:
    validator = InferenceValidator(context=mock_pipeline_context)

    model_schema = {
        "features": {
            "numerical": [
                "recency_days",
                "frequency",
                "monetary_total",
            ],
            "categorical": {},
        }
    }

    builder_schema = [
        {
            "name": "customer_unique_id",
            "physical_type": "VARCHAR",
        },
        {
            "name": "snapshot_date",
            "physical_type": "VARCHAR",
        },
        {
            "name": "recency_days",
            "physical_type": "BIGINT",
        },
    ]

    is_valid, errors = validator._validate_data_contract(
        model_schema=model_schema,
        builder_schema=builder_schema,
    )

    assert is_valid is False
    assert len(errors) == 1
    assert (
        "missing predictive features"
        in errors[0].lower()
    )
    assert "frequency" in errors[0]
    assert "monetary_total" in errors[0]


def test_validate_data_contract_missing_system_columns(
    mock_pipeline_context: MagicMock,
) -> None:
    validator = InferenceValidator(context=mock_pipeline_context)

    model_schema = {
        "features": {
            "numerical": ["recency_days"],
            "categorical": {},
        }
    }

    builder_schema = [
        {
            "name": "snapshot_date",
            "physical_type": "VARCHAR",
        },
        {
            "name": "recency_days",
            "physical_type": "BIGINT",
        },
    ]

    is_valid, errors = validator._validate_data_contract(
        model_schema=model_schema,
        builder_schema=builder_schema,
    )

    assert is_valid is False
    assert len(errors) == 1
    assert (
        "missing system/entity columns"
        in errors[0].lower()
    )
    assert "customer_unique_id" in errors[0]


def test_generate_validation_report_success(
    mock_pipeline_context: MagicMock,
) -> None:
    validator = InferenceValidator(context=mock_pipeline_context)

    m_open = mock_open()

    with patch("builtins.open", m_open):
        validator._generate_validation_report(
            is_valid=False,
            validation_errors=["Error 1"],
        )

    m_open.assert_called_once_with(
        validator.config.report_file_path,
        "w",
        encoding="utf-8",
    )

    handle = m_open()
    written_data = "".join(
        call.args[0]
        for call in handle.write.call_args_list
    )

    parsed_report = json.loads(written_data)

    assert parsed_report["validation_status"] == "FAILED"
    assert parsed_report["errors"] == ["Error 1"]
    assert "timestamp_utc" in parsed_report


def test_generate_metadata_success(
    mock_pipeline_context: MagicMock,
) -> None:
    validator = InferenceValidator(context=mock_pipeline_context)

    execution_time = 0.42
    m_open = mock_open()

    with patch("builtins.open", m_open):
        validator._generate_metadata(
            is_valid=True,
            execution_time=execution_time,
        )

    m_open.assert_called_once_with(
        validator.config.metadata_file_path,
        "w",
        encoding="utf-8",
    )

    handle = m_open()
    written_data = "".join(
        call.args[0]
        for call in handle.write.call_args_list
    )

    parsed_metadata = json.loads(written_data)

    assert (
        parsed_metadata["pipeline_stage"]
        == "Inference Validator"
    )
    assert (
        parsed_metadata["inference_run_id"]
        == mock_pipeline_context.run_id
    )
    assert (
        parsed_metadata["execution_time_seconds"]
        == execution_time
    )
    assert (
        parsed_metadata["validation_outcome"]["is_valid"]
        is True
    )
    assert (
        parsed_metadata["validation_outcome"]["report_file"]
        == validator.config.report_file_path
    )