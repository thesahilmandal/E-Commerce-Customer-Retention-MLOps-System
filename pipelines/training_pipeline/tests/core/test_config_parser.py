import yaml
import pytest

from pipelines.training_pipeline.src.core.config_parser import ConfigParser
from shared_core.exceptions.custom_exception import CustomException


def test_config_parser_initialization_success(mock_config_parser, sample_config_dict):
    global_config = mock_config_parser.get_global_config()
    data_processor_config = mock_config_parser.get_data_processor_config()
    model_trainer_config = mock_config_parser.get_model_trainer_config()
    model_evaluator_config = mock_config_parser.get_model_evaluator_config()
    model_registry_config = mock_config_parser.get_model_registry_config()

    expected_config = sample_config_dict["training_pipeline"]

    assert global_config["s3_bucket_name"] == expected_config["global"]["s3_bucket_name"]
    assert data_processor_config["val_size"] == expected_config["data_processor"]["val_size"]
    assert model_trainer_config["mlflow_experiment_name"] == expected_config["model_trainer"]["mlflow_experiment_name"]
    assert model_evaluator_config["min_eroi_threshold"] == expected_config["model_evaluator"]["min_eroi_threshold"]
    assert model_registry_config["deployment_environment"] == expected_config["model_registry"]["deployment_environment"]


def test_config_parser_missing_file():
    missing_path = "non_existent_config_file.yaml"

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_filepath=missing_path)

    assert "Configuration file not found at" in str(exc_info.value)


def test_config_parser_empty_file(tmp_path):
    empty_file_path = tmp_path / "empty_config.yaml"
    empty_file_path.touch()

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_filepath=str(empty_file_path))

    assert "Configuration file is empty or invalid" in str(exc_info.value)


def test_config_parser_missing_required_section(tmp_path, sample_config_dict):
    del sample_config_dict["training_pipeline"]["model_trainer"]

    malformed_config_path = tmp_path / "missing_section_config.yaml"

    with open(malformed_config_path, "w") as f:
        yaml.dump(sample_config_dict, f)

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_filepath=str(malformed_config_path))

    assert "Missing required configuration sections" in str(exc_info.value)
    assert "model_trainer" in str(exc_info.value)


def test_config_parser_invalid_section_type(tmp_path, sample_config_dict):
    sample_config_dict["training_pipeline"]["data_processor"] = None

    invalid_section_config_path = tmp_path / "invalid_section_config.yaml"

    with open(invalid_section_config_path, "w") as f:
        yaml.dump(sample_config_dict, f)

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_filepath=str(invalid_section_config_path))

    assert "are empty or invalid" in str(exc_info.value)
    assert "data_processor" in str(exc_info.value)


def test_config_parser_fallback_to_root_namespace(tmp_path, sample_config_dict):
    root_config = sample_config_dict["training_pipeline"]

    root_namespace_config_path = tmp_path / "root_namespace_config.yaml"

    with open(root_namespace_config_path, "w") as f:
        yaml.dump(root_config, f)

    parser = ConfigParser(config_filepath=str(root_namespace_config_path))

    global_config = parser.get_global_config()

    assert global_config["target_column"] == root_config["global"]["target_column"]