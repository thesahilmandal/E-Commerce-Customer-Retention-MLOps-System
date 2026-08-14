import os
import yaml
import pytest

from pipelines.inference_pipeline.src.core.config_parser import ConfigParser
from shared_core.exceptions.custom_exception import CustomException


def test_config_parser_successful_initialization(mock_global_config_path):
    """
    Tests that the ConfigParser successfully initializes and parses a valid
    configuration file without raising any exceptions.
    """
    parser = ConfigParser(config_path=mock_global_config_path)

    assert parser.config is not None
    assert isinstance(parser.config, dict)

    for key in ConfigParser.REQUIRED_TOP_LEVEL_KEYS:
        assert key in parser.config


def test_config_parser_file_not_found(temp_workspace):
    """
    Tests that a CustomException is raised when the configuration file does not exist.
    """
    non_existent_path = os.path.join(temp_workspace, "non_existent_config.yaml")

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_path=non_existent_path)

    assert "not found" in str(exc_info.value).lower()


def test_config_parser_invalid_yaml_type(temp_workspace):
    """
    Tests that a CustomException is raised if the YAML file is successfully parsed
    but the resulting object is not a dictionary (e.g., a list).
    """
    invalid_yaml_path = os.path.join(temp_workspace, "invalid_type.yaml")

    with open(invalid_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(["this", "is", "a", "list", "not", "a", "dict"], f)

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_path=invalid_yaml_path)

    assert "must be a dictionary" in str(exc_info.value).lower()


def test_config_parser_malformed_yaml(temp_workspace):
    """
    Tests that a CustomException is raised if the YAML file is structurally malformed
    and cannot be parsed by yaml.safe_load.
    """
    malformed_yaml_path = os.path.join(temp_workspace, "malformed.yaml")

    with open(malformed_yaml_path, "w", encoding="utf-8") as f:
        f.write("system: \n  - item1\n - item2\nbad_indentation")

    with pytest.raises(CustomException):
        ConfigParser(config_path=malformed_yaml_path)


def test_config_parser_missing_required_keys(temp_workspace):
    """
    Tests that a CustomException is raised if the parsed dictionary is missing
    any of the contractually required top-level keys.
    """
    missing_keys_yaml_path = os.path.join(temp_workspace, "missing_keys.yaml")

    incomplete_config = {
        "system": {"pipeline_name": "inference_pipeline"},
        "business_logic": {"target_column": "target_is_churn"}
        # Intentionally omitting 'cloud_storage' and 'components'
    }

    with open(missing_keys_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(incomplete_config, f)

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_path=missing_keys_yaml_path)

    assert "missing required top-level keys" in str(exc_info.value).lower()


def test_config_parser_getters_return_correct_data(mock_global_config_path):
    """
    Tests that all getter methods return the correct sub-dictionaries from the parsed configuration.
    """
    parser = ConfigParser(config_path=mock_global_config_path)

    system_cfg = parser.get_system_config()
    assert isinstance(system_cfg, dict)
    assert system_cfg["pipeline_name"] == "inference_pipeline"

    business_cfg = parser.get_business_logic_config()
    assert isinstance(business_cfg, dict)
    assert business_cfg["target_column"] == "target_is_churn"
    assert business_cfg["churn_probability_threshold"] == 0.5

    cloud_cfg = parser.get_cloud_storage_config()
    assert isinstance(cloud_cfg, dict)
    assert "data_lake" in cloud_cfg
    assert "model_registry" in cloud_cfg

    components_cfg = parser.get_components_config()
    assert isinstance(components_cfg, dict)
    assert "model_loader" in components_cfg
    assert "feature_matrix_builder" in components_cfg