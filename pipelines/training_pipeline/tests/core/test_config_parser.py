import pytest
from pathlib import Path

from pipelines.training_pipeline.src.core.config_parser import ConfigParser
from shared_core.exceptions.custom_exception import CustomException


def test_config_parser_success(
    env_vars: None,
    dummy_yaml_config: str,
) -> None:
    parser = ConfigParser(config_filepath=dummy_yaml_config)

    global_cfg = parser.get_global_config()
    assert global_cfg["s3_bucket_name"] == "test-pipeline-artifacts-bucket"
    assert global_cfg["target_column"] == "target_is_churn"

    dp_cfg = parser.get_data_processor_config()
    assert dp_cfg["val_size"] == 0.15
    assert "snapshot_date" in dp_cfg["system_columns_to_drop"]

    mt_cfg = parser.get_model_trainer_config()
    assert mt_cfg["optuna_n_trials"] == 2
    assert mt_cfg["calibration_method"] == "isotonic"

    me_cfg = parser.get_model_evaluator_config()
    assert me_cfg["min_eroi_threshold"] == 0.05
    assert me_cfg["business_assumptions"]["campaign_cost"] == 10.0

    mr_cfg = parser.get_model_registry_config()
    assert mr_cfg["deployment_environment"] == "testing"


def test_missing_config_file() -> None:
    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_filepath="nonexistent_config.yaml")

    assert "not found at" in str(exc_info.value)
    assert "nonexistent_config.yaml" in str(exc_info.value)


def test_empty_config_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_filepath=str(empty_file))

    assert "empty or invalid" in str(exc_info.value)


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("global: [unclosed list")

    with pytest.raises(CustomException):
        ConfigParser(config_filepath=str(invalid_file))


def test_missing_required_section(tmp_path: Path) -> None:
    yaml_content = """
training_pipeline:
global: {}
data_processor: {}
model_trainer: {}
model_registry: {}
"""

    bad_file = tmp_path / "missing_section.yaml"
    bad_file.write_text(yaml_content)

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_filepath=str(bad_file))

    assert "Missing required configuration sections" in str(exc_info.value)
    assert "'model_evaluator'" in str(exc_info.value)


def test_empty_section_raises_validation_error(tmp_path: Path) -> None:
    yaml_content = """
training_pipeline:
global: {}
data_processor: {}
model_trainer: {}
model_evaluator: {}
model_registry:
"""

    bad_file = tmp_path / "empty_section.yaml"
    bad_file.write_text(yaml_content)

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_filepath=str(bad_file))

    assert "are empty or invalid" in str(exc_info.value)
    assert "missing YAML indentation" in str(exc_info.value)


def test_resolve_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXISTING_VAR", "injected_value")

    yaml_content = """
global:
  s3_bucket_name: "${EXISTING_VAR}"
data_processor:
  val_size: "${MISSING_VAR_WITH_DEFAULT:-0.2}"
model_trainer:
  mlflow_experiment_name: "${MISSING_VAR}"
model_evaluator: {}
model_registry: {}
"""

    file_path = tmp_path / "env.yaml"
    file_path.write_text(yaml_content)

    parser = ConfigParser(config_filepath=str(file_path))

    assert parser.get_global_config()["s3_bucket_name"] == "injected_value"
    assert parser.get_data_processor_config()["val_size"] == "0.2"
    assert parser.get_model_trainer_config()["mlflow_experiment_name"] == ""


def test_fallback_without_root_namespace(tmp_path: Path) -> None:
    yaml_content = """
global:
  key: "val"
data_processor: {}
model_trainer: {}
model_evaluator: {}
model_registry: {}
"""

    file_path = tmp_path / "no_namespace.yaml"
    file_path.write_text(yaml_content)

    parser = ConfigParser(config_filepath=str(file_path))

    assert parser.get_global_config()["key"] == "val"