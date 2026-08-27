
import pytest
from unittest.mock import MagicMock, patch, mock_open

from pipelines.inference_pipeline.src.core.config_parser import ConfigParser
from shared_core.exceptions.custom_exception import CustomException


VALID_YAML_CONTENT = """
system:
  artifact_dir: "artifacts"

business_logic:
  target_column: "churn"

cloud_storage:
  data_lake: "s3://lake"

components:
  model_loader: {}
"""


@patch("pipelines.inference_pipeline.src.core.config_parser.load_dotenv")
@patch(
    "pipelines.inference_pipeline.src.core.config_parser.os.path.exists"
)
def test_config_parser_initialization_success(
    mock_exists: MagicMock,
    mock_load_dotenv: MagicMock
) -> None:
    mock_exists.return_value = True

    with patch(
        "builtins.open",
        mock_open(read_data=VALID_YAML_CONTENT),
    ):
        parser = ConfigParser(config_path="dummy.yaml")

        assert parser.get_system_config() == {
            "artifact_dir": "artifacts"
        }
        assert parser.get_business_logic_config() == {
            "target_column": "churn"
        }
        assert parser.get_cloud_storage_config() == {
            "data_lake": "s3://lake"
        }
        assert parser.get_components_config() == {
            "model_loader": {}
        }


@patch("pipelines.inference_pipeline.src.core.config_parser.load_dotenv")
@patch(
    "pipelines.inference_pipeline.src.core.config_parser.os.path.exists"
)
def test_config_parser_file_not_found(
    mock_exists: MagicMock,
    mock_load_dotenv: MagicMock
) -> None:
    mock_exists.return_value = False

    with pytest.raises(CustomException) as exc_info:
        ConfigParser(config_path="missing.yaml")

    assert "not found" in str(exc_info.value).lower()


@patch("pipelines.inference_pipeline.src.core.config_parser.load_dotenv")
@patch(
    "pipelines.inference_pipeline.src.core.config_parser.os.path.exists"
)
def test_config_parser_invalid_yaml_format(
    mock_exists: MagicMock,
    mock_load_dotenv: MagicMock
) -> None:
    mock_exists.return_value = True

    invalid_yaml = "- system\n- business_logic\n"

    with patch(
        "builtins.open",
        mock_open(read_data=invalid_yaml),
    ):
        with pytest.raises(CustomException) as exc_info:
            ConfigParser(config_path="dummy.yaml")

        assert "must be a dictionary" in str(exc_info.value).lower()


@patch("pipelines.inference_pipeline.src.core.config_parser.load_dotenv")
@patch(
    "pipelines.inference_pipeline.src.core.config_parser.os.path.exists"
)
def test_config_parser_missing_required_keys(
    mock_exists: MagicMock,
    mock_load_dotenv: MagicMock
) -> None:
    mock_exists.return_value = True

    missing_keys_yaml = "system: {}\n"

    with patch(
        "builtins.open",
        mock_open(read_data=missing_keys_yaml),
    ):
        with pytest.raises(CustomException) as exc_info:
            ConfigParser(config_path="dummy.yaml")

        assert (
            "missing required top-level keys"
            in str(exc_info.value).lower()
        )


@patch("pipelines.inference_pipeline.src.core.config_parser.load_dotenv")
@patch("pipelines.inference_pipeline.src.core.config_parser.os.path.exists")
def test_resolve_env_vars(
    mock_exists: MagicMock,
    mock_load_dotenv: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:    
    mock_exists.return_value = True
    
    monkeypatch.setenv(
        "EXISTING_VAR",
        "injected_artifact_dir",
    )

    env_yaml = """
system:
  artifact_dir: "${EXISTING_VAR}"

business_logic:
  target_column: "${MISSING_VAR:-default_target}"

cloud_storage:
  data_lake: "${MISSING_NO_DEFAULT}"

components: {}
"""

    with patch(
        "builtins.open",
        mock_open(read_data=env_yaml),
    ):
        parser = ConfigParser(config_path="dummy.yaml")

        sys_config = parser.get_system_config()
        assert sys_config["artifact_dir"] == "injected_artifact_dir"

        biz_config = parser.get_business_logic_config()
        assert biz_config["target_column"] == "default_target"

        cloud_config = parser.get_cloud_storage_config()
        assert cloud_config["data_lake"] == ""