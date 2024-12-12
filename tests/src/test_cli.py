import pytest
from click.testing import CliRunner
from unittest.mock import patch

from src.cli import cli


@pytest.fixture
def mock_api():
    with patch("src.cli.api") as MockAPI:
        yield MockAPI


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def components():
    return ["create_data_asset", "preprocess_data"]


@pytest.fixture
def pipelines():
    return ["model_training"]


def test_job_command(mock_api, runner, components):
    mock_api.return_value
    for component in components:
        result = runner.invoke(cli, ["job", component])

        assert result.exit_code == 0


def test_component_command(mock_api, runner, components):
    mock_api.return_value
    for component in components:
        result = runner.invoke(cli, ["component", component])

        assert result.exit_code == 0


def test_pipeline_command(mock_api, runner, pipelines):
    mock_api.return_value
    for pipeline in pipelines:
        result = runner.invoke(cli, ["pipeline", pipeline])

        assert result.exit_code == 0


def test_update_command_conda(mock_api, runner):
    mock_api.return_value
    result = runner.invoke(cli, ["update", "conda"])

    assert result.exit_code == 0


def test_update_command_unsupported_entity(mock_api, runner):
    mock_api.return_value
    result = runner.invoke(cli, ["update", "unsupported-entity"])

    assert result.exit_code != 0
    assert pytest.raises(NotImplementedError)
