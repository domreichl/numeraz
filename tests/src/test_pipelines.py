import pytest
from unittest.mock import MagicMock


from src.pipelines import Pipelines


@pytest.fixture
def mock_config():
    mock = MagicMock()
    mock.compute_instance = "test-compute"
    mock.experiment_name = "test-experiment"
    mock.data_asset_uri = "test-uri"
    return mock


@pytest.fixture
def mock_ml_client():
    return MagicMock()


@pytest.fixture
def pipelines(mock_config, mock_ml_client):
    return Pipelines(config=mock_config, ml_client=mock_ml_client)


def test_get_pipeline(pipelines: Pipelines):
    mock_pipeline = MagicMock()
    pipelines._model_training = MagicMock(return_value=mock_pipeline)
    result = pipelines.get_pipeline("model_training")

    assert result == mock_pipeline
    pipelines._model_training.assert_called_once()
