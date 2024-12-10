from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, Workspace

from src import utils
from src.config import Config


config = Config()
ml_client: MLClient = utils.get_ml_client(config)


def test_ml_client():
    ws: Workspace = ml_client.workspaces.get("numeraz-ws")

    assert ws.location == "germanywestcentral"
    assert ws.resource_group == "numeraz-rg"


def test_get_latest_env():
    env: Environment = utils.get_latest_env(config, ml_client)

    assert env.name == "numeraz-env"


def test_set_latest_env_name():
    config.latest_env_name = utils.get_latest_env_name(config, ml_client)
    name, version = config.latest_env_name.split(":")

    assert name == "numeraz-env"
    assert int(version) >= 0
