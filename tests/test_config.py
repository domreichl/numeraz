from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, Workspace

from src.config import Config


config = Config()
ml_client: MLClient = config.get_ml_client()


def test_ml_client():
    ws: Workspace = ml_client.workspaces.get("numeraz-ws")

    assert ws.location == "germanywestcentral"
    assert ws.resource_group == "numeraz-rg"


def test_get_latest_env():
    env: Environment = config.get_latest_env(ml_client)

    assert env.name == "numeraz-env"


def test_get_latest_env_name():
    env_name: str = config.get_latest_env_name(ml_client)
    name, version = env_name.split(":")

    assert name == "numeraz-env"
    assert int(version) >= 0
