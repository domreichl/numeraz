from azure.ai.ml.entities import Data, Environment, Workspace

from src.api import NumerazAPI


api = NumerazAPI()


def test_ml_client():
    ws: Workspace = api.ml_client.workspaces.get("numeraz-ws")

    assert ws.location == "germanywestcentral"
    assert ws.resource_group == "numeraz-rg"


def test_get_data_asset():
    data: Data = api._get_data_asset()

    assert data.name == "numerai"


def test_get_latest_env():
    env: Environment = api._get_latest_env()

    assert env.name == "numeraz-env"


def test_get_latest_env_name():
    env_name: str = api._get_latest_env_name()
    name, version = env_name.split(":")

    assert name == "numeraz-env"
    assert int(version) >= 0
