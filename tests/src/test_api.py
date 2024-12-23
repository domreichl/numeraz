import pytest
from azure.ai.ml.entities import Data, Environment, Workspace

from src.api import NumerazAPI


@pytest.fixture
def numeraz_api() -> NumerazAPI:
    return NumerazAPI()


def test_ml_client(numeraz_api: NumerazAPI):
    ws: Workspace = numeraz_api.ml_client.workspaces.get("numeraz-ws")

    assert ws.location == "germanywestcentral"
    assert ws.resource_group == "numeraz-rg"


def test_get_data_asset(numeraz_api: NumerazAPI):
    data: Data = numeraz_api._get_data_asset()

    assert data.name == "numerai"


def test_get_latest_env(numeraz_api: NumerazAPI):
    env: Environment = numeraz_api._get_latest_env()

    assert env.name == "numeraz-env"


def test_get_latest_env_name(numeraz_api: NumerazAPI):
    env_name: str = numeraz_api._get_latest_env_name()
    name, version = env_name.split(":")

    assert name == "numeraz-env"
    assert int(version) >= 0
