import json
import datetime as dt
from azure.identity import DefaultAzureCredential
from typing import TypedDict


class MLClientConfig(TypedDict):
    credential: DefaultAzureCredential
    subscription_id: str
    resource_group_name: str
    workspace_name: str


class Config(TypedDict):
    subscription_id: str
    resource_group: str
    workspace_name: str
    compute_instance: str
    local_data_dir: str
    numerai_data_version: str
    current_version: str
    ml_client_config: MLClientConfig


def load_config() -> Config:
    LOCAL_DATA_DIR = "./data"
    NUMERAI_DATA_VERSION = "v5.0"

    with open("resources.json") as cfg_file:
        config = Config(
            **json.load(cfg_file),
            local_data_dir=LOCAL_DATA_DIR,
            numerai_data_version=NUMERAI_DATA_VERSION,
            current_version=dt.datetime.now().strftime("%Y-%m"),
        )
    config["ml_client_config"] = MLClientConfig(
        credential=DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )

    return config
