import json
from dataclasses import dataclass
import datetime as dt
from azure.identity import DefaultAzureCredential
from typing import TypedDict


class MLClientConfig(TypedDict):
    credential: DefaultAzureCredential
    subscription_id: str
    resource_group_name: str
    workspace_name: str


@dataclass
class Config:
    def __init__(self):
        self.numerai_data_version = "v5.0"
        self.current_version = dt.datetime.now().strftime("%Y-%m")
        with open("resources.json") as file:
            resources = json.load(file)
        self.subscription_id = resources["subscription_id"]
        self.resource_group = resources["resource_group"]
        self.workspace_name = resources["workspace_name"]
        self.compute_instance = resources["compute_instance"]
        self.environment_name = resources["environment_name"]
        self.ml_client_config = MLClientConfig(
            credential=DefaultAzureCredential(),
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name,
        )
