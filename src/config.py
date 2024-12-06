import json
from dataclasses import dataclass
import datetime as dt
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from pathlib import Path


@dataclass
class Config:
    def __init__(self):
        self.src_path = str(Path(__file__).parent)
        self.experiment_name = "initial"
        self.numerai_data_version = "v5.0"
        self.data_asset_name = "numerai"
        self.data_asset_version = dt.datetime.now().strftime("%Y-%m")
        with open("resources.json") as file:
            resources = json.load(file)
        self.subscription_id = resources["subscription_id"]
        self.resource_group = resources["resource_group"]
        self.workspace_name = resources["workspace_name"]
        self.compute_instance = resources["compute_instance"]
        self.environment_name = resources["environment_name"]

    def get_ml_client(self) -> MLClient:
        return MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name,
        )

    def get_latest_env(self, ml_client: MLClient) -> str:
        env: Environment = max(
            ml_client.environments.list(self.environment_name),
            key=lambda env: env.version,
        )
        return f"{env.name}:{env.version}"
