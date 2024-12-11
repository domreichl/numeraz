import json
from dataclasses import dataclass
import datetime as dt
from pathlib import Path


@dataclass
class Config:
    def __init__(self):
        self.src_path = str(Path(__file__).parent)
        self.feature_set = "medium"
        self.experiment_name = "initial"
        self.numerai_data_version = "v5.0"
        self.data_asset_name = "numerai"
        self.data_asset_version = dt.datetime.now().strftime("%Y-%m")
        self.data_asset_uri = (
            f"azureml:{self.data_asset_name}:{self.data_asset_version}"
        )
        with open("resources.json") as file:
            resources = json.load(file)
        self.subscription_id = resources["subscription_id"]
        self.resource_group = resources["resource_group"]
        self.workspace_name = resources["workspace_name"]
        self.compute_instance = resources["compute_instance"]
        self.environment_name = resources["environment_name"]
        self.conda_file_path = str(Path(__file__).parent.parent / "conda.yml")
        self.latest_env_name = None
