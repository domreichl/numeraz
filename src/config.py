import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # PARAMS
    experiment_name = "initial"
    feature_set = "small"  # "medium"
    hparams = {
        "n_estimators": 10000,  # 15000,
        "max_depth": 5,
        "num_leaves": 21,
        "learning_rate": 0.01,
        "colsample_bytree": 0.1,
        "min_child_samples": 200,
    }

    # DATA
    main_target = "target_cyrusd_20"
    numerai_data_version = "v5.0"
    data_asset_name = "numerai"
    data_asset_version = dt.datetime.now().strftime("%Y-%m")
    data_asset_uri = f"azureml:{data_asset_name}:{data_asset_version}"

    # AZURE
    with open("resources.json") as file:
        resources = json.load(file)
    subscription_id = resources["subscription_id"]
    resource_group = resources["resource_group"]
    workspace_name = resources["workspace_name"]
    compute_instance = resources["compute_instance"]
    environment_name = resources["environment_name"]

    latest_env_name = ""

    # PATHS
    src_path = str(Path(__file__).parent)
    conda_file_path = str(Path(__file__).parent.parent / "conda.yml")
