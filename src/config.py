import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # PARAMS
    model_name = "WISIGERNAZ"
    experiment_name = "initial"
    feature_set = "small"  # "medium"
    hparams = {
        "n_estimators": 9000,  # 15000,
        "max_depth": 5,
        "num_leaves": 21,
        "learning_rate": 0.01,
        "colsample_bytree": 0.1,
        "min_child_samples": 200,
    }
    component_inputs = {
        "train_data": "azureml:azureml_994d2f8d-a4d6-410f-8bb1-e62d14dfc5d6_output_data_train_data:1",
        "test_data": "azureml:azureml_994d2f8d-a4d6-410f-8bb1-e62d14dfc5d6_output_data_test_data:1",
        "base_models_dir": "azureml:azureml_cca8ff38-4ad2-46d0-8420-40016e50bec6_output_data_base_models_dir:1",
        "best_ensemble": "azureml:azureml_bb3d40d4-e0d7-4340-bc8b-b68770b97c2f_output_data_best_ensemble:1",
    }

    # DATA
    main_target = "target_cyrusd_20"
    numerai_data_version = "v5.0"
    data_asset_name = "numerai"
    data_asset_version = dt.datetime.now().strftime("%Y-%m")
    data_asset_uri = f"azureml:{data_asset_name}:{data_asset_version}"
    prod_model_info = "azureml:prod_model_info:0"

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
