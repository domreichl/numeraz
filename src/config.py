import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    # PARAMS
    model_name = "AENCAZ"  # "WISIGERNAZ"
    experiment_name = "aencaz"  # "initial"
    feature_set = "aenc"  # "small"
    hparams = {
        "n_estimators": 15000,  # 8000,
        "max_depth": 5,
        "num_leaves": 21,
        "learning_rate": 0.01,
        "colsample_bytree": 0.1,
        "min_child_samples": 200,
    }
    encoder_uri = "models:/autoenc_orig/4"
    component_inputs = {
        "zero_cols": "azureml:azureml_6f0d9d80-eda8-4a6e-974a-15346b4583af_output_data_zero_cols:1",
        "train_data": "azureml:azureml_6f0d9d80-eda8-4a6e-974a-15346b4583af_output_data_train_data:1",
        "test_data": "azureml:azureml_6f0d9d80-eda8-4a6e-974a-15346b4583af_output_data_test_data:1",
        "base_models_dir": "azureml:azureml_4c00d3c8-608e-421c-a314-71e78007f769_output_data_base_models_dir:1",
        "best_ensemble": "azureml:azureml_85c3a023-90d1-4570-b5ef-4a60c3f9a06f_output_data_best_ensemble:1",
        # "train_data": "azureml:azureml_994d2f8d-a4d6-410f-8bb1-e62d14dfc5d6_output_data_train_data:1",
        # "test_data": "azureml:azureml_994d2f8d-a4d6-410f-8bb1-e62d14dfc5d6_output_data_test_data:1",
        # "base_models_dir": "azureml:azureml_cca8ff38-4ad2-46d0-8420-40016e50bec6_output_data_base_models_dir:1",
        # "best_ensemble": "azureml:azureml_bb3d40d4-e0d7-4340-bc8b-b68770b97c2f_output_data_best_ensemble:1",
    }

    # DATA
    main_target = "target_cyrusd_20"
    numerai_data_version = "v5.0"
    data_asset_name = "numerai"
    data_asset_version = dt.datetime.now().strftime("%Y-%m")
    data_asset_uri = f"azureml:{data_asset_name}:{data_asset_version}"
    prod_model_info = "azureml:prod_model_info:0"

    # AZURE
    if not os.environ.get("SUBSCRIPTION_ID"):
        load_dotenv()
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group = os.environ.get("RESOURCE_GROUP")
    workspace_name = os.environ.get("WORKSPACE_NAME")
    compute_instance = os.environ.get("COMPUTE_INSTANCE")
    environment_name = os.environ.get("ENVIRONMENT_NAME")
    latest_env_name = ""
    src_path = str(Path(__file__).parent)
