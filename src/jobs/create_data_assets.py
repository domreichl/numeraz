import json

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from numerapi import NumerAPI

from components import parse_args

args: dict = parse_args(
    [
        "data_asset_name",
        "data_asset_version",
        "subscription_id",
        "resource_group",
        "workspace_name",
        "numerai_data_version",
    ]
)
numerapi = NumerAPI()
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=args["subscription_id"],
    resource_group_name=args["resource_group"],
    workspace_name=args["workspace_name"],
)

try:
    data_asset: Data = ml_client.data.get(
        name=args["data_asset_name"], version=args["data_asset_version"]
    )
    print(
        f"Data asset '{args['data_asset_name']}' with version '{args['data_asset_version']}' already exists"
    )
except:
    data_dir = "./data"
    for fn in [
        "features.json",
        "live.parquet",
        "meta_model.parquet",
        "train.parquet",
        "validation.parquet",
    ]:
        numerapi.download_dataset(
            filename=f"{args['numerai_data_version']}/{fn}",
            dest_path=f"{data_dir}/{fn}",
        )
    data_asset: Data = ml_client.data.create_or_update(
        Data(
            name=args["data_asset_name"],
            version=args["data_asset_version"],
            description=f"Numerai {args['numerai_data_version']} Data",
            path=data_dir,
            type="uri_folder",
        )
    )
    print(
        f"Created data asset '{args['data_asset_name']}' with version '{args['data_asset_version']}'"
    )

print("Numerai data path:", data_asset.path)


file_name = "prod_model_info"
try:
    data_asset: Data = ml_client.data.get(name=file_name, version="0")
    print(f"Data asset '{file_name}' already exists")
except:
    with open(file_name + ".json", "w") as file:
        json.dump(
            {
                "MED_TE3_WA": {
                    "date": "2024-10-02",
                    "experiment_name": "WISIGERNO_WA",
                    "run_name": "amusing-squid-772",
                    "run_id": "77ac74d14de342759df4acb3f73f63a8",
                    "main_target": "target_cyrusd_20",
                    "feature_set": "medium",
                    "train_eras": "0301-1129",
                    "test_eras": "0521-0574",
                    "base_model": "lightgbm",
                    "base_model_hparams": {
                        "n_estimators": 15000,
                        "max_depth": 5,
                        "num_leaves": 21,
                        "learning_rate": 0.01,
                        "colsample_bytree": 0.1,
                        "min_child_samples": 200,
                    },
                    "ensemble": "cyrusd-sam-caroline",
                    "ensembling_method": "weighted",
                    "metrics": {
                        "corr_mean": 0.05122,
                        "corr_std": 0.01663,
                        "corr_sharpe": 3.07956,
                        "corr_max_drawdown": 0,
                    },
                }
            },
            file,
        ),
    data_asset: Data = ml_client.data.create_or_update(
        Data(
            name=file_name,
            version="0",
            description="Information about production models",
            path=file_name + ".json",
            type="uri_file",
        )
    )
    print(f"Created data asset '{file_name}'")

print("Prod model info path:", data_asset.path)
