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
    data_asset = ml_client.data.get(
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
    data_asset = ml_client.data.create_or_update(
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

print("Path:", data_asset.path)
