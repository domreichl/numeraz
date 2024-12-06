from argparse import ArgumentParser
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from numerapi import NumerAPI


LOCAL_DATA_DIR = "./data"

parser = ArgumentParser()
parser.add_argument("-v", "--version", type=str, required=True)
parser.add_argument("-dan", "--data_asset_name", type=str, required=True)
parser.add_argument("-dav", "--data_asset_version", type=str, required=True)
parser.add_argument("-sid", "--subscription_id", type=str, required=True)
parser.add_argument("-rg", "--resource_group", type=str, required=True)
parser.add_argument("-ws", "--workspace_name", type=str, required=True)
args = parser.parse_args()

numerapi = NumerAPI()
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=args.subscription_id,
    resource_group_name=args.resource_group,
    workspace_name=args.workspace_name,
)

try:
    data_asset = ml_client.data.get(
        name=args.data_asset_name, version=args.data_asset_version
    )
    print(
        f"Data asset '{args.data_asset_name}' with version '{args.data_asset_version}' already exists."
    )
except:
    for fn in [
        "features.json",
        "live.parquet",
        "meta_model.parquet",
        "train.parquet",
        "validation.parquet",
    ]:
        numerapi.download_dataset(
            filename=f"{args.version}/{fn}", dest_path=f"{LOCAL_DATA_DIR}/{fn}"
        )
    data_asset = ml_client.data.create_or_update(
        Data(
            name=args.data_asset_name,
            version=args.data_asset_version,
            description=f"Numerai {args.version} Data",
            path=LOCAL_DATA_DIR,
            type="uri_folder",
        )
    )

print("Path:", data_asset.path)
