from argparse import ArgumentParser
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from numerapi import NumerAPI


parser = ArgumentParser()
parser.add_argument("-n", "--name", type=str, required=True)
parser.add_argument("-v", "--version", type=str, required=True)
parser.add_argument("-ndv", "--numerai_data_version", type=str, required=True)
parser.add_argument("-sid", "--subscription_id", type=str, required=True)
parser.add_argument("-rg", "--resource_group", type=str, required=True)
parser.add_argument("-ws", "--workspace_name", type=str, required=True)
parser.add_argument("-exp", "--experiment_name", type=str, required=True)
parser.add_argument("-dir", "--data_dir", type=str, required=False, default="./data")
args = parser.parse_args()

numerapi = NumerAPI()
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=args.subscription_id,
    resource_group_name=args.resource_group,
    workspace_name=args.workspace_name,
)

try:
    data_asset = ml_client.data.get(name=args.name, version=args.version)
    print(f"Data asset '{args.name}' with version '{args.version}' already exists.")
except:
    for fn in [
        "features.json",
        "live.parquet",
        "meta_model.parquet",
        "train.parquet",
        "validation.parquet",
    ]:
        numerapi.download_dataset(
            filename=f"{args.numerai_data_version}/{fn}",
            dest_path=f"{args.data_dir}/{fn}",
        )
    data_asset = ml_client.data.create_or_update(
        Data(
            name=args.name,
            version=args.version,
            description=f"Numerai {args.numerai_data_version} Data",
            path=args.data_dir,
            type="uri_folder",
        )
    )

print("Path:", data_asset.path)
