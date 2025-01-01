import json
import os

import mlflow
import pandas as pd
from azureml.fsspec import AzureMachineLearningFileSystem
from sklearn.preprocessing import MinMaxScaler

from components import parse_args
from core.preprocessing import DataPreprocessor

args: dict = parse_args(
    ["data_uri", "encoder_uri", "zero_cols", "train_data", "test_data"]
)
dp = DataPreprocessor(args["data_uri"])
fs = AzureMachineLearningFileSystem(args["data_uri"])
encoder = mlflow.keras.load_model(args["encoder_uri"])

with fs.open(fs.glob("LocalUpload/*/*/features.json")[0]) as f:
    features_data = json.load(f)
    features, targets = dp.get_features_and_targets(features_data, "all")
df = dp.load_full_data([], targets)


def _encode_era(data_dir: str, era: str, features: list[str]):
    print(f"Processing era {era}")
    args = dict(columns=features, filters=[("era", "==", era)])
    era_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"), **args)
    if len(era_df) == 0:
        era_df = pd.read_parquet(os.path.join(data_dir, "validation.parquet"), **args)
    scaled = MinMaxScaler().fit_transform(era_df.to_numpy())
    preds = pd.DataFrame(encoder(scaled))
    preds.columns = [f"feat_{col}" for col in preds.columns]
    return preds[sorted(preds.columns)]


with mlflow.start_run():
    mlflow.set_tags(
        {
            "data_uri": args["data_uri"],
            "encoder_uri": args["encoder_uri"],
        },
    )
    encoded = pd.concat(
        [_encode_era(args["data_uri"], era, features) for era in df["era"].unique()]
    ).reset_index(drop=True)
    zero_columns = list(encoded.columns[encoded.mean() == 0])
    with open(args["zero_cols"], "w") as f:
        [f.write(f"{line}\n") for line in zero_columns]
    encoded = (
        encoded.drop(columns=zero_columns)
        .rank(pct=True)
        .apply(lambda col: pd.cut(col, bins=5, labels=list(range(5))))
    )
    df = pd.concat([df.reset_index(drop=True), encoded], axis=1)
    train_df, test_df = dp.split_train_test(df)
    mlflow.log_params(
        {
            "train_eras": dp.get_era_range(train_df),
            "test_eras": dp.get_era_range(test_df),
        }
    )
    mlflow.log_metrics(
        {
            "n_features_all": len(features),
            "n_features_enc": len(encoded.columns),
            "n_features_enc_nonzero": len(
                [col for col in df if col.startswith("feat")]
            ),
            "n_zero_cols": len(zero_columns),
            "n_targets": len(targets),
        }
    )

train_df.to_csv(args["train_data"], index=False)
test_df.to_csv(args["test_data"], index=False)
