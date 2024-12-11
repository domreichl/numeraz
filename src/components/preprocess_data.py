import json, mlflow
from azureml.fsspec import AzureMachineLearningFileSystem

from components import parse_args
from core.preprocessing import DataPreprocessor


args: dict = parse_args(["feature_set", "data_uri", "train_data", "test_data"])

dp = DataPreprocessor(args["data_uri"])
fs = AzureMachineLearningFileSystem(args["data_uri"])

with fs.open(fs.glob("LocalUpload/*/*/features.json")[0]) as f:
    features_data = json.load(f)
    features, targets = dp.get_features_and_targets(features_data, args["feature_set"])
df = dp.load_full_data(features, targets)
train_df, test_df = dp.split_train_test(df)

with mlflow.start_run():
    mlflow.set_tags(
        {
            "data_uri": args["data_uri"],
            "feature_set": args["feature_set"],
        },
    )
    mlflow.log_params(
        {
            "train_eras": dp.get_era_range(train_df),
            "test_eras": dp.get_era_range(test_df),
        }
    )
    mlflow.log_metrics(
        {
            "n_features": len(features),
            "n_targets": len(targets),
        }
    )

train_df.to_csv(args["train_data"], index=False)
test_df.to_csv(args["test_data"], index=False)
