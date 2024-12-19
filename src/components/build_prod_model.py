import datetime as dt
import json
import os

import cloudpickle
import mlflow
import numpy as np
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from azureml.fsspec import AzureMachineLearningFileSystem

from components import parse_args
from core.evaluation import evaluate_ensembles
from core.model import Model
from core.preprocessing import DataPreprocessor

args: dict = parse_args(
    [
        "subscription_id",
        "resource_group",
        "workspace_name",
        "experiment_name",
        "model_name",
        "feature_set",
        "main_target",
        "hparams",
        "data_uri",
        "train_data",
        "test_data",
        "best_ensemble",
        "prod_model_info",
        "prod_model_dir",
    ]
)
model_name = args["model_name"]
main_target = args["main_target"]
hparams = json.loads(args["hparams"])
models_dir = args["prod_model_dir"]

with open(args["best_ensemble"], "r") as file:
    best_ensemble: dict = json.load(file)
ensembling_method: str = best_ensemble["ensembling_method"]
target_names = [m.lstrip("pred_") for m in best_ensemble["models"]]
targets = [f"target_{name}_20" for name in target_names]
ensemble = "-".join(target_names)

dp = DataPreprocessor(args["data_uri"])
x_train, y_train = dp.split_x_y(pd.read_csv(args["train_data"]))
x_test, y_test = dp.split_x_y(pd.read_csv(args["test_data"]))

base_models = []
with mlflow.start_run() as run:
    mlflow.set_tag("model_name", model_name)
    mlflow.log_params(hparams)
    mlflow.log_param("n_targets", len(targets))
    mlflow.log_param("ensemble", ensemble)

    # TRAIN BASE MODELS
    pred_cols: list[str] = []
    for target, target_name in zip(targets, target_names):
        base_name = f"base_{target_name}"
        with mlflow.start_run(run_name=base_name, nested=True):
            model = Model(name=base_name, main_target=main_target, hparams=hparams)
            model.fit(x_train, y_train, target)
            base_models.append(model.model)
            model.save(models_dir)
            predictions = model.predict(x_test)
            metrics = model.evaluate(predictions, y_test)
            mlflow.set_tag("target", target)
            mlflow.log_metrics(metrics)
            mlflow.lightgbm.log_model(
                model.model, base_name, input_example=x_test.iloc[:2]
            )
            pred_cols.append(f"pred_{target_name}")
            y_test[pred_cols[-1]] = predictions

    # EVALUATE ENSEMBLE
    _, metrics = evaluate_ensembles(
        {"prod": pred_cols}, y_test, main_target, ensembling_method
    )
    mlflow.log_metrics(metrics)

    # UPDATE PROD MODEL INFO
    file_name = "prod_model_info"
    file_path = os.path.join(models_dir, file_name + ".json")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=args["subscription_id"],
        resource_group_name=args["resource_group"],
        workspace_name=args["workspace_name"],
    )
    data_asset: Data = ml_client.data.get(name=file_name, label="latest")
    fs = AzureMachineLearningFileSystem(data_asset.path)
    with fs.open(fs.glob(f"LocalUpload/*/{file_name}.json")[0]) as file:
        model_info: dict = json.load(file)
        model_info[model_name] = {
            "date": str(dt.datetime.today().date()),
            "experiment_name": args["experiment_name"],
            "run_name": run.info.run_name,
            "run_id": run.info.run_id,
            "main_target": main_target,
            "feature_set": args["feature_set"],
            "train_eras": dp.get_era_range(y_train),
            "test_eras": dp.get_era_range(y_test),
            "base_model": "lightgbm",
            "base_model_hparams": hparams,
            "ensemble": ensemble,
            "ensembling_method": ensembling_method,
            "metrics": metrics,
        }
    with open(file_path, "w") as file:
        json.dump(model_info, file)
    updated_data_asset = Data(
        name=file_name,
        version=str(int(data_asset.version) + 1),
        path=file_path,
        type=data_asset.type,
        tags=data_asset.tags,
    )
    ml_client.data.create_or_update(updated_data_asset)

    # CREATE PERFORMANCE PLOTS
    performance = pd.DataFrame(
        {k: v["metrics"] for k, v in model_info.items()}
    ).transpose()
    for col in performance.columns:
        performance[col].plot(kind="bar").figure.savefig(f"{col}.jpg")
        mlflow.log_artifact(f"{col}.jpg")

    # LOAD FEATURES FOR PREDICT FUNCTION
    fs = AzureMachineLearningFileSystem(args["data_uri"])
    with fs.open(fs.glob("LocalUpload/*/*/features.json")[0]) as f:
        features_data = json.load(f)
    FEATURES, _ = dp.get_features_and_targets(features_data, args["feature_set"])
    BASE_MODELS = tuple(base_models)

    # SET PREDICT FUNCTION
    if ensembling_method == "simple":

        def predict(live_features: pd.DataFrame) -> pd.DataFrame:
            features = live_features[FEATURES]
            live_preds = pd.DataFrame(
                {i: model.predict(features) for i, model in enumerate(BASE_MODELS)}
            )
            live_preds["prediction"] = live_preds.rank(pct=True).mean(axis=1)
            live_preds.index = live_features.index
            submission = live_preds[["prediction"]]

            return submission

    elif ensembling_method == "weighted":

        def predict(live_features: pd.DataFrame) -> pd.DataFrame:
            features = live_features[FEATURES]
            live_preds = pd.DataFrame(
                {i: model.predict(features) for i, model in enumerate(BASE_MODELS)}
            )
            corrmat = np.corrcoef(live_preds.values.T)
            np.fill_diagonal(corrmat, 0.0)
            weights = 1 / np.mean(corrmat, axis=1)
            weights = weights / sum(weights)
            weighted_preds = live_preds.dot(weights)
            live_preds["prediction"] = weighted_preds
            live_preds.index = live_features.index
            submission = live_preds[["prediction"]]

            return submission

    else:
        raise NotImplementedError(
            f"Ensembling method {ensembling_method} is not implemented"
        )

    # SAVE & REGISTER PROD MODEL
    with open(model_name + ".pkl", "wb") as file:
        cloudpickle.dump(predict, file)
    mlflow.log_artifact(model_name + ".pkl")
    mlflow.register_model(f"runs:/{run.info.run_id}/{model_name}.pkl", model_name)

    # VERIFY PROD MODEL
    prod_model = pd.read_pickle(model_name + ".pkl")
    live_data = pd.read_parquet(
        path=os.path.join(args["data_uri"], "live.parquet"),
        columns=FEATURES,
    )
    submission = prod_model(live_data)
    print("\nSUBMISSION:\n", submission)
    assert submission.index.name == "id"
    assert submission.columns[0] == "prediction"
    assert len(submission.columns) == 1
