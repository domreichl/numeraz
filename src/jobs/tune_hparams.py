import argparse
import json

import mlflow
import pandas as pd

from core.model import Model
from core.preprocessing import DataPreprocessor

parser = argparse.ArgumentParser()
parser.add_argument("--main_target", type=str, required=True)
parser.add_argument("--hparams", type=str, required=True)
parser.add_argument("--train_data", type=str, required=True)
parser.add_argument("--n_estimators", type=int, required=True)
args: dict = vars(parser.parse_args())

dp = DataPreprocessor()
df = pd.read_csv(args["train_data"])
(x_train, y_train), (x_val, y_val) = dp.load_x_y_splits(df)

hparams = json.loads(args["hparams"])
hparams["n_estimators"] = args["n_estimators"]
target = args["main_target"]

with mlflow.start_run():
    mlflow.set_tag("target", target)
    mlflow.log_params(hparams)
    mlflow.log_param("train_eras", dp.get_era_range(y_train))
    mlflow.log_param("val_eras", dp.get_era_range(y_val))

    model = Model(name="tune", main_target=target, hparams=hparams)
    model.fit(x_train, y_train, target)
    predictions = model.predict(x_val)
    metrics = model.evaluate(predictions, y_val)
    corr_sharpe = metrics["corr_sharpe"]

    print(f"Correlation results: {corr_sharpe} (sharpe), {metrics['corr_mean']} (mean)")
    mlflow.log_metrics(metrics)
