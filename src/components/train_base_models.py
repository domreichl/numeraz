import json
import os

import mlflow
import pandas as pd

from components import parse_args
from core.model import Model
from core.preprocessing import DataPreprocessor

args: dict = parse_args(["main_target", "hparams", "train_data", "base_models_dir"])

dp = DataPreprocessor()
df = pd.read_csv(args["train_data"])
(x_train, y_train), (x_val, y_val) = dp.load_x_y_splits(df)
del df

models_dir = args["base_models_dir"]
hparams = json.loads(args["hparams"])
target_cols = [col for col in y_train.columns if "target" in col]
corrs = {}

with mlflow.start_run():
    mlflow.log_params(hparams)
    for col in target_cols:
        target = col.split("_")[1]
        model_name = f"base_{target}"
        model = Model(name=model_name, main_target=args["main_target"], hparams=hparams)
        model.fit(x_train, y_train, col)
        predictions = model.predict(x_val)
        metrics = model.evaluate(predictions, y_val)
        with mlflow.start_run(run_name=target, nested=True):
            mlflow.set_tag("target", col)
            mlflow.log_metrics(**metrics)
            mlflow.lightgbm.log_model(model, model_name)
            model.save(models_dir)
        y_val[f"pred_{target}"] = predictions
        corrs[model_name] = metrics["corr_mean"]

y_val.to_csv(os.path.join(models_dir, "predictions.csv"), index=False)
with open(os.path.join(models_dir, "model_corrs.json"), "w") as f:
    json.dump(corrs, f)
