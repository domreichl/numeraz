import json

import mlflow
import pandas as pd

from components import parse_args
from core.model import Model
from core.preprocessing import DataPreprocessor

args: dict = parse_args(
    ["main_target", "hparams", "train_data", "base_models_dir", "val_predictions"]
)

dp = DataPreprocessor()
df = pd.read_csv(args["train_data"])
(x_train, y_train), (x_val, y_val) = dp.load_x_y_splits(df)
del df

hparams = json.loads(args["hparams"])
target_cols = [col for col in y_train.columns if "target" in col]

with mlflow.start_run():
    mlflow.log_params(hparams)
    for col in target_cols:
        target = col.split("_")[1]
        model_name = f"base_{target}"
        model = Model(name=model_name, main_target=args["main_target"], hparams=hparams)
        model.fit(x_train, y_train, col)
        model.save(args["base_models_dir"])
        predictions = model.predict(x_val)
        metrics = model.evaluate(predictions, y_val)
        with mlflow.start_run(run_name=target, nested=True):
            mlflow.set_tag("target", col)
            mlflow.lightgbm.log_model(model, model_name)
            mlflow.log_metrics(**metrics)
        y_val[f"pred_{target}"] = predictions

y_val.to_csv(args["val_predictions"], index=False)
