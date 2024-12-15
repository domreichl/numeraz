import json
import os

import mlflow
import pandas as pd

from components import parse_args
from core.evaluation import evaluate_ensembles, rank_models

args: dict = parse_args(["main_target", "base_models_dir", "best_ensemble"])
main_target = args["main_target"]
main_target_name = main_target.split("_")[1]
main_model = f"base_{main_target_name}"
models_dir = args["base_models_dir"]
predictions = pd.read_csv(os.path.join(models_dir, "predictions.csv"))
with open(os.path.join(models_dir, "model_corrs.json"), "r") as f:
    corrs: dict = json.load(f)
corrs_df: pd.DataFrame = rank_models(corrs)

with mlflow.start_run():
    mlflow.set_tag("main_model", main_model)
    mlflow.set_tag("top_base_model", corrs_df["model"][corrs_df["corr"].argmax()])
    mlflow.log_metric("main_corr", corrs[main_model])
    mlflow.log_metric("top_base_corr", corrs_df["corr"].max())
    ensembles = {
        "main": [f"pred_{main_target_name}"],
        "all": [col for col in predictions.columns if col.startswith("pred_")],
    }
    for n in range(1, 6):
        top_n_models = corrs_df["model"].iloc[:n].tolist()
        ensembles[f"top{n}"] = [m.replace("base_", "pred_") for m in top_n_models]
    top_ensemble, top_metrics = evaluate_ensembles(ensembles, predictions, main_target)
    mlflow.set_tag("top_ensemble", top_ensemble)
    mlflow.set_tag("top_ensemble_models", ", ".join(ensembles[top_ensemble]))
    mlflow.log_metric("top_ensemble_corr", top_metrics["corr_mean"])
    mlflow.log_metric("top_ensemble_corr_sharpe", top_metrics["corr_sharpe"])

with open(args["best_ensemble"], "w") as f:
    for model_name in ensembles[top_ensemble]:
        f.write(model_name + "\n")
