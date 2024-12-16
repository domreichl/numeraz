import warnings

import numpy as np
import pandas as pd
from numerai_tools.scoring import correlation_contribution, numerai_corr


def evaluate_predictions(y: pd.DataFrame, target: str) -> dict:
    per_era_corr = y.groupby("era").apply(
        lambda era: numerai_corr(era[["prediction"]], era[target])
    )
    corr_mean = per_era_corr.mean().iloc[0]
    corr_std = per_era_corr.std().iloc[0]
    corr_max_drawdown = (
        (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum())
        .max()
        .iloc[0]
    )
    metrics = {
        "pred_std": y["prediction"].std(),
        "corr_mean": corr_mean,
        "corr_std": corr_std,
        "corr_sharpe": corr_mean / corr_std,
        "corr_max_drawdown": corr_max_drawdown,
    }
    try:
        per_era_mmc = y.groupby("era").apply(
            lambda era: correlation_contribution(
                era[["prediction"]], era["numerai_meta_model"], era[target]
            )
        )
        metrics["mmc_mean"] = per_era_mmc.mean().iloc[0]
        metrics["score"] = 0.5 * corr_mean + 2.0 * metrics["mmc_mean"]
        metrics["mmc_std"] = per_era_mmc.std().iloc[0]
        metrics["mmc_max_drawdown"] = (
            (per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum())
            .max()
            .iloc[0]
        )
        try:
            metrics["mmc_sharpe"] = metrics["mmc_mean"] / metrics["mmc_std"]
        except:
            metrics["mmc_sharpe"] = 0
    except:
        warnings.warn(f"Failed to compute mmc scores")
    return {k: round(float(v), 5) for k, v in metrics.items()}


def evaluate_ensembles(
    ensembles: dict[str, list[str]],
    predictions: pd.DataFrame,
    main_target: str,
    ensembling_method: str,
) -> tuple[str, dict]:
    top_ensemble = None
    top_metrics = {"corr_sharpe": 0}
    predictions = predictions[
        ["era", main_target]
        + [col for col in predictions.columns if col.startswith("pred_")]
    ]
    for name, te in ensembles.items():
        pred_cols = [col for col in predictions.columns if col in te]
        if len(pred_cols) == 1 or ensembling_method == "simple":
            ensemble = predictions.groupby("era")[pred_cols].rank(pct=True).mean(axis=1)
        elif ensembling_method == "weighted":
            corrmat = np.corrcoef(
                predictions[pred_cols].values.T
            )  # inter-model correlations
            np.fill_diagonal(corrmat, 0.0)  # removes models' autocorrelations
            weights = 1 / np.mean(
                corrmat, axis=1
            )  # calculates the weight proportional to each model's average correlation with other models
            weights /= sum(weights)  # normalize so that sum equals 1
            ensemble = predictions[pred_cols].dot(
                weights
            )  # computes weighted sum of predictions
        else:
            raise NotImplementedError(
                f"Ensembling method {ensembling_method} is not implemented"
            )
        metrics = evaluate_predictions(
            predictions.assign(prediction=ensemble), main_target
        )
        print(f"Metrics for ensemble {name}: {metrics}")
        if metrics["corr_sharpe"] > top_metrics["corr_sharpe"]:
            top_metrics = metrics
            top_ensemble = name

    return top_ensemble, top_metrics


def rank_models(mean_corr_dict: dict) -> pd.DataFrame:
    corrs_df = pd.DataFrame(mean_corr_dict, index=[0]).transpose().reset_index()
    corrs_df.columns = ["model", "corr"]
    corrs_df.sort_values("corr", ascending=False, inplace=True)

    return corrs_df
