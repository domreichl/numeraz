import warnings

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
