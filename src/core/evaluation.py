import pandas as pd
from numerai_tools.scoring import numerai_corr, correlation_contribution


def evaluate_predictions(y: pd.DataFrame, target: str) -> dict:
    per_era_corr = y.groupby("era").apply(
        lambda era: numerai_corr(era[["prediction"]], era[target])
    )
    per_era_mmc = y.groupby("era").apply(
        lambda era: correlation_contribution(
            era[["prediction"]], era["numerai_meta_model"], era[target]
        )
    )
    corr_mean = per_era_corr.mean().iloc[0]
    corr_std = per_era_corr.std().iloc[0]
    corr_max_drawdown = (
        (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum())
        .max()
        .iloc[0]
    )
    mmc_mean = per_era_mmc.mean().iloc[0]
    mmc_std = per_era_mmc.std().iloc[0]
    mmc_max_drawdown = (
        (per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum())
        .max()
        .iloc[0]
    )
    score = 0.5 * corr_mean + 2.0 * mmc_mean
    metrics = {
        "pred_std": y["prediction"].std(),
        "corr_mean": corr_mean,
        "corr_std": corr_std,
        "corr_sharpe": corr_mean / corr_std,
        "corr_max_drawdown": corr_max_drawdown,
        "mmc_mean": mmc_mean,
        "mmc_std": mmc_std,
        "mmc_sharpe": mmc_mean / mmc_std,
        "mmc_max_drawdown": mmc_max_drawdown,
        "score": score,
    }
    return {k: round(float(v), 5) for k, v in metrics.items()}
