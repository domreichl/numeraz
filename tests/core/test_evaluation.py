import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.core import evaluation


@pytest.fixture
def examples_dir() -> Path:
    return Path(__file__).parent / "examples"


@pytest.fixture
def corrs_df(examples_dir: Path) -> pd.DataFrame:
    with open(examples_dir / "model_corrs.json", "r") as f:
        corrs: dict = json.load(f)
    return evaluation.rank_models(corrs)


@pytest.fixture
def predictions(examples_dir: Path) -> pd.DataFrame:
    return pd.read_csv(examples_dir / "predictions.csv")


@pytest.fixture
def main_target() -> str:
    return "target_cyrusd_20"


def test_evaluate_predictions(predictions: pd.DataFrame, main_target: str):
    warnings.filterwarnings("ignore")

    predictions["prediction"] = predictions["pred_cyrusd"]
    metrics = evaluation.evaluate_predictions(predictions, main_target)

    for name, value in metrics.items():
        assert isinstance(value, float)
        if "drawdown" in name:
            assert value == 0
        else:
            assert value > 0


def test_evaluate_ensembles(
    corrs_df: pd.DataFrame, predictions: pd.DataFrame, main_target: str
):
    warnings.filterwarnings("ignore")

    ensembles = {
        "main": "pred_" + main_target.split("_")[1],
        "all": [col for col in predictions.columns if col.startswith("pred_")],
    }
    for n in range(1, 3):
        top_n_models = corrs_df["model"].iloc[:n].tolist()
        ensembles[f"top{n}"] = [m.replace("base_", "pred_") for m in top_n_models]

    top_ensemble, top_metrics = evaluation.evaluate_ensembles(
        ensembles, predictions, main_target
    )
    assert top_ensemble == "all"
    assert len(top_metrics) == 5
