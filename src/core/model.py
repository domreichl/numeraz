import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.core.evaluation import evaluate_predictions


class Model:
    def __init__(self, name: str, main_target: str, hparams: dict):
        self.name = name
        self.main_target = main_target
        self.model = LGBMRegressor(**hparams, force_row_wise=True, verbose=-1)

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self.model.fit(x_train, y_train[self.main_target])

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x_test)

    def evaluate(self, predictions: np.ndarray, y_test: pd.DataFrame) -> dict:
        metrics = evaluate_predictions(
            y_test.assign(prediction=predictions), self.main_target
        )

        return metrics
