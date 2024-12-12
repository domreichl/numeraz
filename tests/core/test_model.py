import warnings
import pytest
import numpy as np
import pandas as pd

from src.core.model import Model


@pytest.fixture
def mock_data() -> tuple:
    x_train = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [5, 6, 9, 4, 8, 9],
        }
    )
    y_train = pd.DataFrame(
        {
            "id": [str(i) for i in range(6)],
            "era": ["488", "488", "501", "501", "503", "503"],
            "target_20": [1, 2, 3, 4, 5, 2],
            "numerai_meta_model": [1, 2, 3, 4, 4, 3],
        }
    )
    x_val = x_train.copy()
    y_val = y_train.copy()

    return (x_train, y_train), (x_val, y_val)


def test_fit_predict_evaluate(mock_data: tuple):
    warnings.filterwarnings("ignore")

    (x_train, y_train), (x_val, y_val) = mock_data
    model = Model("test-model", "target_20", {})
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)

    assert len(predictions) == 6

    predictions = np.array([4, 1, 3, 5, 8, 2])
    metrics = model.evaluate(predictions, y_val)

    for name, value in metrics.items():
        if "mmc_sharpe" in name:
            continue
        assert isinstance(value, float)
        assert value >= 0
