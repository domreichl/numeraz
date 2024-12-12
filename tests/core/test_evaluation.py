import pytest
import pandas as pd

from src.core import evaluation


@pytest.fixture
def mock_data():
    data = {
        "era": ["1", "1", "2", "2", "2"],
        "prediction": [1, 3, 3, 2, 5],
        "target": [1, 2, 2, 2, 5],
        "numerai_meta_model": [1, 2, 3, 3, 5],
    }
    return pd.DataFrame(data)


def test_evaluate_predictions(mock_data):
    metrics = evaluation.evaluate_predictions(mock_data, "target")

    for name, value in metrics.items():
        assert isinstance(value, float)
        if "drawdown" in name:
            assert value == 0
        else:
            assert value > 0
