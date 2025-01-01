from unittest.mock import patch

import pandas as pd
import pytest

from src.core.preprocessing import DataPreprocessor


@pytest.fixture
def mock_data_dir():
    return "."


@pytest.fixture
def data_preprocessor(mock_data_dir):
    return DataPreprocessor(data_dir=mock_data_dir)


def test_get_era_range(data_preprocessor):
    df = pd.DataFrame({"era": [101, 102, 103]})
    result = data_preprocessor.get_era_range(df)

    assert result == "0101-0103"


def test_get_features_and_targets(data_preprocessor):
    features_data = {
        "targets": ["target_10", "target_20"],
        "feature_sets": {"set1": ["feature1", "feature2"]},
    }
    features, targets = data_preprocessor.get_features_and_targets(
        features_data, "set1"
    )

    assert features == ["feature1", "feature2"]
    assert targets == ["target_20"]


@patch("src.core.preprocessing.DataPreprocessor._load_parquet")
def test_load_full_data(mock_load_parquet, data_preprocessor: DataPreprocessor):
    mock_load_parquet.side_effect = [
        pd.DataFrame({"era": [101], "feature1": [1], "target_20": [0]}),
        pd.DataFrame({"era": [201], "feature1": [2], "target_20": [1]}),
        pd.DataFrame({"numerai_meta_model": [0.5]}),
    ]
    features = ["feature1"]
    targets = ["target_20"]
    result = data_preprocessor.load_full_data(features, targets)

    assert len(result) == 2
    assert "numerai_meta_model" in result.columns
    assert "feature1" in result.columns


def test_load_x_y_splits(data_preprocessor: DataPreprocessor):
    df = pd.DataFrame(
        {
            "era": [488, 499, 500, 501, 502, 503, 504],
            "feature": [1, 2, 3, 4, 5, 6, 7],
            "numerai_meta_model": [1, 2, 5, 5, 5, 5, 7],
        }
    )
    df = df.assign(id=df["era"], target_20=df["feature"])
    data_preprocessor.eras_to_embargo = 1
    (x_train, y_train), (x_val, y_val) = data_preprocessor.load_x_y_splits(df)

    assert len(x_train) == len(y_train)
    assert len(x_val) == len(y_val)
    assert len(x_train.columns) == len(x_val.columns)
    assert len(y_train.columns) == len(y_val.columns)


def test_split_train_test(data_preprocessor: DataPreprocessor):
    df = pd.DataFrame(
        {
            "era": [488, 499, 500, 501, 502, 503, 504],
            "feature": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    data_preprocessor.eras_to_embargo = 1
    train, test = data_preprocessor.split_train_test(
        df, start_test_era=500, n_test_eras=3
    )

    assert len(train) == 2
    assert len(test) == 3
    assert train["feature"].tolist() == [1, 7]
    assert test["feature"].tolist() == [3, 4, 5]


def test_split_train_val(data_preprocessor: DataPreprocessor):
    df = pd.DataFrame({"era": [101, 102, 103, 104, 105], "feature": [1, 2, 3, 4, 5]})
    data_preprocessor.eras_to_embargo = 1
    train, val = data_preprocessor.split_train_val(df, train_size=0.6)

    assert len(train) == 3
    assert len(val) == 1
    assert train["era"].tolist() == [101, 102, 103]
    assert val["feature"].tolist() == [5]


def test_split_x_y(data_preprocessor: DataPreprocessor):
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "era": ["0101", "0102"],
            "feature1": [5, 4],
            "target_20": [0, 1],
            "numerai_meta_model": [0.8, 0.9],
        }
    )

    x, y = data_preprocessor.split_x_y(df)

    assert list(x.columns) == ["feature1"]
    assert len(y.columns) == 4


def test_impute_missing(data_preprocessor: DataPreprocessor):
    df = pd.DataFrame({"era": [101, 102], "target_20": [None, 1]})
    result = data_preprocessor._impute_missing(df)

    assert result["target_20"].iloc[0] == 1
