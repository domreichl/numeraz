import os

import pandas as pd


class DataPreprocessor:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir
        self.eras_to_embargo = 4

    def get_era_range(self, df: pd.DataFrame) -> str:
        era_range = f"{str(df['era'].min()).zfill(4)}-{str(df['era'].max()).zfill(4)}"
        print(f"Era range: {era_range}")

        return era_range

    def get_features_and_targets(
        self, features_data: dict, feature_set: str
    ) -> tuple[list, list]:
        targets = [
            target for target in features_data["targets"] if target.endswith("_20")
        ]
        features = features_data["feature_sets"][feature_set]
        print(f"Loaded {len(features)} features & {len(targets)} targets")

        return features, targets

    def load_full_data(self, features: list[str], targets: list[str]) -> pd.DataFrame:
        df = pd.concat(
            [
                self._load_parquet("train", features, targets),
                self._load_parquet("validation", features, targets),
            ]
        )
        df = self._impute_missing(df)
        df = df.join(self._load_parquet("meta_model")).reset_index()
        print(f"Loaded full data with {len(df)} rows and {len(df.columns)} columns")

        return df

    def load_x_y_splits(self, df) -> tuple:
        train, val = self.split_train_val(df)
        x_train, y_train = self.split_x_y(train)
        x_val, y_val = self.split_x_y(val)

        return (x_train, y_train), (x_val, y_val)

    def split_train_test(
        self, df: pd.DataFrame, start_test_era: int = 501, n_test_eras: int = 150
    ) -> tuple:
        eras = set(df["era"].unique().astype(int).tolist())
        test_eras = eras.intersection(
            set(range(start_test_era, start_test_era + n_test_eras))
        )
        train_eras = eras.difference(
            set(
                range(
                    start_test_era - self.eras_to_embargo,
                    start_test_era + n_test_eras + self.eras_to_embargo,
                )
            )
        )
        train = df[df["era"].astype(int).isin(train_eras)]
        test = df[df["era"].astype(int).isin(test_eras)]
        print(
            f"Split data ({len(eras)} eras) into {train['era'].nunique()} train and {test['era'].nunique()} test eras"
        )

        return train, test

    def split_train_val(self, df: pd.DataFrame, train_size: float = 0.85) -> tuple:
        eras = list(df["era"].unique())
        split = int(len(eras) * train_size)
        train = df[df["era"].isin(eras[:split])]
        test = df[df["era"].isin(eras[split + self.eras_to_embargo :])]
        return train, test

    def split_x_y(self, df: pd.DataFrame) -> tuple:
        df = df.reset_index()
        x = df[sorted([col for col in df.columns if col.startswith("feat")])]
        y = df[
            ["id", "era", "numerai_meta_model"]
            + [col for col in df.columns if col.startswith("target")]
        ]
        print(f"Split data into {len(x.columns)} x and {len(y.columns)} y columns")

        return x, y

    def _load_parquet(
        self, name: str, features: list[str] = [], targets: list[str] = []
    ) -> pd.DataFrame:
        path = os.path.join(self.data_dir, name + ".parquet")
        if name in ["train", "validation"]:
            columns = ["era", "data_type"] + features + targets
            df = pd.read_parquet(path, columns=columns)
            df = df[df["data_type"] != "test"]
            del df["data_type"]
        elif name == "meta_model":
            df = pd.read_parquet(path)["numerai_meta_model"]

        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            nans = df[df[col].isna()]
            if len(nans) > 0:
                if col.startswith("target"):
                    mode = float(df[col].mode().iloc[0])
                    print(f"Imputing {len(nans)} NaNs in {col} with {mode} (mode)")
                    df[col] = df[col].fillna(mode)
                else:
                    raise Exception(f"{col} contains NaNs")

        return df
