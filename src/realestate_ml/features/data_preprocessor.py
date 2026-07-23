import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from realestate_ml.config import CONFIG


class DataPreprocessor:
    def __init__(self):
        self.split = CONFIG["split"]

        self.processed_path = Path(CONFIG["data"]["processed_path"])

    def train_test_split(self, df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
        target_col = self.split["target_column"]
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.split["test_size"],
            random_state=self.split["random_state"],
        )

        if save:
            path = Path(self.processed_path)
            path.mkdir(parents=True, exist_ok=True)
            X_train.to_csv(path / "X_train.csv", index=False)
            X_test.to_csv(path / "X_test.csv", index=False)
            y_train.to_csv(path / "y_train.csv", index=False)
            y_test.to_csv(path / "y_test.csv", index=False)

        return X_train, X_test, y_train, y_test
