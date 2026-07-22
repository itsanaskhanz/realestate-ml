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

    def _save(self, data: pd.DataFrame, filename: str) -> None:
        path = self.processed_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)

    def train_test_split(self, df: pd.DataFrame) -> pd.DataFrame:
        target_col = self.split["target_column"]
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.split["test_size"],
            random_state=self.split["random_state"],
        )

        self._save(X_train, "X_train.csv")
        self._save(X_test, "X_test.csv")
        self._save(y_train, "y_train.csv")
        self._save(y_test, "y_test.csv")

        return X_train, X_test, y_train, y_test
