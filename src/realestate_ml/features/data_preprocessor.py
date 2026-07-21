import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from realestate_ml.config import CONFIG


class DataPreprocessor:
    def __init__(self):
        self.data_config = CONFIG["data"]
        self.processed_path = Path(self.data_config["processed_path"])

    def save(self, data: pd.DataFrame, filename: str) -> None:
        path = self.processed_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)

    def train_test_split(self, df: pd.DataFrame) -> pd.DataFrame:
        target_col = self.data_config["target_column"]
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.data_config["test_size"],
            random_state=self.data_config["random_state"],
        )

        self.save(X_train, "X_train.csv")
        self.save(X_test, "X_test.csv")
        self.save(y_train, "y_train.csv")
        self.save(y_test, "y_test.csv")

        return X_train, X_test, y_train, y_test
