import os
import yaml
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from realestate_ml.config import CONFIG
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.split = CONFIG["split"]
        self.df = df
        self.processed_path = Path(CONFIG["data"]["processed_path"])

    def build_processor(self) -> ColumnTransformer:
        top_5_cities = self.df["city"].value_counts().head(5).index.tolist()

        encoder = OneHotEncoder(
            categories=[top_5_cities], handle_unknown="ignore", sparse_output=False
        )
        target_col = self.split["target_column"]

        numerical_columns = self.df.drop(columns=["city", target_col]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("city", encoder, ["city"]),
                ("num", StandardScaler(), numerical_columns),
            ]
        )

        return preprocessor

    def train_test_split(
        self, save: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        target_col = self.split["target_column"]
        X = self.df.drop(target_col, axis=1)
        y = self.df[target_col]

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
