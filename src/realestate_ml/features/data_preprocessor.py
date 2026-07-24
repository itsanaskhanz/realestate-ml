from typing import Tuple

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from realestate_ml.config import CONFIG


def build_column_transformer(
    X: pd.DataFrame, target_column: str = "price"
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer from training features only.

    City categories are derived from X to avoid leaking test-set information.
    """
    top_5_cities = X["city"].value_counts().head(5).index.tolist()
    encoder = OneHotEncoder(
        categories=[top_5_cities], handle_unknown="ignore", sparse_output=False
    )
    numerical_columns = X.drop(columns=["city", target_column], errors="ignore").columns

    return ColumnTransformer(
        transformers=[
            ("city", encoder, ["city"]),
            ("num", StandardScaler(), numerical_columns),
        ]
    )


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.split = CONFIG["split"]
        self.df = df
        self.processed_path = Path(CONFIG["data"]["processed_path"])

    def build_processor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessor using training features only."""
        return build_column_transformer(
            X_train, target_column=self.split["target_column"]
        )

    def train_test_split(
        self, save: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        target_col = self.split["target_column"]
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.split["test_size"],
            random_state=self.split["random_state"],
        )

        if save:
            path = self.processed_path
            path.mkdir(parents=True, exist_ok=True)
            X_train.to_csv(path / "X_train.csv", index=False)
            X_test.to_csv(path / "X_test.csv", index=False)
            y_train.to_csv(path / "y_train.csv", index=False, header=False)
            y_test.to_csv(path / "y_test.csv", index=False, header=False)

        return X_train, X_test, y_train, y_test
