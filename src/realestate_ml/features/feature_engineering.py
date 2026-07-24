import pandas as pd
from realestate_ml.config import CONFIG
from pathlib import Path


class FeatureEngineer:
    """
    Class for feature engineering.

    This class handles the creation of new features from existing columns in a DataFrame.
    """

    def __init__(self, df):
        """
        Initialize with a pandas DataFrame.

        Args:
            df: Input DataFrame
        """
        self.df = df.copy()
        if "date" in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["date"])

    def engineer(self, save: bool = False):
        """
        Execute all feature engineering steps in sequence.

        Returns:
            DataFrame with engineered features and unnecessary columns removed
        """
        self._create_date_features()
        self._create_property_features()
        feature_engineered_data = self._drop_unnecessary_columns()
        if save:
            path = Path(CONFIG["data"]["interim_path"])
            path.mkdir(parents=True, exist_ok=True)
            feature_engineered_data.to_csv(
                path / "data_engineered.csv",
                index=False,
            )
        return feature_engineered_data

    def _create_date_features(self):
        """
        Extract temporal features from date column.
        Adds year, month, and day as separate integer columns.
        """
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day"] = self.df["date"].dt.day

    def _create_property_features(self):
        """
        Create property-based features:
        - age_at_sale: Property age at time of sale
        - was_renovated: Binary flag (1 if renovated)
        - years_since_renovation: Years since last renovation (0 if never)
        - total_sqft: Combined living and basement square footage
        """
        self.df["age_at_sale"] = self.df["year"] - self.df["yr_built"]

        self.df["was_renovated"] = self.df["yr_renovated"].apply(
            lambda x: 1 if x > 0 else 0
        )
        self.df["years_since_renovation"] = self.df.apply(
            lambda x: x["year"] - x["yr_renovated"] if x["yr_renovated"] > 0 else 0,
            axis=1,
        )
        self.df["total_sqft"] = self.df["sqft_living"] + self.df["sqft_basement"]

    def _drop_unnecessary_columns(self):
        """
        Remove columns no longer needed after feature engineering.

        Returns:
            DataFrame with irrelevant columns removed
        """
        cols_to_drop = [
            "date",
            "year",
            "street",
            # "city",
            "statezip",
            "country",
            "yr_built",
            "yr_renovated",
        ]
        self.df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        return self.df
