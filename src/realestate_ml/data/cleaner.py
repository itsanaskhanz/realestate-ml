import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean housing data"""

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier  # For outlier detection
        self.stats = {}  # Track cleaning statistics

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main cleaning pipeline"""
        df = df.copy()
        initial_rows = len(df)

        # Remove invalid records
        df = self._remove_invalid(df)
        after_invalid = len(df)

        # Cap outliers
        df = self._cap_outliers(df)

        # Fix data types
        df = self._fix_dtypes(df)

        # Store statistics
        self.stats = {
            "initial_rows": initial_rows,
            "after_invalid_removal": after_invalid,
            "removed_rows": initial_rows - after_invalid,
            "final_rows": len(df),
        }
        logger.info(f"Cleaned: {initial_rows} -> {len(df)} rows")
        return df

    def _remove_invalid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid values"""
        df = df[df["price"] > 0]
        df = df[df["bathrooms"] > 0]
        df = df[df["bedrooms"] > 0]
        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers using IQR method"""
        outliers_col = [
            "price",
            "sqft_living",
            "sqft_lot",
            "sqft_above",
        ]
        for col in outliers_col:
            if col not in df.columns:
                continue
            # Calculate IQR bounds
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            IQR = q3 - q1
            lower_bound = q1 - (self.iqr_multiplier * IQR)
            upper_bound = q3 + (self.iqr_multiplier * IQR)

            # Cap values
            df[col] = df[col].clip(lower_bound, upper_bound)
        return df

    def _fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types"""
        # Convert numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df

    def get_stats(self):
        """Return cleaning statistics"""
        return self.stats
