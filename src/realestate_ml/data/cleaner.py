from pathlib import Path
import pandas as pd
import logging
from realestate_ml.config import CONFIG

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean and preprocess data.

    This class handles:
    - Removing invalid values (price, bedrooms, bathrooms <= 0)
    - Capping outliers using IQR method
    - Fixing data types
    """

    OUTLIER_COLUMNS = ["price", "sqft_living", "sqft_lot", "sqft_above"]
    POSITIVE_COLUMNS = ["price", "bathrooms", "bedrooms"]

    def __init__(self, iqr_multiplier: float = 1.5):
        """
        Initialize DataCleaner.

        Args:
            iqr_multiplier: Multiplier for IQR outlier detection (default: 1.5)
        """
        self.iqr_multiplier = iqr_multiplier
        self.data_config = CONFIG["data"]
        self.interim_path = Path(self.data_config["interim_path"])
        self.stats = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete cleaning pipeline.

        Args:
            df: Raw DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        initial_rows = len(df)

        df = self._remove_invalid(df)
        df = self._cap_outliers(df)
        df = self._fix_dtypes(df)

        final_rows = len(df)
        removed = initial_rows - final_rows

        self.stats = {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "removed_rows": removed,
        }

        if removed > 0:
            logger.info(
                f"Cleaning complete: {initial_rows} -> {final_rows} rows (removed {removed})"
            )
        else:
            logger.info(f"Cleaning complete: {initial_rows} rows (no rows removed)")
        path = self.interim_path / "data_cleaned.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df

    def _remove_invalid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid values (<= 0)."""
        before = len(df)

        for col in self.POSITIVE_COLUMNS:
            if col in df.columns:
                df = df[df[col] > 0]

        removed = before - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with invalid values")

        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers using IQR method."""
        for col in self.OUTLIER_COLUMNS:
            if col not in df.columns:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - (self.iqr_multiplier * iqr)
            upper = q3 + (self.iqr_multiplier * iqr)

            outliers = ((df[col] < lower) | (df[col] > upper)).sum()

            df[col] = df[col].clip(lower, upper)

            if outliers > 0:
                logger.info(f"Capped {outliers} outliers in {col}")

        return df

    def _fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for numeric and date columns."""
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            logger.info("Converted date column to datetime")

        return df

    def get_stats(self) -> dict:
        """Return cleaning statistics."""
        return self.stats
