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

    def __init__(
        self,
    ):
        """
        Initialize DataCleaner.

        Args:
            iqr_multiplier: Multiplier for IQR outlier detection (default: 1.5)
        """
        self.data_config = CONFIG["data"]
        self.interim_path = Path(self.data_config["interim_path"])
        self.stats = {}

    def clean(self, df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
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
        if save:
            path = Path(self.interim_path)
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "data_cleaned.csv", index=False)
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
