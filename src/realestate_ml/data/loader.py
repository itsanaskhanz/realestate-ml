import pandas as pd
from pathlib import Path
from .interfaces import CSVDataSource
from .cleaner import DataCleaner
from .validator import DataValidator
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and clean housing data"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.validator = DataValidator()
        self.cleaner = DataCleaner()

    def load_and_clean(self, filename: str = "data.csv") -> pd.DataFrame:
        """Load raw data, validate, clean, and save"""
        # Load raw data
        source = CSVDataSource(self.data_dir / "raw" / filename)
        df = source.load()
        logger.info(f"Loaded {len(df)} rows")

        # Validate data
        is_valid, errors = self.validator.validate(df)
        if not is_valid:
            logger.warning(f"Data has issues:\n" + "\n".join(errors))

        # Clean data
        df = self.cleaner.clean(df)

        # Save cleaned data
        self._save_cleaned(df)
        logger.info(
            f"Saved cleaned data to {self.data_dir / 'interim' / 'data_cleaned.csv'}"
        )

        return df

    def _save_cleaned(self, df=pd.DataFrame):
        """Save cleaned data to interim folder"""
        interim_dir = self.data_dir / "interim"
        interim_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(interim_dir / "data_cleaned.csv", index=False)
        logger.info(f"Saved cleaned data to {interim_dir / 'data_cleaned.csv'}")

    def load_cleaned(self):
        """Load previously cleaned data"""
        filepath = self.data_dir / "interim" / "data_cleaned.csv"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Cleaned data not found. Run load_and_clean() first."
            )

        return pd.read_csv(filepath)
