import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load data from CSV files.

    This class handles loading raw data from the filesystem.
    It provides basic error handling and logging for data loading operations.
    """

    def load(self, file_path: Path) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            file_path: Path to the CSV file to load

        Returns:
            DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty or malformed
        """
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")

            df = pd.read_csv(file_path)

            if df.empty:
                logger.warning(f"File is empty: {file_path}")
                raise ValueError(f"File is empty: {file_path}")

            logger.info(
                f"Loaded {len(df)} rows, {len(df.columns)} columns from {file_path.name}"
            )
            return df

        except pd.errors.EmptyDataError:
            logger.error(f"Empty CSV file: {file_path}")
            raise ValueError(f"File is empty: {file_path}")

        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error in {file_path}: {e}")
            raise ValueError(f"Invalid CSV format: {e}")

        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")
            raise
