import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Base class for all data sources"""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data into DataFrame"""
        pass

    @abstractmethod
    def exists(self) -> bool:
        """Check if data source exists"""
        pass


class CSVDataSource(DataSource):
    """CSV file data source"""

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        """Load CSV file"""
        if not self.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        df = pd.read_csv(self.filepath)
        logger.info(f"Loaded {len(df)} rows")
        return df

    def exists(self) -> bool:
        """Check if CSV file exists"""
        return self.filepath.exists()
