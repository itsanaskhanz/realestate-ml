import pytest
import pandas as pd
from realestate_ml.data.loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""

    def test_load(self, csv_file):
        """Test loading CSV file returns a non-empty DataFrame."""
        loader = DataLoader()
        df = loader.load(csv_file)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
