import pytest
import pandas as pd
from realestate_ml.data.cleaner import DataCleaner


class TestDataCleaner:
    """Test suite for DataCleaner class."""

    def test_clean_normal_data(self, sample_data):
        """Test cleaning works on normal data without removing rows."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean(sample_data)

        assert len(cleaned_df) == len(sample_data)

    def test_clean_outliers_data(self, sample_data_outliers):
        """Test outlier capping works correctly (max sqft_living ≤ 2700)."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean(sample_data_outliers)
        assert cleaned_df["sqft_living"].max() <= 2700

    def test_clean_missing_data(self, sample_data_missing):
        """Test missing values are removed completely."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean(sample_data_missing)
        assert cleaned_df.isnull().sum().sum() == 0
