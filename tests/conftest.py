import pytest
import pandas as pd


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Clean sample dataset for testing."""
    return pd.DataFrame(
        {
            "date": [
                "2024-01-15",
                "2024-02-20",
                "2024-03-10",
                "2024-04-05",
                "2024-05-12",
            ],
            "price": [500000, 350000, 750000, 250000, 600000],
            "bedrooms": [3, 2, 4, 1, 3],
            "bathrooms": [2.5, 1.5, 3.0, 1.0, 2.0],
            "sqft_living": [1800, 1200, 2500, 800, 1600],
            "sqft_lot": [5000, 3500, 8000, 2000, 6000],
            "floors": [1.5, 1.0, 2.0, 1.0, 1.0],
            "waterfront": [0, 0, 1, 0, 0],
            "view": [0, 1, 2, 0, 0],
            "condition": [3, 4, 5, 2, 3],
            "sqft_above": [1600, 1200, 2200, 800, 1400],
            "sqft_basement": [200, 0, 300, 0, 200],
            "yr_built": [2010, 2005, 2018, 1995, 2008],
            "yr_renovated": [0, 2015, 0, 0, 2020],
            "city": ["Seattle", "Bellevue", "Redmond", "Sammamish", "Kirkland"],
        }
    )


@pytest.fixture
def sample_data_invalid() -> pd.DataFrame:
    """Data with invalid values (zeros and negatives)."""
    return pd.DataFrame(
        {
            "date": ["2024-01-15", "2024-02-20", "2024-03-10"],
            "price": [500000, 350000, 750000],
            "bedrooms": [3, -1, 0],
            "bathrooms": [2.5, 0, -1.0],
            "sqft_living": [1800, 0, -500],
            "sqft_lot": [5000, 3500, 8000],
            "floors": [1.5, 1.0, 2.0],
            "waterfront": [0, 0, 1],
            "view": [0, 1, 2],
            "condition": [3, 4, 5],
            "sqft_above": [1600, 1200, 2200],
            "sqft_basement": [200, 0, 300],
            "yr_built": [2010, 2005, 2018],
            "yr_renovated": [0, 2015, 0],
            "city": ["Seattle", "Bellevue", "Redmond"],
        }
    )


@pytest.fixture
def sample_data_outliers() -> pd.DataFrame:
    """Data with outliers for IQR testing."""
    return pd.DataFrame(
        {
            "date": [
                "2024-01-15",
                "2024-02-20",
                "2024-03-10",
                "2024-04-05",
                "2024-05-12",
            ],
            "price": [500000, 350000, 750000, 250000, 600000],
            "bedrooms": [3, 2, 4, 1, 3],
            "bathrooms": [2.5, 1.5, 3.0, 1.0, 2.0],
            "sqft_living": [1800, 1200, 15000, 800, 1600],
            "sqft_lot": [5000, 3500, 8000000, 2000, 6000],
            "sqft_above": [1600, 1200, 14000, 800, 1400],
            "floors": [1.5, 1.0, 2.0, 1.0, 1.0],
            "waterfront": [0, 0, 1, 0, 0],
            "view": [0, 1, 2, 0, 0],
            "condition": [3, 4, 5, 2, 3],
            "sqft_basement": [200, 0, 300, 0, 200],
            "yr_built": [2010, 2005, 2018, 1995, 2008],
            "yr_renovated": [0, 2015, 0, 0, 2020],
            "city": ["Seattle", "Bellevue", "Redmond", "Seattle", "Kirkland"],
        }
    )


@pytest.fixture
def sample_data_missing() -> pd.DataFrame:
    """Data with missing values."""
    return pd.DataFrame(
        {
            "date": ["2024-01-15", "2024-02-20", None, "2024-04-05", "2024-05-12"],
            "price": [500000, 350000, 750000, 250000, 600000],
            "bedrooms": [3, 2, 4, 1, None],
            "bathrooms": [2.5, 1.5, None, 1.0, 2.0],
            "sqft_living": [1800, 1200, 2500, 800, None],
            "sqft_lot": [5000, 3500, 8000, 2000, 6000],
            "floors": [1.5, 1.0, 2.0, 1.0, 1.0],
            "waterfront": [0, 0, 1, 0, 0],
            "view": [0, 1, 2, 0, 0],
            "condition": [3, 4, 5, 2, 3],
            "sqft_above": [1600, 1200, 2200, 800, 1400],
            "sqft_basement": [200, 0, 300, 0, 200],
            "yr_built": [2010, 2005, 2018, 1995, 2008],
            "yr_renovated": [0, 2015, 0, 0, 2020],
            "city": ["Seattle", "Bellevue", "Redmond", "Seattle", "Kirkland"],
        }
    )


@pytest.fixture
def engineered_data():
    """Fixture with already engineered data (22 columns)."""
    return pd.DataFrame(
        {
            "price": [500000, 350000, 750000, 250000, 600000, 900000],
            "bedrooms": [3, 2, 4, 1, 3, 5],
            "bathrooms": [2.5, 1.5, 3.0, 1.0, 2.0, 3.5],
            "sqft_living": [1800, 1200, 2500, 800, 1600, 3000],
            "sqft_lot": [5000, 3500, 8000, 2000, 6000, 10000],
            "floors": [1.5, 1.0, 2.0, 1.0, 1.0, 2.0],
            "waterfront": [0, 0, 1, 0, 0, 0],
            "view": [0, 1, 2, 0, 0, 3],
            "condition": [3, 4, 5, 2, 3, 4],
            "sqft_above": [1600, 1200, 2200, 800, 1400, 2700],
            "sqft_basement": [200, 0, 300, 0, 200, 300],
            "month": [1, 2, 3, 4, 5, 6],
            "day": [15, 20, 10, 5, 12, 1],
            "age_at_sale": [14, 21, 8, 31, 18, 11],
            "was_renovated": [0, 1, 0, 0, 1, 0],
            "years_since_renovation": [0, 11, 0, 0, 6, 0],
            "total_sqft": [2000, 1200, 2800, 800, 1800, 3300],
            "city": [
                "Seattle",
                "Bellevue",
                "Redmond",
                "Seattle",
                "Kirkland",
                "Seattle",
            ],
        }
    )


@pytest.fixture
def csv_file(sample_data, tmp_path):
    """Temporary CSV file for testing."""
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path, index=False)
    return file_path
