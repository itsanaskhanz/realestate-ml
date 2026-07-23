import pytest
import pandas as pd
from pathlib import Path
from realestate_ml.models import Predictor
from realestate_ml.config import CONFIG


class TestPredictor:
    def test_predictor_initialization(self):
        """Test that Predictor initializes correctly."""
        test_house = {
            "date": "2015-06-15",
            "price": 0,
            "sqft_living": 1850,
            "bedrooms": 3,
            "bathrooms": 2.5,
            "yr_built": 2008,
            "city": "Seattle",
            "sqft_lot": 6500,
            "floors": 2.0,
            "view": 1,
            "condition": 3,
            "sqft_above": 1650,
            "sqft_basement": 200,
            "yr_renovated": 2019,
            "waterfront": 0,
        }

        predictor = Predictor(test_house)
        assert predictor.user_input is not None
        assert isinstance(predictor.user_input, pd.DataFrame)
        assert len(predictor.user_input) == 1

    def test_predictor_model_path(self):
        """Test that model path is correct."""
        test_house = {"date": "2015-06-15", "price": 0, "sqft_living": 1850}
        predictor = Predictor(test_house)
        expected_path = Path(CONFIG["data"]["models_path"]) / "best_model.pkl"
        assert predictor.model_path == expected_path
