import joblib
import numpy as np
from pathlib import Path
from realestate_ml.models import Predictor
from realestate_ml.data import DataLoader


class TestPredictor:
    def test_predict(self):
        model = joblib.load("models/best_model.pkl")
        predictor = Predictor(model)
        loader = DataLoader()
        X_test = loader.load(Path("data/processed/X_test.csv"))
        y_pred = predictor.predict(X_test)
        assert isinstance(y_pred, np.ndarray)
