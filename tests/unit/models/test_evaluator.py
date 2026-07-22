import pytest
import joblib
import pandas as pd
from pathlib import Path
from realestate_ml.models import Evaluator, Predictor
from realestate_ml.data import DataLoader


class TestEvaluator:
    def test_evaluate(self):
        loader = DataLoader()
        y_true = loader.load(Path("data/processed/y_test.csv"))
        X_test = loader.load(Path("data/processed/X_test.csv"))
        model = joblib.load("models/best_model.pkl")
        predictor = Predictor(model)
        y_pred = predictor.predict(X_test)
        evaluator = Evaluator(y_true, y_pred)
        result = evaluator.evaluate()
        assert isinstance(result, pd.DataFrame)
