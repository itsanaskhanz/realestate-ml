import numpy as np
import pandas as pd
from realestate_ml.models import Predictor


class TestPredictor:
    def test_predict(self):
        class DummyModel:
            def predict(self, X):
                return np.array([1.1, 2.2, 3.3, 4.4, 5.5])

        # Create dummy data
        X_test = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )

        # Test predictor
        model = DummyModel()
        predictor = Predictor(model)
        y_pred = predictor.predict(X_test)

        assert isinstance(y_pred, np.ndarray)
