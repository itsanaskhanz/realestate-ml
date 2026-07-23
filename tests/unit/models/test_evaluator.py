import pytest
import pandas as pd
import numpy as np
from realestate_ml.models import Evaluator


class TestEvaluator:
    def test_evaluate(self):
        # Simple dummy data
        y_test = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([1.1, 1.9, 3.2, 3.8, 5.1])

        evaluator = Evaluator(y_test, y_pred)
        result = evaluator.evaluate()

        assert isinstance(result, pd.DataFrame)
