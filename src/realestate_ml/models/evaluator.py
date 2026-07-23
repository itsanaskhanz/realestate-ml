from realestate_ml.config import CONFIG
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path


class Evaluator:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def evaluate(self, save: bool = False) -> pd.DataFrame:
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        result = pd.DataFrame({"MAE": mae, "R2": r2}, index=[0])
        if save:
            path = Path(CONFIG["data"]["reports_path"])
            path.mkdir(parents=True, exist_ok=True)
            result.to_csv(path / "final_evaluation.csv", index=False)
        return result
