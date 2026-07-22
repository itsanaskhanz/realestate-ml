from realestate_ml.config import CONFIG
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from realestate_ml.data import DataLoader, DataCleaner, DataValidator
from realestate_ml.features import FeatureEngineer, DataPreprocessor
from .evaluator import Evaluator


class Trainer:

    def train(self):
        loader = DataLoader()
        raw_data = loader.load(Path(CONFIG["data"]["raw_path"]))

        cleaner = DataCleaner()
        cleaned_data = cleaner.clean(raw_data)

        validator = DataValidator()
        is_valid, errors = validator.validate(cleaned_data)
        if is_valid:
            cleaned_data.to_csv(
                Path(CONFIG["data"]["interim_path"]) / "cleaned_data.csv",
                index=False,
            )
        else:
            raise ValueError("Data validation failed")

        feature_engineer = FeatureEngineer(cleaned_data)
        engineered_data = feature_engineer.engineer()

        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(
            engineered_data
        )

        model = Pipeline(
            [
                ("preprocessor", StandardScaler()),
                ("model", XGBRegressor(**CONFIG["model"]["params"])),
            ]
        )
        model.fit(X_train, y_train)
        path = Path(CONFIG["data"]["models_path"])
        joblib.dump(model, path / "best_model.pkl")
        y_pred = model.predict(X_test)
        evaluator = Evaluator(y_test, y_pred)
        evaluator.evaluate()
        return model
