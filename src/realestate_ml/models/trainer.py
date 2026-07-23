from realestate_ml.config import CONFIG
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from realestate_ml.data import DataLoader, DataCleaner, DataValidator
from realestate_ml.features import FeatureEngineer, DataPreprocessor
from .evaluator import Evaluator


class Trainer:

    def train(self, save: bool = False):
        loader = DataLoader()
        raw_data = loader.load(Path(CONFIG["data"]["raw_path"]))

        cleaner = DataCleaner()
        cleaned_data = cleaner.clean(raw_data, save=True)

        validator = DataValidator()
        is_valid, errors = validator.validate(cleaned_data)
        if not is_valid:
            raise ValueError(f"Data validation failed: {', '.join(errors)}")

        feature_engineer = FeatureEngineer(cleaned_data)
        engineered_data = feature_engineer.engineer(save=True)

        data_preprocessor = DataPreprocessor(engineered_data)
        X_train, X_test, y_train, y_test = data_preprocessor.train_test_split(save=True)

        # Build the preprocessor
        column_transformer = data_preprocessor.build_processor()

        model = Pipeline(
            [
                ("preprocessor", column_transformer),
                ("model", XGBRegressor(**CONFIG["model"]["params"])),
            ]
        )

        model.fit(X_train, y_train)
        path = Path(CONFIG["data"]["models_path"])
        joblib.dump(model, path / "best_model.pkl")
        y_pred = model.predict(X_test)
        evaluator = Evaluator(y_test, y_pred)
        evaluator.evaluate(save=True)
        return model


if __name__ == "__main__":
    trainer = Trainer()
    model = trainer.train(save=True)
    print("Training completed successfully!")
