import joblib
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from realestate_ml.config import CONFIG
from realestate_ml.data import DataCleaner, DataLoader, DataValidator
from realestate_ml.features import DataPreprocessor, FeatureEngineer
from .evaluator import Evaluator


class Trainer:
    def train(self, save: bool = False):
        loader = DataLoader()
        raw_data = loader.load(Path(CONFIG["data"]["raw_path"]))

        cleaner = DataCleaner()
        cleaned_data = cleaner.clean(raw_data, save=save)

        validator = DataValidator()
        is_valid, errors = validator.validate(cleaned_data)
        if not is_valid:
            raise ValueError(f"Data validation failed: {', '.join(errors)}")

        feature_engineer = FeatureEngineer(cleaned_data)
        engineered_data = feature_engineer.engineer(save=save)

        data_preprocessor = DataPreprocessor(engineered_data)
        X_train, X_test, y_train, y_test = data_preprocessor.train_test_split(
            save=save
        )

        column_transformer = data_preprocessor.build_processor(X_train)

        model_params = {
            "random_state": CONFIG["split"]["random_state"],
            **CONFIG["model"]["params"],
        }
        model = Pipeline(
            [
                ("preprocessor", column_transformer),
                ("model", XGBRegressor(**model_params)),
            ]
        )

        model.fit(X_train, y_train)

        models_path = Path(CONFIG["data"]["models_path"])
        models_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_path / "best_model.pkl")

        y_pred = model.predict(X_test)
        evaluator = Evaluator(y_test, y_pred)
        evaluator.evaluate(save=save)
        return model


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(save=True)
    print("Training completed successfully!")
