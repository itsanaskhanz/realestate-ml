from realestate_ml.config import CONFIG
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from realestate_ml.data import DataLoader, DataCleaner, DataValidator
from realestate_ml.features import FeatureEngineer, DataPreprocessor
from .evaluator import Evaluator
from sklearn.compose import ColumnTransformer


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

        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(
            engineered_data, save=True
        )
        top_5_cities = X_train["city"].value_counts().head(5).index.tolist()

        encoder = OneHotEncoder(
            categories=[top_5_cities], handle_unknown="ignore", sparse_output=False
        )

        numerical_columns = X_train.drop(columns=["city"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("city", encoder, ["city"]),
                ("num", StandardScaler(), numerical_columns),
            ]
        )

        model = Pipeline(
            [
                ("preprocessor", preprocessor),
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


if __name__ == "__main__":
    trainer = Trainer()
    model = trainer.train(save=True)
    print("Training completed successfully!")
