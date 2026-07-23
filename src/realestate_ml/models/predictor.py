import pandas as pd
import joblib
from pathlib import Path
from realestate_ml.config import CONFIG
from realestate_ml.features import FeatureEngineer


class Predictor:
    def __init__(self, user_input: dict):
        """
        Initialize predictor with user input as dictionary.

        Args:
            user_input: Dictionary with feature values
                       Example: {'city': 'New York', 'bedrooms': 3, 'bathrooms': 2, ...}
        """
        self.user_input = pd.DataFrame([user_input])  # Convert dict to DataFrame
        self.model_path = Path(CONFIG["data"]["models_path"]) / "best_model.pkl"

    def predict(self) -> float:
        """
        Make single prediction for API.

        Returns:
            float: Predicted price
        """
        # Load the trained model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        model = joblib.load(self.model_path)

        # Apply feature engineering to user input
        feature_engineer = FeatureEngineer(self.user_input)
        engineered_input = feature_engineer.engineer(save=False)

        # Make prediction
        prediction = model.predict(engineered_input)

        return float(prediction[0])  # Return as float for JSON response
