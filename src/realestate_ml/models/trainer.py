from realestate_ml.config import CONFIG
import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Trainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def train(self):

        model = Pipeline(
            [
                ("preprocessor", StandardScaler()),
                ("model", XGBRegressor(**CONFIG["model"]["params"])),
            ]
        )
        model.fit(self.X_train, self.y_train)
        return model
