from realestate_ml.config import CONFIG
import pandas as pd
from xgboost import XGBRegressor


class Trainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def train(self):
        model = XGBRegressor(**CONFIG["model"]["params"])
        model.fit(self.X_train, self.y_train)
        return model
