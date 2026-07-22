import pandas as pd


class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, X_test):
        return self.model.predict(X_test)
