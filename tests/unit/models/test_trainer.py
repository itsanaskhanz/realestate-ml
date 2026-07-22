from realestate_ml.models import Trainer
from xgboost import XGBRegressor
from realestate_ml.data import DataLoader
from pathlib import Path


class TestTrainer:
    def test_train(self):
        loader = DataLoader()
        X_train = loader.load(Path("data/processed/X_train.csv"))
        y_train = loader.load(Path("data/processed/y_train.csv"))
        trainer = Trainer(X_train, y_train)
        model = trainer.train()

        assert isinstance(model, XGBRegressor)
