import pytest
from realestate_ml.models import Trainer


class TestTrainer:
    def test_trainer_initialization(self):
        """Test that Trainer initializes correctly."""
        trainer = Trainer()
        assert trainer is not None

    def test_trainer_has_train_method(self):
        """Test that Trainer has a train method."""
        trainer = Trainer()
        assert hasattr(trainer, "train")
        assert callable(trainer.train)
