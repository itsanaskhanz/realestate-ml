from realestate_ml.features import FeatureEngineer
import pandas as pd


class TestFeatureEngineer:
    def test_engineer(self, sample_data):
        fe = FeatureEngineer(sample_data)
        df = fe.engineer(save=False)
        assert df.shape[1] == 18
