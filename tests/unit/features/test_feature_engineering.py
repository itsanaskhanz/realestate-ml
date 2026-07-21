from realestate_ml.features import FeatureEngineer


class TestFeatureEngineer:
    def test_engineer(self, sample_data):
        fe = FeatureEngineer(sample_data)
        df = fe.engineer()
        assert df.shape[1] == 22
