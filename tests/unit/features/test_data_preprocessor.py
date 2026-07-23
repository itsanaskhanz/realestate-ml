from realestate_ml.features import DataPreprocessor


class TestDataPreprocessor:
    def test_train_test_split(self, engineered_data):
        preprocessor = DataPreprocessor(engineered_data)
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(save=False)
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
