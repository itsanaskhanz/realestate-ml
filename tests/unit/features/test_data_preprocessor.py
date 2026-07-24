from realestate_ml.features import DataPreprocessor, build_column_transformer


class TestDataPreprocessor:
    def test_train_test_split(self, engineered_data):
        preprocessor = DataPreprocessor(engineered_data)
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(save=False)
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert "price" not in X_train.columns
        assert "price" not in X_test.columns

    def test_build_processor(self, engineered_data):
        preprocessor = DataPreprocessor(engineered_data)
        X_train, _, _, _ = preprocessor.train_test_split(save=False)
        processor = preprocessor.build_processor(X_train)
        assert processor is not None

    def test_build_processor_uses_training_cities_only(self, engineered_data):
        preprocessor = DataPreprocessor(engineered_data)
        X_train, _, _, _ = preprocessor.train_test_split(save=False)

        processor = build_column_transformer(X_train, target_column="price")
        city_encoder = processor.transformers[0][1]
        expected_cities = X_train["city"].value_counts().head(5).index.tolist()

        assert city_encoder.categories[0] == expected_cities
