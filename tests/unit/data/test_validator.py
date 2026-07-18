import pytest
from realestate_ml.data.validator import DataValidator


class TestDataValidator:
    """Test suite for DataValidator class."""

    def test_validate(self, sample_data):
        """Test validation passes for clean data with no errors."""
        validator = DataValidator()
        is_valid, errors = validator.validate(sample_data)
        assert is_valid
        assert not errors
