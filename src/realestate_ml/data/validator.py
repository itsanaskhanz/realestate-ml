import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate data quality and integrity.

    This class checks for:
    - Required columns exist
    - No invalid values (<= 0) in critical columns
    - No missing values
    """

    REQUIRED_COLUMNS = [
        "date",
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "street",
        "city",
        "statezip",
        "country",
    ]

    POSITIVE_COLUMNS = ["price", "bedrooms", "bathrooms", "sqft_living"]

    def validate(self, df: pd.DataFrame) -> tuple:
        """
        Validate data quality.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid: bool, errors: list)

        Raises:
            ValueError: If required columns are missing
        """
        errors = []

        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        for col in self.POSITIVE_COLUMNS:
            invalid_count = (df[col] <= 0).sum()
            if invalid_count > 0:
                errors.append(f"{col}: {invalid_count} rows with <= 0")
                logger.warning(f"Found {invalid_count} rows with {col} <= 0")

        null_count = df.isnull().sum().sum()
        if null_count > 0:
            errors.append(f"Found {null_count} missing values")
            logger.warning(f"Found {null_count} missing values")

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Data validation passed ✓")
        else:
            logger.warning(f"Validation failed with {len(errors)} issues")

        return is_valid, errors
