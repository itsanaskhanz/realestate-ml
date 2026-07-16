import pandas as pd


class DataValidator:
    """Validates housing data"""

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check data quality and return validation results"""

        # Required columns for housing data
        required = [
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

        # Check for missing columns
        missing = [col for col in required if col not in df.columns]
        errors = []

        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        # Validate numeric fields
        if (df["price"] <= 0).any():
            errors.append(f"Found {sum(df['price'] <= 0)} rows with price <= 0")

        if (df["bedrooms"] <= 0).any():
            errors.append(
                f"Found {(df['bedrooms'] <= 0).sum()} rows with bedrooms <= 0"
            )

        if (df["bathrooms"] <= 0).any():
            errors.append(
                f"Found {(df['bathrooms'] <= 0).sum()} rows with bathrooms <= 0"
            )

        if (df["sqft_living"] <= 0).any():
            errors.append(
                f"Found {sum(df['sqft_living'] <= 0)} rows with sqft_living <= 0"
            )

        # Check for null values
        total_missing = df.isnull().sum().sum()

        if total_missing > 0:
            errors.append(f"Found {total_missing} missing values")

        return len(errors) == 0, errors
