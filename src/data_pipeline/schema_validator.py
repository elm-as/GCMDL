"""Schema and statistical sanity validator for cryptocurrency telemetry data."""

import pandas as pd
from src.core.config import DataConfig


def validate_raw_telemetry(df: pd.DataFrame, config: DataConfig) -> None:
    """Validate raw telemetry DataFrame against schema and boundary constraints.

    Raises:
        ValueError: If row count, columns, or data boundaries are violated.
    """
    if df.empty:
        raise ValueError("Raw telemetry dataset is empty.")

    if len(df) != config.expected_row_count:
        raise ValueError(
            f"Expected {config.expected_row_count} records, found {len(df)}."
        )

    missing_numerical = [c for c in config.raw_numerical_columns if c not in df.columns]
    if missing_numerical:
        raise ValueError(f"Missing required numerical columns: {missing_numerical}")

    missing_categorical = [
        c for c in config.categorical_annotation_columns if c not in df.columns
    ]
    if missing_categorical:
        raise ValueError(f"Missing required categorical columns: {missing_categorical}")

    if df["id"].isnull().any() or df["symbol"].isnull().any():
        raise ValueError("Null values detected in core asset identifiers (id, symbol).")

    if (df["market_cap_rank"] <= 0).any():
        raise ValueError("Invalid market_cap_rank values <= 0 detected.")

    if (df["current_price"] < 0).any():
        raise ValueError("Negative current_price values detected.")

    if (df["market_cap"] < 0).any():
        raise ValueError("Negative market_cap values detected.")


def validate_feature_matrix_shapes(
    features: pd.DataFrame,
    target_turnover: pd.Series,
    target_tail_risk: pd.Series,
) -> None:
    """Validate consistency of engineered feature matrix and target arrays.

    Raises:
        ValueError: If shapes are mismatched or null values exist.
    """
    if len(features) != len(target_turnover) or len(features) != len(target_tail_risk):
        raise ValueError("Length mismatch between features and target series.")

    if features.isnull().any().any():
        null_counts = features.isnull().sum()[lambda x: x > 0].to_dict()
        raise ValueError(f"Null values present in engineered features: {null_counts}")

