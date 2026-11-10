"""Unit tests for telemetry schema validation and boundary checks."""

import pandas as pd
import pytest
from src.core.config import DataConfig
from src.data_pipeline.schema_validator import (
    validate_feature_matrix_shapes,
    validate_raw_telemetry,
)


def test_schema_validator_rejects_empty_dataframe() -> None:
    """Verify that an empty DataFrame triggers an immediate ValueError."""
    config = DataConfig()
    with pytest.raises(ValueError, match="empty"):
        validate_raw_telemetry(pd.DataFrame(), config)


def test_schema_validator_rejects_missing_columns() -> None:
    """Verify that omission of expected numerical columns triggers ValueError."""
    config = DataConfig()
    incomplete = pd.DataFrame(
        {
            "id": ["bitcoin"] * 1000,
            "symbol": ["BTC"] * 1000,
            "market_cap_rank": list(range(1, 1001)),
            "current_price": [50000.0] * 1000,
            "market_cap": [1e9] * 1000,
        }
    )
    with pytest.raises(ValueError, match="Missing required numerical columns"):
        validate_raw_telemetry(incomplete, config)


def test_schema_validator_rejects_negative_market_cap() -> None:
    """Verify that negative financial values trigger an immediate ValueError."""
    config = DataConfig()
    df = pd.DataFrame(
        {col: [1.0] * 1000 for col in config.raw_numerical_columns}
    )
    for cat in config.categorical_annotation_columns:
        df[cat] = "test_cat"
    df["id"] = "test_id"
    df["symbol"] = "TST"
    df["market_cap_rank"] = 1
    df.loc[0, "market_cap"] = -500.0

    with pytest.raises(ValueError, match="Negative market_cap"):
        validate_raw_telemetry(df, config)


def test_feature_matrix_validator_detects_shape_mismatches() -> None:
    """Verify that length discrepancies between X and y are caught."""
    x = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
    y1 = pd.Series([0, 1])
    y2 = pd.Series([0, 1, 0])
    with pytest.raises(ValueError, match="Length mismatch"):
        validate_feature_matrix_shapes(x, y1, y2)

