"""Target encoding, label mapping, and feature matrix builder."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.core.types import FeatureMatrix


TURNOVER_LABEL_MAPPING: Dict[str, int] = {
    "Illiquid_Constrained": 0,
    "Moderate_Liquid": 1,
    "High_Liquid_Active": 2,
    "Hyper_Liquid_Turnover": 3,
}

TURNOVER_INVERSE_MAPPING: Dict[int, str] = {
    v: k for k, v in TURNOVER_LABEL_MAPPING.items()
}

ENGINEERED_FEATURE_COLUMNS: Tuple[str, ...] = (
    "log_market_cap",
    "log_volume_24h",
    "turnover_ratio",
    "amihud_illiquidity_proxy",
    "corwin_schultz_spread",
    "high_low_range_pct",
    "circulating_supply_ratio",
    "fdv_overhang_multiplier",
    "volume_to_fdv_ratio",
    "is_capped_supply",
    "ath_drawdown_depth",
    "atl_expansion_log",
    "market_cap_velocity_ratio",
    "price_change_percentage_24h",
)


def encode_feature_matrix(df: pd.DataFrame) -> FeatureMatrix:
    """Build clean FeatureMatrix containing predictors, targets, and identifiers.

    Args:
        df: Feature-engineered dataframe.

    Returns:
        FeatureMatrix: Immutable structure with numerical features and targets.
    """
    feature_cols = [c for c in ENGINEERED_FEATURE_COLUMNS if c in df.columns]
    x_matrix = df[feature_cols].copy()

    # Fill any edge case non-finite with column medians
    for col in feature_cols:
        x_matrix[col] = x_matrix[col].replace([np.inf, -np.inf], np.nan)
        if x_matrix[col].isnull().any():
            x_matrix[col] = x_matrix[col].fillna(x_matrix[col].median())

    # Map multi-class target
    y_turnover = df["liquidity_turnover_regime"].map(TURNOVER_LABEL_MAPPING).values
    if np.isnan(y_turnover).any():
        raise ValueError("Unmapped values found in liquidity_turnover_regime.")

    y_tail_risk = df["is_catastrophic_drawdown"].values

    identifiers = df[
        [
            "id",
            "symbol",
            "name",
            "market_cap_rank",
            "current_price",
            "market_cap",
            "total_volume",
            "market_cap_tier",
            "drawdown_from_ath_severity",
            "dilution_risk_profile",
        ]
    ].copy()

    return FeatureMatrix(
        features=x_matrix,
        feature_names=tuple(feature_cols),
        target_turnover=y_turnover.astype(int),
        target_tail_risk=y_tail_risk.astype(int),
        asset_identifiers=identifiers,
    )
