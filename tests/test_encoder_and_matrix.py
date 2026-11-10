"""Unit tests for target encoding and FeatureMatrix construction."""

import numpy as np
import pandas as pd
from src.core.types import FeatureMatrix
from src.feature_engineering.encoder import TURNOVER_LABEL_MAPPING, encode_feature_matrix
from src.feature_engineering.microstructure_features import compute_microstructure_features


def test_encoder_constructs_valid_feature_matrix() -> None:
    """Verify that encode_feature_matrix produces non-empty, finite FeatureMatrix."""
    raw_df = pd.DataFrame(
        {
            "id": ["btc", "eth"],
            "symbol": ["BTC", "ETH"],
            "name": ["Bitcoin", "Ethereum"],
            "market_cap_rank": [1, 2],
            "current_price": [80000.0, 2500.0],
            "market_cap": [1.6e12, 3.0e11],
            "total_volume": [2.5e10, 1.0e10],
            "high_24h": [81000.0, 2550.0],
            "low_24h": [79000.0, 2450.0],
            "price_change_percentage_24h": [-1.2, 0.8],
            "market_cap_change_24h": [-2e10, 5e9],
            "circulating_supply": [1.9e7, 1.2e8],
            "total_supply": [1.9e7, 1.2e8],
            "max_supply": [2.1e7, 0.0],
            "fully_diluted_valuation": [1.7e12, 3.0e11],
            "ath_change_percentage": [-35.0, -50.0],
            "atl_change_percentage": [100000.0, 50000.0],
            "liquidity_turnover_regime": ["Illiquid_Constrained", "Moderate_Liquid"],
            "market_cap_tier": ["Mega_Cap_Sovereign", "Mega_Cap_Sovereign"],
            "drawdown_from_ath_severity": ["Moderate_Correction", "Moderate_Correction"],
            "dilution_risk_profile": ["Zero_Inflation_Capped", "Low_Moderate_Dilution"],
        }
    )

    engineered = compute_microstructure_features(raw_df)
    matrix = encode_feature_matrix(engineered)

    assert isinstance(matrix, FeatureMatrix)
    assert len(matrix.features) == 2
    assert len(matrix.target_turnover) == 2
    assert matrix.target_turnover[0] == TURNOVER_LABEL_MAPPING["Illiquid_Constrained"]
    assert matrix.target_turnover[1] == TURNOVER_LABEL_MAPPING["Moderate_Liquid"]
    assert matrix.target_tail_risk[0] == 0
    assert matrix.target_tail_risk[1] == 0
    assert not matrix.features.isnull().any().any()
