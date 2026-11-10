"""Unit tests for microstructure, Amihud proxy, and tokenomics feature calculations."""

import numpy as np
import pandas as pd
from src.feature_engineering.microstructure_features import compute_microstructure_features


def test_microstructure_features_are_pure_and_deterministic() -> None:
    """Verify that feature extraction is pure and does not mutate input dataframe."""
    sample_df = pd.DataFrame(
        {
            "current_price": [100.0, 50.0],
            "market_cap": [1_000_000.0, 500_000.0],
            "total_volume": [100_000.0, 50_000.0],
            "high_24h": [105.0, 52.0],
            "low_24h": [95.0, 48.0],
            "price_change_percentage_24h": [2.5, -3.0],
            "market_cap_change_24h": [25000.0, -15000.0],
            "circulating_supply": [10_000.0, 10_000.0],
            "total_supply": [12_000.0, 15_000.0],
            "max_supply": [20_000.0, 0.0],
            "fully_diluted_valuation": [2_000_000.0, 500_000.0],
            "ath_change_percentage": [-20.0, -88.5],
            "atl_change_percentage": [500.0, 120.0],
        }
    )
    original_copy = sample_df.copy()

    augmented = compute_microstructure_features(sample_df)

    # Prove immutability
    pd.testing.assert_frame_equal(sample_df, original_copy)

    # Prove correct feature calculations
    assert "turnover_ratio" in augmented.columns
    assert np.isclose(augmented.loc[0, "turnover_ratio"], 100_000.0 / 1_000_001.0)

    assert "amihud_illiquidity_proxy" in augmented.columns
    expected_amihud = 2.5 / ((100_000.0 / 1e6) + 1.0)
    assert np.isclose(augmented.loc[0, "amihud_illiquidity_proxy"], expected_amihud)

    assert "corwin_schultz_spread" in augmented.columns
    assert np.isclose(augmented.loc[0, "corwin_schultz_spread"], (105.0 - 95.0) / 100.0)

    assert "circulating_supply_ratio" in augmented.columns
    assert np.isclose(augmented.loc[0, "circulating_supply_ratio"], 10_000.0 / 12_000.0)

    # Prove tail risk indicator correctly flags >85% collapse
    assert augmented.loc[0, "is_catastrophic_drawdown"] == 0
    assert augmented.loc[1, "is_catastrophic_drawdown"] == 1

