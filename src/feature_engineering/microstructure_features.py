"""Microstructure, liquidity turnover, and tokenomics dilution feature extraction."""

import numpy as np
import pandas as pd


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute financial market microstructure and tokenomic risk features purely.

    Args:
        df: Raw validated telemetry dataframe (not mutated).

    Returns:
        pd.DataFrame: New DataFrame containing engineered analytical features.
    """
    out = df.copy()

    # Liquidity turnover and orderbook depth proxies
    out["turnover_ratio"] = out["total_volume"] / (out["market_cap"] + 1.0)
    out["log_market_cap"] = np.log10(np.maximum(out["market_cap"], 0.0) + 1.0)
    out["log_volume_24h"] = np.log10(np.maximum(out["total_volume"], 0.0) + 1.0)

    # Amihud (2002) Illiquidity Proxy: Price response per $1M volume
    price_impact = np.abs(out["price_change_percentage_24h"])
    norm_volume = (out["total_volume"] / 1_000_000.0) + 1.0
    out["amihud_illiquidity_proxy"] = price_impact / norm_volume

    # Corwin-Schultz (2012) High-Low Intraday Volatility Spread
    safe_price = np.maximum(out["current_price"], 1e-8)
    out["corwin_schultz_spread"] = (out["high_24h"] - out["low_24h"]) / safe_price
    out["high_low_range_pct"] = (
        (out["high_24h"] - out["low_24h"]) / np.maximum(out["low_24h"], 1e-8)
    ) * 100.0

    # Tokenomics Dilution and Supply Overhang
    safe_total_supply = np.maximum(out["total_supply"], 1e-8)
    out["circulating_supply_ratio"] = np.clip(
        out["circulating_supply"] / safe_total_supply, 0.0, 1.0
    )
    out["fdv_overhang_multiplier"] = out["fully_diluted_valuation"] / (
        out["market_cap"] + 1.0
    )
    out["volume_to_fdv_ratio"] = out["total_volume"] / (
        out["fully_diluted_valuation"] + 1.0
    )
    out["is_capped_supply"] = (out["max_supply"] > 0).astype(int)

    # Multi-Cycle Valuation and Drawdown Telemetry
    out["ath_drawdown_depth"] = np.abs(out["ath_change_percentage"]) / 100.0
    out["atl_expansion_log"] = np.log1p(np.maximum(0.0, out["atl_change_percentage"]))
    out["market_cap_velocity_ratio"] = np.abs(out["market_cap_change_24h"]) / (
        out["market_cap"] + 1.0
    )

    # Binary Tail-Risk Indicator (Catastrophic Drawdown > 85% from ATH)
    out["is_catastrophic_drawdown"] = (out["ath_change_percentage"] <= -85.0).astype(
        int
    )

    return out

