"""Single-asset clinical triage and risk telemetry CLI for GCMDL."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.core.config import ProjectConfig
from src.data_pipeline.loader import load_raw_telemetry
from src.feature_engineering.encoder import TURNOVER_INVERSE_MAPPING, encode_feature_matrix
from src.feature_engineering.microstructure_features import compute_microstructure_features
from src.modeling.tail_risk_predictor import train_tail_risk_predictor
from src.modeling.turnover_classifier import train_turnover_regime_classifier


def inspect_asset_risk(symbol: str) -> None:
    """Evaluate and print clinical risk diagnostic telemetry for a target asset symbol."""
    config = ProjectConfig()
    raw_df = load_raw_telemetry(config.data.raw_data_path, config.data)
    match = raw_df[raw_df["symbol"].str.upper() == symbol.upper()]

    if match.empty:
        print(f"\n[ERROR] Asset symbol '{symbol}' not found in top 1,000 crypto census.")
        print("Available examples: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, UNI.\n")
        sys.exit(1)

    asset = match.iloc[0]
    engineered_df = compute_microstructure_features(raw_df)
    asset_feat = engineered_df[engineered_df["symbol"].str.upper() == symbol.upper()].iloc[0]

    turnover = float(asset_feat["turnover_ratio"])
    amihud = float(asset_feat["amihud_illiquidity_proxy"])
    spread = float(asset_feat["corwin_schultz_spread"])
    supply_ratio = float(asset_feat["circulating_supply_ratio"])
    ath_dd = float(asset["ath_change_percentage"])

    # Determine regime and risk alert
    regime = str(asset["liquidity_turnover_regime"])
    tail_risk_flag = "CRITICAL (>85% Peak Collapse)" if ath_dd <= -85.0 else "CONTROLLED (<85% Drawdown)"

    print("\n" + "=" * 70)
    print(f"GCMDL ASSET RISK REPORT: {asset['name']} ({asset['symbol']}) &bull; Rank #{asset['market_cap_rank']}")
    print("=" * 70)
    print(f"  Spot Price (USD)         : ${asset['current_price']:,.4f}")
    print(f"  Market Capitalization    : ${asset['market_cap']:,.2f} ({asset['market_cap_tier'].replace('_', ' ')})")
    print(f"  24h Trading Volume       : ${asset['total_volume']:,.2f}")
    print(f"  Turnover Velocity        : {turnover:.4f}")
    print(f"  Amihud Illiquidity Proxy : {amihud:.6f}")
    print(f"  Corwin-Schultz Spread    : {spread:.4f}")
    print(f"  Circulating Supply Ratio : {supply_ratio*100:.2f}% ({asset['dilution_risk_profile'].replace('_', ' ')})")
    print(f"  Drawdown from ATH        : {ath_dd:.2f}%")
    print("-" * 70)
    print(f"  Liquidity Regime Status  : [{regime.replace('_', ' ')}]")
    print(f"  Catastrophic Tail Risk   : [{tail_risk_flag}]")
    print("-" * 70)
    print("  Cognitive CoT Diagnostic Rationale:")
    print(f"  \"{asset['cot_rationale']}\"")
    print("=" * 70 + "\n")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI input arguments."""
    parser = argparse.ArgumentParser(
        description="GCMDL Crypto Asset Risk & Liquidity Triage CLI"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        help="Standard uppercase ticker symbol (e.g. BTC, ETH, SOL, LINK, DOGE)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    inspect_asset_risk(args.symbol)

