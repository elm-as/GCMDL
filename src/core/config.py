"""Global configuration and hyperparameters for GCMDL."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DataConfig:
    """Paths and technical specifications for raw and processed telemetry."""

    raw_data_path: Path = Path("data/raw/crypto_market_dynamics_1000.csv")
    processed_data_path: Path = Path("data/processed/crypto_market_engineered.parquet")
    artifacts_dir: Path = Path("artifacts")
    images_dir: Path = Path("images")
    notebooks_dir: Path = Path("notebooks")

    expected_row_count: int = 1000
    expected_column_count: int = 32

    raw_numerical_columns: Tuple[str, ...] = (
        "current_price",
        "market_cap",
        "market_cap_rank",
        "fully_diluted_valuation",
        "total_volume",
        "high_24h",
        "low_24h",
        "price_change_24h",
        "price_change_percentage_24h",
        "market_cap_change_24h",
        "market_cap_change_percentage_24h",
        "circulating_supply",
        "total_supply",
        "max_supply",
        "ath",
        "ath_change_percentage",
        "atl",
        "atl_change_percentage",
    )

    categorical_annotation_columns: Tuple[str, ...] = (
        "market_cap_tier",
        "liquidity_turnover_regime",
        "drawdown_from_ath_severity",
        "dilution_risk_profile",
        "short_term_momentum_regime",
    )


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters and cross-validation parameters."""

    random_seed: int = 42
    n_splits: int = 5

    # LightGBM Hyperparameters
    lgb_n_estimators: int = 300
    lgb_learning_rate: float = 0.04
    lgb_max_depth: int = 4
    lgb_num_leaves: int = 15
    lgb_subsample: float = 0.85
    lgb_colsample_bytree: float = 0.85

    # CatBoost Hyperparameters
    cb_iterations: int = 350
    cb_learning_rate: float = 0.04
    cb_depth: int = 4
    cb_l2_leaf_reg: float = 5.0

    # Drawdown Severity Threshold
    catastrophic_drawdown_cutoff_pct: float = -85.0


@dataclass(frozen=True)
class ProjectConfig:
    """Root configuration aggregator for GCMDL."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

