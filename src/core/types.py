"""Strongly typed domain models and results for GCMDL."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketTelemetryRecord:
    """Immutable representation of a single cryptocurrency asset record."""

    id: str
    symbol: str
    name: str
    current_price: float
    market_cap: float
    market_cap_rank: int
    fully_diluted_valuation: float
    total_volume: float
    high_24h: float
    low_24h: float
    price_change_percentage_24h: float
    circulating_supply: float
    total_supply: float
    max_supply: float
    ath: float
    ath_change_percentage: float
    atl: float
    atl_change_percentage: float
    liquidity_turnover_regime: str
    drawdown_from_ath_severity: str
    dilution_risk_profile: str


@dataclass(frozen=True)
class FeatureMatrix:
    """Preprocessed features and targets ready for estimation."""

    features: pd.DataFrame
    feature_names: Tuple[str, ...]
    target_turnover: np.ndarray
    target_tail_risk: np.ndarray
    asset_identifiers: pd.DataFrame


@dataclass(frozen=True)
class ClassificationMetrics:
    """Multi-class performance metrics for liquidity turnover regime."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    multi_class_log_loss: float
    brier_score: float
    confusion_matrix: np.ndarray
    per_class_f1: Dict[str, float]


@dataclass(frozen=True)
class TailRiskMetrics:
    """Binary performance metrics for catastrophic drawdown (>85%) prediction."""

    roc_auc: float
    pr_auc: float
    optimal_threshold: float
    sensitivity: float
    specificity: float
    brier_score: float
    fpr_array: np.ndarray
    tpr_array: np.ndarray
    precision_array: np.ndarray
    recall_array: np.ndarray


@dataclass(frozen=True)
class PipelineRunResult:
    """Aggregated outcome of the complete empirical pipeline execution."""

    total_records: int
    turnover_metrics: ClassificationMetrics
    tail_risk_metrics: TailRiskMetrics
    top_liquidity_drivers: List[Tuple[str, float]]
    top_tail_risk_drivers: List[Tuple[str, float]]
    dashboard_path: str
    artifacts_dir: str

