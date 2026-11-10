"""Integration tests for model estimation, metrics, and cross-validation bounds."""

import numpy as np
import pandas as pd
from src.core.config import ModelConfig
from src.core.types import ClassificationMetrics, FeatureMatrix, TailRiskMetrics
from src.modeling.tail_risk_predictor import train_tail_risk_predictor
from src.modeling.turnover_classifier import train_turnover_regime_classifier


def _build_synthetic_feature_matrix(n_samples: int = 120) -> FeatureMatrix:
    """Helper creating reproducible mini synthetic FeatureMatrix for test isolation."""
    rng = np.random.default_rng(42)
    n_features = 8
    feature_names = tuple(f"feat_{i}" for i in range(n_features))
    x_data = rng.normal(size=(n_samples, n_features))
    features_df = pd.DataFrame(x_data, columns=feature_names)

    # 4-class target for turnover, stratified
    target_turnover = np.tile(np.arange(4), n_samples // 4)
    # Binary target for tail risk
    target_tail_risk = rng.binomial(1, 0.45, size=n_samples)

    identifiers = pd.DataFrame(
        {
            "id": [f"asset_{i}" for i in range(n_samples)],
            "symbol": [f"AST{i}" for i in range(n_samples)],
            "name": [f"Asset {i}" for i in range(n_samples)],
            "market_cap_rank": list(range(1, n_samples + 1)),
            "current_price": [10.0] * n_samples,
            "market_cap": [1e8] * n_samples,
            "total_volume": [1e6] * n_samples,
            "market_cap_tier": ["Small_Cap_Ecosystem"] * n_samples,
            "drawdown_from_ath_severity": ["Moderate_Correction"] * n_samples,
            "dilution_risk_profile": ["Low_Moderate_Dilution"] * n_samples,
        }
    )

    return FeatureMatrix(
        features=features_df,
        feature_names=feature_names,
        target_turnover=target_turnover,
        target_tail_risk=target_tail_risk,
        asset_identifiers=identifiers,
    )


def test_turnover_classifier_produces_bounded_metrics() -> None:
    """Verify that multi-class turnover classifier completes and yields valid metrics."""
    data = _build_synthetic_feature_matrix(n_samples=100)
    config = ModelConfig(
        n_splits=3,
        lgb_n_estimators=30,
        cb_iterations=30,
        random_seed=42,
    )

    metrics, oof_probs, importances = train_turnover_regime_classifier(data, config)

    assert isinstance(metrics, ClassificationMetrics)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert metrics.multi_class_log_loss > 0.0
    assert oof_probs.shape == (100, 4)
    assert np.allclose(oof_probs.sum(axis=1), 1.0)


def test_tail_risk_predictor_produces_bounded_metrics() -> None:
    """Verify that binary tail risk predictor yields valid ROC-AUC and curves."""
    data = _build_synthetic_feature_matrix(n_samples=100)
    config = ModelConfig(
        n_splits=3,
        lgb_n_estimators=30,
        cb_iterations=30,
        random_seed=42,
    )

    metrics, oof_probs, shap_vals, importances = train_tail_risk_predictor(
        data, config
    )

    assert isinstance(metrics, TailRiskMetrics)
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.sensitivity <= 1.0
    assert 0.0 <= metrics.specificity <= 1.0
    assert len(oof_probs) == 100
    assert (oof_probs >= 0.0).all() and (oof_probs <= 1.0).all()

