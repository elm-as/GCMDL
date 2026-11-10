"""Multi-class GBDT and linear ensemble for liquidity turnover regime classification."""

from typing import Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from src.core.config import ModelConfig
from src.core.types import ClassificationMetrics, FeatureMatrix
from src.feature_engineering.encoder import TURNOVER_INVERSE_MAPPING


def train_turnover_regime_classifier(
    data: FeatureMatrix,
    config: ModelConfig,
) -> Tuple[ClassificationMetrics, np.ndarray, Dict[str, np.ndarray]]:
    """Train 5-fold cross-validated multi-class ensemble for liquidity regimes.

    Args:
        data: Preprocessed FeatureMatrix.
        config: Model hyperparameters.

    Returns:
        Tuple: (ClassificationMetrics, oof_probabilities, feature_importances)
    """
    x = data.features.values
    y = data.target_turnover
    n_samples, n_classes = len(y), len(np.unique(y))

    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.random_seed
    )
    oof_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    lgb_importances = np.zeros(x.shape[1], dtype=np.float64)
    cb_importances = np.zeros(x.shape[1], dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        x_tr, y_tr = x[train_idx], y[train_idx]
        x_va, y_va = x[val_idx], y[val_idx]

        # Model 1: LightGBM Multiclass
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=config.lgb_n_estimators,
            learning_rate=config.lgb_learning_rate,
            max_depth=config.lgb_max_depth,
            num_leaves=config.lgb_num_leaves,
            subsample=config.lgb_subsample,
            colsample_bytree=config.lgb_colsample_bytree,
            random_state=config.random_seed + fold,
            objective="multiclass",
            num_class=n_classes,
            verbose=-1,
        )
        clf_lgb.fit(x_tr, y_tr)
        probs_lgb = clf_lgb.predict_proba(x_va)
        lgb_importances += clf_lgb.feature_importances_ / config.n_splits

        # Model 2: CatBoost Multiclass
        clf_cb = CatBoostClassifier(
            iterations=config.cb_iterations,
            learning_rate=config.cb_learning_rate,
            depth=config.cb_depth,
            l2_leaf_reg=config.cb_l2_leaf_reg,
            random_seed=config.random_seed + fold,
            loss_function="MultiClass",
            verbose=0,
        )
        clf_cb.fit(x_tr, y_tr)
        probs_cb = clf_cb.predict_proba(x_va)
        cb_importances += clf_cb.get_feature_importance() / config.n_splits

        # 50/50 Soft-Voting Blend on Fold Validation
        oof_probs[val_idx] = 0.50 * probs_lgb + 0.50 * probs_cb

    # Evaluate Out-Of-Fold Global Metrics
    y_pred = np.argmax(oof_probs, axis=1)
    acc = float(accuracy_score(y, y_pred))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    m_f1 = float(f1_score(y, y_pred, average="macro"))
    w_f1 = float(f1_score(y, y_pred, average="weighted"))
    loss = float(log_loss(y, oof_probs))

    # Brier Multi-Class Score: sum of squared differences from one-hot target / N
    one_hot_y = np.eye(n_classes)[y]
    brier = float(np.mean(np.sum((oof_probs - one_hot_y) ** 2, axis=1)))

    cm = confusion_matrix(y, y_pred)
    class_f1s = f1_score(y, y_pred, average=None)
    per_class_dict = {
        TURNOVER_INVERSE_MAPPING[i]: float(class_f1s[i]) for i in range(n_classes)
    }

    metrics = ClassificationMetrics(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_f1=m_f1,
        weighted_f1=w_f1,
        multi_class_log_loss=loss,
        brier_score=brier,
        confusion_matrix=cm,
        per_class_f1=per_class_dict,
    )

    importances = {
        "lightgbm": lgb_importances,
        "catboost": cb_importances,
        "mean_importance": 0.5 * (lgb_importances + cb_importances),
    }

    return metrics, oof_probs, importances

