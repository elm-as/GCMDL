"""Binary tail-risk model predicting catastrophic cyclical drawdown (>85% from ATH)."""

from typing import Dict, Tuple
import lightgbm as lgb
import numpy as np
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from src.core.config import ModelConfig
from src.core.types import FeatureMatrix, TailRiskMetrics


def train_tail_risk_predictor(
    data: FeatureMatrix,
    config: ModelConfig,
) -> Tuple[TailRiskMetrics, np.ndarray, np.ndarray, np.ndarray]:
    """Train 5-fold cross-validated model predicting catastrophic drawdown (>85%).

    Args:
        data: FeatureMatrix containing predictors and binary target.
        config: Model parameters.

    Returns:
        Tuple: (TailRiskMetrics, oof_probabilities, mean_shap_values, importances)
    """
    x = data.features.values
    y = data.target_tail_risk
    n_samples = len(y)

    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.random_seed
    )
    oof_probs = np.zeros(n_samples, dtype=np.float64)
    lgb_importances = np.zeros(x.shape[1], dtype=np.float64)

    fitted_models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        x_tr, y_tr = x[train_idx], y[train_idx]
        x_va, y_va = x[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=config.lgb_n_estimators,
            learning_rate=config.lgb_learning_rate,
            max_depth=config.lgb_max_depth,
            num_leaves=config.lgb_num_leaves,
            subsample=config.lgb_subsample,
            colsample_bytree=config.lgb_colsample_bytree,
            random_state=config.random_seed + fold,
            objective="binary",
            verbose=-1,
        )
        clf_lgb.fit(x_tr, y_tr)
        probs_lgb = clf_lgb.predict_proba(x_va)[:, 1]
        lgb_importances += clf_lgb.feature_importances_ / config.n_splits

        clf_cb = CatBoostClassifier(
            iterations=config.cb_iterations,
            learning_rate=config.cb_learning_rate,
            depth=config.cb_depth,
            l2_leaf_reg=config.cb_l2_leaf_reg,
            random_seed=config.random_seed + fold,
            loss_function="Logloss",
            verbose=0,
        )
        clf_cb.fit(x_tr, y_tr)
        probs_cb = clf_cb.predict_proba(x_va)[:, 1]

        oof_probs[val_idx] = 0.50 * probs_lgb + 0.50 * probs_cb
        fitted_models.append(clf_lgb)

    # Global ROC and PR Curves
    roc_auc = float(roc_auc_score(y, oof_probs))
    fpr, tpr, roc_thresholds = roc_curve(y, oof_probs)
    precisions, recalls, _ = precision_recall_curve(y, oof_probs)
    pr_auc = float(auc(recalls, precisions))
    brier = float(brier_score_loss(y, oof_probs))

    # Youden's Index Optimization: J = sensitivity + specificity - 1 = tpr - fpr
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    optimal_thresh = float(roc_thresholds[optimal_idx])

    binary_pred = (oof_probs >= optimal_thresh).astype(int)
    cm = confusion_matrix(y, binary_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # SHAP Explanations via TreeExplainer on Fold 0
    explainer = shap.TreeExplainer(fitted_models[0])
    shap_vals = explainer.shap_values(x)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    metrics = TailRiskMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        optimal_threshold=optimal_thresh,
        sensitivity=sensitivity,
        specificity=specificity,
        brier_score=brier,
        fpr_array=fpr,
        tpr_array=tpr,
        precision_array=precisions,
        recall_array=recalls,
    )

    return metrics, oof_probs, shap_vals, lgb_importances
