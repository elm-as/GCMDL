"""Feature importance extraction, ranking, and attribution helpers."""

from typing import List, Tuple
import numpy as np
import pandas as pd


def rank_feature_importances(
    feature_names: Tuple[str, ...],
    importance_values: np.ndarray,
) -> List[Tuple[str, float]]:
    """Rank features by descending importance score purely.

    Args:
        feature_names: Names of input variables.
        importance_values: Corresponding numeric importance values.

    Returns:
        List of (feature_name, normalized_score) sorted descending.
    """
    total = np.sum(np.abs(importance_values))
    safe_total = total if total > 0 else 1.0
    normalized = np.abs(importance_values) / safe_total

    pairs = list(zip(feature_names, normalized.tolist()))
    return sorted(pairs, key=lambda x: x[1], reverse=True)


def build_importance_dataframe(
    feature_names: Tuple[str, ...],
    turnover_importance: np.ndarray,
    tail_risk_importance: np.ndarray,
) -> pd.DataFrame:
    """Build unified comparison DataFrame of feature importances across tasks."""
    t_rank = dict(rank_feature_importances(feature_names, turnover_importance))
    r_rank = dict(rank_feature_importances(feature_names, tail_risk_importance))

    records = []
    for f in feature_names:
        records.append(
            {
                "feature": f,
                "turnover_importance": t_rank.get(f, 0.0),
                "tail_risk_importance": r_rank.get(f, 0.0),
                "mean_importance": 0.5 * (t_rank.get(f, 0.0) + r_rank.get(f, 0.0)),
            }
        )

    return pd.DataFrame(records).sort_values("mean_importance", ascending=False)
