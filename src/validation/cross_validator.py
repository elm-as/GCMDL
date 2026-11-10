"""Cross-validation orchestration and tabular metric summarizer."""

from typing import Dict, List
import numpy as np
import pandas as pd
from src.core.types import ClassificationMetrics, TailRiskMetrics


def summarize_evaluation_report(
    turnover_metrics: ClassificationMetrics,
    tail_risk_metrics: TailRiskMetrics,
) -> str:
    """Format publication-grade text summary of all empirical validation results."""
    lines: List[str] = [
        "=" * 72,
        "GCMDL: GLOBAL CRYPTO MARKET DYNAMICS & LIQUIDITY BENCHMARK REPORT",
        "=" * 72,
        "",
        "TASK 1: MULTI-CLASS LIQUIDITY TURNOVER REGIME CLASSIFICATION",
        "-" * 72,
        f"  Overall Accuracy        : {turnover_metrics.accuracy:.4f} ({turnover_metrics.accuracy*100:.2f}%)",
        f"  Balanced Accuracy       : {turnover_metrics.balanced_accuracy:.4f}",
        f"  Macro F1 Score          : {turnover_metrics.macro_f1:.4f}",
        f"  Weighted F1 Score       : {turnover_metrics.weighted_f1:.4f}",
        f"  Multi-Class Log-Loss    : {turnover_metrics.multi_class_log_loss:.4f}",
        f"  Brier Multi-Score       : {turnover_metrics.brier_score:.4f}",
        "",
        "  Per-Class F1 Breakdown:",
    ]

    for cls_name, f1 in turnover_metrics.per_class_f1.items():
        lines.append(f"    - {cls_name:24s}: {f1:.4f}")

    lines.extend(
        [
            "",
            "TASK 2: TAIL-RISK CATASTROPHIC DRAWDOWN (>85% FROM ATH) PREDICTION",
            "-" * 72,
            f"  ROC-AUC Score           : {tail_risk_metrics.roc_auc:.4f}",
            f"  PR-AUC Score            : {tail_risk_metrics.pr_auc:.4f}",
            f"  Optimal Youden Threshold: {tail_risk_metrics.optimal_threshold:.4f}",
            f"  Sensitivity (Recall)    : {tail_risk_metrics.sensitivity:.4f} ({tail_risk_metrics.sensitivity*100:.2f}%)",
            f"  Specificity             : {tail_risk_metrics.specificity:.4f} ({tail_risk_metrics.specificity*100:.2f}%)",
            f"  Brier Score             : {tail_risk_metrics.brier_score:.4f}",
            "=" * 72,
        ]
    )

    return "\n".join(lines)

