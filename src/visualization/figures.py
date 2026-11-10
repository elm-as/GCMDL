"""Publication-grade scientific charts for crypto market dynamics and liquidity."""

from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.core.types import ClassificationMetrics, FeatureMatrix, TailRiskMetrics
from src.feature_engineering.encoder import TURNOVER_INVERSE_MAPPING


# High-contrast publication palette
PALETTE = {
    "primary": "#1E3A8A",
    "secondary": "#0D9488",
    "accent": "#E11D48",
    "warning": "#F59E0B",
    "neutral_dark": "#1F2937",
    "neutral_light": "#F9FAFB",
    "regimes": {
        "Illiquid_Constrained": "#EF4444",
        "Moderate_Liquid": "#F59E0B",
        "High_Liquid_Active": "#10B981",
        "Hyper_Liquid_Turnover": "#6366F1",
    },
}


def plot_liquidity_microstructure_map(
    df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Generate 4-quadrant Market Cap vs 24h Trading Volume liquidity map."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    fig.patch.set_facecolor(PALETTE["neutral_light"])
    ax.set_facecolor("white")

    # Plot points by regime
    for regime, color in PALETTE["regimes"].items():
        subset = df[df["liquidity_turnover_regime"] == regime]
        ax.scatter(
            subset["log_market_cap"],
            subset["log_volume_24h"],
            c=color,
            label=regime.replace("_", " "),
            alpha=0.65,
            edgecolors="none",
            s=40,
        )

    # Highlight top flagship assets
    anchors = ["BTC", "ETH", "USDT", "SOL", "BNB", "XRP"]
    for _, row in df[df["symbol"].isin(anchors)].iterrows():
        ax.scatter(
            row["log_market_cap"],
            row["log_volume_24h"],
            color="#0F172A",
            s=120,
            edgecolors="#FFFFFF",
            linewidth=1.8,
            zorder=5,
        )
        ax.annotate(
            row["symbol"],
            (row["log_market_cap"], row["log_volume_24h"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontweight="bold",
            fontsize=9,
            color="#0F172A",
            zorder=6,
        )

    # Median quadrant split lines
    med_mcap = df["log_market_cap"].median()
    med_vol = df["log_volume_24h"].median()
    ax.axvline(med_mcap, color="#94A3B8", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(med_vol, color="#94A3B8", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_title(
        "Global Crypto Microstructure: Market Cap vs. 24h Volume (Top 1,000 Assets)",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color=PALETTE["neutral_dark"],
    )
    ax.set_xlabel("Log10 Market Capitalization (USD)", fontsize=10, labelpad=8)
    ax.set_ylabel("Log10 24h Trading Volume (USD)", fontsize=10, labelpad=8)
    ax.legend(title="Liquidity Regime", loc="upper left", frameon=True, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_catastrophic_drawdown_curves(
    metrics: TailRiskMetrics,
    save_path: Path,
) -> None:
    """Generate dual-panel ROC and Precision-Recall curves for tail-risk prediction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    fig.patch.set_facecolor(PALETTE["neutral_light"])

    # Left: ROC Curve
    ax1.set_facecolor("white")
    ax1.plot(
        metrics.fpr_array,
        metrics.tpr_array,
        color=PALETTE["primary"],
        lw=2.2,
        label=f"LightGBM + CatBoost (AUC = {metrics.roc_auc:.4f})",
    )
    ax1.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", lw=1.2)
    ax1.scatter(
        [1 - metrics.specificity],
        [metrics.sensitivity],
        color=PALETTE["accent"],
        s=90,
        zorder=5,
        label=f"Youden J Cutoff ({metrics.optimal_threshold:.2f})",
    )
    ax1.set_title(
        "ROC: Catastrophic Drawdown (>85% Peak Decline)",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["neutral_dark"],
    )
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=9)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right: Precision-Recall Curve
    ax2.set_facecolor("white")
    ax2.plot(
        metrics.recall_array,
        metrics.precision_array,
        color=PALETTE["secondary"],
        lw=2.2,
        label=f"PR-AUC = {metrics.pr_auc:.4f}",
    )
    ax2.set_title(
        "Precision-Recall: Tail-Risk Detection",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["neutral_dark"],
    )
    ax2.set_xlabel("Recall (Coverage of Crashed Assets)", fontsize=9)
    ax2.set_ylabel("Precision (Correctly Flagged Risk)", fontsize=9)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_tokenomics_dilution_distribution(
    df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Generate tokenomics dilution overhang distribution across market cap tiers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    fig.patch.set_facecolor(PALETTE["neutral_light"])

    ax1.set_facecolor("white")
    order = [
        "Micro_Cap_Speculative",
        "Small_Cap_Ecosystem",
        "Mid_Cap_Established",
        "Large_Cap_Institutional",
        "Mega_Cap_Sovereign",
    ]
    existing_order = [o for o in order if o in df["market_cap_tier"].unique()]
    sns.boxplot(
        data=df,
        x="market_cap_tier",
        y="circulating_supply_ratio",
        order=existing_order,
        ax=ax1,
        palette="Blues_r",
    )
    ax1.set_title(
        "Circulating Supply Ratio by Scale Tier", fontsize=11, fontweight="bold"
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Circulating / Total Supply", fontsize=9)
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.set_facecolor("white")
    sns.countplot(
        data=df,
        x="dilution_risk_profile",
        palette=["#10B981", "#F59E0B", "#EF4444"],
        ax=ax2,
    )
    ax2.set_title(
        "Dilution Overhang Severity Profile (1,000 Assets)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("Asset Count", fontsize=9)
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(True, linestyle=":", alpha=0.5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_feature_importance_comparison(
    importance_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Plot top predictive drivers for liquidity turnover vs catastrophic drawdown."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor(PALETTE["neutral_light"])
    ax.set_facecolor("white")

    top = importance_df.head(10).iloc[::-1]
    y_pos = np.arange(len(top))
    bar_height = 0.38

    ax.barh(
        y_pos + bar_height / 2,
        top["turnover_importance"],
        height=bar_height,
        color=PALETTE["secondary"],
        label="Turnover Regime Classifier",
        alpha=0.9,
    )
    ax.barh(
        y_pos - bar_height / 2,
        top["tail_risk_importance"],
        height=bar_height,
        color=PALETTE["accent"],
        label="Catastrophic Drawdown Predictor",
        alpha=0.9,
    )

    clean_names = [f.replace("_", " ").title() for f in top["feature"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names, fontsize=9)
    ax.set_xlabel("Normalized Feature Importance (Gain / Permutation)", fontsize=9)
    ax.set_title(
        "Top Microstructural & Tokenomic Risk Drivers", fontsize=12, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_turnover_confusion_matrix(
    metrics: ClassificationMetrics,
    save_path: Path,
) -> None:
    """Generate normalized confusion matrix heatmap for liquidity classification."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    fig.patch.set_facecolor(PALETTE["neutral_light"])

    cm = metrics.confusion_matrix
    norm_cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = [TURNOVER_INVERSE_MAPPING[i] for i in range(len(cm))]
    clean_labels = [l.replace("_", " ") for l in labels]

    sns.heatmap(
        norm_cm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar=False,
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(
        f"Liquidity Turnover Confusion Matrix (Macro F1 = {metrics.macro_f1:.4f})",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Predicted Turnover Regime", fontsize=9, labelpad=6)
    ax.set_ylabel("True Cognitive CoT Regime", fontsize=9, labelpad=6)
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

