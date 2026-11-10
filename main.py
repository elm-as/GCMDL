"""Production pipeline orchestrator for Global Crypto Market Dynamics & Liquidity (GCMDL)."""

import json
from pathlib import Path
import pandas as pd
from src.core.config import ProjectConfig
from src.core.types import PipelineRunResult
from src.data_pipeline.loader import load_raw_telemetry, save_processed_telemetry
from src.feature_engineering.encoder import encode_feature_matrix
from src.feature_engineering.microstructure_features import compute_microstructure_features
from src.interpretability.feature_importance import (
    build_importance_dataframe,
    rank_feature_importances,
)
from src.modeling.tail_risk_predictor import train_tail_risk_predictor
from src.modeling.turnover_classifier import train_turnover_regime_classifier
from src.validation.cross_validator import summarize_evaluation_report
from src.visualization.dashboard_builder import generate_interactive_dashboard
from src.visualization.figures import (
    plot_catastrophic_drawdown_curves,
    plot_feature_importance_comparison,
    plot_liquidity_microstructure_map,
    plot_tokenomics_dilution_distribution,
    plot_turnover_confusion_matrix,
)


def run_pipeline() -> PipelineRunResult:
    """Execute end-to-end empirical estimation and artifact generation pipeline."""
    config = ProjectConfig()
    print("=" * 72)
    print("GCMDL 2026: INITIATING CRYPTO MICROSTRUCTURE PIPELINE")
    print("=" * 72)

    # 1. Ingestion & Validation
    print(f"\n[1/6] Ingesting telemetry: {config.data.raw_data_path}...")
    raw_df = load_raw_telemetry(config.data.raw_data_path, config.data)
    print(f"      Loaded {len(raw_df)} assets with {raw_df.shape[1]} raw attributes.")

    # 2. Feature Engineering
    print("\n[2/6] Engineering microstructural and tokenomic features...")
    engineered_df = compute_microstructure_features(raw_df)
    save_processed_telemetry(engineered_df, config.data.processed_data_path)
    feature_matrix = encode_feature_matrix(engineered_df)
    print(
        f"      Constructed FeatureMatrix with {len(feature_matrix.feature_names)} active predictors."
    )

    # 3. Model 1: Turnover Regime Classification
    print("\n[3/6] Estimating 5-Fold Multi-Class Liquidity Turnover Classifier...")
    t_metrics, oof_turnover, t_importances = train_turnover_regime_classifier(
        feature_matrix, config.model
    )
    print(
        f"      Turnover Macro F1: {t_metrics.macro_f1:.4f} | Balanced Acc: {t_metrics.balanced_accuracy:.2%}"
    )

    # 4. Model 2: Catastrophic Drawdown Tail-Risk Predictor
    print(
        "\n[4/6] Estimating 5-Fold Binary Tail-Risk Predictor (>85% Peak Decline)..."
    )
    r_metrics, oof_tail, shap_vals, r_importances = train_tail_risk_predictor(
        feature_matrix, config.model
    )
    print(
        f"      Tail-Risk ROC-AUC: {r_metrics.roc_auc:.4f} | PR-AUC: {r_metrics.pr_auc:.4f} | Sensitivity: {r_metrics.sensitivity:.2%}"
    )

    # 5. Interpretability & Importance Ranking
    print("\n[5/6] Extracting feature attribution and importance hierarchies...")
    importance_df = build_importance_dataframe(
        feature_matrix.feature_names,
        t_importances["mean_importance"],
        r_importances,
    )
    importance_csv_path = (
        config.data.artifacts_dir / "feature_importance_benchmark.csv"
    )
    config.data.artifacts_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_csv_path, index=False)

    top_liquidity = rank_feature_importances(
        feature_matrix.feature_names, t_importances["mean_importance"]
    )[:5]
    top_tail = rank_feature_importances(feature_matrix.feature_names, r_importances)[:5]

    # 6. Visualization & Interactive Dashboard
    print("\n[6/6] Rendering publication figures and interactive HTML5 dashboard...")
    plot_liquidity_microstructure_map(
        engineered_df, config.data.images_dir / "liquidity_microstructure_map.png"
    )
    plot_catastrophic_drawdown_curves(
        r_metrics, config.data.images_dir / "catastrophic_drawdown_roc_pr.png"
    )
    plot_tokenomics_dilution_distribution(
        engineered_df,
        config.data.images_dir / "tokenomics_dilution_overhang.png",
    )
    plot_feature_importance_comparison(
        importance_df, config.data.images_dir / "feature_importance_shap.png"
    )
    plot_turnover_confusion_matrix(
        t_metrics, config.data.images_dir / "confusion_matrix_turnover.png"
    )

    dashboard_path = config.data.artifacts_dir / "dashboard.html"
    generate_interactive_dashboard(
        engineered_df, t_metrics, r_metrics, dashboard_path
    )

    # Export structured metrics JSON
    metrics_summary = {
        "census_size": len(raw_df),
        "turnover_classification": {
            "macro_f1": t_metrics.macro_f1,
            "weighted_f1": t_metrics.weighted_f1,
            "balanced_accuracy": t_metrics.balanced_accuracy,
            "multi_class_log_loss": t_metrics.multi_class_log_loss,
            "brier_score": t_metrics.brier_score,
            "per_class_f1": t_metrics.per_class_f1,
        },
        "tail_risk_prediction": {
            "roc_auc": r_metrics.roc_auc,
            "pr_auc": r_metrics.pr_auc,
            "optimal_threshold": r_metrics.optimal_threshold,
            "sensitivity": r_metrics.sensitivity,
            "specificity": r_metrics.specificity,
            "brier_score": r_metrics.brier_score,
        },
        "top_liquidity_features": [f[0] for f in top_liquidity],
        "top_tail_risk_features": [f[0] for f in top_tail],
    }
    (config.data.artifacts_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2), encoding="utf-8"
    )

    print("\n" + summarize_evaluation_report(t_metrics, r_metrics))
    print(f"\nArtifacts successfully exported to: {config.data.artifacts_dir.resolve()}")
    print(f"Interactive Dashboard ready at: {dashboard_path.resolve()}\n")

    return PipelineRunResult(
        total_records=len(raw_df),
        turnover_metrics=t_metrics,
        tail_risk_metrics=r_metrics,
        top_liquidity_drivers=top_liquidity,
        top_tail_risk_drivers=top_tail,
        dashboard_path=str(dashboard_path.resolve()),
        artifacts_dir=str(config.data.artifacts_dir.resolve()),
    )


if __name__ == "__main__":
    run_pipeline()

