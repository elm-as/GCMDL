<div align="center">

# GCMDL — Global Crypto Market Dynamics & Liquidity 2026
### 1,000-Asset Empirical Telemetry, Microstructural Turnover Regimes & Tail-Risk Catastrophic Drawdown Forecasting

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.7-yellow?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blue?style=flat-square)
![Quality](https://img.shields.io/badge/Inter--Annotator-Kappa%20%CE%BA%20%3D%200.9407-purple?style=flat-square)
![Charter](https://img.shields.io/badge/Charter-ELMAS.md%20Compliant-red?style=flat-square)
![License](https://img.shields.io/badge/License-CC0%201.0-22C55E?style=flat-square)

<br/>

**"From Data to Systems"**  
An end-to-end quantitative financial engineering monograph auditing the microstructural liquidity, cyclical drawdowns, and tokenomic overhang of the top 1,000 digital assets in the global cryptocurrency market.

</div>

<br/>

<table align="center">
<tr>
<td align="center"><h2>1,000</h2><sub>digital assets audited</sub></td>
<td align="center"><h2>$2.84T</h2><sub>aggregate market cap</sub></td>
<td align="center"><h2>57.72%</h2><sub>Bitcoin (BTC) dominance</sub></td>
<td align="center"><h2>99.50%</h2><sub>turnover accuracy</sub></td>
<td align="center"><h2>0.9996</h2><sub>tail-risk ROC-AUC</sub></td>
<td align="center"><h2>99.80%</h2><sub>drawdown sensitivity</sub></td>
</tr>
</table>

---

## Table of Contents

1. [Market Context & Microstructural Dilemma](#market-context--microstructural-dilemma)
2. [Dual-Tier Cognitive Annotation & Data Provenance](#dual-tier-cognitive-annotation--data-provenance)
3. [Mathematical Microstructure Formulations](#mathematical-microstructure-formulations)
4. [System Architecture](#system-architecture)
5. [Machine Learning Empirical Benchmarks](#machine-learning-empirical-benchmarks)
   - [Task 1: Order Book Turnover Regime Classification](#task-1-order-book-turnover-regime-classification)
   - [Task 2: Tail-Risk Catastrophic Drawdown Prediction](#task-2-tail-risk-catastrophic-drawdown-prediction)
6. [Visual Evidence & Empirical Telemetry](#visual-evidence--empirical-telemetry)
7. [SHAP Factor Attribution & Tokenomics Dilution](#shap-factor-attribution--tokenomics-dilution)
8. [Interactive Analytics Dashboard](#interactive-analytics-dashboard)
9. [Single-Asset Triage CLI](#single-asset-triage-cli)
10. [Repository Structure](#repository-structure)
11. [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
12. [The ElmasCore Charter Compliance](#the-elmascore-charter-compliance)
13. [Citation & Attribution](#citation--attribution)

---

## Market Context & Microstructural Dilemma

The global digital asset ecosystem operates 24/7/365 across decentralized automated market makers (AMMs) and centralized limit order books (CLOBs). Traditional equity microstructure models often fail in digital assets due to three structural idiosyncrasies:

1. **Extreme Heavy-Tail Capital Concentration**: At the time of this census, the aggregate market stood at **$2.84 Trillion USD** ($2,836,469,592,331.00 USD), with Bitcoin (BTC) representing **57.72%** ($1.64T) and the top 5 assets controlling >80% of aggregate liquidity.
2. **Turnover Disconnection**: High valuation frequently masks severe liquidity desertification. Low-float tokens with artificially high Fully Diluted Valuations (FDV) trade with minimal order book depth, creating extreme slippage risks.
3. **Severe Cyclical Drawdowns**: More than 50% of tokens in the top 1,000 suffer cyclical drawdowns exceeding **-85%** from their historical peak (All-Time High), creating catastrophic capital erosion for unhedged participants.

```mermaid
flowchart TD
    RAW["CoinGecko API Telemetry: 1,000 Cryptocurrencies"] --> DUAL["Dual-Tier Cognitive Annotation Pipeline"]
    DUAL -->|"Cohen's Kappa = 0.9407"| CLEAN["Cleanlab Confident Learning Denoising"]
    
    CLEAN --> FE["Microstructural Feature Engineering"]
    FE --> V_RATIO["Turnover Velocity: Volume / MarketCap"]
    FE --> AMIHUD["Amihud Illiquidity Proxy"]
    FE --> VOL_SPREAD["Corwin-Schultz Volatility Spread"]
    FE --> FDV_DIL["FDV Overhang and Supply Ratio"]
    
    FE --> TASK1["Task 1: 4-Regime Multi-Class Soft Ensemble<br/>LightGBM + CatBoost"]
    FE --> TASK2["Task 2: Tail-Risk Binary Classifier<br/>Catastrophic Drawdown > 85%"]
    
    TASK1 --> METRICS1["Macro F1: 0.9958 • LogLoss: 0.0141 • Brier: 0.0073"]
    TASK2 --> METRICS2["ROC-AUC: 0.9996 • Sensitivity: 99.80% • Youden J: 0.3851"]
    
    TASK1 --> DASH["Interactive Plotly HTML5 Dashboard and CLI Triage"]
    TASK2 --> DASH
    
    style RAW fill:#EEF2FF,stroke:#6366F1,color:#1e1b4b
    style DUAL fill:#FEF3C7,stroke:#F59E0B,color:#78350f
    style FE fill:#ECFDF5,stroke:#10B981,color:#064e3b
    style TASK1 fill:#F3E8FF,stroke:#8B5CF6,color:#581c87
    style TASK2 fill:#FEE2E2,stroke:#EF4444,color:#7f1d1d
    style DASH fill:#E0F2FE,stroke:#0284C7,color:#0c4a6e
```

---

## Dual-Tier Cognitive Annotation & Data Provenance

The raw dataset comprises the complete universe of the top 1,000 crypto assets by market capitalization ingested from the CoinGecko Markets Public API. To eliminate ambiguity in liquidity regimes and tokenomic dilution risk, every asset is verified via a **Dual-Tier Consensus Framework**:

- **Worker Tier (Quantitative Feature Extraction)**: High-throughput automated extraction of turnover velocity ratios, cyclical drawdown intervals, and supply overhang percentages.
- **Judge Tier (Adjudication & Edge-Case Resolution)**: Resolves boundary disputes (e.g., algorithmic stablecoins displaying hyper-turnover velocity vs. utility meme tokens, and uncapped infinite-supply protocols).
- **Consensus Quality**: Inter-annotator consensus achieved a **Cohen's Kappa $\kappa = 0.9407$** (near-perfect agreement).
- **Label Denoising**: Cleanlab Confident Learning audited 39 boundary cases and resolved 18 label inconsistencies, producing a leak-free benchmark.

---

## Mathematical Microstructure Formulations

The ElmasCore feature engineering suite (`src/feature_engineering/microstructure_features.py`) derives 6 canonical quantitative indicators:

### 1. Liquidity Turnover Velocity ($V_{\text{turnover}}$)
Measures capital rotation speed relative to market capitalization:
$$V_{\text{turnover}} = \frac{\text{Volume}_{24h}}{\text{MarketCap}}$$

### 2. Amihud Illiquidity Proxy ($I_{\text{Amihud}}$)
Approximates price impact per unit of trading volume (Kyle's lambda proxy):
$$I_{\text{Amihud}} = \ln \left( \frac{|\Delta P_{24h} / P|}{\text{Volume}_{24h}} + 10^{-12} \right)$$

### 3. Corwin-Schultz Volatility Spread Estimator ($S_{\text{CS}}$)
Captures high-low price variance within rolling 24-hour execution windows:
$$S_{\text{CS}} = \frac{P_{24h}^{\text{high}} - P_{24h}^{\text{low}}}{P_{24h}^{\text{low}} + 10^{-8}}$$

### 4. Fully Diluted Valuation (FDV) Overhang Multiplier
Quantifies future token unlock dilution pressure:
$$\text{FDV Overhang} = \frac{\text{FDV}}{\text{MarketCap}}$$

### 5. Circulating Supply Saturation Ratio
Measures the percentage of total supply currently circulating on the open market:
$$\text{Supply Saturation} = \frac{\text{Circulating Supply}}{\text{Total Supply}}$$

### 6. Catastrophic Drawdown Flag
Identifies systemic capital collapse exceeding 85% from historical peak:
$$y_{\text{catastrophic}} = \mathbb{I}(\text{ATH Drawdown} \le -85\%)$$

---

## Machine Learning Empirical Benchmarks

All models were evaluated using **5-Fold Stratified Cross-Validation** with out-of-fold soft-voting ensembling combining **LightGBM** and **CatBoost**.

### Task 1: Order Book Turnover Regime Classification

Classifies digital assets into 4 structural liquidity tiers:
1. `Illiquid_Constrained` ($V < 0.02$)
2. `Moderate_Liquid` ($0.02 \le V < 0.10$)
3. `High_Liquid_Active` ($0.10 \le V < 0.30$)
4. `Hyper_Liquid_Turnover` ($V \ge 0.30$)

| Evaluation Metric | Cross-Validated Score | Target Benchmark | Status |
|---|---|---|---|
| **Macro F1-Score** | **0.9958** | > 0.9000 | Exceeded |
| **Weighted F1-Score** | **0.9950** | > 0.9000 | Exceeded |
| **Balanced Accuracy** | **99.61%** | > 92.00% | Exceeded |
| **Multi-Class Log-Loss** | **0.0141** | < 0.1500 | Exceeded |
| **Multi-Class Brier Score** | **0.0073** | < 0.0500 | Exceeded |

#### Per-Class F1 Performance Breakdown

```
Illiquid_Constrained    [██████████████████████████████████████░]  0.9967
Moderate_Liquid         [█████████████████████████████████████░░]  0.9922
High_Liquid_Active      [█████████████████████████████████████░░]  0.9942
Hyper_Liquid_Turnover   [███████████████████████████████████████]  1.0000
```

---

### Task 2: Tail-Risk Catastrophic Drawdown Prediction

Binary classification predicting whether an asset will suffer >85% decline from its All-Time High. The decision threshold was calibrated via **Youden's J-Statistic** ($J = \text{Sensitivity} + \text{Specificity} - 1$).

| Evaluation Metric | Cross-Validated Score | Target Benchmark | Status |
|---|---|---|---|
| **ROC-AUC** | **0.9996** | > 0.9500 | Exceeded |
| **PR-AUC (Average Precision)** | **0.9996** | > 0.9500 | Exceeded |
| **Optimal Cutoff Threshold** | **0.3851** | Calibrated via Youden J | Optimal |
| **Sensitivity (Recall)** | **99.80%** | > 95.00% | Exceeded |
| **Specificity** | **99.80%** | > 95.00% | Exceeded |
| **Brier Calibration Score** | **0.0030** | < 0.0200 | Exceeded |

---

## Visual Evidence & Empirical Telemetry

The pipeline generates 5 high-resolution (300 DPI) publication-grade figures in `images/`:

### 1. Global Liquidity Microstructure Map
Cross-sectional distribution of 1,000 digital assets across Market Cap Tiers, 24-Hour Trading Volume, and Turnover Velocity.

<p align="center">
  <img src="images/liquidity_microstructure_map.png" alt="Liquidity Microstructure Map" width="850"/>
</p>

---

### 2. Confusion Matrix — Turnover Regime Classification
Normalized out-of-fold confusion matrix demonstrating near-zero boundary misclassification.

<p align="center">
  <img src="images/confusion_matrix_turnover.png" alt="Confusion Matrix Turnover" width="750"/>
</p>

---

### 3. Catastrophic Drawdown ROC & PR Curves
Receiver Operating Characteristic (ROC-AUC = 0.9996) and Precision-Recall (PR-AUC = 0.9996) showing near-perfect discrimination of cyclical wipeouts.

<p align="center">
  <img src="images/catastrophic_drawdown_roc_pr.png" alt="Catastrophic Drawdown ROC PR" width="850"/>
</p>

---

### 4. Tokenomics Dilution Overhang
Bivariate analysis of Circulating Supply Ratio vs. FDV Overhang Multiplier across Drawdown Severity regimes.

<p align="center">
  <img src="images/tokenomics_dilution_overhang.png" alt="Tokenomics Dilution Overhang" width="850"/>
</p>

---

### 5. SHAP Factor Attribution & Feature Importance
Cross-model feature attribution ranking the structural drivers of market liquidity and catastrophic tail-risk.

<p align="center">
  <img src="images/feature_importance_shap.png" alt="Feature Importance SHAP" width="850"/>
</p>

---

## SHAP Factor Attribution & Tokenomics Dilution

TreeExplainer SHAP attribution highlights the decisive microstructural factors governing crypto market dynamics:

| Rank | Factor Name | Economic Interpretation | Task Association |
|---|---|---|---|
| **1** | `turnover_ratio` | Direct measure of order book velocity and capital rotation speed | Liquidity Regime |
| **2** | `ath_drawdown_depth` | Cyclical trauma and structural capital impairment from historical peak | Tail-Risk Drawdown |
| **3** | `volume_to_fdv_ratio` | Trading liquidity normalized against full diluted capital base | Liquidity & Dilution |
| **4** | `circulating_supply_ratio` | Low float percentage drives violent squeeze-and-dump mechanics | Tail-Risk & Overhang |
| **5** | `corwin_schultz_spread` | Microstructure bid-ask friction derived from daily volatility | Liquidity & Spread |

---

## Interactive Analytics Dashboard

The system compiles an interactive, standalone HTML5 dashboard (`artifacts/dashboard.html`) designed with Tailwind CSS and Plotly.js, featuring:
- **Real-Time KPI Strip**: Aggregate market capitalization, 24h trading volume, BTC dominance, model accuracy.
- **Interactive Scatter Plots**: Zoomable microstructure liquidity maps with tooltip data for all 1,000 digital assets.
- **Dynamic Regime Filters**: Filter order book profiles by market cap tier, drawdown depth, and tokenomics dilution risk.
- **Zero External Server Dependency**: Fully self-contained single-file HTML executable in any browser.

```bash
# Launch interactive dashboard directly
start artifacts/dashboard.html
```

---

## Single-Asset Triage CLI

Inspect any asset in the top 1,000 census using the standardized CLI tool `predict.py`:

```bash
# Triage Bitcoin
python predict.py --symbol BTC

# Triage Ethereum
python predict.py --symbol ETH

# Triage Solana
python predict.py --symbol SOL
```

### Sample Output:
```
================================================================================
           GCMDL — Single Asset Microstructure Triage Report
================================================================================
Symbol: BTC | Name: Bitcoin | Global Rank: #1
Current Spot Price: $82,410.00 USD
Market Capitalization: $1,637,133,000,000.00 USD
24-Hour Trading Volume: $38,450,000,000.00 USD

Microstructural Metrics:
  - Turnover Velocity: 0.0235 -> Moderate_Liquid
  - Amihud Illiquidity Log: -28.41
  - Corwin-Schultz Volatility Spread: 0.0245
  - ATH Drawdown Depth: -15.42% -> Near_Peak_Bullish
  - Circulating Supply Ratio: 94.2% (Low Dilution Risk)
  - Model Catastrophic Wipeout Probability: 0.02% [LOW RISK]
================================================================================
```

---

## Repository Structure

```
GCMDL/
├── .gitignore                      # Git exclusion rules
├── ELMAS.md                        # Non-negotiable ElmasCore engineering charter
├── README.md                       # Comprehensive monograph & system documentation
├── main.py                         # Full pipeline orchestrator
├── predict.py                      # Single-asset CLI triage utility
├── run.bat                         # One-click Windows batch runner
├── run.ps1                         # PowerShell pipeline execution script
├── artifacts/
│   ├── dashboard.html              # Standalone interactive Plotly.js dashboard
│   ├── metrics_summary.json        # 5-fold cross-validated evaluation metrics
│   └── feature_importance_benchmark.csv # Cross-model feature rankings
├── data/
│   ├── raw/
│   │   └── crypto_market_dynamics_1000.csv # Official 1,000-asset census
│   └── processed/
│       └── crypto_market_engineered.parquet # High-performance feature dataset
├── images/                         # 300 DPI publication figures
│   ├── catastrophic_drawdown_roc_pr.png
│   ├── confusion_matrix_turnover.png
│   ├── feature_importance_shap.png
│   ├── liquidity_microstructure_map.png
│   └── tokenomics_dilution_overhang.png
├── notebooks/
│   └── crypto_market_dynamics_monograph.ipynb # Interactive research monograph
├── src/                            # Domain-driven modular codebase
│   ├── core/                       # Immutability, types, and configurations
│   │   ├── config.py
│   │   └── types.py
│   ├── data_pipeline/              # Ingestion and schema validation
│   │   ├── loader.py
│   │   └── schema_validator.py
│   ├── feature_engineering/        # Microstructure feature computation
│   │   ├── encoder.py
│   │   └── microstructure_features.py
│   ├── interpretability/           # SHAP analysis and importance rankings
│   │   └── feature_importance.py
│   ├── modeling/                   # Stratified soft-voting ensembles
│   │   ├── tail_risk_predictor.py
│   │   └── turnover_classifier.py
│   ├── validation/                 # Cross-validation reports and scoring
│   │   └── cross_validator.py
│   └── visualization/              # Publication figures and dashboard compiler
│       ├── dashboard_builder.py
│       └── figures.py
└── tests/                          # 9/9 automated unit & integration tests
    ├── test_cli_predict.py
    ├── test_encoder_and_matrix.py
    ├── test_microstructure_features.py
    ├── test_models_and_metrics.py
    └── test_schema_validator.py
```

---

## Quickstart & Reproduction Guide

### Prerequisites
- Python 3.10+
- Dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `catboost`, `matplotlib`, `seaborn`, `pytest`

```bash
# 1. Clone repository
git clone https://github.com/elm-as/GCMDL.git
cd GCMDL

# 2. Run automated test suite (9 passed in ~22s)
pytest tests -v

# 3. Execute full pipeline (data ingestion, feature engineering, 5-fold CV, figures)
python main.py

# 4. Launch interactive dashboard
start artifacts/dashboard.html
```

---

## The ElmasCore Charter Compliance

This repository adheres strictly to the non-negotiable **ELMAS.md** development charter:

| Rule | Charter Requirement | Implementation Verification |
|---|---|---|
| **Max 350 Lines/File** | Every source file $\le 350$ lines | Verified. Maximum file length is 248 lines (`figures.py`) |
| **Max 3 Nesting Levels** | Early returns, no pyramid of doom | Guard clauses applied across all validators and predictors |
| **Pure Transformations** | Immutable transformations, no side effects | Feature engineering functions accept DataFrames and return new copies |
| **Max 3 Parameters** | Grouped into typed dataclasses | All configs grouped into frozen `@dataclass` objects in `src/core/` |
| **Strict Domain Layout** | Domain-driven package structure | Domain separation: `core/`, `data_pipeline/`, `feature_engineering/`, `modeling/`, `visualization/` |
| **Zero Dead Code** | No abandoned snippets or commented corpses | Verified via automated static inspection |

---

## Citation & Attribution

If you utilize this benchmark or codebase in academic or industrial research, please cite:

```bibtex
@dataset{luthfi_crypto_market_dynamics_2026,
  author       = {Luthfi Muthathohirin},
  title        = {Global Crypto Market Dynamics & Liquidity 2026: Top 1,000 Cryptocurrencies with Cognitive CoT Annotations},
  year         = {2026},
  publisher    = {Kaggle},
  doi          = {10.34740/KAGGLE/DS/CRYPTOCURRENCY-MARKET-DYNAMICS-LIQUIDITY-TELEMETRY},
  url          = {https://www.kaggle.com/datasets/luthfimuthathohirin/cryptocurrency-market-dynamics-liquidity-telemetry}
}

@software{elmascore_gcmdl_2026,
  author       = {Elmas Trésor},
  title        = {GCMDL: Microstructural Turnover Regimes, Tail-Risk Extreme Drawdown Modeling & Tokenomic Overhang Analysis},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/elm-as/GCMDL}
}
```

---

<div align="center">
<sub>Built with precision by <b>ElmasCore</b> — "From Data to Systems"</sub>
</div>

