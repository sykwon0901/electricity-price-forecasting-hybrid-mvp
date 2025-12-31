# Electricity Price Forecasting with Deep Learning and Machine Learning
**Multi-asset, 10-step ahead forecasting on 15-minute EPEX OHLCV data**

## Project Continuation (Repo 1 → Repo 2)
This repository is a follow-up to **Repo 1: Electricity Price Forecasting (Preliminary Benchmark Report)** (https://github.com/sykwon0901/electricity-price-forecasting-preliminary).

Repo 1 established three key findings:
- **Full-timeline metrics are often dominated by a strong persistence baseline** under heavy inactivity.
- **Volume is the main bottleneck** (rare active periods are hard and matter).
- **Activity-aware modeling is essential** to avoid instability under extreme zero-volume imbalance.

Based on these findings, Repo 2 focuses on a pragmatic MVP design:
- keep **price** (High/Low/Close) on a strong persistence baseline,
- model **volume** with specialized hurdle learners (activity + magnitude),
- evaluate candidates under a consistent harness and export artifacts to `results/`.

**Winner (under this setup):** **ML Hurdle LGBM hybrid** (baseline price + modeled volume) achieved the best overall trade-off.

---

## Notebooks

- Full notebook (nbviewer): https://nbviewer.org/github/sykwon0901/electricity-price-forecasting-hybrid-mvp/tree/main/notebooks/

- `01_EDA.ipynb` — EDA on 15-min OHLCV  
  - Outputs: `results/plots/eda/*`

- `02_Feature_Engineering.ipynb` — Build leakage-safe per-ID feature grids  
  - Outputs: `results/cache/*`

- `03_LGBM_Baseline_hybrid.ipynb` — Baseline price + LGBM hurdle volume  
  - Outputs: `results/models/ml_hurdle_lgbm_best.txt`, `results/metrics/*`

- `04_TCN_Baseline_hybrid.ipynb` — Baseline price + TCN hurdle volume  
  - Outputs: `results/models/dl_hurdle_tcn_best.pt`, `results/metrics/*`

- `05_GatedTransformer_Baseline_hybrid.ipynb` — Baseline price + gated Transformer hurdle volume  
  - Outputs: `results/models/dl_gated_transformer_best.pt`, `results/metrics/*`

- `06_LSTM_Baseline_hybrid.ipynb` — Baseline price + LSTM hurdle volume  
  - Outputs: `results/models/dl_hurdle_lstm_best.pt`, `results/metrics/*`

- `07_Results_Comparison.ipynb` — Consolidate model results  
  - Outputs: `results/metrics/consolidated_metrics_by_model.csv`
---

## Overview

Short-horizon electricity forecasting with a strong baseline and volume-specialized models.

This project predicts:
- **10 future steps** (150 minutes)
- for **multiple delivery-window assets**
- on **15-minute OHLCV** EPEX data

Key challenges:
- **Non-stationary price behavior**
- **Severe zero-volume imbalance** (inactive majority)

Repo 2 purpose:
- Turn the preliminary findings into a **reproducible MVP evaluation pipeline**
- Compare multiple **volume-only hybrid** methods under the same window index and metrics export format

---

## Dataset Access

The dataset is large (4.3GB), so it is not stored in this repository.

Dataset location:
- Data is not included in this repository.
- In Colab, mount Drive and place/clone this repo at:
  `/content/drive/MyDrive/0.Portfolio/electricity_price_forecasting`
- Place the dataset under:
  `/content/drive/MyDrive/0.Portfolio/electricity_price_forecasting/data/`

---

## Colab Setup

- Mount Drive and open the notebook from:
  `/content/drive/MyDrive/0.Portfolio/electricity_price_forecasting`
- All outputs are written under `results/` (metrics/models/window_index/logs/plots/cache).
- Notebooks are self-contained (no separate bootstrap file needed).
- Note: The fixed path in notebooks is a portfolio-only tradeoff for simplicity and reviewer convenience; this is not standard production practice.

---

## Project Conventions

- Inputs: `data/` (parquet time series) and `src/` (shared preprocessing + model code)
- Outputs: `results/models/`, `results/metrics/`, `results/plots/` (all notebooks write here)
- How to Run: open any notebook and **Run All**; the first setup cells mount Drive and prepare paths/dirs automatically
- Notes: clarity and reproducibility take priority over minimizing code length

---

## Key Results (Hybrid Volume Modeling on a Fixed Price Baseline)

### What is being compared
All methods in this repo use the same hybrid structure:

- **Price targets (High/Low/Close):** always predicted by **persistence baseline**
- **Volume:** predicted by a **hurdle model** (activity + magnitude), then substituted into the baseline output
- Therefore, differences in Total metrics are driven by **Volume only**, while price metrics remain identical by design.

### Shared evaluation conditions (applies to 03–06)
- `WIN = 128`, `HORIZON = 10`
- `n_test_windows_used = 200,000` (global cap)
- `n_ids_test = 672`
- `active_ratio_test ≈ 0.04584` (≈ 4.58% active)
- Thresholds selected on **validation**
- Same `window_index_hash_test` across runs (apples-to-apples)

**Note on reproducibility:**  
Metrics can vary slightly across runs due to nondeterministic GPU operations and/or the use of capped window sampling. A fixed seed is used to reduce drift, but minor variation is still possible.

---

### Model Metrics (Active + Full)

| Model                                     | Active High | Active Low | Active Close | Active Volume | Active Total | Full High | Full Low | Full Close | Full Volume | Full Total   |
|:------------------------------------------|------------:|-----------:|-------------:|--------------:|-------------:|----------:|---------:|-----------:|------------:|-------------:|
| Baseline (Persistence)                    | 0.227735    | 0.230610   | 0.231246     | 1.34683       | **0.509106** | 0.027610  | 0.027890 | 0.027855   | 0.089302    | 0.043164     |
| DL Hurdle TCN (Volume only)               | 0.227735    | 0.230610   | 0.231246     | 1.929596      | 0.654797     | 0.027610  | 0.027890 | 0.027855   | 0.092827    | 0.044045     |
| ML Hurdle LGBM (Volume only)              | 0.227735    | 0.230610   | 0.231246     | 1.425700      | 0.528823     | 0.027610  | 0.027890 | 0.027855   | **0.074345**| **0.039425** |
| DL Gated Transformer Hurdle (Volume only) | 0.227735    | 0.230610   | 0.231246     | 1.430663      | 0.530064     | 0.027610  | 0.027890 | 0.027855   | 0.083867    | 0.041805     |
| DL Hurdle LSTM (Volume only)              | 0.227735    | 0.230610   | 0.231246     | 1.836802      | 0.631598     | 0.027610  | 0.027890 | 0.027855   | 0.091725    | 0.043770     |

---

### Ranking (Full Total sMAPE, lower is better)
1. **ML Hurdle LGBM (Volume only)** — **0.039425**  
2. **DL Gated Transformer Hurdle (Volume only)** — 0.041805  
3. **Baseline (Persistence)** — 0.043164  
4. **DL Hurdle LSTM (Volume only)** — 0.043770  
5. **DL Hurdle TCN (Volume only)** — 0.044045  

### Ranking (Active Total sMAPE, lower is better)
1. **Baseline (Persistence)** — **0.509106**  
2. **ML Hurdle LGBM (Volume only)** — 0.528823  
3. **DL Gated Transformer Hurdle (Volume only)** — 0.530064  
4. **DL Hurdle LSTM (Volume only)** — 0.631598  
5. **DL Hurdle TCN (Volume only)** — 0.654797  

---

## Key Report Summary (What we learned)

1) **Price is strongly autoregressive at this horizon**  
Because High/Low/Close are kept on persistence in every hybrid, the experiment confirms a practical reality: for short horizons, a persistence-style baseline is extremely competitive and often hard to beat reliably.

2) **Volume is the bottleneck and determines the winner**  
All separation in Total metrics comes from how well Volume is handled under extreme zero-inflation.

3) **Full vs Active objectives produce different “winners”**
- **Full timeline (stability under heavy inactivity):**  
  **LGBM hybrid wins**, driven by a strong Full Volume improvement (0.0893 → 0.0743).
- **Active periods (rare but important trading windows):**  
  **Baseline wins** in this setup; learned models slightly worsen Active Total.

4) **Deep sequence models did not justify their complexity here**
- **TCN and LSTM** degrade both Active and Full totals vs baseline under this evaluation.
- **Gated Transformer** is competitive and improves Full vs baseline, but still trails **LGBM** on the main Full KPI.

---

## Decision Guidance (Which model would you deploy?)

### Scenario A — Full timeline accuracy matters most (planning / reporting / settlement-style aggregation)
**Choose: ML Hurdle LGBM hybrid**
- Best **Full Total** and best **Full Volume**
- Good practical MVP: typically fast and stable for tabular features, and does not require a GPU.

### Scenario B — Active-period accuracy matters most (operations / dispatch decisions during actual trading)
**Choose: Baseline (Persistence)** as the default under this configuration
- Best **Active Total** here
- With only ~4.6% active timestamps, models optimized for Full can still regress on the rare active windows

### Scenario C — You want a robust hybrid with some DL expressivity but still stable
**Consider: DL Gated Transformer hybrid**
- Much more stable than ungated DL approaches under zero-inflation
- Improves Full vs baseline (but not as strong as LGBM here)

### Practical playbook (recommended)
- Use **Baseline for price (H/L/C)** consistently.
- Use **LGBM hurdle for volume** when the KPI is overall Full stability.
- Keep **Baseline-only fallback** for regimes where active windows matter disproportionately, unless you retrain/tune specifically for Active optimization.

---

## What this repository demonstrates
- A reproducible evaluation harness with consistent output artifacts under `results/`.
- A clean “baseline-first” modeling philosophy: preserve what works for price and target the real bottleneck (volume).
- A portfolio-ready MVP pattern: **multiple candidate approaches evaluated under a shared window index**, leading to a data-driven selection of **LGBM hybrid**.

