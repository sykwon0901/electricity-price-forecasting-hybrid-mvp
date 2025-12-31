"""
Diagnostics and auxiliary evaluation utilities.

Includes:
- per-ID metrics for robustness checks
- bootstrap confidence intervals
- standard threshold sweep for hurdle-style volume prediction
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import smape_flat


def per_id_smape_full_active(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    ids_for_windows: np.ndarray,
    volume_index: int,
    active_eps: float = 0.0,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Compute per-ID Full/Active sMAPE using the canonical flatten-based definition.

    Y_true/Y_pred: (N, H, C)
    ids_for_windows: (N,)
    """
    if Y_true.shape != Y_pred.shape:
        raise ValueError("Shape mismatch.")
    if Y_true.ndim != 3:
        raise ValueError("Expected (N, H, C).")

    ids_for_windows = np.asarray(ids_for_windows).astype(np.int64)
    unique_ids = np.unique(ids_for_windows)

    rows = []
    for _id in unique_ids:
        mask_w = (ids_for_windows == _id)
        yt = Y_true[mask_w]
        yp = Y_pred[mask_w]

        full = smape_flat(yt.reshape(-1), yp.reshape(-1), eps=eps)

        active_mask = (yt[..., volume_index] > float(active_eps))
        if np.any(active_mask):
            active = smape_flat(yt[active_mask].reshape(-1), yp[active_mask].reshape(-1), eps=eps)
        else:
            active = float("nan")

        rows.append((_id, float(full), float(active), int(mask_w.sum())))

    out = pd.DataFrame(rows, columns=["ID", "full_smape", "active_smape", "n_windows"])
    return out.sort_values("ID").reset_index(drop=True)


def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for the mean.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = x.size
    for i in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        means[i] = samp.mean()

    mean = float(x.mean())
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return {"mean": mean, "ci_low": low, "ci_high": high}


def sweep_threshold_volume(
    y_true_volume: np.ndarray,
    y_proba_active: np.ndarray,
    y_pred_logvol: np.ndarray,
    thresholds: np.ndarray,
    eps_smape: float = 1e-8,
) -> Tuple[float, pd.DataFrame]:
    """
    Threshold sweep for hurdle volume prediction.

    Inputs are flattened-compatible:
    - y_true_volume: (N, H) or (N, H, 1)
    - y_proba_active: (N, H)
    - y_pred_logvol: (N, H) predicted log1p(volume) for active regime

    Prediction rule:
      vol_hat = (proba >= thr) * (expm1(logvol_hat).clip(min=0))

    Objective:
      pick thr minimizing FULL sMAPE on volume (flatten-based)
    """
    yt = np.asarray(y_true_volume, dtype=np.float64).reshape(-1)
    pa = np.asarray(y_proba_active, dtype=np.float64).reshape(-1)
    lv = np.asarray(y_pred_logvol, dtype=np.float64).reshape(-1)

    curves = []
    best_thr = None
    best_score = float("inf")

    for thr in thresholds:
        active_hat = (pa >= float(thr)).astype(np.float64)
        vol_hat = active_hat * np.expm1(lv)
        vol_hat = np.clip(vol_hat, 0.0, None)

        score_full = smape_flat(yt, vol_hat, eps=eps_smape)

        # Active-only (truth-based)
        mask_active = (yt > 0)
        if np.any(mask_active):
            score_active = smape_flat(yt[mask_active], vol_hat[mask_active], eps=eps_smape)
        else:
            score_active = float("nan")

        curves.append((float(thr), float(score_full), float(score_active)))

        if score_full < best_score:
            best_score = score_full
            best_thr = float(thr)

    curve_df = pd.DataFrame(curves, columns=["threshold", "full_smape", "active_smape"])
    return float(best_thr), curve_df
