"""Train/evaluate helper for ML Hurdle LGBM volume-only."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from ..models.ml_hurdle_lgbm_hwise import train_hurdle_hwise, predict_hurdle_hwise, apply_hurdle_threshold


def threshold_sweep_on_validation(
    y_true_vol_va: np.ndarray,   # (N,H)
    proba_va: np.ndarray,        # (N,H)
    logvol_va: np.ndarray,       # (N,H)
    thresholds: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[float, pd.DataFrame]:
    """Pick threshold minimizing FULL sMAPE on volume (flattened)."""
    yt = y_true_vol_va.reshape(-1).astype(np.float64)
    rows = []
    best_thr = None
    best_score = float("inf")

    for thr in thresholds:
        vol_hat = apply_hurdle_threshold(proba_va, logvol_va, float(thr)).reshape(-1).astype(np.float64)
        num = 2.0 * np.abs(yt - vol_hat)
        den = np.abs(yt) + np.abs(vol_hat) + eps
        full = float(np.mean(num / den))

        mask = (yt > 0)
        if mask.any():
            num_a = 2.0 * np.abs(yt[mask] - vol_hat[mask])
            den_a = np.abs(yt[mask]) + np.abs(vol_hat[mask]) + eps
            active = float(np.mean(num_a / den_a))
        else:
            active = float("nan")

        rows.append((float(thr), full, active))
        if full < best_score:
            best_score = full
            best_thr = float(thr)

    df = pd.DataFrame(rows, columns=["threshold", "val_full_smape_volume", "val_active_smape_volume"])
    return float(best_thr), df
