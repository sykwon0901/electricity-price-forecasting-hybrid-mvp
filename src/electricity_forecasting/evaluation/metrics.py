"""
Canonical evaluation metrics.

Key properties (legacy-compatible):
- Full sMAPE: computed once on flattened (N, H, C) tensors (do NOT average per-target sMAPEs).
- Active sMAPE: active mask is derived from ground-truth volume channel (Y_true[..., volume_index] > active_threshold),
  then computed once on flattened active elements only.
- Robust to denominator==0 cases: 0/0 contributes 0 (no NaNs, no RuntimeWarning).
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def smape_flat(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    sMAPE on flattened arrays (single scalar).
    Robust: if |y_true|+|y_pred| == 0, contribution is 0.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()

    den = np.abs(yt) + np.abs(yp)

    # numerator: 2*|yt-yp|
    num = 2.0 * np.abs(yp - yt)

    # Safe divide: where den>0 use num/(den+eps), else 0
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den + float(eps), out=out, where=den > 0)

    return float(np.mean(out))


def build_active_mask_from_volume(
    Y_true: np.ndarray,
    volume_index: int,
    active_threshold: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    Build active mask from volume channel.

    Backward compatibility:
    - Some older code passes `eps=` or `active_eps=`; we treat them as the threshold.
    """
    if "eps" in kwargs:
        active_threshold = float(kwargs["eps"])
    if "active_eps" in kwargs:
        active_threshold = float(kwargs["active_eps"])

    if Y_true.ndim != 3:
        raise ValueError(f"Expected Y_true with ndim=3, got shape={getattr(Y_true, 'shape', None)}")

    return (Y_true[..., int(volume_index)] > float(active_threshold))


def evaluate_full_smape(Y_true: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Full sMAPE: flatten all elements once.
    """
    return smape_flat(Y_true, Y_pred, eps=eps)


def evaluate_active_smape(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    volume_index: int,
    eps: float = 1e-8,
    active_eps: float = 0.0,
) -> float:
    """
    Active sMAPE: compute mask from ground-truth volume, then flatten active elements once.
    """
    mask = build_active_mask_from_volume(Y_true, volume_index=volume_index, active_eps=active_eps)


    yt = Y_true[mask]
    yp = Y_pred[mask]

    if yt.size == 0:
        # No active elements: define as 0.0 to avoid NaNs and crashes (should be rare).
        return 0.0

    return smape_flat(yt, yp, eps=eps)


def evaluate_predictions(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    volume_index: int,
    eps: float = 1e-8,
    active_eps: float = 0.0,
) -> Dict[str, float]:
    """
    Convenience wrapper returning both full and active sMAPE.
    """
    full = evaluate_full_smape(Y_true, Y_pred, eps=eps)
    active = evaluate_active_smape(Y_true, Y_pred, volume_index=volume_index, eps=eps, active_eps=active_eps)
    return {"full_smape": float(full), "active_smape": float(active)}
