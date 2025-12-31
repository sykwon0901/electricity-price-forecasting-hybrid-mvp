"""Leakage-safe scaling utilities (canonical).

These helpers are intentionally lightweight (no sklearn dependency) and work with:
- X tensors: (N, WIN, F) or (N, F)
- Y tensors: (N, H, C)

Policy:
- If active_only=True, the scaler for the volume target is fit on active elements only
  (where y_true_volume > 0). Other targets use all elements.

Returned object is a plain dictionary so it can be saved as JSON if needed.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Sequence, Optional
import numpy as np


def _safe_mean_std(x: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_scalers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    active_only: bool = True,
    volume_index: int | None = None,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Fit simple (mean, std) scalers for X and Y.

    Args:
      X_train: (N, WIN, F) or (N, F)
      y_train: (N, H, C)
      active_only: apply active-only fit to the volume channel if volume_index is provided
      volume_index: index of the volume channel in y_train last dim
      eps: numerical stability

    Returns:
      scalers dict with keys:
        - X_mean, X_std
        - Y_mean, Y_std
        - active_only, volume_index, eps
    """
    X = np.asarray(X_train)
    Y = np.asarray(y_train)
    if Y.ndim != 3:
        raise ValueError(f"y_train must be 3D (N,H,C). Got shape {Y.shape}")

    # X: flatten over window dimension if present
    if X.ndim == 3:
        X_flat = X.reshape(-1, X.shape[-1])
    elif X.ndim == 2:
        X_flat = X
    else:
        raise ValueError(f"X_train must be 2D or 3D. Got shape {X.shape}")

    X_mean, X_std = _safe_mean_std(X_flat, eps=eps)

    # Y: fit per target channel
    C = Y.shape[-1]
    Y_mean = np.zeros((C,), dtype=np.float32)
    Y_std = np.ones((C,), dtype=np.float32)

    for c in range(C):
        y_c = Y[..., c].reshape(-1).astype(np.float64)
        if active_only and (volume_index is not None) and (c == int(volume_index)):
            mask = (y_c > 0.0)
            if mask.any():
                y_c = y_c[mask]
        m, s = _safe_mean_std(y_c, eps=eps)
        # _safe_mean_std returns arrays; here scalars
        Y_mean[c] = float(np.asarray(m).reshape(()))
        Y_std[c] = float(np.asarray(s).reshape(()))

    return {
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "active_only": bool(active_only),
        "volume_index": None if volume_index is None else int(volume_index),
        "eps": float(eps),
    }


def transform_with_scalers(
    X: np.ndarray,
    y: np.ndarray,
    scalers: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply fitted scalers to X and y."""
    X_mean = np.asarray(scalers["X_mean"], dtype=np.float32)
    X_std = np.asarray(scalers["X_std"], dtype=np.float32)
    Y_mean = np.asarray(scalers["Y_mean"], dtype=np.float32)
    Y_std = np.asarray(scalers["Y_std"], dtype=np.float32)

    X_in = np.asarray(X, dtype=np.float32)
    Y_in = np.asarray(y, dtype=np.float32)

    if X_in.ndim == 3:
        X_out = (X_in - X_mean.reshape(1, 1, -1)) / X_std.reshape(1, 1, -1)
    elif X_in.ndim == 2:
        X_out = (X_in - X_mean.reshape(1, -1)) / X_std.reshape(1, -1)
    else:
        raise ValueError(f"X must be 2D or 3D. Got shape {X_in.shape}")

    if Y_in.ndim != 3:
        raise ValueError(f"y must be 3D (N,H,C). Got shape {Y_in.shape}")

    Y_out = (Y_in - Y_mean.reshape(1, 1, -1)) / Y_std.reshape(1, 1, -1)

    return X_out.astype(np.float32), Y_out.astype(np.float32)
