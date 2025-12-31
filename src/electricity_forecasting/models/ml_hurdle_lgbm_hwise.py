"""Horizon-wise hurdle LightGBM (volume-only).

Two-stage approach:
- classifier predicts P(active)
- regressor predicts log1p(volume) on active samples
- a threshold gate converts to final volume prediction

This matches the notebook implementation (no early stopping by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import lightgbm as lgb


@dataclass
class HurdleLGBMHWise:
    models_cls: Dict[int, lgb.LGBMClassifier]
    models_reg: Dict[int, lgb.LGBMRegressor]
    params_cls: dict
    params_reg: dict


def train_hurdle_hwise(
    X_tr: np.ndarray,
    y_tr_vol: np.ndarray,  # (N, H)
    X_va: np.ndarray,
    y_va_vol: np.ndarray,  # (N, H)
    horizon: int,
    params_cls: dict,
    params_reg: dict,
) -> Tuple[HurdleLGBMHWise, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train horizon-wise models and return validation/test-ready prediction tensors."""
    proba_va = np.zeros((X_va.shape[0], horizon), dtype=np.float32)
    logvol_va = np.zeros((X_va.shape[0], horizon), dtype=np.float32)

    models_cls = {}
    models_reg = {}

    for h in range(1, horizon + 1):
        ytr = y_tr_vol[:, h - 1]
        yva = y_va_vol[:, h - 1]

        ytr_active = (ytr > 0).astype(np.int32)
        yva_active = (yva > 0).astype(np.int32)

        clf = lgb.LGBMClassifier(**params_cls)
        clf.fit(X_tr, ytr_active)
        models_cls[h] = clf
        proba_va[:, h - 1] = clf.predict_proba(X_va)[:, 1].astype(np.float32)

        idx_tr = np.where(ytr_active == 1)[0]
        if idx_tr.size == 0:
            raise RuntimeError(f"No active train samples for horizon h={h}")

        Xtr_reg = X_tr[idx_tr]
        ytr_reg = np.log1p(ytr[idx_tr]).astype(np.float32)

        reg = lgb.LGBMRegressor(**params_reg)
        reg.fit(Xtr_reg, ytr_reg)
        models_reg[h] = reg
        logvol_va[:, h - 1] = reg.predict(X_va).astype(np.float32)

    bundle = HurdleLGBMHWise(models_cls=models_cls, models_reg=models_reg, params_cls=params_cls, params_reg=params_reg)
    return bundle, proba_va, logvol_va


def predict_hurdle_hwise(
    bundle: HurdleLGBMHWise,
    X: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (proba_active, logvol_hat) for all horizons."""
    proba = np.zeros((X.shape[0], horizon), dtype=np.float32)
    logvol = np.zeros((X.shape[0], horizon), dtype=np.float32)

    for h in range(1, horizon + 1):
        proba[:, h - 1] = bundle.models_cls[h].predict_proba(X)[:, 1].astype(np.float32)
        logvol[:, h - 1] = bundle.models_reg[h].predict(X).astype(np.float32)

    return proba, logvol


def apply_hurdle_threshold(proba: np.ndarray, logvol_hat: np.ndarray, threshold: float) -> np.ndarray:
    """Convert (proba, logvol_hat) to volume prediction with a gate."""
    vol_reg = np.expm1(logvol_hat.astype(np.float64))
    vol_reg = np.clip(vol_reg, 0.0, None)
    gate = (proba.astype(np.float64) >= float(threshold)).astype(np.float64)
    return (gate * vol_reg).astype(np.float32)
