"""Training utilities for hurdle-style PyTorch sequence models.

This module is intentionally model-agnostic and only assumes the model returns:
  active_logits, logvol_hat
with shapes (B, HORIZON).

It supports models whose forward() accepts either:
  - model(x)                      where x is a Tensor
  - model(*x)                     where x is a tuple/list of Tensors
  - model(**x)                    where x is a dict of Tensors

The training objective is a two-head hurdle loss:
  - Head1: BCEWithLogitsLoss on active mask (volume > active_eps)
  - Head2: masked SmoothL1Loss (Huber) on log1p(volume), active-only

Threshold selection is performed on validation only via a sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import contextlib

import numpy as np

import torch
import torch.nn.functional as F


@dataclass
class TrainConfig:
    """Simple configuration holder for training."""

    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 10
    patience: int = 3
    grad_clip: float = 1.0
    w_bce: float = 1.0
    w_reg: float = 1.0
    huber_delta: float = 1.0
    active_eps: float = 0.0
    use_amp: bool = False


def _to_device(obj, device: torch.device, non_blocking: bool = True):
    """Recursively move tensors to device (Tensor / tuple / list / dict)."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_device(x, device, non_blocking=non_blocking) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
    return obj


def _forward_model(model: torch.nn.Module, x):
    """Call model with a flexible x container."""
    if isinstance(x, (tuple, list)):
        return model(*x)
    if isinstance(x, dict):
        return model(**x)
    return model(x)


def _autocast_ctx(device: torch.device, enabled: bool):
    """Compatibility wrapper for AMP autocast (torch.amp.autocast preferred)."""
    if not enabled:
        return contextlib.nullcontext()

    # New API (PyTorch 2.x): torch.amp.autocast(device_type=..., enabled=...)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)

    # Legacy API
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return contextlib.nullcontext()


def _make_grad_scaler(device: torch.device, enabled: bool):
    """Compatibility wrapper for GradScaler (torch.amp.GradScaler preferred)."""
    if not enabled:
        # Still return a scaler-like object for unified code paths
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler(device_type=device.type, enabled=False)
            except TypeError:
                return torch.amp.GradScaler(enabled=False)
        return torch.cuda.amp.GradScaler(enabled=False)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type=device.type, enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)

    return torch.cuda.amp.GradScaler(enabled=True)


def compute_pos_weight_from_loader(
    loader: Iterable,
    *,
    volume_index: int,
    active_eps: float = 0.0,
    max_batches: Optional[int] = 200,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Compute pos_weight for BCE to handle class imbalance.

    pos_weight = n_negative / n_positive over all elements in (B, HORIZON).

    If no positive elements are observed, returns None.
    """
    pos = 0
    total = 0
    n_batches = 0

    for batch in loader:
        # Expected batch: (X, Y, ...) where Y: (B, H, C)
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Loader must yield at least (X, Y)")
        y = batch[1]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)

        if device is not None:
            y = y.to(device)

        y_vol = y[..., volume_index]
        mask = (y_vol > float(active_eps))

        pos += int(mask.sum().item())
        total += int(mask.numel())

        n_batches += 1
        if max_batches is not None and n_batches >= int(max_batches):
            break

    if pos <= 0:
        return None
    neg = total - pos
    w = float(neg) / float(pos)
    return torch.tensor(w, dtype=torch.float32)


def hurdle_losses(
    active_logits: torch.Tensor,
    logvol_hat: torch.Tensor,
    y_true: torch.Tensor,
    *,
    volume_index: int,
    pos_weight: Optional[torch.Tensor] = None,
    huber_delta: float = 1.0,
    active_eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute two-head hurdle loss.

    Returns:
        loss_bce, loss_reg, active_rate, n_active
    """
    if y_true.ndim != 3:
        raise ValueError(f"Expected y_true with shape (B, H, C). Got {tuple(y_true.shape)}")
    y_vol = y_true[..., volume_index]
    y_active = (y_vol > float(active_eps)).to(dtype=torch.float32)
    y_logvol = torch.log1p(torch.clamp(y_vol, min=0.0))

    if active_logits.shape != y_active.shape:
        raise ValueError(
            f"Shape mismatch: active_logits={tuple(active_logits.shape)} vs y_active={tuple(y_active.shape)}"
        )
    if logvol_hat.shape != y_active.shape:
        raise ValueError(
            f"Shape mismatch: logvol_hat={tuple(logvol_hat.shape)} vs y_active={tuple(y_active.shape)}"
        )

    # BCE over all elements
    if pos_weight is not None:
        pw = pos_weight.to(device=active_logits.device)
        loss_bce = F.binary_cross_entropy_with_logits(active_logits, y_active, pos_weight=pw)
    else:
        loss_bce = F.binary_cross_entropy_with_logits(active_logits, y_active)

    # Regression head: masked Huber on active-only
    mask = (y_active > 0.5)
    n_active = mask.sum().to(dtype=torch.float32)
    if n_active.item() > 0:
        # smooth_l1_loss beta is the Huber delta
        loss_reg = F.smooth_l1_loss(logvol_hat[mask], y_logvol[mask], beta=float(huber_delta), reduction="mean")
    else:
        loss_reg = torch.tensor(0.0, device=active_logits.device)

    active_rate = y_active.mean()
    return loss_bce, loss_reg, active_rate, n_active


def train_hurdle_tcn(
    *,
    model: torch.nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    volume_index: int,
    device: torch.device,
    cfg: TrainConfig,
    pos_weight: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """Train with early stopping on validation loss.

    Despite the name, this works for any model that returns (active_logits, logvol_hat).
    """
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    use_amp = bool(cfg.use_amp) and (device.type == "cuda")
    scaler = _make_grad_scaler(device=device, enabled=use_amp)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_bce": [],
        "train_reg": [],
        "val_bce": [],
        "val_reg": [],
        "train_active_rate": [],
        "val_active_rate": [],
    }

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(int(cfg.max_epochs)):
        # Allow IterableDataset to reshuffle each epoch
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        model.train()
        tr_loss = 0.0
        tr_bce = 0.0
        tr_reg = 0.0
        tr_active = 0.0
        tr_batches = 0

        for batch in train_loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Train loader must yield at least (X, Y)")

            x = _to_device(batch[0], device=device, non_blocking=True)
            y = _to_device(batch[1], device=device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with _autocast_ctx(device=device, enabled=use_amp):
                active_logits, logvol_hat = _forward_model(model, x)
                loss_bce_t, loss_reg_t, active_rate, _ = hurdle_losses(
                    active_logits,
                    logvol_hat,
                    y,
                    volume_index=volume_index,
                    pos_weight=pos_weight,
                    huber_delta=float(cfg.huber_delta),
                    active_eps=float(cfg.active_eps),
                )
                loss = float(cfg.w_bce) * loss_bce_t + float(cfg.w_reg) * loss_reg_t

            scaler.scale(loss).backward()
            if float(cfg.grad_clip) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()

            tr_loss += float(loss.detach().item())
            tr_bce += float(loss_bce_t.detach().item())
            tr_reg += float(loss_reg_t.detach().item())
            tr_active += float(active_rate.detach().item())
            tr_batches += 1

        tr_loss /= max(tr_batches, 1)
        tr_bce /= max(tr_batches, 1)
        tr_reg /= max(tr_batches, 1)
        tr_active /= max(tr_batches, 1)

        # Validation
        model.eval()
        va_loss = 0.0
        va_bce = 0.0
        va_reg = 0.0
        va_active = 0.0
        va_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    raise ValueError("Val loader must yield at least (X, Y)")

                x = _to_device(batch[0], device=device, non_blocking=True)
                y = _to_device(batch[1], device=device, non_blocking=True)

                with _autocast_ctx(device=device, enabled=use_amp):
                    active_logits, logvol_hat = _forward_model(model, x)

                loss_bce_t, loss_reg_t, active_rate, _ = hurdle_losses(
                    active_logits,
                    logvol_hat,
                    y,
                    volume_index=volume_index,
                    pos_weight=pos_weight,
                    huber_delta=float(cfg.huber_delta),
                    active_eps=float(cfg.active_eps),
                )
                loss = float(cfg.w_bce) * loss_bce_t + float(cfg.w_reg) * loss_reg_t

                va_loss += float(loss.detach().item())
                va_bce += float(loss_bce_t.detach().item())
                va_reg += float(loss_reg_t.detach().item())
                va_active += float(active_rate.detach().item())
                va_batches += 1

        va_loss /= max(va_batches, 1)
        va_bce /= max(va_batches, 1)
        va_reg /= max(va_batches, 1)
        va_active /= max(va_batches, 1)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_bce"].append(tr_bce)
        history["train_reg"].append(tr_reg)
        history["val_bce"].append(va_bce)
        history["val_reg"].append(va_reg)
        history["train_active_rate"].append(tr_active)
        history["val_active_rate"].append(va_active)

        if verbose:
            print(
                f"Epoch {epoch+1:02d}/{cfg.max_epochs} | "
                f"train_loss={tr_loss:.4f} (bce={tr_bce:.4f}, reg={tr_reg:.4f}) | "
                f"val_loss={va_loss:.4f} (bce={va_bce:.4f}, reg={va_reg:.4f})"
            )

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg.patience):
                if verbose:
                    print(f"Early stopping triggered (patience={cfg.patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def collect_hurdle_outputs(
    *,
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    volume_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect Y_true, Y_base, proba, logvol_hat from a loader.

    The loader must yield at least (X, Y_true, Y_base, ...). Only the first three items are used.

    Notes:
      - X can be a Tensor, tuple/list of Tensors, or dict of Tensors.
      - proba/logvol_hat are returned with shape (N, H).
    """
    model = model.to(device)
    model.eval()

    y_list: List[np.ndarray] = []
    yb_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []
    lv_list: List[np.ndarray] = []

    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 3:
            raise ValueError("Loader must yield at least (X, Y_true, Y_base)")

        x = _to_device(batch[0], device=device, non_blocking=True)
        y = _to_device(batch[1], device=device, non_blocking=True)
        yb = _to_device(batch[2], device=device, non_blocking=True)

        active_logits, logvol_hat = _forward_model(model, x)
        proba = torch.sigmoid(active_logits)

        y_list.append(y.detach().cpu().numpy().astype(np.float32))
        yb_list.append(yb.detach().cpu().numpy().astype(np.float32))
        p_list.append(proba.detach().cpu().numpy().astype(np.float32))
        lv_list.append(logvol_hat.detach().cpu().numpy().astype(np.float32))

    Y_true = np.concatenate(y_list, axis=0) if y_list else np.empty((0, 0, 0), dtype=np.float32)
    Y_base = np.concatenate(yb_list, axis=0) if yb_list else np.empty((0, 0, 0), dtype=np.float32)
    proba = np.concatenate(p_list, axis=0) if p_list else np.empty((0, 0), dtype=np.float32)
    logvol_hat = np.concatenate(lv_list, axis=0) if lv_list else np.empty((0, 0), dtype=np.float32)

    # Basic sanity
    if Y_true.ndim != 3 or Y_base.ndim != 3:
        raise ValueError("Collected Y arrays must have shape (N, H, C)")
    if proba.ndim != 2 or logvol_hat.ndim != 2:
        raise ValueError("Collected hurdle outputs must have shape (N, H)")

    if Y_true.shape[:2] != proba.shape:
        raise ValueError(f"Mismatch: Y_true {Y_true.shape} vs proba {proba.shape}")
    if proba.shape != logvol_hat.shape:
        raise ValueError(f"Mismatch: proba {proba.shape} vs logvol_hat {logvol_hat.shape}")

    return Y_true, Y_base, proba, logvol_hat


def apply_hurdle_threshold(
    proba: np.ndarray,
    logvol_hat: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Apply a probability threshold to get final volume predictions.

    Returns volume_hat with shape (N, H).
    """
    proba = np.asarray(proba, dtype=np.float64)
    logvol_hat = np.asarray(logvol_hat, dtype=np.float64)

    vol_reg = np.expm1(logvol_hat)
    vol_reg = np.clip(vol_reg, 0.0, None)
    gate = (proba >= float(threshold)).astype(np.float64)
    return (gate * vol_reg).astype(np.float32)


def threshold_sweep_on_validation(
    y_true_vol: np.ndarray,
    proba: np.ndarray,
    logvol_hat: np.ndarray,
    thresholds: np.ndarray,
    *,
    eps: float = 1e-8,
) -> Tuple[float, "pd.DataFrame"]:
    """Pick threshold minimizing FULL sMAPE on volume (flattened).

    Returns:
      best_thr, sweep_df
    """
    import pandas as pd

    yt = np.asarray(y_true_vol, dtype=np.float64).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    full_scores = np.zeros_like(thresholds, dtype=np.float64)
    active_scores = np.zeros_like(thresholds, dtype=np.float64)

    mask_active = (yt > 0)

    for i, thr in enumerate(thresholds):
        yhat = apply_hurdle_threshold(proba, logvol_hat, float(thr)).reshape(-1).astype(np.float64)
        num = 2.0 * np.abs(yt - yhat)
        den = np.abs(yt) + np.abs(yhat) + float(eps)
        full_scores[i] = float(np.mean(num / den))

        if mask_active.any():
            num_a = 2.0 * np.abs(yt[mask_active] - yhat[mask_active])
            den_a = np.abs(yt[mask_active]) + np.abs(yhat[mask_active]) + float(eps)
            active_scores[i] = float(np.mean(num_a / den_a))
        else:
            active_scores[i] = np.nan

    best_idx = int(np.nanargmin(full_scores))
    best_thr = float(thresholds[best_idx])

    sweep_df = pd.DataFrame(
        {
            "threshold": thresholds.astype(np.float64),
            "val_volume_full_smape": full_scores.astype(np.float64),
            "val_volume_active_smape": active_scores.astype(np.float64),
        }
    )

    return best_thr, sweep_df
