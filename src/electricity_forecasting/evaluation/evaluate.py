"""
Standard evaluation and serialization utilities.

This module enforces a consistent metrics JSON schema across all models,
aligned with the legacy definition:
- Full sMAPE: flatten-based
- Active sMAPE: ground-truth volume mask

It also provides deterministic window_index persistence for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .metrics import evaluate_full_smape, evaluate_active_smape


# -----------------------------
# JSON helpers
# -----------------------------

def _to_python(obj: Any) -> Any:
    """
    Convert numpy/pandas types into JSON-serializable Python types.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        # Keep ISO string including timezone if present
        return obj.isoformat()
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


def save_json(obj: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    Save a dictionary as JSON (ensuring numpy/pandas types are serializable).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_python(obj), f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON from disk.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# window_index persistence
# -----------------------------

def save_window_index(window_index: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Persist canonical window_index to CSV.
    Required columns: ["ID", "start_pos"].

    This file should be committed (or versioned) under results/window_index/
    to ensure exact reproducibility across model notebooks.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    required = {"ID", "start_pos"}
    if not required.issubset(set(window_index.columns)):
        raise ValueError(f"window_index must include columns {required}. Got: {list(window_index.columns)}")

    wi = window_index[["ID", "start_pos"]].copy()
    wi["ID"] = wi["ID"].astype(np.int64)
    wi["start_pos"] = wi["start_pos"].astype(np.int64)
    wi = wi.sort_values(["ID", "start_pos"]).reset_index(drop=True)

    wi.to_csv(path, index=False)


def load_window_index(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load canonical window_index from CSV.
    """
    path = Path(path)
    wi = pd.read_csv(path)
    if "ID" not in wi.columns or "start_pos" not in wi.columns:
        raise ValueError(f"Invalid window_index CSV: {path}. Must have columns ID,start_pos")
    wi["ID"] = wi["ID"].astype(np.int64)
    wi["start_pos"] = wi["start_pos"].astype(np.int64)
    wi = wi.sort_values(["ID", "start_pos"]).reset_index(drop=True)
    return wi


def hash_window_index(window_index: pd.DataFrame) -> str:
    """
    Stable hash for window_index content (to track exact sampling).
    """
    required = {"ID", "start_pos"}
    if not required.issubset(set(window_index.columns)):
        raise ValueError(f"window_index must include columns {required}. Got: {list(window_index.columns)}")

    wi = window_index[["ID", "start_pos"]].copy()
    wi["ID"] = wi["ID"].astype(np.int64)
    wi["start_pos"] = wi["start_pos"].astype(np.int64)
    wi = wi.sort_values(["ID", "start_pos"]).reset_index(drop=True)

    # Create deterministic bytes without external deps
    arr = wi.to_numpy(dtype=np.int64, copy=False)
    import hashlib
    h = hashlib.sha256(arr.tobytes()).hexdigest()
    return h


# -----------------------------
# Metric evaluation (canonical)
# -----------------------------

def evaluate_predictions(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    volume_index: int,
    eps: float = 1e-8,
    active_eps: float = 0.0,
) -> Dict[str, float]:
    """
    Compute canonical Full/Active sMAPE.
    """
    full = evaluate_full_smape(Y_true, Y_pred, eps=eps)
    active = evaluate_active_smape(Y_true, Y_pred, volume_index=volume_index, eps=eps, active_eps=active_eps)
    return {"full_smape": float(full), "active_smape": float(active)}


def active_ratio_from_truth(
    Y_true: np.ndarray,
    volume_index: int,
    active_eps: float = 0.0,
) -> float:
    """
    Active ratio based on ground-truth volume mask on Y_true.
    """
    mask = (Y_true[..., volume_index] > float(active_eps))
    return float(mask.mean())


def build_metrics_record(
    *,
    model_name: str,
    run_tag: str,
    constants: Dict[str, Any],
    split_info: Dict[str, Any],
    sampling_info: Dict[str, Any],
    shapes: Optional[Dict[str, Any]] = None,
    baseline_metrics: Dict[str, float],
    model_metrics: Dict[str, float],
    active_ratio_test: float,
    threshold_info: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a standardized metrics JSON record.

    Required fields are explicit to keep every notebook consistent.
    """
    rec = {
        "model_name": model_name,
        "run_tag": run_tag,
        "constants": constants,         # WIN, HORIZON, MAX_TEST_WINDOWS, target_cols order, etc.
        "split_info": split_info,       # valid_ratio, n_train_windows, n_val_windows, etc.
        "sampling_info": sampling_info, # n_ids_test, n_test_windows_used, sampling_rule, hashes, etc.
        "shapes": shapes or {},
        "baseline": baseline_metrics,   # full_smape, active_smape
        "model": model_metrics,         # full_smape, active_smape
        "active_ratio_test": float(active_ratio_test),
    }

    if threshold_info is not None:
        rec["threshold_info"] = threshold_info  # must state threshold_selected_on="validation"
    if extra is not None:
        rec["extra"] = extra

    return rec


def save_metrics_record(
    record: Dict[str, Any],
    out_json: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Save a standardized metrics record JSON.

    This is a thin wrapper so notebooks do not call save_json directly.
    """
    save_json(record, out_json, indent=indent)


def assert_no_leakage_policy(threshold_info: Optional[Dict[str, Any]]) -> None:
    """
    Enforce the project policy: any threshold selection must be done on validation, not test.
    """
    if threshold_info is None:
        return
    if "threshold_selected_on" not in threshold_info:
        raise ValueError("threshold_info must include 'threshold_selected_on' field.")
    if str(threshold_info["threshold_selected_on"]).lower() != "validation":
        raise ValueError("Leakage policy violated: threshold must be selected on validation only.")


def log_run_header(
    *,
    model_name: str,
    constants: Dict[str, Any],
    shapes: Dict[str, Any],
    sampling_info: Dict[str, Any],
    threshold_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print a compact run header for notebook reproducibility.
    """
    print("=== RUN HEADER ===")
    print("Model:", model_name)
    print("Constants:", constants)
    print("Shapes:", shapes)
    print("Sampling:", sampling_info)
    if threshold_info is not None:
        print("Threshold:", threshold_info)
    print("==================")

def hash_id_list(ids: list[Any]) -> str:
    """Stable SHA256 hash of an ID list (order-insensitive).

    This is useful for tracking exactly which IDs were included in a run.
    """
    import hashlib

    ids_sorted = sorted(str(x) for x in ids)
    payload = ("|".join(ids_sorted)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def log_run_header_from_record(record: Dict[str, Any]) -> None:
    """Print the reproducibility header directly from a saved metrics record."""
    model_name = str(record.get("model_name", ""))
    constants = record.get("constants", {})
    split_info = record.get("split_info", {})
    sampling_info = record.get("sampling_info", {})
    active_ratio_test = record.get("active_ratio_test", None)
    threshold_info = record.get("threshold_info", None)

    print("=== RUN HEADER ===")
    print("Model:", model_name)
    print("Constants:", constants)
    print("Split:", split_info)
    print("Sampling:", sampling_info)
    if active_ratio_test is not None:
        print("Active ratio on test:", active_ratio_test)
    if threshold_info is not None:
        print("Threshold:", threshold_info)
    print("==================")
