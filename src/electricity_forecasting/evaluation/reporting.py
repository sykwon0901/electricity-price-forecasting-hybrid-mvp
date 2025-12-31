# src/electricity_forecasting/evaluation/reporting.py
"""
Reporting utilities for consistent, portfolio-friendly evaluation outputs.

This module builds and saves a standardized metrics table:
- Full sMAPE per target (high/low/close/volume) + Full Total
- Active sMAPE per target (mask from ground-truth volume) + Active Total

Totals are computed with the legacy definition (flatten-based), NOT by averaging per-target sMAPEs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import evaluate_full_smape, smape_flat


# -----------------------------
# Core metric helpers
# -----------------------------


def full_smape_per_target(Y_true: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-8) -> List[float]:
    """
    Compute Full sMAPE per target channel.
    Y_true/Y_pred: (N, H, C)
    """
    if Y_true.shape != Y_pred.shape:
        raise ValueError(f"Shape mismatch: Y_true={Y_true.shape}, Y_pred={Y_pred.shape}")
    if Y_true.ndim != 3:
        raise ValueError("Expected Y tensors with shape (N, H, C).")

    C = Y_true.shape[-1]
    return [smape_flat(Y_true[..., k], Y_pred[..., k], eps=eps) for k in range(C)]


def active_smape_per_target(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    active_mask: np.ndarray,
    eps: float = 1e-8,
) -> List[float]:
    """
    Compute Active sMAPE per target channel, where active_mask is derived from truth volume.
    active_mask: (N, H) boolean
    """
    if Y_true.shape != Y_pred.shape:
        raise ValueError(f"Shape mismatch: Y_true={Y_true.shape}, Y_pred={Y_pred.shape}")
    if Y_true.ndim != 3:
        raise ValueError("Expected Y tensors with shape (N, H, C).")

    if active_mask.shape != Y_true.shape[:2]:
        raise ValueError(f"active_mask shape {active_mask.shape} must match (N, H) = {Y_true.shape[:2]}")

    C = Y_true.shape[-1]
    out: List[float] = []
    for k in range(C):
        yt = Y_true[..., k][active_mask]
        yp = Y_pred[..., k][active_mask]
        out.append(smape_flat(yt, yp, eps=eps))
    return out


# -----------------------------
# Table builder
# -----------------------------

def make_metrics_row(
    model_name: str,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    target_cols: Sequence[str],
    volume_index: int,
    eps: float = 1e-8,
    active_eps: float = 0.0,
    total_mode: str = "mean",  # kept for backward compatibility (ignored)
) -> Dict[str, Any]:
    """
    Build a single row for the metrics table.

    Notes:
      - Totals are computed with the legacy definition (flatten-based), NOT by averaging per-target sMAPEs.
      - `active_eps` is the ACTIVE MASK threshold applied to ground-truth volume:
            active_mask = (Y_true[..., volume_index] > active_eps)
      - `total_mode` is kept for backward compatibility but is ignored.
    """
    if len(target_cols) != Y_true.shape[-1]:
        raise ValueError("target_cols length must match last dim of Y tensors.")

    active_mask = (Y_true[..., volume_index] > float(active_eps))

    full_vals = full_smape_per_target(Y_true, Y_pred, eps=eps)
    act_vals = active_smape_per_target(Y_true, Y_pred, active_mask, eps=eps)

    # Legacy totals (flatten-based, do NOT average per-target sMAPEs)
    full_total = float(evaluate_full_smape(Y_true, Y_pred))

    yt_all = Y_true[active_mask]
    yp_all = Y_pred[active_mask]
    act_total = float(smape_flat(yt_all, yp_all, eps=eps))


    # Map per-target to expected named columns (high/low/close/volume ...)
    name_map_full = {f"Full {t.title()}": v for t, v in zip(target_cols, full_vals)}
    name_map_act = {f"Active {t.title()}": v for t, v in zip(target_cols, act_vals)}

    row = {"Model": model_name}
    row.update(name_map_act)
    row["Active Total"] = act_total
    row.update(name_map_full)
    row["Full Total"] = full_total
    return row


def make_metrics_table(
    Y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    target_cols: Sequence[str] = ("high", "low", "close", "volume"),
    volume_index: int = 3,
    eps: float = 1e-8,
    active_eps: float = 0.0,
    total_mode: str = "mean",  # kept for backward compatibility (ignored)
    model_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a standardized metrics table from multiple model predictions.

    preds: dict mapping model_name -> Y_pred tensor of shape (N, H, C)
    model_order: optional list to enforce row ordering.
    """
    if Y_true.ndim != 3:
        raise ValueError("Y_true must have shape (N, H, C).")

    if model_order is None:
        model_names = list(preds.keys())
    else:
        model_names = model_order

    rows = []
    for name in model_names:
        if name not in preds:
            raise ValueError(f"Model '{name}' not found in preds.")
        row = make_metrics_row(
            model_name=name,
            Y_true=Y_true,
            Y_pred=preds[name],
            target_cols=target_cols,
            volume_index=volume_index,
            eps=eps,
            active_eps=active_eps,
            total_mode=total_mode,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Standard column ordering: Active (per target) + Active Total, then Full (per target) + Full Total
    act_cols = [f"Active {t.title()}" for t in target_cols] + ["Active Total"]
    full_cols = [f"Full {t.title()}" for t in target_cols] + ["Full Total"]
    col_order = ["Model"] + act_cols + full_cols
    df = df[col_order]

    return df


# -----------------------------
# Saving utilities
# -----------------------------

def save_metrics_table(
    df: pd.DataFrame,
    out_csv: Union[str, Path],
    out_json: Union[str, Path],
) -> None:
    """
    Save the metrics table as CSV and JSON.
    JSON is stored as list of records (row-wise dictionaries).
    """
    out_csv = Path(out_csv)
    out_json = Path(out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)


def save_metrics_table_figure(
    df: pd.DataFrame,
    out_png: Union[str, Path],
    title: str = "Model Comparison (sMAPE)",
    dpi: int = 200,
    float_fmt: str = "{:.4f}",
) -> None:
    """
    Render the metrics DataFrame as an image (PNG).

    This uses matplotlib's table artist for a clean, dependency-free export.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Format floats for readability
    df_disp = df.copy()

    def _fmt_cell(x):
        try:
            if x is None:
                return ""
            if pd.isna(x):
                return ""
        except Exception:
            pass

        try:
            return float_fmt.format(float(x))
        except Exception:
            return ""

    for c in df_disp.columns:
        if c != "Model":
            df_disp[c] = df_disp[c].apply(_fmt_cell)

    nrows, ncols = df_disp.shape

    # Heuristic figure sizing
    fig_w = max(10.0, 1.25 * ncols)
    fig_h = max(2.5, 0.55 * (nrows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=df_disp.values,
        colLabels=df_disp.columns.tolist(),
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    # Bold header row
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------
# Compact cross-model metrics table (single CSV for readers)
# ---------------------------------------------------------------------


def _compact_row_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a metrics record into a single-row schema for the global table."""
    constants = record.get("constants", {}) or {}
    sampling = record.get("sampling_info", {}) or {}
    threshold = record.get("threshold_info", {}) or {}

    baseline = record.get("baseline", {}) or {}
    model = record.get("model", {}) or {}

    row = {
        "model_name": record.get("model_name", ""),
        "run_tag": record.get("run_tag", ""),
        "baseline_full": baseline.get("full_smape", None),
        "baseline_active": baseline.get("active_smape", None),
        "model_full": model.get("full_smape", None),
        "model_active": model.get("active_smape", None),
        "active_ratio_test": record.get("active_ratio_test", None),
        "n_test_windows_used": sampling.get("n_test_windows_used", None),
        "n_ids_test": sampling.get("n_ids_test", None),
        "WIN": constants.get("WIN", None),
        "HORIZON": constants.get("HORIZON", None),
        "MAX_TEST_WINDOWS": constants.get("MAX_TEST_WINDOWS", None),
        "test_window_sampling_rule": sampling.get("test_window_sampling_rule", None),
        "threshold_selected_on": threshold.get("threshold_selected_on", None),
    }

    # Optional: keep hashes if present
    hashes = sampling.get("window_index_hashes", None)
    if isinstance(hashes, dict) and "test" in hashes:
        row["window_index_hash_test"] = hashes.get("test")

    return row


def load_metrics_records(metrics_dir: str | Path) -> List[Dict[str, Any]]:
    """Load all metrics JSON records (one per model) from a directory."""
    metrics_dir = Path(metrics_dir)
    if not metrics_dir.exists():
        return []

    records: List[Dict[str, Any]] = []
    for p in sorted(metrics_dir.glob("*.json")):
        # Skip derived tables
        if p.name.startswith("metrics_table") or p.name.startswith("demo_"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(data, dict) and "model_name" in data and "baseline" in data and "model" in data:
            records.append(data)

    return records


def build_compact_metrics_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a compact DataFrame with one row per model."""
    rows = [_compact_row_from_record(r) for r in records]
    df = pd.DataFrame(rows)

    # Stable ordering
    order = [
        "model_name",
        "run_tag",
        "baseline_full",
        "baseline_active",
        "model_full",
        "model_active",
        "active_ratio_test",
        "n_test_windows_used",
        "n_ids_test",
        "WIN",
        "HORIZON",
        "MAX_TEST_WINDOWS",
        "test_window_sampling_rule",
        "threshold_selected_on",
        "window_index_hash_test",
    ]
    cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    df = df.loc[:, cols]

    if "model_name" in df.columns:
        df = df.sort_values("model_name").reset_index(drop=True)

    return df


def save_compact_metrics_table(
    df: pd.DataFrame,
    out_csv: str | Path,
    out_xlsx: str | Path | None = None,
) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if out_xlsx is not None:
        out_xlsx = Path(out_xlsx)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        # Avoid any styling; keep it simple and portable.
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="metrics")


def update_metrics_table(
    *,
    metrics_dir: str | Path,
    out_csv: str | Path,
    out_xlsx: str | Path | None = None,
    out_png: str | Path | None = None,
    title: str = "Model performance metrics (canonical)",
) -> pd.DataFrame:
    """Rebuild the global metrics table from JSON records."""
    records = load_metrics_records(metrics_dir)
    df = build_compact_metrics_table(records)
    save_compact_metrics_table(df, out_csv=out_csv, out_xlsx=out_xlsx)

    if out_png is not None:
        save_metrics_table_figure(df, out_png, title=title)

    return df
