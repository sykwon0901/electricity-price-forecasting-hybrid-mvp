"""
Update a shared metrics table from standardized metrics JSON records.

This keeps reporting consistent across notebooks and scripts:
- results/metrics/<model>.json (one per run/model)
- results/metrics/metrics_table.csv (single consolidated table)
- results/metrics/metrics_table.xlsx
- images/eval/tables/metrics_table.png

Notes:
- This module does NOT compute metrics. It only aggregates already-saved JSON records.
- It is intentionally tolerant of missing keys and schema drift, but it will emit warnings
  (file-name only) when it cannot parse a JSON file.
- NaN/Inf values are preserved in the CSV/XLSX by default, but are made safe for PNG rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json
import math

import pandas as pd
import matplotlib.pyplot as plt


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite_or_nan(x: Any) -> Any:
    if _is_number(x):
        return x if math.isfinite(float(x)) else float("nan")
    return x


def _make_row(rec: Dict[str, Any], filename: str) -> Dict[str, Any]:
    baseline_full = _safe_get(rec, ["baseline", "full_smape"])
    baseline_active = _safe_get(rec, ["baseline", "active_smape"])
    model_full = _safe_get(rec, ["model", "full_smape"])
    model_active = _safe_get(rec, ["model", "active_smape"])

    sampling = rec.get("sampling_info", {}) if isinstance(rec.get("sampling_info"), dict) else {}
    constants = rec.get("constants", {}) if isinstance(rec.get("constants"), dict) else {}

    # Normalize numerics (Inf -> NaN) to avoid breaking downstream rendering/sorting.
    row = {
        "model_name": rec.get("model_name"),
        "run_tag": rec.get("run_tag"),
        "baseline_full": _finite_or_nan(baseline_full),
        "baseline_active": _finite_or_nan(baseline_active),
        "model_full": _finite_or_nan(model_full),
        "model_active": _finite_or_nan(model_active),
        "n_windows_used": _finite_or_nan(sampling.get("n_test_windows_used")),
        "n_ids": _finite_or_nan(sampling.get("n_ids_test")),
        "WIN": _finite_or_nan(constants.get("WIN")),
        "HORIZON": _finite_or_nan(constants.get("HORIZON")),
        "MAX_TEST_WINDOWS": _finite_or_nan(constants.get("MAX_TEST_WINDOWS")),
        "active_ratio_test": _finite_or_nan(rec.get("active_ratio_test")),
        "source_file": filename,
    }
    return row


def _format_float(x: Any, ndigits: int = 6) -> Any:
    if _is_number(x) and x is not None and math.isfinite(float(x)):
        return f"{float(x):.{ndigits}f}"
    return x


def _save_table_png(df: pd.DataFrame, out_png: Union[str, Path], dpi: int = 160) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Limit width by selecting key columns for the figure
    cols = [
        "model_name",
        "run_tag",
        "baseline_full",
        "baseline_active",
        "model_full",
        "model_active",
        "n_windows_used",
        "n_ids",
        "active_ratio_test",
    ]
    cols = [c for c in cols if c in df.columns]
    df_fig = df[cols].copy()

    # Format floats for readability (PNG only)
    for c in ["baseline_full", "baseline_active", "model_full", "model_active", "active_ratio_test"]:
        if c in df_fig.columns:
            df_fig[c] = df_fig[c].map(lambda v: _format_float(v, ndigits=6))

    fig = plt.figure(figsize=(min(18, 1.8 + 1.2 * len(df_fig.columns)), 0.6 + 0.35 * max(3, len(df_fig))))
    ax = fig.add_subplot(111)
    ax.axis("off")

    table = ax.table(
        cellText=df_fig.values,
        colLabels=list(df_fig.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def update_metrics_table(
    metrics_dir: Union[str, Path],
    out_csv: Union[str, Path],
    out_xlsx: Union[str, Path],
    out_png: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build and save a consolidated metrics table from all *.json files in metrics_dir.

    Args:
        metrics_dir: Directory containing per-run metrics JSONs.
        out_csv: Output CSV path.
        out_xlsx: Output XLSX path.
        out_png: Optional PNG path to render a compact table image.
        verbose: If True, prints warnings (file-name only) for unreadable JSONs.

    Returns:
        pd.DataFrame (sorted, one row per json record)
    """
    metrics_dir = Path(metrics_dir)
    out_csv = Path(out_csv)
    out_xlsx = Path(out_xlsx)

    json_files = sorted([p for p in metrics_dir.glob("*.json") if p.is_file()], key=lambda p: p.name)

    rows: List[Dict[str, Any]] = []
    bad_files: List[str] = []

    for p in json_files:
        try:
            rec = _read_json(p)
            rows.append(_make_row(rec, filename=p.name))
        except Exception:
            bad_files.append(p.name)

    if verbose and bad_files:
        # File-name only to avoid leaking local/Drive absolute paths.
        print("WARN: Skipped unreadable metrics JSON files:", ", ".join(bad_files))

    df = pd.DataFrame(rows)

    # Deterministic ordering
    sort_cols = [c for c in ["model_name", "run_tag", "source_file"] if c in df.columns]
    if sort_cols and len(df) > 0:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Ensure output dirs exist
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    # Write outputs even if empty to keep pipeline predictable
    df.to_csv(out_csv, index=False)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="metrics", index=False)

    if out_png is not None:
        _save_table_png(df, out_png)

    return df
