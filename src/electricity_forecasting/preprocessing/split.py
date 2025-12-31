"""
Time-based split utilities.

Canonical split for this project is WINDOW-BASED:
- Compute a per-ID cutoff on the 15-min grid using valid_ratio (last 10% of time steps is validation).
- Train windows: targets must be fully within the train region.
- Val windows: targets must start within the val region.

This avoids leakage while still allowing deterministic window definitions.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def ensure_sorted_grid(
    df_grid: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    pos_col: str = "pos",
) -> pd.DataFrame:
    """
    Ensure df_grid is sorted by ID and time, and has per-ID contiguous pos.
    """
    if id_col not in df_grid.columns:
        raise ValueError(f"Missing required column: {id_col}")
    if time_col not in df_grid.columns:
        raise ValueError(f"Missing required column: {time_col}")

    df = df_grid.copy()
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    if pos_col not in df.columns:
        df[pos_col] = df.groupby(id_col, sort=False).cumcount().astype(np.int32)
    else:
        rebuilt = df.groupby(id_col, sort=False).cumcount().astype(np.int32)
        if not np.array_equal(rebuilt.to_numpy(), df[pos_col].to_numpy(dtype=np.int32, copy=False)):
            df[pos_col] = rebuilt

    return df


def compute_time_cut_per_id(
    df_grid: pd.DataFrame,
    valid_ratio: float = 0.1,
    ids: Optional[Sequence[int]] = None,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    pos_col: str = "pos",
) -> pd.DataFrame:
    """
    Compute a per-ID time cutoff index on the 15-min grid.

    cutoff_pos is the first position that belongs to validation region
    (i.e., validation starts at cutoff_pos and goes until the end).

    cutoff_pos = floor(T * (1 - valid_ratio))

    Returns DataFrame columns:
      [ID, T, cutoff_pos]
    """
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError("valid_ratio must be in (0, 1)")

    df = ensure_sorted_grid(df_grid, id_col=id_col, time_col=time_col, pos_col=pos_col)
    if ids is not None:
        ids_set = set(map(int, ids))
        df = df[df[id_col].isin(ids_set)].copy()

    # T per ID
    max_pos = df.groupby(id_col, sort=False)[pos_col].max().astype(np.int64)
    T = (max_pos + 1).astype(np.int64)

    cutoff = np.floor(T.to_numpy(dtype=np.float64) * (1.0 - float(valid_ratio))).astype(np.int64)
    cutoff = np.clip(cutoff, 0, T.to_numpy(dtype=np.int64))

    out = pd.DataFrame({
        id_col: T.index.astype(np.int64),
        "T": T.values.astype(np.int64),
        "cutoff_pos": cutoff.astype(np.int64),
    }).sort_values(id_col).reset_index(drop=True)

    return out


def split_window_index_by_id_time(
    df_grid: pd.DataFrame,
    win: int,
    horizon: int,
    valid_ratio: float = 0.1,
    ids: Optional[Sequence[int]] = None,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    pos_col: str = "pos",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Canonical split: return window_index_train and window_index_val defined on the SAME df_grid.

    Definitions per ID with length T and cutoff_pos:
    - Train windows must have targets entirely within train region:
        start_pos + win + horizon <= cutoff_pos
      => start_pos <= cutoff_pos - win - horizon
    - Val windows must have targets starting in validation region:
        start_pos + win >= cutoff_pos
      and still must have complete horizon:
        start_pos <= T - win - horizon

    Returned formats:
      window_index_train: DataFrame ["ID", "start_pos"]
      window_index_val  : DataFrame ["ID", "start_pos"]

    This design keeps evaluation strictly on future targets while preserving past context.
    """
    if win <= 0 or horizon <= 0:
        raise ValueError("win and horizon must be positive")

    df = ensure_sorted_grid(df_grid, id_col=id_col, time_col=time_col, pos_col=pos_col)

    if ids is None:
        ids_sorted = np.array(sorted(df[id_col].unique().tolist()), dtype=np.int64)
    else:
        ids_sorted = np.array(sorted(set(map(int, ids))), dtype=np.int64)

    cuts = compute_time_cut_per_id(
        df, valid_ratio=valid_ratio, ids=ids_sorted, id_col=id_col, time_col=time_col, pos_col=pos_col
    ).set_index(id_col)

    train_rows = []
    val_rows = []

    for _id in ids_sorted.tolist():
        if _id not in cuts.index:
            continue

        T = int(cuts.loc[_id, "T"])
        cutoff_pos = int(cuts.loc[_id, "cutoff_pos"])

        max_start = T - win - horizon
        if max_start < 0:
            continue

        # Train start range: [0, cutoff_pos - win - horizon]
        max_train_start = cutoff_pos - win - horizon
        if max_train_start >= 0:
            train_starts = np.arange(0, min(max_train_start, max_start) + 1, dtype=np.int64)
            train_rows.extend((int(_id), int(s)) for s in train_starts.tolist())

        # Val start range: [cutoff_pos - win, max_start] (because start_pos+win >= cutoff_pos)
        min_val_start = cutoff_pos - win
        if min_val_start <= max_start:
            val_starts = np.arange(max(0, min_val_start), max_start + 1, dtype=np.int64)
            val_rows.extend((int(_id), int(s)) for s in val_starts.tolist())

    w_tr = pd.DataFrame(train_rows, columns=[id_col, "start_pos"])
    w_va = pd.DataFrame(val_rows, columns=[id_col, "start_pos"])

    for w in (w_tr, w_va):
        if len(w) > 0:
            w[id_col] = w[id_col].astype(np.int64)
            w["start_pos"] = w["start_pos"].astype(np.int64)
            w.sort_values([id_col, "start_pos"], inplace=True)
            w.reset_index(drop=True, inplace=True)

    meta = {
        "valid_ratio": float(valid_ratio),
        "n_ids": float(len(ids_sorted)),
        "n_train_windows": float(len(w_tr)),
        "n_val_windows": float(len(w_va)),
        "win": float(win),
        "horizon": float(horizon),
    }
    return w_tr, w_va, meta


def report_window_split_stats(
    window_index: pd.DataFrame,
    id_col: str = "ID",
) -> Dict[str, float]:
    """
    Summarize a window_index for reproducibility logs.
    """
    if window_index is None or len(window_index) == 0:
        return {
            "n_windows": 0.0,
            "n_ids": 0.0,
            "min_windows_per_id": 0.0,
            "median_windows_per_id": 0.0,
            "max_windows_per_id": 0.0,
        }

    counts = window_index.groupby(id_col, sort=False).size().to_numpy(dtype=np.int64)
    return {
        "n_windows": float(len(window_index)),
        "n_ids": float(window_index[id_col].nunique()),
        "min_windows_per_id": float(counts.min()),
        "median_windows_per_id": float(np.median(counts)),
        "max_windows_per_id": float(counts.max()),
    }
