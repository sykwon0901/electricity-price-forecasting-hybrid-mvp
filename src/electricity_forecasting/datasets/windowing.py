"""
Windowing utilities (canonical).

This module defines the project's canonical window index format and provides
deterministic, ID-balanced sampling with a global cap (e.g., 200k test windows),
plus tensor builders shared across ML/DL models.

Canonical window_index format:
- DataFrame with columns: ["ID", "start_pos"]
  * ID: int
  * start_pos: int  (0-based position within each ID's 15-min grid)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# -----------------------------
# Helpers: validation / indexing
# -----------------------------

def ensure_sorted_grid(
    df_grid: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    pos_col: str = "pos",
) -> pd.DataFrame:
    """
    Ensure df_grid is sorted by ID and time, and has a per-ID contiguous 0..T-1 position column.

    Requirements for downstream window slicing:
    - Rows must be grouped contiguously per ID.
    - Within each ID, rows must be ordered by time.
    - pos must be 0..T-1 without gaps for each ID.
    """
    if id_col not in df_grid.columns:
        raise ValueError(f"Missing required column: {id_col}")
    if time_col not in df_grid.columns:
        raise ValueError(f"Missing required column: {time_col}")

    df = df_grid.copy()

    # Sort
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # Ensure pos
    if pos_col not in df.columns:
        df[pos_col] = df.groupby(id_col, sort=False).cumcount().astype(np.int32)
    else:
        # Rebuild pos if it's inconsistent
        pos = df[pos_col].to_numpy()
        if np.any(pos < 0):
            df[pos_col] = df.groupby(id_col, sort=False).cumcount().astype(np.int32)
        else:
            # Validate monotonic within each ID (soft check)
            # If violated, rebuild to be safe.
            rebuilt = df.groupby(id_col, sort=False).cumcount().astype(np.int32)
            if not np.array_equal(rebuilt.to_numpy(), pos.astype(np.int32)):
                df[pos_col] = rebuilt

    return df


def _compute_id_slices(
    df_sorted: pd.DataFrame,
    id_col: str = "ID",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute contiguous slices per ID for a df that is already sorted and grouped by ID.
    Returns:
      - ids: unique IDs in encounter order
      - starts: start row index for each ID
      - ends: end row index (exclusive) for each ID
    """
    id_arr = df_sorted[id_col].to_numpy()
    if id_arr.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    change = np.flatnonzero(id_arr[1:] != id_arr[:-1]) + 1
    starts = np.r_[0, change].astype(np.int64)
    ends = np.r_[change, id_arr.size].astype(np.int64)
    ids = id_arr[starts].astype(np.int64)
    return ids, starts, ends


def count_windows_per_id(
    df_grid: pd.DataFrame,
    win: int,
    horizon: int,
    ids: Optional[Sequence[int]] = None,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    pos_col: str = "pos",
) -> pd.DataFrame:
    """
    Count grid length T and available number of windows for each ID.

    n_windows = max(T - win - horizon + 1, 0)

    Returns DataFrame with columns:
      ["ID", "T", "n_windows"]
    """
    df = ensure_sorted_grid(df_grid, id_col=id_col, time_col=time_col, pos_col=pos_col)

    if ids is not None:
        ids_set = set(map(int, ids))
        df = df[df[id_col].isin(ids_set)].copy()

    grp = df.groupby(id_col, sort=False)[pos_col].max()
    # pos is 0..T-1, so max pos == T-1
    T = (grp + 1).astype(np.int64)
    n_windows = (T - int(win) - int(horizon) + 1).clip(lower=0).astype(np.int64)

    out = pd.DataFrame({id_col: T.index.astype(np.int64), "T": T.values, "n_windows": n_windows.values})
    out = out.sort_values(id_col).reset_index(drop=True)
    return out


def sample_windows_id_balanced(
    df_grid: pd.DataFrame,
    ids: Sequence[int],
    max_windows: int,
    win: int,
    horizon: int,
    seed: int = 42,
    strategy: str = "spread",
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    pos_col: str = "pos",
) -> pd.DataFrame:
    """
    Deterministically sample windows with an ID-balanced allocation under a global cap.

    Canonical return format: window_index DataFrame with columns ["ID", "start_pos"].

    Allocation rule:
      q = max_windows // n_ids
      r = max_windows % n_ids
      alloc_i = min(n_windows_i, q + (1 if id_rank < r else 0))

    Selection strategies (per ID):
      - "spread": evenly spread start_pos across [0, n_windows_i - 1] using linspace
      - "first": take earliest windows [0..alloc_i-1]
      - "random": sample alloc_i unique start_pos with fixed RNG seed (still deterministic)
    """
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    if win <= 0 or horizon <= 0:
        raise ValueError("win and horizon must be positive")

    df = ensure_sorted_grid(df_grid, id_col=id_col, time_col=time_col, pos_col=pos_col)

    ids_sorted = np.array(sorted(set(map(int, ids))), dtype=np.int64)
    if ids_sorted.size == 0:
        return pd.DataFrame({id_col: pd.Series([], dtype="int64"), "start_pos": pd.Series([], dtype="int64")})

    # Count windows per ID
    stats = count_windows_per_id(df, win=win, horizon=horizon, ids=ids_sorted, id_col=id_col, time_col=time_col, pos_col=pos_col)
    stats = stats.set_index(id_col)

    n_ids = int(ids_sorted.size)
    q = int(max_windows // n_ids)
    r = int(max_windows % n_ids)

    rows: List[Tuple[int, int]] = []
    rng = np.random.default_rng(seed)

    for rank, _id in enumerate(ids_sorted.tolist()):
        n_w = int(stats.loc[_id, "n_windows"]) if _id in stats.index else 0
        if n_w <= 0:
            continue

        alloc = q + (1 if rank < r else 0)
        alloc = int(min(alloc, n_w))
        if alloc <= 0:
            continue

        if strategy == "spread":
            if alloc == 1:
                picks = np.array([0], dtype=np.int64)
            else:
                picks = np.linspace(0, n_w - 1, num=alloc, dtype=np.int64)
            # Ensure unique and sorted
            picks = np.unique(picks)
            # If uniqueness reduced count, top up deterministically
            if picks.size < alloc:
                missing = alloc - picks.size
                candidates = np.setdiff1d(np.arange(n_w, dtype=np.int64), picks, assume_unique=True)
                # deterministically take earliest remaining
                picks = np.r_[picks, candidates[:missing]]
                picks = np.sort(picks)

        elif strategy == "first":
            picks = np.arange(alloc, dtype=np.int64)

        elif strategy == "random":
            if alloc == n_w:
                picks = np.arange(n_w, dtype=np.int64)
            else:
                picks = rng.choice(n_w, size=alloc, replace=False)
                picks = np.sort(picks.astype(np.int64))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        rows.extend((int(_id), int(p)) for p in picks.tolist())

    window_index = pd.DataFrame(rows, columns=[id_col, "start_pos"])
    window_index[id_col] = window_index[id_col].astype(np.int64)
    window_index["start_pos"] = window_index["start_pos"].astype(np.int64)
    window_index = window_index.sort_values([id_col, "start_pos"]).reset_index(drop=True)

    # Enforce global cap deterministically (safety)
    if len(window_index) > max_windows:
        window_index = window_index.iloc[:max_windows].copy()

    return window_index


def make_window_tensors(
    df_grid: pd.DataFrame,
    window_index: pd.DataFrame,
    win: int,
    horizon: int,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    pos_col: str = "pos",
    dtype: Union[str, np.dtype] = np.float32,
    batch_size: int = 8192,
    return_meta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Build window tensors from a sorted 15-min grid and canonical window_index.

    X shape: (N, WIN, n_features)
    Y shape: (N, HORIZON, n_targets)

    Definition:
      For each (ID, start_pos):
        X = grid[start_pos : start_pos+WIN]
        Y = grid[start_pos+WIN : start_pos+WIN+HORIZON] on target_cols

    batch_size controls fill chunks to reduce peak Python overhead.
    """
    if win <= 0 or horizon <= 0:
        raise ValueError("win and horizon must be positive")

    if id_col not in window_index.columns or "start_pos" not in window_index.columns:
        raise ValueError("window_index must have columns ['ID', 'start_pos']")

    df = ensure_sorted_grid(df_grid, id_col=id_col, time_col=time_col, pos_col=pos_col)

    # Validate columns
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")
    for c in target_cols:
        if c not in df.columns:
            raise ValueError(f"Missing target column: {c}")

    # Compute ID slices (contiguous blocks)
    ids_unique, starts, ends = _compute_id_slices(df, id_col=id_col)
    id_to_slice: Dict[int, Tuple[int, int]] = {int(i): (int(s), int(e)) for i, s, e in zip(ids_unique, starts, ends)}

    # Prepare arrays
    N = int(len(window_index))
    n_feat = int(len(feature_cols))
    n_tgt = int(len(target_cols))

    X = np.empty((N, win, n_feat), dtype=dtype)
    Y = np.empty((N, horizon, n_tgt), dtype=dtype)

    # Extract values once for speed
    feat_vals = df.loc[:, list(feature_cols)].to_numpy(dtype=dtype, copy=False)
    tgt_vals = df.loc[:, list(target_cols)].to_numpy(dtype=dtype, copy=False)

    ids_w = window_index[id_col].to_numpy(dtype=np.int64, copy=False)
    spos_w = window_index["start_pos"].to_numpy(dtype=np.int64, copy=False)

    # Fill in batches
    for b0 in range(0, N, batch_size):
        b1 = min(N, b0 + batch_size)
        for i in range(b0, b1):
            _id = int(ids_w[i])
            start_pos = int(spos_w[i])

            if _id not in id_to_slice:
                raise ValueError(f"ID {_id} not found in df_grid")

            s, e = id_to_slice[_id]
            T = e - s
            max_start = T - win - horizon
            if start_pos < 0 or start_pos > max_start:
                raise ValueError(
                    f"Invalid start_pos for ID={_id}: {start_pos} (allowed 0..{max_start})"
                )

            g0 = s + start_pos
            g1 = g0 + win
            g2 = g1 + horizon

            X[i, :, :] = feat_vals[g0:g1, :]
            Y[i, :, :] = tgt_vals[g1:g2, :]

    meta = None
    if return_meta:
        meta = {
            "ids": ids_w.astype(np.int64, copy=False),
            "start_pos": spos_w.astype(np.int64, copy=False),
        }

    return X, Y, meta


def make_baseline_persistence_preds(
    df_grid: pd.DataFrame,
    window_index: pd.DataFrame,
    win: int,
    horizon: int,
    target_cols: Sequence[str],
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    pos_col: str = "pos",
    dtype: Union[str, np.dtype] = np.float32,
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Persistence baseline predictions aligned to the canonical window set.

    For each window:
      y_hat[t+h] = value at the last input step (start_pos + WIN - 1)
    repeated for h=1..HORIZON.

    Returns:
      Y_pred shape: (N, HORIZON, n_targets)
    """
    if id_col not in window_index.columns or "start_pos" not in window_index.columns:
        raise ValueError("window_index must have columns ['ID', 'start_pos']")

    df = ensure_sorted_grid(df_grid, id_col=id_col, time_col=time_col, pos_col=pos_col)

    # Validate columns
    for c in target_cols:
        if c not in df.columns:
            raise ValueError(f"Missing target column: {c}")

    ids_unique, starts, ends = _compute_id_slices(df, id_col=id_col)
    id_to_slice: Dict[int, Tuple[int, int]] = {int(i): (int(s), int(e)) for i, s, e in zip(ids_unique, starts, ends)}

    N = int(len(window_index))
    n_tgt = int(len(target_cols))
    Y_pred = np.empty((N, horizon, n_tgt), dtype=dtype)

    tgt_vals = df.loc[:, list(target_cols)].to_numpy(dtype=dtype, copy=False)
    ids_w = window_index[id_col].to_numpy(dtype=np.int64, copy=False)
    spos_w = window_index["start_pos"].to_numpy(dtype=np.int64, copy=False)

    for b0 in range(0, N, batch_size):
        b1 = min(N, b0 + batch_size)
        for i in range(b0, b1):
            _id = int(ids_w[i])
            start_pos = int(spos_w[i])

            if _id not in id_to_slice:
                raise ValueError(f"ID {_id} not found in df_grid")

            s, e = id_to_slice[_id]
            T = e - s
            max_start = T - win - horizon
            if start_pos < 0 or start_pos > max_start:
                raise ValueError(f"Invalid start_pos for ID={_id}: {start_pos} (allowed 0..{max_start})")

            last_input_global = s + start_pos + win - 1
            last_val = tgt_vals[last_input_global, :]  # (n_targets,)
            # Repeat across horizon
            Y_pred[i, :, :] = np.repeat(last_val.reshape(1, -1), repeats=horizon, axis=0)

    return Y_pred


def summarize_window_index(
    window_index: pd.DataFrame,
    id_col: str = "ID",
) -> Dict[str, float]:
    """
    Quick summary for reproducibility logging.
    """
    if len(window_index) == 0:
        return {
            "n_windows": 0,
            "n_ids": 0,
            "min_windows_per_id": 0,
            "median_windows_per_id": 0,
            "max_windows_per_id": 0,
        }

    counts = window_index.groupby(id_col, sort=False).size().to_numpy(dtype=np.int64)
    return {
        "n_windows": float(len(window_index)),
        "n_ids": float(window_index[id_col].nunique()),
        "min_windows_per_id": float(counts.min()),
        "median_windows_per_id": float(np.median(counts)),
        "max_windows_per_id": float(counts.max()),
    }

# Add the following at the bottom of src/electricity_forecasting/datasets/windowing.py

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from ..preprocessing.split import split_window_index_by_id_time
from ..evaluation.evaluate import save_window_index, hash_window_index


def prepare_and_persist_window_indices(
    *,
    df_train_grid: pd.DataFrame,
    df_test_grid: pd.DataFrame,
    ids_train: Optional[Sequence[int]] = None,
    ids_test: Optional[Sequence[int]] = None,
    win: int = 128,
    horizon: int = 10,
    valid_ratio: float = 0.1,
    max_test_windows: int = 200_000,
    test_sampling_strategy: str = "spread",
    out_dir: str | Path = "results/window_index",
    seed: int = 42,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    pos_col: str = "pos",
) -> Dict[str, Any]:
    """
    Create and persist canonical window_index files for train/val/test.

    - Train/Val: split by per-ID time cutoff using split_window_index_by_id_time()
    - Test: ID-balanced sampling across all IDs with a global cap (max_test_windows)

    Output files:
      train_windows_all_ids.csv
      val_windows_all_ids.csv
      test_windows_all_ids_cap200k.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine ID sets
    if ids_train is None:
        ids_train = sorted(df_train_grid[id_col].unique().tolist())
    else:
        ids_train = sorted(set(map(int, ids_train)))

    if ids_test is None:
        ids_test = sorted(df_test_grid[id_col].unique().tolist())
    else:
        ids_test = sorted(set(map(int, ids_test)))

    # Train/Val windows on train grid
    w_tr, w_va, split_meta = split_window_index_by_id_time(
        df_grid=df_train_grid,
        win=win,
        horizon=horizon,
        valid_ratio=valid_ratio,
        ids=ids_train,
        id_col=id_col,
        time_col=time_col,
        pos_col=pos_col,
    )

    # Test windows on test grid (ID-balanced + cap)
    w_test = sample_windows_id_balanced(
        df_grid=df_test_grid,
        ids=ids_test,
        max_windows=max_test_windows,
        win=win,
        horizon=horizon,
        seed=seed,
        strategy=test_sampling_strategy,
        id_col=id_col,
        time_col=time_col,
        pos_col=pos_col,
    )

    # Persist
    p_tr = out_dir / "train_windows_all_ids.csv"
    p_va = out_dir / "val_windows_all_ids.csv"
    p_te = out_dir / f"test_windows_all_ids_cap{max_test_windows}.csv"

    save_window_index(w_tr, p_tr)
    save_window_index(w_va, p_va)
    save_window_index(w_test, p_te)

    meta = {
        "paths": {"train": str(p_tr), "val": str(p_va), "test": str(p_te)},
        "hashes": {
            "train": hash_window_index(w_tr),
            "val": hash_window_index(w_va),
            "test": hash_window_index(w_test),
        },
        "counts": {
            "n_train_windows": int(len(w_tr)),
            "n_val_windows": int(len(w_va)),
            "n_test_windows_used": int(len(w_test)),
            "n_ids_train": int(w_tr[id_col].nunique()) if len(w_tr) else 0,
            "n_ids_val": int(w_va[id_col].nunique()) if len(w_va) else 0,
            "n_ids_test": int(w_test[id_col].nunique()) if len(w_test) else 0,
        },
        "split_meta": split_meta,
        "test_sampling": {
            "strategy": test_sampling_strategy,
            "seed": int(seed),
            "max_test_windows": int(max_test_windows),
            "id_order": "sorted",
            "within_id_order": "start_pos ascending",
        },
        "constants": {"win": int(win), "horizon": int(horizon), "valid_ratio": float(valid_ratio)},
    }

    return meta
