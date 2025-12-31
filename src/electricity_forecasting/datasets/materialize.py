"""Materialize tabular datasets from window_index (ID-by-ID, RAM-safe).

This matches the notebook:
- Build per-ID grid from observed data
- Use features at t_end (window end)
- Targets are the next H steps for canonical target_cols (high,low,close,volume)
- Baseline persistence is last observed at t_end repeated for horizon
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd

from ..preprocessing.grid_single_id import load_one_id_observed, build_15min_grid_single_id
from ..preprocessing.features import add_time_features, add_lag_features, add_rolling_features


def infer_feature_cols(df_grid: pd.DataFrame) -> List[str]:
    cols = list(df_grid.columns)
    feat: List[str] = []

    if "logvol" in cols:
        feat.append("logvol")
    feat += sorted([c for c in cols if c.startswith("logvol_lag")])
    feat += sorted([c for c in cols if c.startswith("logvol_roll")])

    for c in ["close", "high", "low"]:
        if c in cols and c not in feat:
            feat.append(c)

    feat += sorted([c for c in cols if "berlin" in c.lower()])

    seen = set()
    out: List[str] = []
    for c in feat:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def build_one_id_grid_with_features(
    parquet_path: str,
    raw_id: str,
    id_code: int,
    start_utc: "pd.Timestamp",
    end_utc: "pd.Timestamp",
    freq: str,
    berlin_tz: str,
) -> pd.DataFrame:
    df_obs = load_one_id_observed(parquet_path, raw_id, start_utc, end_utc)
    df_grid = build_15min_grid_single_id(df_obs, raw_id, start_utc, end_utc, freq=freq)

    # Canonical ID code
    df_grid["ID"] = int(id_code)

    # FE on grid
    df_grid["logvol"] = np.log1p(df_grid["volume"].astype(np.float32))
    df_grid = add_time_features(df_grid, tz=berlin_tz, time_col="ExecutionTime", prefix="berlin", add_cyclical=True)
    df_grid = add_lag_features(df_grid, cols=["logvol"], lags=(1, 2), time_col="ExecutionTime")
    df_grid = add_rolling_features(df_grid, cols=["logvol"], windows=(4, 16), shift=1, time_col="ExecutionTime")

    # Fill NaNs in feature columns
    feat_cols = infer_feature_cols(df_grid)
    for c in feat_cols:
        df_grid[c] = df_grid[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    return df_grid


def extract_windows_for_id(
    df_grid: pd.DataFrame,
    w_idx_id: pd.DataFrame,
    win: int,
    horizon: int,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    volume_index: int,
):
    if w_idx_id is None or len(w_idx_id) == 0:
        return None

    start_pos = w_idx_id["start_pos"].to_numpy(dtype=np.int64)
    t_end = start_pos + (win - 1)

    feat = df_grid[list(feature_cols)].to_numpy(dtype=np.float32)
    tgt = df_grid[list(target_cols)].to_numpy(dtype=np.float32)

    X = feat[t_end, :]  # (N,F)
    steps = t_end[:, None] + np.arange(1, horizon + 1, dtype=np.int64)[None, :]
    Y_true = tgt[steps, :]  # (N,H,C)

    last = tgt[t_end, :]  # (N,C)
    Y_base = np.repeat(last[:, None, :], repeats=horizon, axis=1)

    y_true_vol = Y_true[:, :, volume_index]
    return X, Y_true, Y_base, y_true_vol


def materialize_from_window_index(
    parquet_path: str,
    window_index: pd.DataFrame,
    code_to_id: Dict[int, str],
    start_utc: "pd.Timestamp",
    end_utc: "pd.Timestamp",
    freq: str,
    berlin_tz: str,
    win: int,
    horizon: int,
    target_cols: Sequence[str],
    volume_index: int,
    feature_cols: Sequence[str] | None = None,
):
    X_list, Y_list, Yb_list, yv_list, ids_list = [], [], [], [], []

    # Build feature cols once if needed
    if feature_cols is None:
        sample_code = int(window_index["ID"].iloc[0])
        raw_id = code_to_id[sample_code]
        df_grid_sample = build_one_id_grid_with_features(parquet_path, raw_id, sample_code, start_utc, end_utc, freq, berlin_tz)
        feature_cols = infer_feature_cols(df_grid_sample)

    for id_code, w_id in window_index.groupby("ID", sort=True):
        id_code = int(id_code)
        raw_id = code_to_id[id_code]

        df_grid = build_one_id_grid_with_features(parquet_path, raw_id, id_code, start_utc, end_utc, freq, berlin_tz)
        out = extract_windows_for_id(df_grid, w_id, win, horizon, feature_cols, target_cols, volume_index)
        if out is None:
            continue
        X, Y_true, Y_base, y_true_vol = out

        X_list.append(X)
        Y_list.append(Y_true)
        Yb_list.append(Y_base)
        yv_list.append(y_true_vol)
        ids_list.append(np.full((X.shape[0],), id_code, dtype=np.int64))

    if len(X_list) == 0:
        raise RuntimeError("No windows materialized. Check window_index and grid bounds.")

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    Y_all = np.concatenate(Y_list, axis=0).astype(np.float32)
    Yb_all = np.concatenate(Yb_list, axis=0).astype(np.float32)
    yv_all = np.concatenate(yv_list, axis=0).astype(np.float32)
    ids_all = np.concatenate(ids_list, axis=0).astype(np.int64)

    return X_all, Y_all, Yb_all, yv_all, ids_all, list(feature_cols)
