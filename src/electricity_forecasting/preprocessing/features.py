"""
Feature engineering (canonical).

Includes:
- build_15min_grid: convert observed sparse rows into a dense 15-min grid per ID.
- time features (Berlin time)
- leakage-safe lag and rolling features
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def build_15min_grid(
    df_obs: pd.DataFrame,
    ids: Optional[Sequence[int]] = None,
    start_utc: Optional[pd.Timestamp] = None,
    end_utc: Optional[pd.Timestamp] = None,
    freq: str = "15min",
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    price_cols: Sequence[str] = ("high", "low", "close"),
    volume_col: str = "volume",
    add_pos: bool = True,
) -> pd.DataFrame:
    """
    Build a dense 15-min grid per ID.

    Rules:
    - ExecutionTime is enforced as tz-aware UTC.
    - volume missing -> 0
    - prices missing -> forward-fill within each ID
    - output is sorted by [ID, ExecutionTime]
    - optionally add per-ID pos column (0..T-1)

    start_utc/end_utc:
    - If None, inferred from df_obs global min/max.
    - For 2024 full-year, callers should pass explicit bounds to avoid drift.
    """
    if id_col not in df_obs.columns or time_col not in df_obs.columns:
        raise ValueError("df_obs must include ID and ExecutionTime columns.")

    df = df_obs.copy()

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col])

    if ids is not None:
        # Robust ID filtering (raw IDs are often strings)
        ids_set = set(str(x) for x in ids)
        df = df[df[id_col].astype(str).isin(ids_set)].copy()

    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # Snap timestamps to the grid and deduplicate per (ID, time)
    df[time_col] = df[time_col].dt.floor(freq)
    df = df.sort_values([id_col, time_col]).groupby([id_col, time_col], as_index=False).last()

    if start_utc is None:
        start_utc = df[time_col].min()
    if end_utc is None:
        end_utc = df[time_col].max()

    if start_utc.tz is None:
        start_utc = start_utc.tz_localize("UTC")
    else:
        start_utc = start_utc.tz_convert("UTC")

    if end_utc.tz is None:
        end_utc = end_utc.tz_localize("UTC")
    else:
        end_utc = end_utc.tz_convert("UTC")

    # Normalize bounds to grid
    start_utc = start_utc.floor(freq)
    end_utc = end_utc.floor(freq)

    ids_unique = np.sort(df[id_col].astype(str).unique())
    if ids_unique.size == 0:
        raise ValueError("No IDs found after filtering.")

    full_time_index = pd.date_range(start_utc, end_utc, freq=freq, tz="UTC")
    full_index = pd.MultiIndex.from_product([ids_unique, full_time_index], names=[id_col, time_col])

    # Reindex onto full grid
    use_cols = [id_col, time_col, *price_cols, volume_col]
    use_cols = [c for c in use_cols if c in df.columns]
    df_idx = df[use_cols].set_index([id_col, time_col]).sort_index()
    df_full = df_idx.reindex(full_index)

    # Fill volume with 0
    if volume_col in df_full.columns:
        df_full[volume_col] = df_full[volume_col].fillna(0.0).astype(np.float32)

    # Fill prices (ffill + bfill) to avoid NaNs at the beginning of each ID
    for c in price_cols:
        if c in df_full.columns:
            df_full[c] = df_full[c].groupby(level=0).ffill().bfill()
            df_full[c] = df_full[c].fillna(0.0).astype(np.float32)

    df_full = df_full.reset_index()
    df_full = df_full.sort_values([id_col, time_col]).reset_index(drop=True)

    if add_pos:
        df_full["pos"] = df_full.groupby(id_col, sort=False).cumcount().astype(np.int32)

    return df_full


def add_time_features(
    df_grid: pd.DataFrame,
    tz: str = "Europe/Berlin",
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    prefix: str = "berlin",
    add_cyclical: bool = True,
) -> pd.DataFrame:
    """
    Add Berlin time features from UTC timestamps.

    Adds:
    - hour, dow, month
    - optional cyclical encodings for hour
    """
    df = df_grid.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col])

    t_local = df[time_col].dt.tz_convert(tz)
    df[f"hour_{prefix}"] = t_local.dt.hour.astype(np.int16)
    df[f"dow_{prefix}"] = t_local.dt.dayofweek.astype(np.int16)
    df[f"month_{prefix}"] = t_local.dt.month.astype(np.int16)

    if add_cyclical:
        hour = df[f"hour_{prefix}"].to_numpy(dtype=np.float32)
        df[f"hour_{prefix}_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        df[f"hour_{prefix}_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

    return df


def add_lag_features(
    df_grid: pd.DataFrame,
    cols: Sequence[str],
    lags: Sequence[int] = (1, 2),
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
) -> pd.DataFrame:
    """
    Leakage-safe lag features (uses shift within ID).
    """
    df = df_grid.sort_values([id_col, time_col]).copy()
    g = df.groupby(id_col, sort=False)

    for c in cols:
        for lag in lags:
            out = f"{c}_lag{lag}"
            df[out] = g[c].shift(lag).astype(np.float32)

    return df


def add_rolling_features(
    df_grid: pd.DataFrame,
    cols: Sequence[str],
    windows: Sequence[int] = (4, 16),
    shift: int = 1,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Leakage-safe rolling features:
    rolling is computed on shifted series (default shift=1).
    """
    df = df_grid.sort_values([id_col, time_col]).copy()
    g = df.groupby(id_col, sort=False)

    def _roll(s: pd.Series, w: int) -> pd.Series:
        return s.shift(shift).rolling(window=w, min_periods=min_periods).mean()

    for c in cols:
        for w in windows:
            out = f"{c}_rollmean_{w}"
            df[out] = g[c].apply(lambda s: _roll(s, w)).reset_index(level=0, drop=True).astype(np.float32)

    return df
