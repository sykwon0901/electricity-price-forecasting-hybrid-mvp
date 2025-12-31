"""Robust single-ID loading and 15-min grid building.

This implements the logic used in the notebook:
- Filter Parquet by raw string ID only (robust)
- Parse timestamps as UTC
- Snap timestamps to 15-min grid (floor)
- Deduplicate timestamps (keep last)
- Join onto full 15-min index and fill prices (ffill+bfill), volume missing -> 0
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


DEFAULT_COLS_OBS = ["ExecutionTime", "ID", "high", "low", "close", "volume"]


def load_one_id_observed(
    parquet_path: str,
    raw_id: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    columns: Sequence[str] = DEFAULT_COLS_OBS,
) -> pd.DataFrame:
    """Load sparse observed rows for one raw ID. Filters by ID in pyarrow, by time in pandas."""
    dataset = ds.dataset(parquet_path, format="parquet")
    filt = (ds.field("ID") == pa.scalar(str(raw_id)))
    table = dataset.to_table(columns=list(columns), filter=filt)
    df = table.to_pandas(self_destruct=True)

    df["ExecutionTime"] = pd.to_datetime(df["ExecutionTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["ExecutionTime"])
    df = df[(df["ExecutionTime"] >= start_utc) & (df["ExecutionTime"] <= end_utc)]
    df = df.sort_values("ExecutionTime").reset_index(drop=True)
    return df


def build_15min_grid_single_id(
    df_obs: pd.DataFrame,
    raw_id: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    freq: str = "15min",
    price_cols: Sequence[str] = ("high", "low", "close"),
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Build a dense 15-min grid for one raw ID (string)."""
    time_index = pd.date_range(start_utc.floor(freq), end_utc.floor(freq), freq=freq, tz="UTC")

    df = df_obs.copy()
    df["ExecutionTime"] = pd.to_datetime(df["ExecutionTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["ExecutionTime"])

    # Snap to grid
    df["ExecutionTime"] = df["ExecutionTime"].dt.floor(freq)

    keep_cols = ["ExecutionTime", *price_cols, volume_col]
    df = df[keep_cols].sort_values("ExecutionTime")

    # Deduplicate timestamps (keep last)
    df = df.groupby("ExecutionTime", as_index=False).last()

    df = df.set_index("ExecutionTime")
    df_grid = pd.DataFrame(index=time_index).join(df, how="left")

    # Fill volume
    df_grid[volume_col] = df_grid[volume_col].fillna(0.0).astype(np.float32)

    # Fill prices
    for c in price_cols:
        df_grid[c] = df_grid[c].ffill().bfill()
        df_grid[c] = df_grid[c].fillna(0.0).astype(np.float32)

    df_grid = df_grid.reset_index().rename(columns={"index": "ExecutionTime"})
    df_grid["ID"] = str(raw_id)
    df_grid["pos"] = np.arange(len(df_grid), dtype=np.int32)
    return df_grid
