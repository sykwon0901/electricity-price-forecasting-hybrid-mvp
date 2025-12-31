from __future__ import annotations

from typing import Iterable, Tuple
import pandas as pd

try:
    import pyarrow.dataset as ds
except Exception:
    ds = None


def load_one_id_observed(
    parquet_path: str,
    raw_id: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
) -> pd.DataFrame:
    """
    Load sparse observed rows for a single ID from a Parquet dataset within a UTC time range.

    Notes:
    - Uses Arrow dataset filtering (predicate pushdown) when available.
    - Returns a pandas DataFrame sorted by time_col (UTC).
    """
    start_ts = pd.Timestamp(start_utc)
    end_ts = pd.Timestamp(end_utc)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    if ds is None:
        # Fallback: this may be slower because it can read more data depending on the engine.
        df = pd.read_parquet(parquet_path)
        df = df[(df[id_col].astype(str) == str(raw_id))]
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df[(df[time_col] >= start_ts) & (df[time_col] <= end_ts)]
        return df.sort_values(time_col).reset_index(drop=True)

    dataset = ds.dataset(parquet_path, format="parquet")

    start64 = start_ts.to_datetime64()
    end64 = end_ts.to_datetime64()

    filt = (ds.field(id_col) == str(raw_id)) & (ds.field(time_col) >= start64) & (ds.field(time_col) <= end64)
    table = dataset.to_table(filter=filt)
    df = table.to_pandas()

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)

    return df.sort_values(time_col).reset_index(drop=True)


def build_15min_grid_single_id(
    df_obs: pd.DataFrame,
    raw_id: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    freq: str = "15min",
    price_cols: Tuple[str, str, str] = ("high", "low", "close"),
    volume_col: str = "volume",
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Build a dense time grid for one ID (15-min by default) and left-join sparse observed rows.

    Output:
    - A DataFrame with a complete timestamp grid and merged price/volume columns.
    - Missing values in price/volume columns are filled (default 0.0) for EDA convenience.
    """
    start_ts = pd.Timestamp(start_utc)
    end_ts = pd.Timestamp(end_utc)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    grid = pd.DataFrame({time_col: pd.date_range(start=start_ts, end=end_ts, freq=freq, tz="UTC")})
    grid[id_col] = str(raw_id)

    df = df_obs.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)

    keep_cols = [time_col]
    for c in list(price_cols) + [volume_col]:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].drop_duplicates(subset=[time_col], keep="last")
    out = grid.merge(df, on=time_col, how="left")

    for c in price_cols:
        if c in out.columns:
            out[c] = out[c].astype("float32").fillna(fill_value)

    if volume_col in out.columns:
        out[volume_col] = out[volume_col].astype("float32").fillna(fill_value)

    return out
