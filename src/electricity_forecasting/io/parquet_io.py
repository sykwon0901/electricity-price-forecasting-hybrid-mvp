"""Parquet I/O helpers (canonical).

The legacy notebooks read Parquet files produced by the feature engineering step.
This module provides a thin, reusable interface to avoid ad-hoc pandas reads.

Notes:
- IDs in the raw data are typically strings.
- ExecutionTime is enforced as tz-aware UTC.
- Time filtering is applied in pandas for robustness across Parquet schemas.
"""

from __future__ import annotations

from typing import Iterable, Sequence
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


def load_parquet_filtered(
    parquet_path: str | Path,
    ids: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    time_min: str | pd.Timestamp | None = None,
    time_max: str | pd.Timestamp | None = None,
    *,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
) -> pd.DataFrame:
    """Load a Parquet dataset with optional ID/time filtering.

    Args:
      parquet_path: path to a Parquet file or directory.
      ids: optional list of raw IDs to keep.
      columns: optional list of columns to load.
      time_min/time_max: optional time bounds (inclusive).
      id_col/time_col: column names.

    Returns:
      pandas.DataFrame (not schema-enforced; call enforce_schema() afterwards).
    """
    parquet_path = Path(parquet_path)
    dataset = ds.dataset(str(parquet_path), format="parquet")

    filt = None
    if ids is not None:
        ids_list = [str(x) for x in ids]
        # pyarrow.dataset supports isin() on string columns.
        expr = ds.field(id_col).isin(ids_list)
        filt = expr if filt is None else (filt & expr)

    table = dataset.to_table(columns=list(columns) if columns is not None else None, filter=filt)
    df = table.to_pandas(self_destruct=True)

    # Time filtering (robust across parquet schemas / timezone annotations)
    if time_col in df.columns and (time_min is not None or time_max is not None):
        t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.assign(**{time_col: t}).dropna(subset=[time_col])

        if time_min is not None:
            tmin = pd.Timestamp(time_min)
            tmin = tmin.tz_convert("UTC") if tmin.tzinfo is not None else tmin.tz_localize("UTC")
            df = df[df[time_col] >= tmin]

        if time_max is not None:
            tmax = pd.Timestamp(time_max)
            tmax = tmax.tz_convert("UTC") if tmax.tzinfo is not None else tmax.tz_localize("UTC")
            df = df[df[time_col] <= tmax]

    return df
