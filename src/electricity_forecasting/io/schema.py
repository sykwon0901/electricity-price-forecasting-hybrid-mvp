"""Schema enforcement (canonical).

The project relies on stable dtypes and ordering:
- ExecutionTime: tz-aware UTC timestamps
- Sorting: [ID, ExecutionTime]
- Targets: float32 (high, low, close, volume)
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def enforce_schema(
    df: pd.DataFrame,
    *,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    target_cols: Sequence[str] = ("high", "low", "close", "volume"),
    sort: bool = True,
) -> pd.DataFrame:
    """Enforce dtypes, timezone, and ordering.

    This function is intentionally conservative:
    - It does not drop rows except those with invalid timestamps.
    - It casts target columns to float32 when present.
    """
    if id_col not in df.columns:
        raise ValueError(f"Missing required column: {id_col}")
    if time_col not in df.columns:
        raise ValueError(f"Missing required column: {time_col}")

    out = df.copy()

    # Time: tz-aware UTC
    out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out = out.dropna(subset=[time_col])

    # ID: keep int64 if numeric, else string
    if pd.api.types.is_numeric_dtype(out[id_col]):
        out[id_col] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
        # Convert nullable Int64 to plain int64 if possible
        if out[id_col].isna().any():
            # Fallback to string if IDs are mixed / missing
            out[id_col] = out[id_col].astype(str)
        else:
            out[id_col] = out[id_col].astype(np.int64)
    else:
        out[id_col] = out[id_col].astype(str)

    # Targets: float32
    for c in target_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)

    if sort:
        out = out.sort_values([id_col, time_col]).reset_index(drop=True)

    return out
