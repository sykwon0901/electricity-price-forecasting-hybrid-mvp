"""Window index builder (deterministic, ID-balanced).

This matches the notebook logic:
- Use fixed time bounds to compute grid length and max start_pos
- Train windows sampled in [0, max_start_train]
- Val windows sampled in [min_start_val, max_start_all]
- Test windows sampled in [0, n_windows_test-1] with global cap
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowIndexPaths:
    train_csv: str
    val_csv: str
    test_csv: str


def steps_in_range(start_utc: "pd.Timestamp", end_utc: "pd.Timestamp", freq: str = "15min") -> int:
    idx = pd.date_range(start_utc, end_utc, freq=freq, tz="UTC")
    return int(len(idx))


def sample_windows_id_balanced_in_range(
    ids: List[int],
    start_min: int,
    start_max: int,
    max_windows: int,
    strategy: str = "spread",
    seed: int = 42,
) -> pd.DataFrame:
    if start_max < start_min:
        raise ValueError("Invalid sampling range: start_max < start_min")

    ids_sorted = sorted(set(map(int, ids)))
    n_ids = len(ids_sorted)

    q = max_windows // n_ids
    r = max_windows % n_ids

    n_range = start_max - start_min + 1
    rng = np.random.default_rng(seed)

    rows = []
    for rank, _id in enumerate(ids_sorted):
        alloc = q + (1 if rank < r else 0)
        alloc = int(min(alloc, n_range))
        if alloc <= 0:
            continue

        if strategy == "spread":
            if alloc == 1:
                picks = np.array([start_min], dtype=np.int64)
            else:
                picks = np.linspace(start_min, start_max, num=alloc, dtype=np.int64)
            picks = np.unique(picks)
            if picks.size < alloc:
                missing = alloc - picks.size
                candidates = np.setdiff1d(
                    np.arange(start_min, start_max + 1, dtype=np.int64),
                    picks,
                    assume_unique=True,
                )
                picks = np.r_[picks, candidates[:missing]]
                picks = np.sort(picks)

        elif strategy == "random":
            pool = np.arange(start_min, start_max + 1, dtype=np.int64)
            if alloc == n_range:
                picks = pool
            else:
                picks = rng.choice(pool, size=alloc, replace=False).astype(np.int64)
                picks = np.sort(picks)
        else:
            raise ValueError("Unknown strategy. Use 'spread' or 'random'.")

        rows.extend((int(_id), int(p)) for p in picks.tolist())

    wi = pd.DataFrame(rows, columns=["ID", "start_pos"]).sort_values(["ID", "start_pos"]).reset_index(drop=True)
    if len(wi) > max_windows:
        wi = wi.iloc[:max_windows].copy()
    return wi


def build_window_indices(
    all_id_codes: List[int],
    train_start_utc: "pd.Timestamp",
    train_end_utc: "pd.Timestamp",
    test_start_utc: "pd.Timestamp",
    test_end_utc: "pd.Timestamp",
    win: int,
    horizon: int,
    freq: str,
    max_train_windows: int,
    max_val_windows: int,
    max_test_windows: int,
    valid_ratio: float,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    T_train = steps_in_range(train_start_utc, train_end_utc, freq=freq)
    T_test = steps_in_range(test_start_utc, test_end_utc, freq=freq)

    n_windows_train = T_train - win - horizon + 1
    n_windows_test = T_test - win - horizon + 1
    if n_windows_train <= 0 or n_windows_test <= 0:
        raise ValueError("Invalid (win, horizon) for given time bounds.")

    cut_pos = int(np.floor(T_train * (1.0 - valid_ratio)))
    max_start_all = T_train - win - horizon
    max_start_train = cut_pos - win - horizon
    min_start_val = cut_pos - win

    w_tr = sample_windows_id_balanced_in_range(all_id_codes, 0, max_start_train, max_train_windows, seed=seed)
    w_va = sample_windows_id_balanced_in_range(all_id_codes, min_start_val, max_start_all, max_val_windows, seed=seed)
    w_te = sample_windows_id_balanced_in_range(all_id_codes, 0, n_windows_test - 1, max_test_windows, seed=seed)

    meta = {
        "T_train": int(T_train),
        "T_test": int(T_test),
        "n_windows_per_id_train": int(n_windows_train),
        "n_windows_per_id_test": int(n_windows_test),
        "cut_pos": int(cut_pos),
        "max_start_train": int(max_start_train),
        "min_start_val": int(min_start_val),
        "max_start_all": int(max_start_all),
    }
    return w_tr, w_va, w_te, meta
