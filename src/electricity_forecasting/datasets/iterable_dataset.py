"""
Torch datasets for window-based training/evaluation.

This dataset consumes:
- df_grid: 15-min grid sorted by ID and time (with per-ID contiguous rows)
- window_index: DataFrame ["ID", "start_pos"] (canonical)
and returns samples aligned with the canonical window definition.

X: (WIN, n_features)
Y: (HORIZON, n_targets)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset
except Exception as e:
    torch = None
    Dataset = object  # fallback
    IterableDataset = object  # fallback


def _ensure_sorted_grid(
    df_grid: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "ExecutionTime",
    pos_col: str = "pos",
) -> pd.DataFrame:
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


def _compute_id_slices(df_sorted: pd.DataFrame, id_col: str = "ID") -> Dict[int, Tuple[int, int]]:
    id_arr = df_sorted[id_col].to_numpy()
    if id_arr.size == 0:
        return {}

    change = np.flatnonzero(id_arr[1:] != id_arr[:-1]) + 1
    starts = np.r_[0, change].astype(np.int64)
    ends = np.r_[change, id_arr.size].astype(np.int64)
    ids = id_arr[starts].astype(np.int64)

    return {int(i): (int(s), int(e)) for i, s, e in zip(ids, starts, ends)}


class WindowDataset(Dataset):
    """
    WindowDataset aligned with the project's canonical window definition.

    Each item returns:
      X: torch.FloatTensor (WIN, n_features)
      Y: torch.FloatTensor (HORIZON, n_targets)
    Optionally returns metadata (ID, start_pos) when return_meta=True.
    """

    def __init__(
        self,
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
        return_meta: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use WindowDataset.")

        if win <= 0 or horizon <= 0:
            raise ValueError("win and horizon must be positive")

        if id_col not in window_index.columns or "start_pos" not in window_index.columns:
            raise ValueError("window_index must have columns ['ID', 'start_pos']")

        self.win = int(win)
        self.horizon = int(horizon)
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.id_col = id_col
        self.time_col = time_col
        self.pos_col = pos_col
        self.return_meta = bool(return_meta)

        df = _ensure_sorted_grid(df_grid, id_col=id_col, time_col=time_col, pos_col=pos_col)

        # Validate columns
        for c in self.feature_cols:
            if c not in df.columns:
                raise ValueError(f"Missing feature column: {c}")
        for c in self.target_cols:
            if c not in df.columns:
                raise ValueError(f"Missing target column: {c}")

        # Contiguous slices per ID
        self.id_to_slice = _compute_id_slices(df, id_col=id_col)
        if len(self.id_to_slice) == 0:
            raise ValueError("df_grid has no rows after sorting.")

        # Store arrays (views)
        self._feat = df.loc[:, self.feature_cols].to_numpy(dtype=dtype, copy=False)
        self._tgt = df.loc[:, self.target_cols].to_numpy(dtype=dtype, copy=False)

        # Window index arrays
        self._ids = window_index[id_col].to_numpy(dtype=np.int64, copy=False)
        self._spos = window_index["start_pos"].to_numpy(dtype=np.int64, copy=False)

        # Basic validity check (fast)
        for _id in np.unique(self._ids):
            if int(_id) not in self.id_to_slice:
                raise ValueError(f"ID {_id} in window_index not found in df_grid.")

    def __len__(self) -> int:
        return int(self._ids.shape[0])

    def __getitem__(self, idx: int):
        _id = int(self._ids[idx])
        start_pos = int(self._spos[idx])

        s, e = self.id_to_slice[_id]
        T = e - s
        max_start = T - self.win - self.horizon
        if start_pos < 0 or start_pos > max_start:
            raise ValueError(f"Invalid start_pos for ID={_id}: {start_pos} (allowed 0..{max_start})")

        g0 = s + start_pos
        g1 = g0 + self.win
        g2 = g1 + self.horizon

        x = self._feat[g0:g1, :]
        y = self._tgt[g1:g2, :]

        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))

        if self.return_meta:
            meta = {"ID": _id, "start_pos": start_pos}
            return x_t, y_t, meta

        return x_t, y_t


class ParquetWindowIterableDataset(IterableDataset):
    """IterableDataset that builds per-ID grids on the fly from Parquet.

    This dataset is designed for DL models that need the full input window
    X: (WIN, n_features) (sequence), without materializing all windows in RAM.

    It uses the exact same single-ID grid builder and feature engineering used in
    the ML notebook by reusing:
      datasets.materialize.build_one_id_grid_with_features()

    Expected window_index format:
      DataFrame columns: ["ID", "start_pos"]
      where ID is an integer code and start_pos is the 0-based position in the 15-min grid.
    """

    def __init__(
        self,
        *,
        parquet_path: str,
        window_index: pd.DataFrame,
        code_to_id: Dict[int, str],
        start_utc: "pd.Timestamp",
        end_utc: "pd.Timestamp",
        freq: str,
        berlin_tz: str,
        win: int,
        horizon: int,
        feature_cols: Sequence[str],
        target_cols: Sequence[str],
        id_col: str = "ID",
        cache_dir: Optional[str] = None,
        cache_tag: str = "",
        shuffle_ids: bool = False,
        shuffle_within_id: bool = False,
        seed: int = 42,
        return_id_code: bool = False,
        pack_x_with_id: bool = True,
        return_baseline: bool = True,
        return_meta: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use ParquetWindowIterableDataset.")

        if id_col not in window_index.columns or "start_pos" not in window_index.columns:
            raise ValueError("window_index must have columns ['ID', 'start_pos']")

        self.parquet_path = str(parquet_path)
        self.code_to_id = {int(k): str(v) for k, v in code_to_id.items()}
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.freq = str(freq)
        self.berlin_tz = str(berlin_tz)
        self.win = int(win)
        self.horizon = int(horizon)
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.id_col = str(id_col)

        self.shuffle_ids = bool(shuffle_ids)
        self.shuffle_within_id = bool(shuffle_within_id)
        self.seed = int(seed)
        self._epoch = 0

        self.return_id_code = bool(return_id_code)
        self.pack_x_with_id = bool(pack_x_with_id)

        # If return_id_code=True and pack_x_with_id=True (recommended),
        # the dataset yields: ((x_seq, id_code), y_true, y_base, ...)
        # so that y_true stays at batch[1] for generic training utilities.
        self.return_baseline = bool(return_baseline)
        self.return_meta = bool(return_meta)

        wi = window_index[[id_col, "start_pos"]].copy()
        wi[id_col] = wi[id_col].astype(np.int64)
        wi["start_pos"] = wi["start_pos"].astype(np.int64)
        wi = wi.sort_values([id_col, "start_pos"]).reset_index(drop=True)
        self._wi = wi

        # Pre-group start positions by ID for fast iteration.
        self._id_to_spos: Dict[int, np.ndarray] = {
            int(i): g["start_pos"].to_numpy(dtype=np.int64, copy=False)
            for i, g in wi.groupby(id_col, sort=True)
        }
        self._ids_sorted = np.array(sorted(self._id_to_spos.keys()), dtype=np.int64)
        if self._ids_sorted.size == 0:
            raise ValueError("window_index has no rows.")

        # Optional on-disk cache for per-ID grids
        self.cache_dir = None
        self.cache_tag = str(cache_tag).strip() if cache_tag is not None else ""
        if cache_dir is not None and str(cache_dir).strip() != "":
            from pathlib import Path

            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Sanity: every ID in window_index must exist in code_to_id
        missing = [int(i) for i in self._ids_sorted.tolist() if int(i) not in self.code_to_id]
        if missing:
            raise ValueError(f"Missing code_to_id entries for {len(missing)} IDs, e.g. {missing[:5]}")

    def set_epoch(self, epoch: int) -> None:
        """Set epoch index (used for deterministic reshuffling)."""
        self._epoch = int(epoch)

    def _cache_path(self, id_code: int):
        if self.cache_dir is None:
            return None
        from pathlib import Path

        sub = self.cache_dir
        if self.cache_tag:
            sub = sub / self.cache_tag
            sub.mkdir(parents=True, exist_ok=True)
        return Path(sub) / f"grid_id_{int(id_code)}.npz"

    def _load_or_build_arrays(self, id_code: int):
        """Return (feat, tgt) arrays for one ID."""
        cache_path = self._cache_path(id_code)
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            feat = data["feat"].astype(np.float32, copy=False)
            tgt = data["tgt"].astype(np.float32, copy=False)
            return feat, tgt

        # Build with canonical pipeline
        from .materialize import build_one_id_grid_with_features

        raw_id = self.code_to_id[int(id_code)]
        df_grid = build_one_id_grid_with_features(
            parquet_path=self.parquet_path,
            raw_id=raw_id,
            id_code=int(id_code),
            start_utc=self.start_utc,
            end_utc=self.end_utc,
            freq=self.freq,
            berlin_tz=self.berlin_tz,
        )

        # Extract arrays
        feat = df_grid.loc[:, self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        tgt = df_grid.loc[:, self.target_cols].to_numpy(dtype=np.float32, copy=False)

        if cache_path is not None:
            np.savez(cache_path, feat=feat, tgt=tgt)
        return feat, tgt

    def __iter__(self):
        rng = np.random.default_rng(int(self.seed) + int(self._epoch))

        ids = self._ids_sorted.copy()
        if self.shuffle_ids:
            rng.shuffle(ids)

        # DataLoader workers: shard IDs across workers to avoid duplication.
        try:
            from torch.utils.data import get_worker_info
            info = get_worker_info()
        except Exception:
            info = None

        if info is not None:
            ids = ids[info.id :: info.num_workers]

        for id_code in ids.tolist():
            id_code = int(id_code)
            spos = self._id_to_spos[id_code]
            if self.shuffle_within_id and spos.size > 1:
                spos = spos.copy()
                rng.shuffle(spos)

            feat, tgt = self._load_or_build_arrays(id_code)
            T = int(tgt.shape[0])
            max_start = T - self.win - self.horizon
            if max_start < 0:
                continue

            for start_pos in spos.tolist():
                start_pos = int(start_pos)
                if start_pos < 0 or start_pos > max_start:
                    raise ValueError(
                        f"Invalid start_pos for ID={id_code}: {start_pos} (allowed 0..{max_start})"
                    )

                g0 = start_pos
                g1 = g0 + self.win
                g2 = g1 + self.horizon

                x = feat[g0:g1, :]
                y = tgt[g1:g2, :]

                x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
                y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))

                if self.return_id_code:
                    id_t = torch.tensor(id_code, dtype=torch.long)
                    if self.pack_x_with_id:
                        out = [(x_t, id_t), y_t]
                    else:
                        out = [x_t, id_t, y_t]
                else:
                    out = [x_t, y_t]

                if self.return_baseline:
                    last = tgt[g1 - 1, :]
                    yb = np.repeat(last.reshape(1, -1), repeats=self.horizon, axis=0)
                    yb_t = torch.from_numpy(np.asarray(yb, dtype=np.float32))
                    out.append(yb_t)

                if self.return_meta:
                    out.append({"ID": id_code, "start_pos": start_pos})

                yield tuple(out)
