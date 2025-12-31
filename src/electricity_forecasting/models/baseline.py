"""
Baseline models.

Persistence baseline is aligned to the canonical window set.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..datasets.windowing import make_baseline_persistence_preds


def predict_persistence(
    df_grid: pd.DataFrame,
    window_index: pd.DataFrame,
    win: int,
    horizon: int,
    target_cols: Sequence[str],
) -> np.ndarray:
    """
    Return Y_pred of shape (N, HORIZON, n_targets) aligned to window_index.
    """
    return make_baseline_persistence_preds(
        df_grid=df_grid,
        window_index=window_index,
        win=win,
        horizon=horizon,
        target_cols=target_cols,
    )
