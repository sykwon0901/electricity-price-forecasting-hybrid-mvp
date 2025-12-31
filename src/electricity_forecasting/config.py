"""
Project-wide configuration (canonical).

Keep constants and column ordering in ONE place to avoid subtle mismatches
across notebooks and scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


# Core constants (legacy-compatible)
WIN: int = 128
HORIZON: int = 10
MAX_TEST_WINDOWS: int = 200_000


# Optional global caps used in the legacy notebook (keep identical)
MAX_TRAIN_WINDOWS: int = 300_000
MAX_VAL_WINDOWS: int = 100_000

# Split ratio (last 10% per ID for validation)
VALID_RATIO: float = 0.1
# Time / grid
FREQ: str = "15min"
UTC: str = "UTC"
BERLIN_TZ: str = "Europe/Berlin"

# Canonical columns
TIME_COL: str = "ExecutionTime"
ID_COL: str = "ID"

# Canonical target ordering (do not change once fixed)
TARGET_COLS: List[str] = ["high", "low", "close", "volume"]
VOLUME_INDEX: int = TARGET_COLS.index("volume")

# Default feature columns (minimal baseline features)
# You can extend this in notebooks after FE, but keep TARGET_COLS order fixed.
BASE_FEATURE_COLS: List[str] = ["high", "low", "close", "volume"]


@dataclass(frozen=True)
class Paths:
    """
    Optional convenience holder (not required by core logic).
    """
    project_dir: Path
    data_dir: Path
    results_dir: Path
    images_dir: Path

    @staticmethod
    def from_project_dir(project_dir: str | Path) -> "Paths":
        p = Path(project_dir)
        return Paths(
            project_dir=p,
            data_dir=p / "data",
            results_dir=p / "results",
            images_dir=p / "images",
        )
