"""
Plot utilities (canonical).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(path: str | Path, fig: Optional[plt.Figure] = None, dpi: int = 160) -> None:
    p = ensure_parent_dir(path)
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(p, dpi=dpi)

def save_plot(path: str | Path, fig: Optional[plt.Figure] = None, dpi: int = 160) -> None:
    """Backward-compatible alias for save_fig()."""
    save_fig(path, fig=fig, dpi=dpi)
