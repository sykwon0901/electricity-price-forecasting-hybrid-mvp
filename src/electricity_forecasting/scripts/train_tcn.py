"""Deprecated shim.

Historically, training utilities lived under electricity_forecasting.scripts.
They have been moved to electricity_forecasting.training for cleaner structure.

This module re-exports the public API to keep older imports working.
"""

from __future__ import annotations

from ..training.train_tcn import *  # noqa: F401,F403
