"""
Canonical plotting functions.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .plot_utils import save_fig


def plot_threshold_sweep(curve_df: pd.DataFrame, title: str = "Threshold sweep") -> plt.Figure:
    """
    curve_df must include columns: ["threshold", "full_smape"] and optionally ["active_smape"].
    """
    fig = plt.figure()
    plt.plot(curve_df["threshold"], curve_df["full_smape"], marker="o", label="Full sMAPE")
    if "active_smape" in curve_df.columns:
        plt.plot(curve_df["threshold"], curve_df["active_smape"], marker="o", label="Active sMAPE")
    plt.xlabel("Threshold")
    plt.ylabel("sMAPE")
    plt.title(title)
    plt.legend()
    return fig


def plot_model_vs_baseline_bar(metrics: Dict[str, float], title: str = "Model vs baseline") -> plt.Figure:
    """
    metrics should include:
      baseline_full, baseline_active, model_full, model_active
    """
    keys = ["baseline_full", "model_full", "baseline_active", "model_active"]
    for k in keys:
        if k not in metrics:
            raise ValueError(f"Missing key in metrics: {k}")

    fig = plt.figure()
    labels = ["Baseline Full", "Model Full", "Baseline Active", "Model Active"]
    values = [metrics["baseline_full"], metrics["model_full"], metrics["baseline_active"], metrics["model_active"]]
    plt.bar(labels, values)
    plt.ylabel("sMAPE")
    plt.title(title)
    return fig


def plot_active_rate_heatmap(pivot: pd.DataFrame, title: str = "Active rate heatmap") -> plt.Figure:
    """
    pivot: index = dow, columns = hour, values = active rate
    """
    fig = plt.figure(figsize=(12, 4))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Active rate")
    plt.xticks(ticks=np.arange(pivot.shape[1]), labels=pivot.columns.tolist(), rotation=0)
    plt.yticks(ticks=np.arange(pivot.shape[0]), labels=pivot.index.tolist())
    plt.xlabel("Hour")
    plt.ylabel("Day of week")
    plt.title(title)
    return fig

def plot_and_save_baseline_vs_model_bar(
    metrics_bar: dict,
    out_png: str,
    title: str = "Model vs Baseline (sMAPE)",
    dpi: int = 160,
):
    """
    Save a simple bar chart comparing baseline vs model on Full/Active sMAPE.

    Required keys in metrics_bar:
      - baseline_full, model_full, baseline_active, model_active
    """
    required = ["baseline_full", "model_full", "baseline_active", "model_active"]
    for k in required:
        if k not in metrics_bar:
            raise ValueError(f"Missing key in metrics_bar: {k}")

    fig = plt.figure()
    labels = ["Baseline Full", "Model Full", "Baseline Active", "Model Active"]
    values = [
        float(metrics_bar["baseline_full"]),
        float(metrics_bar["model_full"]),
        float(metrics_bar["baseline_active"]),
        float(metrics_bar["model_active"]),
    ]
    plt.bar(labels, values)
    plt.ylabel("sMAPE")
    plt.title(title)
    save_fig(out_png, fig=fig, dpi=dpi)
    plt.close(fig)