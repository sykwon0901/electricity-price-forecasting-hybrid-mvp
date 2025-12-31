"""Generate cross-model comparison artifacts from saved metrics JSON files.

Outputs:
- results/metrics/metrics_table.csv
- results/metrics/metrics_table.xlsx
- results/plots/eval/tables/metrics_table.png
- results/plots/eval/curves/model_comparison_full_active.png

This script does not run any model training. It only reads existing metrics JSON
records produced by the model notebooks/scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from electricity_forecasting.evaluation.reporting import update_metrics_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", type=str, default=".")
    return p.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()

    results_dir = project_dir / "results"
    plots_dir = results_dir / "plots"

    metrics_dir = results_dir / "metrics"
    (plots_dir / "eval/tables").mkdir(parents=True, exist_ok=True)
    (plots_dir / "eval/curves").mkdir(parents=True, exist_ok=True)

    tbl_csv = metrics_dir / "metrics_table.csv"
    tbl_xlsx = metrics_dir / "metrics_table.xlsx"
    tbl_png = plots_dir / "eval/tables" / "metrics_table.png"

    df = update_metrics_table(
        metrics_dir=metrics_dir,
        out_csv=tbl_csv,
        out_xlsx=tbl_xlsx,
        out_png=tbl_png,
        title="Model performance metrics (canonical)",
    )

    # Plot: model_full/model_active per model
    if df.shape[0] > 0 and "model_name" in df.columns:
        fig = plt.figure()
        x = range(df.shape[0])
        plt.bar([i - 0.2 for i in x], df["model_full"].astype(float), width=0.4, label="Model Full sMAPE")
        plt.bar([i + 0.2 for i in x], df["model_active"].astype(float), width=0.4, label="Model Active sMAPE")
        plt.xticks(list(x), df["model_name"].astype(str).tolist(), rotation=30, ha="right")
        plt.ylabel("sMAPE")
        plt.title("Model comparison (canonical metrics)")
        plt.legend()
        plt.tight_layout()
        out_curve = plots_dir / "eval/curves" / "model_comparison_full_active.png"
        fig.savefig(out_curve, dpi=200)
        plt.close(fig)
        print("Saved:", out_curve)

    print("Saved:", tbl_csv)
    print("Saved:", tbl_xlsx)
    print("Saved:", tbl_png)


if __name__ == "__main__":
    main()
