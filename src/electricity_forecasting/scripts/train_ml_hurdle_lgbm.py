"""Train ML Hurdle LightGBM (volume-only, horizon-wise) and save outputs.

This script matches the notebook logic:
- IDs may be strings; we build a stable ID<->code mapping from TEST parquet
- Deterministic window_index (ID-balanced) using fixed time bounds
- Materialize tabular features at window end t_end and targets for next H steps
- Train horizon-wise hurdle: classifier P(active) + regressor log1p(volume) on active
- Select threshold on validation (volume FULL sMAPE)
- Evaluate on test once (baseline persistence prices + model volume)
- Save:
  - results/metrics/ml_hurdle_lgbm.json
  - results/metrics/metrics_table_ml_hurdle_lgbm_vs_baseline.(csv/json)
- results/plots/models/ml_hurdle_lgbm/baseline_vs_model_full_active.png
- results/plots/eval/curves/ml_hurdle_lgbm_threshold_sweep_volume_full.png
- results/plots/eval/tables/metrics_table_ml_hurdle_lgbm_vs_baseline.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd

# Bootstrap src import when running as a file (python path/to/script.py)
# File location: <repo_root>/src/electricity_forecasting/scripts/*.py
SRC_DIR = Path(__file__).resolve().parents[2]  # <repo_root>/src
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_forecasting.config import WIN, HORIZON, TARGET_COLS, VOLUME_INDEX, FREQ, BERLIN_TZ
from electricity_forecasting.io.id_mapping import load_or_build_id_mapping
from electricity_forecasting.datasets.window_index_builder import build_window_indices
from electricity_forecasting.evaluation.evaluate import (
    save_window_index, load_window_index, hash_window_index,
    evaluate_predictions, active_ratio_from_truth, build_metrics_record, save_json
)
from electricity_forecasting.training.train_lgbm_hurdle import threshold_sweep_on_validation
from electricity_forecasting.datasets.materialize import materialize_from_window_index
from electricity_forecasting.models.ml_hurdle_lgbm_hwise import train_hurdle_hwise, predict_hurdle_hwise, apply_hurdle_threshold
from electricity_forecasting.evaluation.reporting import update_metrics_table
from electricity_forecasting.viz.plots import plot_and_save_baseline_vs_model_bar

import lightgbm as lgb
import matplotlib.pyplot as plt


def _safe_rel(path: Path, project_dir: Path) -> str:
    try:
        return str(path.relative_to(project_dir))
    except ValueError:
        return str(Path(path).name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", type=str, default=str(REPO_DIR))
    p.add_argument("--train_parquet", type=str, default="")
    p.add_argument("--test_parquet", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_train_windows", type=int, default=300_000)
    p.add_argument("--max_val_windows", type=int, default=100_000)
    p.add_argument("--max_test_windows", type=int, default=200_000)
    p.add_argument("--valid_ratio", type=float, default=0.1)

    p.add_argument("--train_start", type=str, default="2021-01-01 00:00:00+00:00")
    p.add_argument("--train_end", type=str, default="2023-12-31 23:45:00+00:00")
    p.add_argument("--test_start", type=str, default="2024-01-01 00:00:00+00:00")
    p.add_argument("--test_end", type=str, default="2024-12-31 23:45:00+00:00")

    return p.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project_dir)

    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    plots_dir = results_dir / "plots"

    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "window_index").mkdir(parents=True, exist_ok=True)
    (plots_dir / "models/ml_hurdle_lgbm").mkdir(parents=True, exist_ok=True)
    (plots_dir / "eval/curves").mkdir(parents=True, exist_ok=True)
    (plots_dir / "eval/tables").mkdir(parents=True, exist_ok=True)

    train_parquet = args.train_parquet or str(data_dir / "TRAIN_Reco_2021_2022_2023_with_time_utc.parquet")
    test_parquet = args.test_parquet or str(data_dir / "TEST_Reco_2024_with_time_utc.parquet")

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)

    print("Project dir:", project_dir)
    print("Train parquet:", train_parquet)
    print("Test parquet:", test_parquet)
    print("Constants:", {"WIN": WIN, "HORIZON": HORIZON, "TARGET_COLS": list(TARGET_COLS), "VOLUME_INDEX": VOLUME_INDEX})

    # 1) ID mapping (shared across ML/DL) - load if exists, else build from TEST and save.
    t0 = time.time()
    mapping_path = results_dir / "window_index" / "id_code_mapping.json"
    id_to_code, code_to_id, all_codes = load_or_build_id_mapping(
        mapping_json_path=mapping_path,
        test_parquet_path=test_parquet,
        id_col="ID",
        batch_size=1_000_000,
    )
    print("IDs:", len(all_codes), "| mapping:", mapping_path.name, "| Elapsed:", f"{(time.time()-t0):.1f}s")

    # 2) Window indices (prefer existing files for fast reruns)
    p_tr = results_dir / "window_index" / f"train_windows_all_ids_cap{args.max_train_windows}.csv"
    p_va = results_dir / "window_index" / f"val_windows_all_ids_cap{args.max_val_windows}.csv"
    p_te = results_dir / "window_index" / f"test_windows_all_ids_cap{args.max_test_windows}.csv"

    if p_tr.exists() and p_va.exists() and p_te.exists():
        w_tr = load_window_index(p_tr)
        w_va = load_window_index(p_va)
        w_te = load_window_index(p_te)
        meta = {"source": "existing_files"}
        print("Loaded window_index:", p_tr.name, p_va.name, p_te.name)
    else:
        w_tr, w_va, w_te, meta = build_window_indices(
            all_id_codes=all_codes,
            train_start_utc=train_start,
            train_end_utc=train_end,
            test_start_utc=test_start,
            test_end_utc=test_end,
            win=WIN,
            horizon=HORIZON,
            freq=FREQ,
            max_train_windows=args.max_train_windows,
            max_val_windows=args.max_val_windows,
            max_test_windows=args.max_test_windows,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
    )

    # Save window_index only if it did not exist (i.e., we built it above)
    if not (p_tr.exists() and p_va.exists() and p_te.exists()):
        save_window_index(w_tr, p_tr)
        save_window_index(w_va, p_va)
        save_window_index(w_te, p_te)
        print("Saved window_index:", p_tr.name, p_va.name, p_te.name)

    print("Meta:", meta)
    print("Hashes:", hash_window_index(w_tr), hash_window_index(w_va), hash_window_index(w_te))

    # 3) Materialize datasets
    t0 = time.time()
    X_tr, Y_tr, Yb_tr, yv_tr, ids_tr, feature_cols = materialize_from_window_index(
        parquet_path=train_parquet,
        window_index=w_tr,
        code_to_id=code_to_id,
        start_utc=train_start,
        end_utc=train_end,
            freq=FREQ,
        berlin_tz=BERLIN_TZ,
            win=WIN,
            horizon=HORIZON,
        target_cols=list(TARGET_COLS),
        volume_index=VOLUME_INDEX,
        feature_cols=None,
    )
    print("Train materialized:", X_tr.shape, Y_tr.shape, "Elapsed:", f"{(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    X_va, Y_va, Yb_va, yv_va, ids_va, _ = materialize_from_window_index(
        parquet_path=train_parquet,
        window_index=w_va,
        code_to_id=code_to_id,
        start_utc=train_start,
        end_utc=train_end,
            freq=FREQ,
        berlin_tz=BERLIN_TZ,
            win=WIN,
            horizon=HORIZON,
        target_cols=list(TARGET_COLS),
        volume_index=VOLUME_INDEX,
        feature_cols=feature_cols,
    )
    print("Val materialized:", X_va.shape, Y_va.shape, "Elapsed:", f"{(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    X_te, Y_te, Yb_te, yv_te, ids_te, _ = materialize_from_window_index(
        parquet_path=test_parquet,
        window_index=w_te,
        code_to_id=code_to_id,
        start_utc=test_start,
        end_utc=test_end,
            freq=FREQ,
        berlin_tz=BERLIN_TZ,
            win=WIN,
            horizon=HORIZON,
        target_cols=list(TARGET_COLS),
        volume_index=VOLUME_INDEX,
        feature_cols=feature_cols,
    )
    print("Test materialized:", X_te.shape, Y_te.shape, "Elapsed:", f"{(time.time()-t0)/60:.1f} min")
    print("Active ratio on test (truth volume):", active_ratio_from_truth(Y_te, volume_index=VOLUME_INDEX))

    # 4) Train hurdle (h-wise)
    # sklearn-style param names to avoid alias warnings
    params_cls = dict(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
    )
    params_reg = dict(
        objective="regression",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
    )

    y_tr_vol = Y_tr[:, :, VOLUME_INDEX]
    y_va_vol = Y_va[:, :, VOLUME_INDEX]
    y_te_vol = Y_te[:, :, VOLUME_INDEX]

    t0 = time.time()
    bundle, proba_va, logvol_va = train_hurdle_hwise(
        X_tr=X_tr,
        y_tr_vol=y_tr_vol,
        X_va=X_va,
        y_va_vol=y_va_vol,
            horizon=HORIZON,
        params_cls=params_cls,
        params_reg=params_reg,
    )
    print("Trained hurdle models. Elapsed:", f"{(time.time()-t0)/60:.1f} min")

    # Predict for test
    proba_te, logvol_te = predict_hurdle_hwise(bundle, X_te, horizon=HORIZON)

    # 5) Threshold sweep on validation (volume FULL)
    thr_grid = np.linspace(0.05, 0.95, 19)
    best_thr, curve_df = threshold_sweep_on_validation(
        y_true_vol_va=y_va_vol,
        proba_va=proba_va,
        logvol_va=logvol_va,
        thresholds=thr_grid,
    )
    print("Best threshold (validation):", best_thr)

    curve_path = plots_dir / "eval/curves" / "ml_hurdle_lgbm_threshold_sweep_volume_full.png"
    plt.figure()
    plt.plot(curve_df["threshold"], curve_df["val_full_smape_volume"], marker="o", label="Val Full sMAPE (volume)")
    plt.plot(curve_df["threshold"], curve_df["val_active_smape_volume"], marker="o", label="Val Active sMAPE (volume)")
    plt.xlabel("Threshold")
    plt.ylabel("sMAPE")
    plt.title("Validation threshold sweep (volume hurdle)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=160)
    plt.close()
    print("Saved:", curve_path)

    # 6) Final predictions on test (baseline prices + model volume)
    vol_hat_te = apply_hurdle_threshold(proba_te, logvol_te, best_thr)  # (N,H)

    Y_pred_baseline = Yb_te.astype(np.float32)
    Y_pred_model = Yb_te.astype(np.float32).copy()
    Y_pred_model[:, :, VOLUME_INDEX] = vol_hat_te.astype(np.float32)

    baseline_metrics = evaluate_predictions(Y_true=Y_te, Y_pred=Y_pred_baseline, volume_index=VOLUME_INDEX)
    model_metrics = evaluate_predictions(Y_true=Y_te, Y_pred=Y_pred_model, volume_index=VOLUME_INDEX)

    print("Baseline:", baseline_metrics)
    print("Model   :", model_metrics)

    # 7) Save bar chart
    bar_path = plots_dir / "models/ml_hurdle_lgbm" / "baseline_vs_model_full_active.png"
    plot_and_save_baseline_vs_model_bar(
        metrics_bar={
            "baseline_full": float(baseline_metrics["full_smape"]),
            "baseline_active": float(baseline_metrics["active_smape"]),
            "model_full": float(model_metrics["full_smape"]),
            "model_active": float(model_metrics["active_smape"]),
        },
        out_png=str(bar_path),
        title="ML Hurdle LGBM vs Baseline (canonical metrics)",
        dpi=160,
    )
    print("Saved:", bar_path)

    # 8) Save metrics record JSON
    threshold_info = {
        "threshold_grid": [float(x) for x in thr_grid.tolist()],
        "best_threshold": float(best_thr),
        "threshold_selected_on": "validation",
        "objective": "minimize_full_smape_on_volume",
    }

    shapes = {
        "X_tr": list(X_tr.shape),
        "Y_tr": list(Y_tr.shape),
        "X_va": list(X_va.shape),
        "Y_va": list(Y_va.shape),
        "X_test": list(X_te.shape),
        "Y_test": list(Y_te.shape),
    }

    record = build_metrics_record(
        model_name="ml_hurdle_lgbm",
        run_tag="ml_hurdle_lgbm_all_ids_window_based",
        constants={
            "WIN": WIN,
            "HORIZON": HORIZON,
            "MAX_TEST_WINDOWS": int(args.max_test_windows),
            "MAX_TRAIN_WINDOWS": int(args.max_train_windows),
            "MAX_VAL_WINDOWS": int(args.max_val_windows),
            "TARGET_COLS": list(TARGET_COLS),
            "VOLUME_INDEX": int(VOLUME_INDEX),
            "FREQ": FREQ,
            "FEATURE_COLS": feature_cols,
        },
        split_info={
            "valid_ratio": float(args.valid_ratio),
            "time_bounds": {
                "train_start_utc": str(train_start),
                "train_end_utc": str(train_end),
                "test_start_utc": str(test_start),
                "test_end_utc": str(test_end),
            },
        },
        sampling_info={
            "n_ids_train": int(np.unique(ids_tr).size),
            "n_ids_val": int(np.unique(ids_va).size),
            "n_ids_test": int(np.unique(ids_te).size),
            "n_train_windows_used": int(X_tr.shape[0]),
            "n_val_windows_used": int(X_va.shape[0]),
            "n_test_windows_used": int(X_te.shape[0]),
            "window_index_paths": {
                "train": _safe_rel(p_tr, project_dir),
                "val": _safe_rel(p_va, project_dir),
                "test": _safe_rel(p_te, project_dir),
            },
            "window_index_hashes": {"train": hash_window_index(w_tr), "val": hash_window_index(w_va), "test": hash_window_index(w_te)},
            "test_window_sampling_rule": "ID-balanced + spread start_pos + global cap",
            "window_index_meta": meta,
        },
        shapes=shapes,
        baseline_metrics=baseline_metrics,
        model_metrics=model_metrics,
        active_ratio_test=active_ratio_from_truth(Y_te, volume_index=VOLUME_INDEX),
        threshold_info=threshold_info,
        extra={
            "lightgbm_params_classifier": params_cls,
            "lightgbm_params_regressor": params_reg,
        },
    )

    out_json = results_dir / "metrics" / "ml_hurdle_lgbm.json"
    save_json(record, out_json)
    print("Saved:", out_json)

    # 9) Global metrics table (single CSV for readers)
    tbl_csv = results_dir / "metrics" / "metrics_table.csv"
    tbl_xlsx = results_dir / "metrics" / "metrics_table.xlsx"
    tbl_png = plots_dir / "eval/tables" / "metrics_table.png"

    df_tbl = update_metrics_table(
        metrics_dir=results_dir / "metrics",
        out_csv=tbl_csv,
        out_xlsx=tbl_xlsx,
        out_png=tbl_png,
        title="Model performance metrics (canonical)",
    )
    print("Saved:", tbl_csv)
    print("Saved:", tbl_xlsx)
    print("Saved:", tbl_png)

    print("DONE.")


if __name__ == "__main__":
    main()
