"""Train DL Gated Transformer hurdle model (volume-only) and save artifacts.

This is the terminal-friendly equivalent of the notebook:
  05_DL_Transformer.ipynb

Key constraints:
- Same split (train=2021-2023, test=2024)
- Same WIN/HORIZON
- window_index-driven sampling (ID_code, start_pos)
- Validation threshold sweep only; test evaluated once

Artifacts (paths relative to PROJECT_DIR):
- results/models/dl_gated_transformer_best.pt
- results/metrics/dl_gated_transformer.json
- results/metrics/metrics_table_dl_gated_transformer_vs_baseline.(csv/json)
- results/plots/models/dl_gated_transformer/baseline_vs_model_full_active.png
- results/plots/eval/curves/dl_gated_transformer_threshold_sweep_volume_full.png
- results/plots/eval/tables/metrics_table_dl_gated_transformer_vs_baseline.png
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Bootstrap src import when running as a file
SRC_DIR = Path(__file__).resolve().parents[2]  # <repo_root>/src
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_forecasting.config import WIN, HORIZON, TARGET_COLS, VOLUME_INDEX, FREQ, BERLIN_TZ
from electricity_forecasting.io.id_mapping import load_or_build_id_mapping
from electricity_forecasting.datasets.window_index_builder import build_window_indices
from electricity_forecasting.evaluation.evaluate import save_window_index, load_window_index, hash_window_index
from electricity_forecasting.datasets.materialize import build_one_id_grid_with_features, infer_feature_cols
from electricity_forecasting.datasets.iterable_dataset import ParquetWindowIterableDataset
from electricity_forecasting.models.dl_gated_transformer import GatedTransformerHurdle
from electricity_forecasting.training.train_tcn import (
    TrainConfig,
    train_hurdle_tcn,
    collect_hurdle_outputs,
    threshold_sweep_on_validation,
    apply_hurdle_threshold,
    compute_pos_weight_from_loader,
)
from electricity_forecasting.evaluation.evaluate import evaluate_predictions, active_ratio_from_truth
from electricity_forecasting.evaluation.evaluate import build_metrics_record, save_metrics_record, log_run_header
from electricity_forecasting.evaluation.reporting import update_metrics_table
from electricity_forecasting.viz.plots import plot_and_save_baseline_vs_model_bar, plot_threshold_sweep


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

    # Training
    p.add_argument("--eval_only", action="store_true", help="Skip training; load checkpoint and run val sweep + test eval.")
    p.add_argument("--ckpt_path", type=str, default="", help="Optional checkpoint path. Default: results/models/dl_gated_transformer_best.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_train", type=int, default=1024)
    p.add_argument("--batch_eval", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true")

    # Hurdle
    p.add_argument("--active_eps", type=float, default=0.0)
    p.add_argument("--lambda_reg", type=float, default=1.0)
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--pos_weight", type=float, default=12.0, help="Fixed for reproducibility. Set <0 to auto-compute.")

    # Model (Gated Transformer)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dim_ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--id_emb_dim", type=int, default=8)
    p.add_argument("--max_len", type=int, default=256)

    # Cache
    p.add_argument("--cache_grids", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    project_dir = Path(args.project_dir)
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    plots_dir = results_dir / "plots"

    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "window_index").mkdir(parents=True, exist_ok=True)
    (plots_dir / "models/dl_gated_transformer").mkdir(parents=True, exist_ok=True)
    (plots_dir / "eval/curves").mkdir(parents=True, exist_ok=True)
    (plots_dir / "eval/tables").mkdir(parents=True, exist_ok=True)

    train_parquet = args.train_parquet or str(data_dir / "TRAIN_Reco_2021_2022_2023_with_time_utc.parquet")
    test_parquet = args.test_parquet or str(data_dir / "TEST_Reco_2024_with_time_utc.parquet")

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp) and (device.type == "cuda")
    print(f"OK: Torch={torch.__version__} | device={device} | use_amp={use_amp}")

    # 1) ID mapping
    mapping_path = results_dir / "window_index" / "id_code_mapping.json"
    id_to_code, code_to_id, all_codes = load_or_build_id_mapping(
        mapping_json_path=mapping_path,
        test_parquet_path=test_parquet,
        id_col="ID",
        batch_size=1_000_000,
    )
    print("IDs:", len(all_codes), "| mapping:", mapping_path.name)

    # 2) Window indices (prefer existing files)
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
            train_start_utc=args.train_start,
            train_end_utc=args.train_end,
            test_start_utc=args.test_start,
            test_end_utc=args.test_end,
            win=WIN,
            horizon=HORIZON,
            freq=FREQ,
            max_train_windows=args.max_train_windows,
            max_val_windows=args.max_val_windows,
            max_test_windows=args.max_test_windows,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        save_window_index(w_tr, p_tr)
        save_window_index(w_va, p_va)
        save_window_index(w_te, p_te)
        print("Saved window_index:", p_tr.name, p_va.name, p_te.name)

    print("window_index hashes:", hash_window_index(w_tr), hash_window_index(w_va), hash_window_index(w_te))

    # 3) Feature columns
    if ckpt is not None and isinstance(ckpt, dict) and "feature_cols" in ckpt:
        feature_cols = list(ckpt["feature_cols"])
        print("Feature cols (from ckpt):", len(feature_cols))
    else:
        sample_code = int(w_tr["ID"].iloc[0]) if len(w_tr) > 0 else int(all_codes[0])
        raw_id = code_to_id[sample_code]
        df_grid = build_one_id_grid_with_features(
            parquet_path=train_parquet,
            raw_id=raw_id,
            id_code=sample_code,
            start_utc=args.train_start,
            end_utc=args.train_end,
            freq=FREQ,
            berlin_tz=BERLIN_TZ,
        )
        feature_cols = infer_feature_cols(df_grid)
        print("Feature cols (inferred):", len(feature_cols))


    # 4) Datasets / loaders
    cache_dir = (results_dir / "cache/grids") if args.cache_grids else None

    train_ds = ParquetWindowIterableDataset(
        parquet_path=train_parquet,
        window_index=w_tr,
        code_to_id=code_to_id,
        start_utc=args.train_start,
        end_utc=args.train_end,
        freq=FREQ,
        berlin_tz=BERLIN_TZ,
        win=WIN,
        horizon=HORIZON,
        feature_cols=feature_cols,
        target_cols=list(TARGET_COLS),
        cache_dir=cache_dir,
        cache_tag="train_2021_2023",
        shuffle_ids=True,
        shuffle_within_id=True,
        seed=int(args.seed),
        return_id_code=True,
        pack_x_with_id=True,
        return_baseline=True,
        return_meta=False,
    )

    val_ds = ParquetWindowIterableDataset(
        parquet_path=train_parquet,
        window_index=w_va,
        code_to_id=code_to_id,
        start_utc=args.train_start,
        end_utc=args.train_end,
        freq=FREQ,
        berlin_tz=BERLIN_TZ,
        win=WIN,
        horizon=HORIZON,
        feature_cols=feature_cols,
        target_cols=list(TARGET_COLS),
        cache_dir=cache_dir,
        cache_tag="train_2021_2023",
        shuffle_ids=False,
        shuffle_within_id=False,
        seed=int(args.seed),
        return_id_code=True,
        pack_x_with_id=True,
        return_baseline=True,
        return_meta=False,
    )

    test_ds = ParquetWindowIterableDataset(
        parquet_path=test_parquet,
        window_index=w_te,
        code_to_id=code_to_id,
        start_utc=args.test_start,
        end_utc=args.test_end,
        freq=FREQ,
        berlin_tz=BERLIN_TZ,
        win=WIN,
        horizon=HORIZON,
        feature_cols=feature_cols,
        target_cols=list(TARGET_COLS),
        cache_dir=cache_dir,
        cache_tag="test_2024",
        shuffle_ids=False,
        shuffle_within_id=False,
        seed=int(args.seed),
        return_id_code=True,
        pack_x_with_id=True,
        return_baseline=True,
        return_meta=False,
    )

    pin = (device.type == "cuda")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(args.batch_train),
        num_workers=int(args.num_workers),
        pin_memory=pin,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(args.batch_eval),
        num_workers=int(args.num_workers),
        pin_memory=pin,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=int(args.batch_eval),
        num_workers=int(args.num_workers),
        pin_memory=pin,
    )

    # 5) Model
    if ckpt is not None and isinstance(ckpt, dict):
        args.d_model = int(ckpt.get('d_model', args.d_model))
        args.nhead = int(ckpt.get('nhead', args.nhead))
        args.n_layers = int(ckpt.get('n_layers', args.n_layers))
        args.dim_ff = int(ckpt.get('dim_ff', args.dim_ff))
        args.dropout = float(ckpt.get('dropout', args.dropout))
        args.id_emb_dim = int(ckpt.get('id_emb_dim', args.id_emb_dim))
        print('Model hparams (from ckpt where available):',
              {'d_model': args.d_model, 'nhead': args.nhead, 'n_layers': args.n_layers, 'dim_ff': args.dim_ff, 'dropout': args.dropout, 'id_emb_dim': args.id_emb_dim})

    model = GatedTransformerHurdle(
        n_features=len(feature_cols),
        n_ids=len(all_codes),
        horizon=HORIZON,
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        n_layers=int(args.n_layers),
        dim_ff=int(args.dim_ff),
        dropout=float(args.dropout),
        id_emb_dim=int(args.id_emb_dim),
        max_len=int(args.max_len),
    ).to(device)

    # 6) Training
    cfg = TrainConfig(
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_epochs=int(args.epochs),
        patience=int(args.patience),
        grad_clip=float(args.grad_clip),
        w_bce=1.0,
        w_reg=float(args.lambda_reg),
        huber_delta=float(args.huber_delta),
        active_eps=float(args.active_eps),
        use_amp=use_amp,
    )

    pos_weight = None
    if not args.eval_only:
        if float(args.pos_weight) > 0:
            pos_weight = torch.tensor(float(args.pos_weight), dtype=torch.float32)
            print("pos_weight (fixed):", float(pos_weight))
        else:
            pos_weight = compute_pos_weight_from_loader(
                train_loader,
                volume_index=VOLUME_INDEX,
                active_eps=float(args.active_eps),
                max_batches=200,
                device=None,
            )
            if pos_weight is None:
                print("WARN: pos_weight could not be computed (no positives observed). Using None.")
            else:
                print("pos_weight (auto):", float(pos_weight))
    else:
        # eval_only: pos_weight does not affect evaluation; prefer ckpt metadata if available.
        if ckpt is not None and isinstance(ckpt, dict) and ckpt.get("pos_weight") is not None:
            try:
                pos_weight = torch.tensor(float(ckpt["pos_weight"]), dtype=torch.float32)
            except Exception:
                pos_weight = None
        elif float(args.pos_weight) > 0:
            pos_weight = torch.tensor(float(args.pos_weight), dtype=torch.float32)



    if args.eval_only:
        model.load_state_dict(ckpt["state_dict"])
        history = ckpt.get("history", {})
        print("OK: Loaded weights from ckpt; skipped training.")
    else:
        t0 = time.time()
        model, history = train_hurdle_tcn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            volume_index=VOLUME_INDEX,
            device=device,
            cfg=cfg,
            pos_weight=pos_weight,
            verbose=True,
        )
        print("Training done | elapsed:", f"{(time.time()-t0)/60:.1f} min")

        ckpt_path = results_dir / "models" / "dl_gated_transformer_best.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "cfg": cfg.__dict__,
                "history": history,
                "seed": int(args.seed),
                "arch": "Gated Transformer + two-head hurdle",
                "d_model": int(args.d_model),
                "nhead": int(args.nhead),
                "n_layers": int(args.n_layers),
                "dim_ff": int(args.dim_ff),
                "dropout": float(args.dropout),
                "id_emb_dim": int(args.id_emb_dim),
                "pos_weight": float(pos_weight) if pos_weight is not None else None,
            },
            ckpt_path,
        )
        print("Saved checkpoint:", ckpt_path)

    # 7) Validation threshold sweep
    Y_va, Yb_va, proba_va, logv_va = collect_hurdle_outputs(
        model=model, loader=val_loader, device=device, volume_index=VOLUME_INDEX
    )
    y_true_vol_va = Y_va[:, :, VOLUME_INDEX].astype(np.float64)

    thr_grid = np.linspace(0.05, 0.95, 19)
    best_thr, sweep_df = threshold_sweep_on_validation(
        y_true_vol_va, proba_va, logv_va, thr_grid, eps=1e-8
    )
    print("Best threshold (val):", best_thr)

    sweep_csv = results_dir / "metrics" / "threshold_sweep_dl_gated_transformer.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    sweep_plot_df = pd.DataFrame(
        {
            "threshold": sweep_df["threshold"].values,
            "full_smape": sweep_df["val_volume_full_smape"].values,
            "active_smape": sweep_df["val_volume_active_smape"].values,
        }
    )
    sweep_png = plots_dir / "eval/curves" / "dl_gated_transformer_threshold_sweep_volume_full.png"
    plot_threshold_sweep(sweep_plot_df, out_png=str(sweep_png), title="DL Gated Transformer: threshold sweep (Volume)")
    print("Saved:", sweep_csv)
    print("Saved:", sweep_png)

    threshold_info = {
        "threshold_grid": thr_grid.tolist(),
        "best_threshold": float(best_thr),
        "selection_metric": "val_volume_full_smape",
        "sweep_csv": _safe_rel(sweep_csv, project_dir),
        "sweep_png": _safe_rel(sweep_png, project_dir),
    }

    # 8) Test evaluation (single run)
    Y_te, Yb_te, proba_te, logv_te = collect_hurdle_outputs(
        model=model, loader=test_loader, device=device, volume_index=VOLUME_INDEX
    )
    vol_hat = apply_hurdle_threshold(proba_te, logv_te, float(best_thr))  # (N, H)

    Y_pred_baseline = Yb_te.astype(np.float32)
    Y_pred_model = Yb_te.copy().astype(np.float32)
    Y_pred_model[:, :, VOLUME_INDEX] = vol_hat.astype(np.float32)

    baseline_metrics = evaluate_predictions(Y_true=Y_te, Y_pred=Y_pred_baseline, volume_index=VOLUME_INDEX)
    model_metrics = evaluate_predictions(Y_true=Y_te, Y_pred=Y_pred_model, volume_index=VOLUME_INDEX)

    print("Baseline metrics:", baseline_metrics)
    print("Model metrics   :", model_metrics)

    # Bar chart
    metrics_bar = {
        "baseline_full": float(baseline_metrics["full_smape"]),
        "baseline_active": float(baseline_metrics["active_smape"]),
        "model_full": float(model_metrics["full_smape"]),
        "model_active": float(model_metrics["active_smape"]),
    }
    bar_path = plots_dir / "models/dl_gated_transformer" / "baseline_vs_model_full_active.png"
    plot_and_save_baseline_vs_model_bar(
        metrics_bar=metrics_bar,
        out_png=str(bar_path),
        title="DL Gated Transformer (Volume only) vs Baseline",
        dpi=160,
    )
    print("Saved:", bar_path)

    # Metrics JSON
    shapes = {
        "X_tr": [int(len(w_tr)), int(WIN), int(len(feature_cols))],
        "Y_tr": [int(len(w_tr)), int(HORIZON), int(len(TARGET_COLS))],
        "X_va": [int(len(w_va)), int(WIN), int(len(feature_cols))],
        "Y_va": [int(len(w_va)), int(HORIZON), int(len(TARGET_COLS))],
        "X_test": [int(len(w_te)), int(WIN), int(len(feature_cols))],
        "Y_test": [int(Y_te.shape[0]), int(Y_te.shape[1]), int(Y_te.shape[2])],
    }

    record = build_metrics_record(
        model_name="dl_gated_transformer",
        run_tag="dl_gated_transformer_window_based",
        constants={
            "WIN": int(WIN),
            "HORIZON": int(HORIZON),
            "MAX_TRAIN_WINDOWS": int(args.max_train_windows),
            "MAX_VAL_WINDOWS": int(args.max_val_windows),
            "MAX_TEST_WINDOWS": int(args.max_test_windows),
            "TARGET_COLS": list(TARGET_COLS),
            "VOLUME_INDEX": int(VOLUME_INDEX),
            "FREQ": str(FREQ),
            "FEATURE_COLS": list(feature_cols),
            "seed": int(args.seed),
        },
        split_info={
            "valid_ratio": float(args.valid_ratio),
            "threshold_selected_on": "validation",
            "time_bounds": {
                "train_start_utc": str(args.train_start),
                "train_end_utc": str(args.train_end),
                "test_start_utc": str(args.test_start),
                "test_end_utc": str(args.test_end),
            },
        },
        sampling_info={
            "n_ids_train": int(w_tr["ID"].nunique()),
            "n_ids_val": int(w_va["ID"].nunique()),
            "n_ids_test": int(w_te["ID"].nunique()),
            "n_train_windows_used": int(len(w_tr)),
            "n_val_windows_used": int(len(w_va)),
            "n_test_windows_used": int(len(w_te)),
            "window_index_paths": {
                "train": _safe_rel(p_tr, project_dir),
                "val": _safe_rel(p_va, project_dir),
                "test": _safe_rel(p_te, project_dir),
            },
            "window_index_hashes": {"train": hash_window_index(w_tr), "val": hash_window_index(w_va), "test": hash_window_index(w_te)},
            "window_index_format": "(ID_code, start_pos)",
            "sampling_rule": "window_index (same as ML experiment); validation time-based; no test leakage",
        },
        shapes=shapes,
        baseline_metrics=baseline_metrics,
        model_metrics=model_metrics,
        active_ratio_test=float(active_ratio_from_truth(Y_te, volume_index=VOLUME_INDEX)),
        threshold_info=threshold_info,
        extra={
            "torch_version": torch.__version__,
            "device": str(device),
            "arch": "Gated Transformer + two-head hurdle (BCE + masked Huber on log1p(volume))",
            "d_model": int(args.d_model),
            "nhead": int(args.nhead),
            "n_layers": int(args.n_layers),
            "dim_ff": int(args.dim_ff),
            "dropout": float(args.dropout),
            "id_emb_dim": int(args.id_emb_dim),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs_ran": int(len(history.get("train_loss", []))),
            "pos_weight": float(pos_weight) if pos_weight is not None else None,
            "lambda_reg": float(args.lambda_reg),
            "huber_delta": float(args.huber_delta),
            "cache_grids": bool(args.cache_grids),
            "use_amp": bool(use_amp),
        },
    )

    out_json = results_dir / "metrics" / "dl_gated_transformer.json"
    save_metrics_record(record, out_json)
    print("Saved metrics JSON:", out_json)

    log_run_header(
        model_name=record["model_name"],
        constants=record["constants"],
        shapes=record["shapes"],
        sampling_info=record["sampling_info"],
    )

    # Global metrics table (single CSV for readers)
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


if __name__ == "__main__":
    main()
