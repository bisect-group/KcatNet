import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from queue import Empty, Queue

import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    normalize_threshold_args,
)
from emulator_bench.tune_optuna import load_training_config, maybe_cache_embeddings, optimizer_defaults_from_config


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"
PREDICT_SCRIPT = REPO_ROOT / "emulator_bench" / "predict_single_target.py"
_STDOUT_LOCK = threading.Lock()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    _ensure_dir(path.parent)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _run_logged_command(cmd: list[str], log_path: Path, env: dict[str, str], live_output: bool = False) -> None:
    _ensure_dir(log_path.parent)
    if not live_output:
        with open(log_path, "w") as handle:
            handle.write(f"$ {shlex.join(cmd)}\n\n")
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = proc.wait()
    else:
        with open(log_path, "wb") as handle:
            handle.write(f"$ {shlex.join(cmd)}\n\n".encode("utf-8", errors="replace"))
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                handle.write(chunk)
                handle.flush()
                with _STDOUT_LOCK:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {log_path}")


def _list_study_names(storage: str) -> list[str]:
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage)
    except Exception:
        return []
    return [summary.study_name for summary in summaries]


def _load_study_with_fallback(study_name: str, storage: str):
    candidate_names = [study_name]
    if study_name.endswith("_best_hparams"):
        candidate_names.append(study_name[: -len("_best_hparams")])
    if study_name.endswith("_best_hparams.json"):
        candidate_names.append(study_name[: -len("_best_hparams.json")])

    seen = set()
    deduped_candidates = []
    for candidate in candidate_names:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        try:
            study = optuna.load_study(study_name=candidate, storage=storage)
            return study, candidate
        except KeyError:
            continue

    available_names = _list_study_names(storage)
    if len(available_names) == 1:
        candidate = available_names[0]
        study = optuna.load_study(study_name=candidate, storage=storage)
        print(
            f"Requested study `{study_name}` not found. Falling back to the only available study `{candidate}`.",
            flush=True,
        )
        return study, candidate

    if available_names:
        raise KeyError(
            "Optuna study not found. "
            f"Requested `{study_name}`. Available studies: {', '.join(sorted(available_names))}."
        )
    raise KeyError(
        "Optuna study not found and no studies were discovered in the provided storage. "
        f"Requested `{study_name}` in storage `{storage}`."
    )


def _load_best_hparams(args):
    if args.hparams_json:
        with open(args.hparams_json, "r") as handle:
            payload = json.load(handle)
        return payload, {
            "source": "hparams_json",
            "hparams_json": str(args.hparams_json),
        }

    if not args.storage:
        raise ValueError("Provide either --hparams_json or --storage for extracting best Optuna params.")

    study, resolved_study_name = _load_study_with_fallback(args.study_name, args.storage)
    payload = dict(study.best_params)
    payload["study_name"] = resolved_study_name
    payload["storage"] = args.storage
    payload["best_trial_number"] = int(study.best_trial.number)
    payload["best_value"] = float(study.best_value)
    payload["direction"] = study.direction.name.lower()
    return payload, {
        "source": "optuna_storage",
        "study_name": resolved_study_name,
        "requested_study_name": args.study_name,
        "storage": args.storage,
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "direction": study.direction.name.lower(),
    }


def _resolve_training_hparams(raw_hparams: dict, args) -> dict:
    config_defaults = args.config_optimizer_defaults

    def choose(key: str, fallback):
        value = raw_hparams.get(key, fallback)
        override = getattr(args, key)
        if override is not None:
            return override
        return value

    resolved = {
        "batch_size": int(choose("batch_size", 16)),
        "lr": float(choose("lr", config_defaults["lr"])),
        "weight_decay": float(choose("weight_decay", config_defaults["weight_decay"])),
        "beta1": float(choose("beta1", config_defaults["beta1"])),
        "beta2": float(choose("beta2", config_defaults["beta2"])),
        "eps": float(choose("eps", config_defaults["eps"])),
        "scheduler": str(choose("scheduler", config_defaults["scheduler"])),
        "lr_decay_factor": float(choose("lr_decay_factor", 0.5)),
        "lr_decay_patience": int(choose("lr_decay_patience", 5)),
        "min_lr": float(choose("min_lr", config_defaults["min_lr"])),
        "lr_warmup_epochs": int(choose("lr_warmup_epochs", config_defaults["lr_warmup_epochs"])),
        "lr_warmup_start_factor": float(choose("lr_warmup_start_factor", 0.1)),
        "clip_grad": float(choose("clip_grad", config_defaults["clip_grad"])),
        "regression_weight": float(choose("regression_weight", 0.9)),
        "cluster_weight": float(choose("cluster_weight", 0.1)),
        "patience": int(choose("patience", 0)),
        "min_delta": float(choose("min_delta", 0.0)),
    }
    raw_amsgrad = raw_hparams.get("amsgrad", config_defaults["amsgrad"])
    resolved["amsgrad"] = bool(args.amsgrad or raw_amsgrad)
    return resolved


def _build_experiments(jobs: list[dict], seeds: list[int], output_root: Path) -> list[dict]:
    experiments = []
    for job in jobs:
        for seed in seeds:
            run_dir = output_root / job["split_group"] / job["split_name"] / f"seed_{seed}"
            experiments.append(
                {
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "train_path": job["train_path"],
                    "val_path": job["val_path"],
                    "test_path": job["test_path"],
                    "seed": int(seed),
                    "run_dir": run_dir,
                }
            )
    return experiments


def _train_command(exp: dict, args, hparams: dict, gpu_local_device: str) -> list[str]:
    train_dir = exp["run_dir"] / "train"
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_path",
        exp["train_path"],
        "--val_path",
        exp["val_path"],
        "--test_path",
        exp["test_path"],
        "--embeddings_dir",
        args.embeddings_dir,
        "--out_dir",
        str(train_dir),
        "--task_name",
        f"{exp['split_group']}_{exp['split_name']}_seed{exp['seed']}",
        "--config_path",
        args.config_path,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
        "--batch_size",
        str(hparams["batch_size"]),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(hparams["lr"]),
        "--weight_decay",
        str(hparams["weight_decay"]),
        "--beta1",
        str(hparams["beta1"]),
        "--beta2",
        str(hparams["beta2"]),
        "--eps",
        str(hparams["eps"]),
        "--scheduler",
        hparams["scheduler"],
        "--lr_decay_factor",
        str(hparams["lr_decay_factor"]),
        "--lr_decay_patience",
        str(hparams["lr_decay_patience"]),
        "--min_lr",
        str(hparams["min_lr"]),
        "--lr_warmup_epochs",
        str(hparams["lr_warmup_epochs"]),
        "--lr_warmup_start_factor",
        str(hparams["lr_warmup_start_factor"]),
        "--clip_grad",
        str(hparams["clip_grad"]),
        "--regression_weight",
        str(hparams["regression_weight"]),
        "--cluster_weight",
        str(hparams["cluster_weight"]),
        "--patience",
        str(hparams["patience"]),
        "--min_delta",
        str(hparams["min_delta"]),
        "--seed",
        str(exp["seed"]),
        "--device",
        gpu_local_device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--val_every",
        str(args.val_every),
        "--protein_cache_items",
        str(args.protein_cache_items),
    ]
    if args.persistent_workers:
        cmd.append("--persistent_workers")
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.preload_proteins:
        cmd.append("--preload_proteins")
    if args.lazy_ligands:
        cmd.append("--lazy_ligands")
    if hparams["amsgrad"]:
        cmd.append("--amsgrad")
    return cmd


def _predict_command(split_name: str, input_path: str, ckpt_path: Path, out_csv: Path, args, gpu_local_device: str, batch_size: int) -> list[str]:
    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--input_path",
        input_path,
        "--embeddings_dir",
        args.embeddings_dir,
        "--ckpt_path",
        str(ckpt_path),
        "--out_csv",
        str(out_csv),
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
        "--batch_size",
        str(batch_size),
        "--device",
        gpu_local_device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--protein_cache_items",
        str(args.protein_cache_items),
    ]
    if args.persistent_workers:
        cmd.append("--persistent_workers")
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.lazy_ligands:
        cmd.append("--lazy_ligands")
    return cmd


def _collect_split_metrics(metrics_csv: Path, split: str) -> dict:
    row = pd.read_csv(metrics_csv).iloc[0].to_dict()
    return {
        "split": split,
        "r2": float(row["r2_score"]),
        "pcc": float(row["pearson"]),
        "scc": float(row["spearman"]),
        "mse": float(row["mse"]),
        "rmse": float(row["rmse"]),
        "mae": float(row["mae"]),
        "r2_score": float(row["r2_score"]),
        "pearson": float(row["pearson"]),
        "spearman": float(row["spearman"]),
    }


def _run_experiment(exp: dict, args, hparams: dict, gpu_id: str) -> dict:
    run_dir = exp["run_dir"]
    train_dir = run_dir / "train"
    pred_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"

    _ensure_dir(train_dir)
    _ensure_dir(pred_dir)
    _ensure_dir(metrics_dir)
    _ensure_dir(logs_dir)

    complete_marker = metrics_dir / "tvt_metrics_long.csv"
    ckpt_path = train_dir / "bestmodel.pth"
    if complete_marker.exists() and ckpt_path.exists() and not args.overwrite:
        return {
            "status": "skipped_exists",
            "gpu_id": gpu_id,
            "error": "",
            "run_dir": str(run_dir),
            "split_group": exp["split_group"],
            "split_name": exp["split_name"],
            "difficulty": exp["difficulty"],
            "seed": exp["seed"],
        }

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = args.device
    if device.startswith("cuda"):
        device = "cuda:0"

    _write_json(
        run_dir / "run_config.json",
        {
            "split_group": exp["split_group"],
            "split_name": exp["split_name"],
            "difficulty": exp["difficulty"],
            "seed": exp["seed"],
            "gpu_id": str(gpu_id),
            "train_hparams": hparams,
            "train_paths": {
                "train": exp["train_path"],
                "val": exp["val_path"],
                "test": exp["test_path"],
            },
        },
    )

    if args.overwrite or not ckpt_path.exists():
        train_cmd = _train_command(exp, args, hparams, device)
        _run_logged_command(
            train_cmd,
            logs_dir / "train.log",
            env=env,
            live_output=args.live_train_logs,
        )
    else:
        with open(logs_dir / "train.log", "a") as handle:
            handle.write("\n[launcher] Reusing existing checkpoint; skipping training step.\n")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint after training: {ckpt_path}")

    split_metrics_rows = []
    pred_records = []
    predict_batch_size = args.predict_batch_size if args.predict_batch_size is not None else hparams["batch_size"]
    split_to_path = {
        "train": exp["train_path"],
        "val": exp["val_path"],
        "test": exp["test_path"],
    }
    for split_name, input_path in split_to_path.items():
        out_csv = pred_dir / f"{split_name}_predictions.csv"
        predict_cmd = _predict_command(
            split_name,
            input_path,
            ckpt_path,
            out_csv,
            args,
            device,
            batch_size=int(predict_batch_size),
        )
        _run_logged_command(
            predict_cmd,
            logs_dir / f"predict_{split_name}.log",
            env=env,
            live_output=args.live_predict_logs,
        )
        metrics_csv = out_csv.with_name(out_csv.stem + "_metrics.csv")
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Missing prediction metrics CSV: {metrics_csv}")
        split_metrics_rows.append(_collect_split_metrics(metrics_csv, split_name))
        pred_records.append(
            {
                "split": split_name,
                "input_path": input_path,
                "predictions_csv": str(out_csv),
                "metrics_csv": str(metrics_csv),
            }
        )

    metrics_long_path = metrics_dir / "tvt_metrics_long.csv"
    pd.DataFrame(split_metrics_rows).to_csv(metrics_long_path, index=False)

    wide_row = {}
    for row in split_metrics_rows:
        split = row["split"]
        for key, value in row.items():
            if key == "split":
                continue
            wide_row[f"{split}_{key}"] = value
    pd.DataFrame([wide_row]).to_csv(metrics_dir / "tvt_metrics_wide.csv", index=False)
    pd.DataFrame(pred_records).to_csv(metrics_dir / "prediction_file_manifest.csv", index=False)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "status": "completed",
            "gpu_id": str(gpu_id),
            "run_dir": str(run_dir),
            "train_dir": str(train_dir),
            "checkpoint_best": str(ckpt_path),
            "checkpoint_last": str(train_dir / "checkpoint_last.pt"),
            "metrics_long_csv": str(metrics_long_path),
            "predictions": pred_records,
        },
    )

    return {
        "status": "completed",
        "gpu_id": gpu_id,
        "error": "",
        "run_dir": str(run_dir),
        "split_group": exp["split_group"],
        "split_name": exp["split_name"],
        "difficulty": exp["difficulty"],
        "seed": exp["seed"],
    }


def _run_parallel(experiments: list[dict], args, hparams: dict) -> list[dict]:
    queue: Queue = Queue()
    for exp in experiments:
        queue.put(exp)

    results: list[dict] = []
    result_lock = threading.Lock()

    def worker(gpu_id: str, slot_idx: int) -> None:
        while True:
            try:
                exp = queue.get_nowait()
            except Empty:
                return

            run_label = f"{exp['split_group']}/{exp['split_name']}/seed_{exp['seed']}"
            worker_label = f"GPU {gpu_id} slot {slot_idx}"
            print(f"[{worker_label}] Starting {run_label}", flush=True)
            try:
                result = _run_experiment(exp, args, hparams, gpu_id)
                print(f"[{worker_label}] Finished {run_label}: {result['status']}", flush=True)
            except Exception as exc:
                result = {
                    "status": "failed",
                    "gpu_id": gpu_id,
                    "gpu_slot": slot_idx,
                    "error": str(exc),
                    "run_dir": str(exp["run_dir"]),
                    "split_group": exp["split_group"],
                    "split_name": exp["split_name"],
                    "difficulty": exp["difficulty"],
                    "seed": exp["seed"],
                }
                print(f"[{worker_label}] Failed {run_label}: {exc}", flush=True)
            else:
                result["gpu_slot"] = slot_idx
            with result_lock:
                results.append(result)
            queue.task_done()

    threads = []
    for gpu_id in args.gpus:
        for slot_idx in range(args.runs_per_gpu):
            thread = threading.Thread(target=worker, args=(str(gpu_id), slot_idx), daemon=True)
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    return results


def _write_global_summaries(output_root: Path, results: list[dict]) -> None:
    _ensure_dir(output_root)
    runs_df = pd.DataFrame(results)
    runs_df.to_csv(output_root / "runs_status.csv", index=False)

    completed = runs_df[runs_df["status"].isin(["completed", "skipped_exists"])] if not runs_df.empty else pd.DataFrame()
    if completed.empty:
        return

    metric_rows = []
    for row in completed.to_dict("records"):
        metrics_path = Path(row["run_dir"]) / "metrics" / "tvt_metrics_long.csv"
        if not metrics_path.exists():
            continue
        frame = pd.read_csv(metrics_path)
        frame["split_group"] = row["split_group"]
        frame["split_name"] = row["split_name"]
        frame["difficulty"] = row["difficulty"]
        frame["seed"] = row["seed"]
        frame["run_dir"] = row["run_dir"]
        metric_rows.append(frame)

    if not metric_rows:
        return

    all_metrics = pd.concat(metric_rows, ignore_index=True)
    all_metrics.to_csv(output_root / "all_tvt_metrics.csv", index=False)

    metric_cols = ["r2", "pcc", "scc", "mse", "rmse", "mae"]
    grouped = (
        all_metrics.groupby(["split_group", "split_name", "difficulty", "split"], dropna=False)[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([part for part in col if part]).rstrip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    grouped.to_csv(output_root / "aggregate_tvt_metrics.csv", index=False)

    test_metrics = all_metrics[all_metrics["split"] == "test"]
    if not test_metrics.empty:
        ranked = (
            test_metrics.groupby(["split_group", "split_name", "difficulty"], dropna=False)[metric_cols]
            .agg(["mean", "var"])
            .reset_index()
        )
        ranked.columns = [
            "_".join([part for part in col if part]).rstrip("_") if isinstance(col, tuple) else col
            for col in ranked.columns
        ]
        if "rmse_mean" in ranked.columns:
            ranked = ranked.sort_values("rmse_mean", ascending=True)
        ranked.to_csv(output_root / "aggregate_test_metrics_ranked.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Extract best Optuna hparams and retrain KcatNet jobs in parallel."
    )
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU ids, e.g. --gpus 0 1 2 3")
    parser.add_argument("--runs_per_gpu", type=int, default=1, help="How many experiments to run concurrently on each listed GPU.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--config_path", type=str, default=str(REPO_ROOT / "config_KcatNet.json"))
    parser.add_argument("--output_root", type=str, default=None)

    parser.add_argument("--study_name", type=str, default="kcatnet_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=None)

    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument(
        "--all_thresholds",
        action="store_true",
        help="Use every discovered threshold under the requested split groups.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=256)
    parser.add_argument("--lazy_ligands", action="store_true")
    parser.add_argument("--predict_batch_size", type=int, default=None)

    # Optional explicit overrides over extracted best hparams.
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default=None)
    parser.add_argument("--lr_decay_factor", type=float, default=None)
    parser.add_argument("--lr_decay_patience", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--lr_warmup_epochs", type=int, default=None)
    parser.add_argument("--lr_warmup_start_factor", type=float, default=None)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--regression_weight", type=float, default=None)
    parser.add_argument("--cluster_weight", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--amsgrad", action="store_true")

    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--prot_t5_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--prot_t5_max_residues", type=int, default=4000)
    parser.add_argument("--prot_t5_max_seq_len", type=int, default=1000)
    parser.add_argument("--prot_t5_max_batch", type=int, default=32)
    parser.add_argument("--ligand_batch_size", type=int, default=256)

    parser.add_argument("--overwrite", action="store_true", help="Rerun experiments even if outputs already exist.")
    parser.add_argument(
        "--live_train_logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream train subprocess output to terminal (shows Rich progress) while still writing train.log.",
    )
    parser.add_argument(
        "--live_predict_logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream prediction subprocess output to terminal while still writing predict_*.log files.",
    )
    args = parser.parse_args()

    if args.runs_per_gpu <= 0:
        parser.error("--runs_per_gpu must be a positive integer.")

    if not args.hparams_json and not args.storage:
        parser.error("Provide either --hparams_json or --storage to select best hyperparameters.")

    args.thresholds = None if args.all_thresholds else normalize_threshold_args(args.thresholds, args.threshold)
    args.model_config = load_training_config(args.config_path)
    args.config_optimizer_defaults = optimizer_defaults_from_config(args.model_config)

    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")

    output_root = Path(args.output_root) if args.output_root else Path(args.base_dir) / "retrain_from_optuna" / args.study_name
    _ensure_dir(output_root)

    raw_hparams, source_meta = _load_best_hparams(args)
    resolved_hparams = _resolve_training_hparams(raw_hparams, args)
    _write_json(
        output_root / "selected_hparams.json",
        {
            "source": source_meta,
            "raw_hparams": raw_hparams,
            "resolved_train_hparams": resolved_hparams,
            "epochs": args.epochs,
            "config_path": args.config_path,
            "config_optimizer_defaults": args.config_optimizer_defaults,
        },
    )

    maybe_cache_embeddings(args)

    experiments = _build_experiments(jobs, args.seeds, output_root)
    pd.DataFrame(
        [
            {
                "split_group": exp["split_group"],
                "split_name": exp["split_name"],
                "difficulty": exp["difficulty"],
                "seed": exp["seed"],
                "run_dir": str(exp["run_dir"]),
                "train_path": exp["train_path"],
                "val_path": exp["val_path"],
                "test_path": exp["test_path"],
            }
            for exp in experiments
        ]
    ).to_csv(output_root / "planned_runs.csv", index=False)

    print(f"Discovered {len(experiments)} training runs.")
    print(f"Output root: {output_root}")
    print(f"Selected hparams: {resolved_hparams}")

    results = _run_parallel(experiments, args, resolved_hparams)
    _write_global_summaries(output_root, results)

    failed = [row for row in results if row.get("status") == "failed"]
    print(f"Completed runs: {sum(row.get('status') == 'completed' for row in results)}")
    print(f"Skipped existing runs: {sum(row.get('status') == 'skipped_exists' for row in results)}")
    print(f"Failed runs: {len(failed)}")
    if failed:
        print("Failed run details:")
        for row in failed:
            print(f"- {row['split_group']}/{row['split_name']}/seed_{row['seed']}: {row['error']}")
        raise RuntimeError("One or more retraining runs failed. Check run-specific logs in each output directory.")


if __name__ == "__main__":
    main()
