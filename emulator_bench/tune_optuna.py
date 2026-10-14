import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import optuna
import pandas as pd
try:
    from src.utils.rich_progress import progress, write
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress, write


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


CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def _metric_direction(metric: str) -> str:
    return "minimize" if metric in {"rmse", "mse", "mae"} else "maximize"


def _sqlite_path_from_storage(storage: str | None) -> Path | None:
    if not storage or not storage.startswith("sqlite:///"):
        return None
    parsed = urlparse(storage)
    if parsed.scheme != "sqlite":
        return None
    raw_path = unquote(parsed.path or "")
    if not raw_path:
        return None
    return Path(raw_path)


def _sqlite_has_optuna_schema(db_path: Path) -> bool:
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    return "version_info" in tables


def prepare_optuna_storage(args) -> None:
    db_path = _sqlite_path_from_storage(args.storage)
    if db_path is None:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        return
    if args.reset_storage:
        db_path.unlink()
        print(f"Removed existing Optuna storage: {db_path}")
        return
    if not _sqlite_has_optuna_schema(db_path):
        raise RuntimeError(
            "Optuna storage exists but does not contain a valid Optuna schema: "
            f"{db_path}. Use a new --storage path or rerun with --reset_storage."
        )


def maybe_cache_embeddings(args):
    if args.skip_cache:
        return
    cmd = [
        sys.executable,
        str(CACHE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--device",
        args.cache_device,
        "--prot_t5_model",
        args.prot_t5_model,
        "--protein_dtype",
        args.protein_dtype,
        "--prot_t5_max_residues",
        str(args.prot_t5_max_residues),
        "--prot_t5_max_seq_len",
        str(args.prot_t5_max_seq_len),
        "--prot_t5_max_batch",
        str(args.prot_t5_max_batch),
        "--ligand_batch_size",
        str(args.ligand_batch_size),
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.cache_overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def load_training_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def optimizer_defaults_from_config(config: dict) -> dict:
    optimizer_cfg = dict(config.get("optimizer", {}))
    betas = optimizer_cfg.get("betas", [0.9, 0.999])
    scheduler = "cosine" if optimizer_cfg.get("schedule_lr", False) else "none"
    return {
        "lr": float(optimizer_cfg.get("lrate", 1e-4)),
        "weight_decay": float(optimizer_cfg.get("weight_decay", 1e-2)),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(optimizer_cfg.get("eps", 1e-8)),
        "amsgrad": bool(optimizer_cfg.get("amsgrad", False)),
        "scheduler": scheduler,
        "min_lr": float(optimizer_cfg.get("min_lrate", 0.0)),
        "lr_warmup_epochs": 0,
        "lr_warmup_start_factor": 0.1,
        "clip_grad": float(optimizer_cfg.get("clip", 1.0)),
    }


def suggest_hparams(trial: optuna.Trial, args) -> dict:
    config_defaults = args.config_optimizer_defaults
    batch_size = int(args.batch_size) if args.batch_size is not None else trial.suggest_categorical("batch_size", [8, 16, 32])
    hparams = {
        "batch_size": batch_size,
        "lr": trial.suggest_float("lr", 3e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True),
        "min_lr": trial.suggest_float("min_lr", 1e-7, 1e-4, log=True),
        "lr_warmup_epochs": trial.suggest_int("lr_warmup_epochs", 0, 8),
        "lr_warmup_start_factor": trial.suggest_float("lr_warmup_start_factor", 0.05, 0.5),
        "clip_grad": trial.suggest_categorical("clip_grad", [0.5, 1.0, 2.0, 5.0]),
        "patience": trial.suggest_categorical("patience", [0, 10, 20, 30]),
        "scheduler": config_defaults["scheduler"],
        "beta1": config_defaults["beta1"],
        "beta2": config_defaults["beta2"],
        "eps": config_defaults["eps"],
        "amsgrad": config_defaults["amsgrad"],
        "regression_weight": 0.9,
        "cluster_weight": 0.1,
        "min_delta": 0.0,
        "lr_decay_factor": 0.5,
        "lr_decay_patience": 5,
    }
    return hparams


def run_trial_job(job: dict, seed: int, hparams: dict, args, trial_number: int) -> float:
    trial_root = (
        Path(job["root_dir"])
        / "kcatnet_optuna_runs"
        / f"trial_{trial_number}"
        / job["split_group"]
        / job["split_name"]
        / f"seed_{seed}"
    )
    trial_root.mkdir(parents=True, exist_ok=True)

    metric_file = trial_root / f"final_results_{args.eval_split}.csv"
    if not metric_file.exists() or args.overwrite_runs:
        cmd = []
        if args.ddp and args.nproc_per_node > 1:
            cmd.extend(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node",
                    str(args.nproc_per_node),
                    str(TRAIN_SCRIPT),
                ]
            )
        else:
            cmd.extend([sys.executable, str(TRAIN_SCRIPT)])
        cmd.extend(
            [
            "--train_path",
            job["train_path"],
            "--val_path",
            job["val_path"],
            "--test_path",
            job["test_path"],
            "--embeddings_dir",
            args.embeddings_dir,
            "--out_dir",
            str(trial_root),
            "--task_name",
            f"optuna_trial_{trial_number}_{job['split_group']}_{job['split_name']}_seed{seed}",
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
            str(seed),
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--prefetch_factor",
            str(args.prefetch_factor),
            "--val_every",
            str(args.val_every),
            "--protein_cache_items",
            str(args.protein_cache_items),
        ]
        )
        if args.persistent_workers:
            cmd.append("--persistent_workers")
        if args.pin_memory:
            cmd.append("--pin_memory")
        if args.preload_proteins:
            cmd.append("--preload_proteins")
        if args.ddp and args.nproc_per_node > 1:
            cmd.append("--ddp")
            cmd.extend(["--ddp_backend", args.ddp_backend])
        if hparams["amsgrad"]:
            cmd.append("--amsgrad")
        if args.lazy_ligands:
            cmd.append("--lazy_ligands")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    metrics = pd.read_csv(metric_file).iloc[0].to_dict()
    if args.metric not in metrics:
        raise RuntimeError(f"Metric `{args.metric}` not found in {metric_file}")
    return float(metrics[args.metric])


def main():
    parser = argparse.ArgumentParser(description="Optuna tuner for KcatNet emulator-bench TVT workflow.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--config_path", type=str, default=str(REPO_ROOT / "config_KcatNet.json"))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument(
        "--all_thresholds",
        action="store_true",
        help="Use every discovered threshold under the requested split groups.",
    )
    parser.add_argument("--max_jobs", type=int, default=0, help="Limit the number of split jobs used for each trial. 0 means all.")
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--ddp_backend", choices=["auto", "nccl", "gloo"], default="auto")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=256)
    parser.add_argument("--lazy_ligands", action="store_true")

    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")

    parser.add_argument("--prot_t5_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--prot_t5_max_residues", type=int, default=4000)
    parser.add_argument("--prot_t5_max_seq_len", type=int, default=1000)
    parser.add_argument("--prot_t5_max_batch", type=int, default=32)
    parser.add_argument("--ligand_batch_size", type=int, default=256)

    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "pearson", "spearman", "r2_score", "mae", "mse"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="kcatnet_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--reset_storage", action="store_true")
    args = parser.parse_args()
    args.thresholds = None if args.all_thresholds else normalize_threshold_args(args.thresholds, args.threshold)
    args.model_config = load_training_config(args.config_path)
    args.config_optimizer_defaults = optimizer_defaults_from_config(args.model_config)

    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    if args.max_jobs and args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    print(f"Using {len(jobs)} jobs for Optuna")
    for job in jobs:
        print(f"- {job['split_group']} / {job['split_name']} ({job['difficulty']})")

    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)

    studies_dir = Path(args.base_dir) / "optuna_studies"
    studies_dir.mkdir(parents=True, exist_ok=True)
    best_hparams_path = studies_dir / f"{args.study_name}_best_hparams.json"
    trials_csv_path = studies_dir / f"{args.study_name}_trials.csv"

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    try:
        study = optuna.create_study(
            direction=_metric_direction(args.metric),
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            sampler=sampler,
        )
    except AssertionError as exc:
        db_path = _sqlite_path_from_storage(args.storage)
        if db_path is not None:
            raise RuntimeError(
                "Optuna failed to open the SQLite storage. The DB likely has an incompatible or "
                f"corrupted schema: {db_path}. Use a new --storage path or rerun with --reset_storage."
            ) from exc
        raise

    def objective(trial: optuna.Trial) -> float:
        hparams = suggest_hparams(trial, args)
        scores = []
        pbar = progress(jobs, desc=f"Trial {trial.number}", unit="job", leave=False)
        for job in pbar:
            for seed in args.seeds:
                score = run_trial_job(job, seed, hparams, args, trial.number)
                scores.append(score)
                pbar.set_postfix(metric=f"{np_mean(scores):.4f}")
        return np_mean(scores)

    study.optimize(objective, n_trials=args.n_trials)

    best_payload = dict(study.best_params)
    best_payload.update(
        {
            "metric": args.metric,
            "eval_split": args.eval_split,
            "n_trials": args.n_trials,
            "epochs": args.epochs,
            "base_dir": args.base_dir,
            "embeddings_dir": args.embeddings_dir,
            "split_groups": list(args.split_groups),
            "thresholds": args.thresholds,
            "seeds": list(args.seeds),
            "batch_size": args.batch_size if args.batch_size is not None else study.best_params.get("batch_size"),
            "config_path": args.config_path,
            "config_optimizer_defaults": args.config_optimizer_defaults,
        }
    )
    best_payload.setdefault("scheduler", args.config_optimizer_defaults["scheduler"])
    best_payload.setdefault("beta1", args.config_optimizer_defaults["beta1"])
    best_payload.setdefault("beta2", args.config_optimizer_defaults["beta2"])
    best_payload.setdefault("eps", args.config_optimizer_defaults["eps"])
    best_payload.setdefault("amsgrad", args.config_optimizer_defaults["amsgrad"])
    best_payload.setdefault("clip_grad", args.config_optimizer_defaults["clip_grad"])
    best_payload.setdefault("lr_decay_factor", 0.5)
    best_payload.setdefault("lr_decay_patience", 5)

    with open(best_hparams_path, "w") as handle:
        json.dump(best_payload, handle, indent=2, sort_keys=True)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_csv_path, index=False)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value ({args.metric}): {study.best_value}")
    print(f"Saved best hparams: {best_hparams_path}")
    print(f"Saved trials csv: {trials_csv_path}")


def np_mean(values):
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()
