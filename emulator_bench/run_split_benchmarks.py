import argparse
import json
import subprocess
import sys
from pathlib import Path

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
    split_sizes,
    summarize_seed_runs,
)

CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def _metric_sort_ascending(metric: str) -> bool:
    return metric in {"rmse", "mse", "mae"}


def maybe_load_hparams(args):
    if not args.hparams_json:
        return args

    with open(args.hparams_json, "r") as handle:
        hparams = json.load(handle)
    for key in [
        "batch_size",
        "lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
        "amsgrad",
        "scheduler",
        "lr_decay_factor",
        "lr_decay_patience",
        "min_lr",
        "lr_warmup_epochs",
        "lr_warmup_start_factor",
        "clip_grad",
        "patience",
        "min_delta",
    ]:
        if key in hparams:
            setattr(args, key, hparams[key])
    return args


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


def run_training(job, seed, args):
    result_root = Path(job["root_dir"]) / "kcatnet_results" / f"seed_{seed}"
    result_root.mkdir(parents=True, exist_ok=True)
    final_test_path = result_root / "final_results_test.csv"
    if final_test_path.exists() and not args.overwrite:
        return result_root

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
        str(result_root),
        "--task_name",
        f"{job['split_group']}_{job['split_name']}_seed{seed}",
        "--config_path",
        args.config_path,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--beta1",
        str(args.beta1),
        "--beta2",
        str(args.beta2),
        "--eps",
        str(args.eps),
        "--scheduler",
        args.scheduler,
        "--lr_decay_factor",
        str(args.lr_decay_factor),
        "--lr_decay_patience",
        str(args.lr_decay_patience),
        "--min_lr",
        str(args.min_lr),
        "--lr_warmup_epochs",
        str(args.lr_warmup_epochs),
        "--lr_warmup_start_factor",
        str(args.lr_warmup_start_factor),
        "--clip_grad",
        str(args.clip_grad),
        "--regression_weight",
        str(args.regression_weight),
        "--cluster_weight",
        str(args.cluster_weight),
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
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
    if args.amsgrad:
        cmd.append("--amsgrad")
    if args.lazy_ligands:
        cmd.append("--lazy_ligands")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return result_root


def main():
    parser = argparse.ArgumentParser(description="Run KcatNet emulator bench across random/enzyme/substrate splits.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--config_path", type=str, default=str(REPO_ROOT / "config_KcatNet.json"))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--ddp_backend", choices=["auto", "nccl", "gloo"], default="auto")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--hparams_json", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="none")
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_warmup_epochs", type=int, default=0)
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--regression_weight", type=float, default=0.9)
    parser.add_argument("--cluster_weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=256)
    parser.add_argument("--lazy_ligands", action="store_true")

    parser.add_argument("--prot_t5_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--prot_t5_max_residues", type=int, default=4000)
    parser.add_argument("--prot_t5_max_seq_len", type=int, default=1000)
    parser.add_argument("--prot_t5_max_batch", type=int, default=32)
    parser.add_argument("--ligand_batch_size", type=int, default=256)

    parser.add_argument("--primary_metric", type=str, default="rmse", choices=["rmse", "pearson", "spearman", "r2_score", "mae", "mse"])
    args = parser.parse_args()
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args = maybe_load_hparams(args)

    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")

    print(f"Discovered {len(jobs)} jobs")
    for job in jobs:
        print(f"- {job['split_group']} / {job['split_name']} ({job['difficulty']})")
    if args.dry_run:
        return

    maybe_cache_embeddings(args)

    run_rows = []
    for job in progress(jobs, desc="KcatNet benchmark", unit="job"):
        size_meta = split_sizes(Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"]))
        for seed in args.seeds:
            result_root = run_training(job, seed, args)
            final_test_path = result_root / "final_results_test.csv"
            final_val_path = result_root / "final_results_val.csv"
            run_summary_path = result_root / "run_summary.csv"

            if not final_test_path.exists():
                continue

            row = pd.read_csv(final_test_path).iloc[0].to_dict()
            val_row = pd.read_csv(final_val_path).iloc[0].to_dict() if final_val_path.exists() else {}
            run_summary = pd.read_csv(run_summary_path).iloc[0].to_dict() if run_summary_path.exists() else {}

            row.update({f"val_{key}": value for key, value in val_row.items()})
            row.update(run_summary)
            row.update(size_meta)
            row["split_group"] = job["split_group"]
            row["split_name"] = job["split_name"]
            row["difficulty"] = job["difficulty"]
            row["seed"] = seed
            row["results_dir"] = str(result_root)
            run_rows.append(row)

    runs_path = Path(args.base_dir) / "kcatnet_summary_runs.csv"
    pd.DataFrame(run_rows).to_csv(runs_path, index=False)

    metric_cols = ["rmse", "pearson", "spearman", "r2_score", "mae", "mse"]
    threshold_df = summarize_seed_runs(
        run_rows,
        group_cols=["split_group", "split_name", "difficulty"],
        metric_cols=metric_cols,
    )
    threshold_path = Path(args.base_dir) / "kcatnet_summary_thresholds.csv"
    threshold_df.to_csv(threshold_path, index=False)

    by_group_rows = []
    for split_group, group in threshold_df.groupby("split_group", sort=False):
        row = {"split_group": split_group, "n_splits": len(group)}
        for metric in metric_cols:
            mean_col = f"{metric}_mean"
            if mean_col in group.columns:
                row[f"{metric}_mean_over_splits"] = float(group[mean_col].mean())
                row[f"{metric}_var_over_splits"] = float(group[mean_col].var(ddof=1)) if len(group) > 1 else 0.0
        by_group_rows.append(row)
    by_group_df = pd.DataFrame(by_group_rows)
    by_group_path = Path(args.base_dir) / "kcatnet_summary_by_split_group.csv"
    by_group_df.to_csv(by_group_path, index=False)

    ranked_df = threshold_df.sort_values(
        f"{args.primary_metric}_mean",
        ascending=_metric_sort_ascending(args.primary_metric),
    )
    ranked_path = Path(args.base_dir) / "kcatnet_summary_ranked.csv"
    ranked_df.to_csv(ranked_path, index=False)

    print(f"Saved runs summary: {runs_path}")
    print(f"Saved split summary: {threshold_path}")
    print(f"Saved group summary: {by_group_path}")
    print(f"Saved ranked summary: {ranked_path}")


if __name__ == "__main__":
    main()
