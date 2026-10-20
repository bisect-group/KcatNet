"""Run KcatNet with its original settings on every discovered split, for every seed."""
import argparse
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT.parents[1]))
    from src.utils.rich_progress import progress

from emulator_bench.esm_deps import setup_esm
from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUT_DIR,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    normalize_threshold_args,
)

CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def cache_embeddings(args):
    cmd = [
        sys.executable, str(CACHE_SCRIPT),
        "--base_dir", args.base_dir,
        "--embeddings_dir", args.embeddings_dir,
        "--sequence_col", args.sequence_col,
        "--smiles_col", args.smiles_col,
        "--device", args.cache_device,
        "--split_groups", *args.split_groups,
    ]
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpus[0])} if args.gpus else None
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def training_command(job, seed, args):
    result_dir = Path(args.out_dir) / Path(args.base_dir).name / job["split_group"] / job["split_name"] / f"seed_{seed}"
    if (result_dir / "final_results_test.csv").exists() and not args.overwrite:
        return None
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--train_path", job["train_path"],
        "--val_path", job["val_path"],
        "--test_path", job["test_path"],
        "--embeddings_dir", args.embeddings_dir,
        "--out_dir", str(result_dir),
        "--task_name", f"{job['split_group']}_{job['split_name']}_seed{seed}",
        "--config_path", args.config_path,
        "--sequence_col", args.sequence_col,
        "--smiles_col", args.smiles_col,
        "--target_col", args.target_col,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--seed", str(seed),
        "--device", "cuda:0" if args.gpus else args.device,
        "--num_workers", str(args.num_workers),
    ]
    return cmd + ["--amp"] if args.amp else cmd


def run_training(cmd, slots):
    gpu = slots.get()
    try:
        env = None if gpu is None else {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    finally:
        slots.put(gpu)


def main():
    parser = argparse.ArgumentParser(description="Evaluate KcatNet (original settings) across the EMULaToR kcat splits and seeds.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR), help="kcat dataset root holding the split folders")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=None, help="default: <out_dir>/embeddings")
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666, 777, 888])
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device when --gpus is not set.")
    parser.add_argument("--gpus", nargs="+", default=None, help="Physical GPU ids, e.g. --gpus 0 1 2. Each run sees one GPU as cuda:0. The cache step uses the first.")
    parser.add_argument("--runs_per_gpu", type=int, default=1, help="Concurrent runs per GPU (per --device when --gpus is not set).")
    parser.add_argument("--amp", action="store_true", help="Mixed precision: bf16 on Ampere or newer, fp16 on older GPUs. Default is fp32.")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.thresholds = normalize_threshold_args(args.thresholds)
    setup_esm(Path(args.out_dir))  # child processes inherit PYTHONPATH and TORCH_HOME
    args.embeddings_dir = args.embeddings_dir or str(Path(args.out_dir) / "embeddings")

    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    print(f"Discovered {len(jobs)} jobs")
    for job in jobs:
        print(f"- {job['split_group']} / {job['split_name']}")
    if args.dry_run:
        return

    if not args.skip_cache:
        cache_embeddings(args)

    commands = [cmd for job in jobs for seed in args.seeds if (cmd := training_command(job, seed, args))]
    slots = queue.Queue()
    for gpu in args.gpus or [None]:
        for _ in range(args.runs_per_gpu):
            slots.put(None if gpu is None else str(gpu))
    with ThreadPoolExecutor(slots.qsize()) as pool:
        list(progress(pool.map(lambda cmd: run_training(cmd, slots), commands), desc="KcatNet benchmark", unit="run", total=len(commands)))
    print(f"Results are in {Path(args.out_dir) / Path(args.base_dir).name}. Run aggregate_kcatnet_results.py to summarize seeds.")


if __name__ == "__main__":
    main()
