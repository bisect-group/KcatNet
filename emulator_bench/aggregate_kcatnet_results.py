"""Aggregate KcatNet results across seeds.

Reads <out_dir>/<task>/<split_group>/<split_name>/seed_N/final_results_{val,test}.csv and writes
<out_dir>/aggregate_kcatnet_results.csv with mean and std per metric for every split.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emulator_bench.common import DEFAULT_OUT_DIR

METRICS = ["rmse", "pearson", "spearman", "r2_score", "mae", "mse"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = []
    for seed_dir in sorted(args.out_dir.glob("*/*/*/seed_*")):
        task, split_group, split_name = seed_dir.parts[-4:-1]
        for tvt_split in ("val", "test"):
            path = seed_dir / f"final_results_{tvt_split}.csv"
            if path.exists():
                rows.append({"task": task, "split_group": split_group, "split_name": split_name, "tvt_split": tvt_split, **pd.read_csv(path).iloc[0].to_dict()})
    if not rows:
        print("No complete results found.")
        return

    group_keys = ["task", "split_group", "split_name", "tvt_split"]
    grouped = pd.DataFrame(rows).groupby(group_keys)
    agg = grouped[METRICS].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_seeds"] = grouped.size()
    out_path = args.out_dir / "aggregate_kcatnet_results.csv"
    agg.reset_index().to_csv(out_path, index=False)
    print(f"Saved {len(agg)} rows to {out_path}")


if __name__ == "__main__":
    main()
