import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EMULATOR_ROOT = REPO_ROOT.parents[1]
DEFAULT_BASE_DIR = EMULATOR_ROOT / "data" / "processed" / "datasets" / "kinetic_dataset" / "kcat"
DEFAULT_OUT_DIR = EMULATOR_ROOT / "outputs" / "baselines" / "KcatNet"
DEFAULT_EMBEDDINGS_DIR = DEFAULT_OUT_DIR / "embeddings"
CACHE_VERSION = 3
PROT_T5_MODEL = "Rostlab/prot_t5_xl_uniref50"
PROT_T5_DTYPE = "float16"
ESM_MODEL = "esm2_t33_650M_UR50D"
ESM_LAYER = 33
DEFAULT_SPLIT_GROUPS = [
    "random_splits",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "substrate_splits",
]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config_KcatNet.json"
RANDOM_SPLIT_GROUP_ALIAS = "random_splits"
RANDOM_SPLIT_GROUP_PREFIX = "random_splits_grouped_"


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_sequence(sequence: str, max_len: int = 1000) -> str:
    seq = str(sequence).strip().upper()[:max_len]
    return seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")


def protein_cache_key(sequence: str, max_len: int = 1000) -> str:
    settings = f"{PROT_T5_MODEL}|{PROT_T5_DTYPE}|{ESM_MODEL}|layer={ESM_LAYER}|max_len={max_len}"
    return _stable_hash(f"{settings}|{normalize_sequence(sequence, max_len=max_len)}")


def ligand_cache_key(smiles: str) -> str:
    return _stable_hash(str(smiles).strip())


def protein_cache_path(embeddings_dir: Path, sequence: str, max_len: int = 1000) -> Path:
    key = protein_cache_key(sequence, max_len=max_len)
    return embeddings_dir / f"proteins_v{CACHE_VERSION}" / key[:2] / f"{key}.npz"


def ligand_cache_path(embeddings_dir: Path, smiles: str) -> Path:
    key = ligand_cache_key(smiles)
    return embeddings_dir / f"ligands_v{CACHE_VERSION}" / key[:2] / f"{key}.npz"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def load_json(path: Path) -> Dict:
    with open(path, "r") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(device: torch.device):
    """bf16 on Ampere or newer, fp16 on older GPUs, None (fp32) on CPU."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def _threshold_value(name: str) -> float:
    try:
        return float(name.split("threshold_")[-1])
    except Exception:
        return math.inf


def _difficulty_labels_for_thresholds(names: List[str]) -> Dict[str, str]:
    ordered = sorted(names, key=_threshold_value)
    if len(ordered) == 1:
        return {ordered[0]: "single"}
    if len(ordered) == 2:
        return {
            ordered[0]: "hard",
            ordered[1]: "easy",
        }
    if len(ordered) == 3:
        return {
            ordered[0]: "hard",
            ordered[1]: "medium",
            ordered[2]: "easy",
        }
    labels = {}
    for rank, name in enumerate(ordered, start=1):
        labels[name] = f"rank_{rank}"
    return labels


def _flat_split_label(split_group: str) -> str:
    if is_random_split_group(split_group):
        return "random"
    if split_group.endswith("_splits"):
        return split_group[: -len("_splits")]
    return split_group


def is_random_split_group(split_group: str) -> bool:
    split_group = str(split_group)
    return split_group == RANDOM_SPLIT_GROUP_ALIAS or split_group.startswith(RANDOM_SPLIT_GROUP_PREFIX)


def expand_split_groups(base_dir: Path, split_groups: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    seen = set()
    grouped_random_dirs: Optional[List[str]] = None

    def add(split_group: str) -> None:
        if split_group not in seen:
            seen.add(split_group)
            expanded.append(split_group)

    for split_group in split_groups:
        split_group = str(split_group)
        if split_group != RANDOM_SPLIT_GROUP_ALIAS:
            add(split_group)
            continue

        if grouped_random_dirs is None:
            grouped_random_dirs = sorted(
                child.name
                for child in Path(base_dir).glob(f"{RANDOM_SPLIT_GROUP_PREFIX}*")
                if child.is_dir()
            )

        if grouped_random_dirs:
            for grouped_split_group in grouped_random_dirs:
                add(grouped_split_group)
        elif (Path(base_dir) / RANDOM_SPLIT_GROUP_ALIAS).exists():
            add(RANDOM_SPLIT_GROUP_ALIAS)

    return expanded


def normalize_threshold_args(thresholds: Optional[Iterable[str]] = None, threshold: Optional[str] = None) -> Optional[List[str]]:
    values: List[str] = []
    if thresholds is not None:
        values.extend([str(value) for value in thresholds if str(value).strip()])
    if threshold is not None and str(threshold).strip():
        values.append(str(threshold))
    if not values:
        return None
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def discover_split_jobs(
    base_dir: Path,
    split_groups: Optional[Iterable[str]] = None,
    thresholds: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    split_groups = expand_split_groups(Path(base_dir), split_groups or DEFAULT_SPLIT_GROUPS)
    threshold_filter = list(thresholds) if thresholds is not None else None
    jobs: List[Dict[str, str]] = []

    for split_group in split_groups:
        group_dir = base_dir / split_group
        if not group_dir.exists():
            continue

        train_path = _find_split_file(group_dir, "train")
        val_path = _find_split_file(group_dir, "val")
        test_path = _find_split_file(group_dir, "test")
        if train_path and val_path and test_path:
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": "all",
                    "difficulty": _flat_split_label(split_group),
                    "root_dir": str(group_dir),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                }
            )
            continue

        candidate_dirs = []
        for child in sorted(group_dir.iterdir()):
            if not child.is_dir():
                continue
            if threshold_filter is not None and child.name not in threshold_filter:
                continue
            if child.name.startswith("threshold_") or child.name in {"easy", "medium", "hard"}:
                candidate_dirs.append(child)

        threshold_names = [path.name for path in candidate_dirs if path.name.startswith("threshold_")]
        threshold_difficulties = _difficulty_labels_for_thresholds(threshold_names)

        for child in candidate_dirs:
            train_path = _find_split_file(child, "train")
            val_path = _find_split_file(child, "val")
            test_path = _find_split_file(child, "test")
            if not (train_path and val_path and test_path):
                continue

            difficulty = child.name
            if child.name.startswith("threshold_"):
                difficulty = threshold_difficulties.get(child.name, child.name)

            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": child.name,
                    "difficulty": difficulty,
                    "root_dir": str(child),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                }
            )

    return jobs


def resolve_single_split_job(base_dir: Path, split_group: str, threshold: Optional[str] = None) -> Dict[str, str]:
    threshold_filter = None if is_random_split_group(split_group) else normalize_threshold_args(threshold=threshold)
    jobs = discover_split_jobs(base_dir, split_groups=[split_group], thresholds=threshold_filter)
    if not jobs:
        detail = f"{split_group}/{threshold}" if threshold else split_group
        raise FileNotFoundError(f"No split job discovered for {detail} in {base_dir}")
    if split_group == RANDOM_SPLIT_GROUP_ALIAS and len(jobs) > 1:
        available = ", ".join(job["split_group"] for job in jobs)
        raise ValueError(
            f"Multiple grouped random split jobs found for {split_group}. Specify one of: {available}"
        )
    if is_random_split_group(split_group) or len(jobs) == 1:
        return jobs[0]
    if threshold is None:
        available = ", ".join(job["split_name"] for job in jobs)
        raise ValueError(
            f"Multiple thresholded jobs found for {split_group}. Specify --threshold. Available: {available}"
        )
    matching = [job for job in jobs if job["split_name"] == threshold]
    if not matching:
        available = ", ".join(job["split_name"] for job in jobs)
        raise FileNotFoundError(
            f"Threshold `{threshold}` not found for {split_group}. Available: {available}"
        )
    return matching[0]


def _find_split_file(directory: Path, stem: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def split_sizes(train_path: Path, val_path: Path, test_path: Path) -> Dict[str, float]:
    train_size = len(read_table(train_path))
    val_size = len(read_table(val_path))
    test_size = len(read_table(test_path))
    total = train_size + val_size + test_size
    if total == 0:
        return {
            "train_size": 0,
            "val_size": 0,
            "test_size": 0,
            "train_ratio": 0.0,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
        }
    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_ratio": train_size / total,
        "val_ratio": val_size / total,
        "test_ratio": test_size / total,
    }


def summarize_seed_runs(
    rows: List[Dict],
    group_cols: Iterable[str],
    metric_cols: Iterable[str],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    runs_df = pd.DataFrame(rows)
    out_rows = []
    for keys, group in runs_df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        for col in runs_df.columns:
            if col in row or col in metric_cols or col == "seed":
                continue
            if col.endswith("_dir"):
                continue
            values = group[col].dropna()
            if len(values) == 0:
                continue
            if pd.api.types.is_numeric_dtype(values):
                row[col] = values.iloc[0]
            else:
                row[col] = values.iloc[0]
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            metric_values = group[metric].dropna()
            if len(metric_values) == 0:
                continue
            row[f"{metric}_mean"] = float(metric_values.mean())
            row[f"{metric}_var"] = float(metric_values.var(ddof=1)) if len(metric_values) > 1 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def write_csv(path: Path, rows: List[Dict]) -> None:
    ensure_parent(path)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)
