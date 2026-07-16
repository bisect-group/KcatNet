import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
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
UTILS_ROOT = REPO_ROOT / "utils"
if str(UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(UTILS_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, read_table, resolve_single_split_job
from emulator_bench.dataset import CachedKcatDataset, LigandEmbeddingStore, ProteinEmbeddingStore
from emulator_bench.train_single_target_tvt import (
    _autocast_context,
    _forward_model,
    _load_model_from_checkpoint,
    _make_loader,
    _resolve_mixed_precision,
)
from metrics import evaluate_reg


def main(args):
    device = torch.device(args.device)
    autocast_dtype, precision_mode, _ = _resolve_mixed_precision(device)
    input_path = Path(args.input_path)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    frame = read_table(input_path)
    required_cols = [args.sequence_col, args.smiles_col]
    for col in required_cols:
        if col not in frame.columns:
            raise ValueError(f"Missing required column `{col}` in {input_path}")

    protein_store = ProteinEmbeddingStore(Path(args.embeddings_dir), max_items=args.protein_cache_items)
    ligand_store = LigandEmbeddingStore(
        Path(args.embeddings_dir),
        smiles_values=frame[args.smiles_col].astype(str).tolist(),
        preload=not args.lazy_ligands,
    )
    target_col = args.target_col if args.target_col in frame.columns else None
    dataset = CachedKcatDataset(
        frame,
        protein_store=protein_store,
        ligand_store=ligand_store,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        target_col=target_col,
    )
    device_pin_memory = args.pin_memory or (device.type == "cuda")
    loader = _make_loader(
        dataset,
        args.batch_size,
        False,
        args.num_workers,
        pin_memory=device_pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model, checkpoint = _load_model_from_checkpoint(Path(args.ckpt_path), device)
    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for batch in progress(loader, desc="Predict", unit="batch"):
            with _autocast_context(device, autocast_dtype=autocast_dtype):
                reg_pred, _ = _forward_model(model, batch, device)
            # NumPy cannot consume bfloat16 tensors directly; cast before host conversion.
            pred_values = reg_pred.squeeze().detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
            preds.extend(pred_values.tolist())
            if target_col is not None:
                label_values = batch.reg_y.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
                labels.extend(label_values.tolist())

    output = frame.copy()
    output["pred_log10_value"] = preds
    output["pred_value"] = np.power(10.0, output["pred_log10_value"])
    output.to_csv(out_csv, index=False)

    if target_col is not None:
        metrics = evaluate_reg(np.asarray(labels, dtype=np.float32), np.asarray(preds, dtype=np.float32))
        metrics["precision_mode"] = precision_mode
        pd.DataFrame([metrics]).to_csv(out_csv.with_name(out_csv.stem + "_metrics.csv"), index=False)
        print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained KcatNet emulator-bench checkpoint on a CSV/parquet split.")
    parser.add_argument("--input_path", default=None, type=str)
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR), type=str)
    parser.add_argument("--split_group", default=None, type=str)
    parser.add_argument("--threshold", default=None, type=str)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], type=str)
    parser.add_argument("--embeddings_dir", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--out_csv", required=True, type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)
    parser.add_argument("--target_col", default="log10_value", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--protein_cache_items", default=256, type=int)
    parser.add_argument("--lazy_ligands", action="store_true")
    args = parser.parse_args()
    if not args.input_path:
        if not args.split_group:
            parser.error("Provide either --input_path or --split_group with optional --threshold.")
        job = resolve_single_split_job(Path(args.base_dir), args.split_group, threshold=args.threshold)
        args.input_path = job[f"{args.split}_path"]
    main(args)
