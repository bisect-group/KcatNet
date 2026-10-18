"""Train KcatNet on one explicit train/val/test split with the original training recipe.

Recipe (utils/trainer.py and models/model_kcat.py of the original code):
AdamW (lr from config_KcatNet.json, default weight decay), per-iteration cosine decay of the
learning rate to min_lrate, gradient clipping, loss 0.9 * MSE + 0.1 * mincut loss, validation
every epoch, best epoch by validation RMSE, no early stopping, fp32 (`--amp` switches to bf16/fp16 autocast).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT.parents[1]))
    from src.utils.rich_progress import progress

from emulator_bench.common import (
    CACHE_VERSION,
    DEFAULT_BASE_DIR,
    DEFAULT_CONFIG_PATH,
    read_table,
    require_columns,
    resolve_amp_dtype,
    resolve_single_split_job,
    save_json,
    set_seed,
)
from emulator_bench.dataset import CachedKcatDataset, LigandEmbeddingStore, ProteinEmbeddingStore, get_or_compute_pna_degrees
from models.model_kcat import KcatNet
from metrics import evaluate_reg

# The original code builds AdamW without weight_decay, so the PyTorch default applies.
WEIGHT_DECAY = 1e-2
REGRESSION_WEIGHT = 0.9
CLUSTER_WEIGHT = 0.1
METRIC_KEYS = ["rmse", "pearson", "spearman", "r2_score", "mae", "mse"]


def _make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        follow_batch=["mol_x", "prot_node_esm"],
    )


def _forward_model(model, batch, device):
    batch = batch.to(device)
    return model(
        mol_x=batch.mol_x,
        mol_x_feat=batch.mol_x_feat,
        mol_total_fea=batch.mol_total_fea,
        residue_esm=batch.prot_node_esm,
        residue_prot5=batch.prot_node_prot5,
        residue_edge_index=batch.prot_edge_index,
        residue_edge_weight=batch.prot_edge_weight,
        mol_batch=batch.mol_x_batch,
        prot_batch=batch.prot_node_esm_batch,
    )


def evaluate_loader(model, loader, device, amp_dtype=None):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                reg_pred, _ = _forward_model(model, batch, device)
            preds.append(reg_pred.float().squeeze().reshape(-1).cpu())
            truths.append(batch.reg_y.squeeze().reshape(-1).cpu())
    preds_np = torch.cat(preds).numpy()
    truths_np = torch.cat(truths).numpy()
    return preds_np, truths_np, evaluate_reg(truths_np, preds_np)


def _build_model(model_config, prot_deg, device):
    keys = [
        "mol_in_channels", "prot_in_channels", "prot_evo_channels", "hidden_channels", "pre_layers",
        "post_layers", "aggregators", "scalers", "total_layer", "K", "heads", "dropout", "dropout_attn_score",
    ]
    return KcatNet(prot_deg.to(device), device=device, **{key: model_config[key] for key in keys}).to(device)


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    amp_dtype = resolve_amp_dtype(device) if args.amp else None
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for name, path in (("train", args.train_path), ("val", args.val_path), ("test", args.test_path)):
        frames[name] = read_table(Path(path))
        require_columns(frames[name], [args.sequence_col, args.smiles_col, args.target_col], Path(path))
    all_frames = pd.concat(frames.values(), ignore_index=True)
    protein_store = ProteinEmbeddingStore(
        Path(args.embeddings_dir),
        sequences=all_frames[args.sequence_col].astype(str).tolist(),
        preload=args.preload_proteins,
        max_items=args.protein_cache_items,
    )
    ligand_store = LigandEmbeddingStore(Path(args.embeddings_dir), smiles_values=all_frames[args.smiles_col].astype(str).tolist())
    datasets = {
        name: CachedKcatDataset(
            frame,
            protein_store=protein_store,
            ligand_store=ligand_store,
            sequence_col=args.sequence_col,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
        )
        for name, frame in frames.items()
    }
    train_loader = _make_loader(datasets["train"], args.batch_size, True, args.num_workers)
    val_loader = _make_loader(datasets["val"], args.batch_size, False, args.num_workers)
    test_loader = _make_loader(datasets["test"], args.batch_size, False, args.num_workers)

    with open(args.config_path, "r") as handle:
        config = json.load(handle)
    prot_deg = get_or_compute_pna_degrees(
        datasets["train"],
        train_path=Path(args.train_path),
        cache_root=Path(args.embeddings_dir) / f"pna_degrees_v{CACHE_VERSION}",
    )
    model = _build_model(config["params"], prot_deg, device)

    opt_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lrate"],
        weight_decay=WEIGHT_DECAY,
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
        amsgrad=opt_cfg["amsgrad"],
    )
    total_iters = args.epochs * len(train_loader)
    lr_max, lr_min = opt_cfg["lrate"], opt_cfg["min_lrate"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda it: (lr_min + 0.5 * (1.0 + math.cos(math.pi * min(it, total_iters) / total_iters)) * (lr_max - lr_min)) / lr_max,
    )
    mse_loss = torch.nn.MSELoss(reduction="mean")

    history = []
    best_val_rmse = float("inf")
    best_model_path = out_dir / "bestmodel.pth"
    started = time.time()
    for epoch in progress(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        model.train()
        running_reg_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                reg_pred, cluster_loss = _forward_model(model, batch, device)
            reg_loss = mse_loss(reg_pred.float().squeeze().reshape(-1), batch.reg_y.squeeze().reshape(-1))
            loss = REGRESSION_WEIGHT * reg_loss + CLUSTER_WEIGHT * cluster_loss.float()
            if scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["clip"])
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["clip"])
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            running_reg_loss += float(reg_loss.item())

        _, _, val_metrics = evaluate_loader(model, val_loader, device, amp_dtype)
        history.append(
            {
                "epoch": epoch,
                "train_rmse": float(np.sqrt(running_reg_loss / len(train_loader))),
                "val_rmse": val_metrics["rmse"],
                "val_pearson": val_metrics["pearson"],
                "val_r2_score": val_metrics["r2_score"],
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        pd.DataFrame(history).to_csv(out_dir / "logfile.csv", index=False)
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": config["params"],
                    "prot_deg": prot_deg.cpu(),
                    "best_val_rmse": best_val_rmse,
                },
                best_model_path,
            )

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model = _build_model(checkpoint["model_config"], checkpoint["prot_deg"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    for name, loader in (("val", val_loader), ("test", test_loader)):
        preds, truths, metrics = evaluate_loader(model, loader, device, amp_dtype)
        pd.DataFrame([{key: metrics[key] for key in METRIC_KEYS}]).to_csv(out_dir / f"final_results_{name}.csv", index=False)
        pd.DataFrame({"pred_log10_value": preds, "label_log10_value": truths}).to_csv(out_dir / f"pred_label_{name}.csv", index=False)
        if name == "test":
            test_rmse = float(metrics["rmse"])

    pd.DataFrame(
        [
            {
                "task_name": args.task_name,
                "train_size": len(datasets["train"]),
                "val_size": len(datasets["val"]),
                "test_size": len(datasets["test"]),
                "best_epoch": int(checkpoint["epoch"]),
                "best_val_rmse": float(checkpoint["best_val_rmse"]),
                "test_rmse": test_rmse,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "elapsed_seconds": round(time.time() - started, 1),
            }
        ]
    ).to_csv(out_dir / "run_summary.csv", index=False)
    save_json(out_dir / "run_state.json", {"task_name": args.task_name, "status": "completed", "train_args": vars(args)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KcatNet with the original recipe on explicit train/val/test splits.")
    parser.add_argument("--train_path", default=None, type=str)
    parser.add_argument("--val_path", default=None, type=str)
    parser.add_argument("--test_path", default=None, type=str)
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR), type=str)
    parser.add_argument("--split_group", default=None, type=str)
    parser.add_argument("--threshold", default=None, type=str)
    parser.add_argument("--embeddings_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--task_name", default="kcatnet", type=str)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG_PATH), type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)
    parser.add_argument("--target_col", default="log10_value", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Mixed precision: bf16 on Ampere or newer, fp16 on older GPUs. Default is the original fp32.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=256)

    args = parser.parse_args()
    if not all([args.train_path, args.val_path, args.test_path]):
        if not args.split_group:
            parser.error("Provide --train_path/--val_path/--test_path or --split_group with optional --threshold.")
        job = resolve_single_split_job(Path(args.base_dir), args.split_group, threshold=args.threshold)
        args.train_path, args.val_path, args.test_path = job["train_path"], job["val_path"], job["test_path"]
    main(args)
