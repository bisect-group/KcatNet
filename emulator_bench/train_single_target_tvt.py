import argparse
import datetime
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_CONFIG_PATH,
    read_table,
    require_columns,
    resolve_single_split_job,
    save_json,
    set_seed,
)
from emulator_bench.dataset import CachedKcatDataset, LigandEmbeddingStore, ProteinEmbeddingStore, get_or_compute_pna_degrees
from models.model_kcat import KcatNet
from metrics import evaluate_reg


def _is_distributed(args) -> bool:
    return bool(int(os.environ.get("WORLD_SIZE", "1")) > 1 or (args.ddp and "LOCAL_RANK" in os.environ))


def _setup_distributed(args):
    distributed = _is_distributed(args)
    if not distributed:
        device = torch.device(args.device)
        return {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_main_process": True,
            "device": device,
        }

    if not dist.is_initialized():
        requested_device = torch.device(args.device)
        if args.ddp_backend == "auto":
            backend = "nccl" if requested_device.type == "cuda" else "gloo"
        else:
            backend = args.ddp_backend
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        init_kwargs = {"backend": backend}
        if backend == "nccl":
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(**init_kwargs)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return {
        "distributed": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main_process": rank == 0,
        "device": device,
    }


def _cleanup_distributed(state):
    if state["distributed"] and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _rank_print(state, message: str) -> None:
    if state["is_main_process"]:
        print(message, flush=True)


def _barrier(state) -> None:
    if state["distributed"] and dist.is_initialized():
        if state["device"].type == "cuda":
            dist.barrier(device_ids=[state["device"].index])
        else:
            dist.barrier()


def _gather_numpy_array(array: np.ndarray, state) -> np.ndarray:
    if not state["distributed"]:
        return array
    gathered = [None for _ in range(state["world_size"])]
    dist.all_gather_object(gathered, np.asarray(array))
    non_empty = [np.asarray(item) for item in gathered if item is not None and len(item) > 0]
    if not non_empty:
        return np.array([], dtype=np.float32)
    return np.concatenate(non_empty, axis=0)


def _resolve_mixed_precision(device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, "fp32", None
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    if major >= 8:
        return torch.bfloat16, "bf16-mixed", device_index
    return torch.float16, "fp16-mixed", device_index


def _autocast_context(device: torch.device, autocast_dtype=None):
    if autocast_dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    return nullcontext()


def _build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.lr_decay_patience,
            min_lr=args.min_lr,
        )
    if args.scheduler == "cosine":
        warmup_epochs = max(0, min(int(args.lr_warmup_epochs), int(args.epochs) - 1))
        cosine_epochs = max(1, int(args.epochs) - warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=float(args.min_lr),
        )
        if warmup_epochs == 0:
            return cosine
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(args.lr_warmup_start_factor),
            end_factor=1.0,
            total_iters=max(1, warmup_epochs),
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def _make_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    sampler=None,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=2,
):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": num_workers,
        "follow_batch": ["mol_x", "prot_node_esm"],
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    kwargs["pin_memory"] = pin_memory
    return DataLoader(dataset, **kwargs)


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


def evaluate_loader(model, loader, device, state, autocast_dtype=None, desc="Evaluation", show_progress=True):
    model.eval()
    reg_preds = []
    reg_truths = []
    running_reg_loss = 0.0
    running_cluster_loss = 0.0
    mse_loss = torch.nn.MSELoss(reduction="mean")
    iterator = progress(loader, desc=desc, unit="batch", leave=False) if state["is_main_process"] and show_progress else loader

    with torch.no_grad():
        for batch in iterator:
            with _autocast_context(device, autocast_dtype=autocast_dtype):
                reg_pred, cluster_loss = _forward_model(model, batch, device)
                reg_pred = reg_pred.squeeze().reshape(-1)
                reg_y = batch.reg_y.to(device).squeeze().reshape(-1)
                reg_loss = mse_loss(reg_pred, reg_y)

            running_reg_loss += float(reg_loss.item())
            running_cluster_loss += float(cluster_loss.item())
            reg_preds.append(reg_pred.detach().float().cpu())
            reg_truths.append(reg_y.detach().cpu())

    reg_preds_np = torch.cat(reg_preds).numpy() if reg_preds else np.array([], dtype=np.float32)
    reg_truths_np = torch.cat(reg_truths).numpy() if reg_truths else np.array([], dtype=np.float32)
    reg_preds_np = _gather_numpy_array(reg_preds_np, state)
    reg_truths_np = _gather_numpy_array(reg_truths_np, state)
    loss_tensor = torch.tensor([running_reg_loss, running_cluster_loss, len(loader)], dtype=torch.float64, device=device)
    if state["distributed"]:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    total_batches = max(1.0, float(loss_tensor[2].item()))
    metrics = evaluate_reg(reg_truths_np, reg_preds_np) if len(reg_truths_np) > 0 else {}
    metrics.update(
        {
            "regression_loss": float(loss_tensor[0].item() / total_batches),
            "cluster_loss": float(loss_tensor[1].item() / total_batches),
        }
    )
    return reg_preds_np, reg_truths_np, metrics


def _load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = KcatNet(
        checkpoint["prot_deg"].to(device),
        mol_in_channels=model_config["mol_in_channels"],
        prot_in_channels=model_config["prot_in_channels"],
        prot_evo_channels=model_config["prot_evo_channels"],
        hidden_channels=model_config["hidden_channels"],
        pre_layers=model_config["pre_layers"],
        post_layers=model_config["post_layers"],
        aggregators=model_config["aggregators"],
        scalers=model_config["scalers"],
        total_layer=model_config["total_layer"],
        K=model_config["K"],
        heads=model_config["heads"],
        dropout=model_config["dropout"],
        dropout_attn_score=model_config["dropout_attn_score"],
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def _write_metrics(path: Path, metrics: dict) -> None:
    pd.DataFrame([metrics]).to_csv(path, index=False)


def main(args):
    dist_state = _setup_distributed(args)
    set_seed(args.seed)
    device = dist_state["device"]
    autocast_dtype, precision_mode, precision_device_index = _resolve_mixed_precision(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))
    out_dir = Path(args.out_dir)
    if dist_state["is_main_process"]:
        out_dir.mkdir(parents=True, exist_ok=True)
    _barrier(dist_state)

    train_df = read_table(Path(args.train_path))
    val_df = read_table(Path(args.val_path))
    test_df = read_table(Path(args.test_path))
    for split_path, frame in (
        (Path(args.train_path), train_df),
        (Path(args.val_path), val_df),
        (Path(args.test_path), test_df),
    ):
        require_columns(frame, [args.sequence_col, args.smiles_col, args.target_col], split_path)

    ligand_smiles = pd.concat(
        [
            train_df[args.smiles_col].astype(str),
            val_df[args.smiles_col].astype(str),
            test_df[args.smiles_col].astype(str),
        ],
        ignore_index=True,
    )
    protein_sequences = pd.concat(
        [
            train_df[args.sequence_col].astype(str),
            val_df[args.sequence_col].astype(str),
            test_df[args.sequence_col].astype(str),
        ],
        ignore_index=True,
    )
    protein_store = ProteinEmbeddingStore(
        Path(args.embeddings_dir),
        sequences=protein_sequences.tolist(),
        preload=args.preload_proteins,
        max_items=args.protein_cache_items,
    )
    ligand_store = LigandEmbeddingStore(
        Path(args.embeddings_dir),
        smiles_values=ligand_smiles.tolist(),
        preload=not args.lazy_ligands,
    )

    train_dataset = CachedKcatDataset(
        train_df,
        protein_store=protein_store,
        ligand_store=ligand_store,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
    )
    val_dataset = CachedKcatDataset(
        val_df,
        protein_store=protein_store,
        ligand_store=ligand_store,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
    )
    test_dataset = CachedKcatDataset(
        test_df,
        protein_store=protein_store,
        ligand_store=ligand_store,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
    )

    train_sampler = None
    val_sampler = None
    test_sampler = None
    if dist_state["distributed"]:
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist_state["world_size"], rank=dist_state["rank"], shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=dist_state["world_size"], rank=dist_state["rank"], shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=dist_state["world_size"], rank=dist_state["rank"], shuffle=False)

    pin_memory = args.pin_memory or (device.type == "cuda")
    train_loader = _make_loader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = _make_loader(
        val_dataset,
        args.batch_size,
        False,
        args.num_workers,
        sampler=val_sampler,
        pin_memory=pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = _make_loader(
        test_dataset,
        args.batch_size,
        False,
        args.num_workers,
        sampler=test_sampler,
        pin_memory=pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    with open(args.config_path, "r") as handle:
        config = json.load(handle)
    model_config = dict(config["params"])

    pna_cache_dir = Path(args.embeddings_dir) / "metadata" / "pna_degrees"
    if dist_state["distributed"]:
        if dist_state["is_main_process"]:
            prot_deg = get_or_compute_pna_degrees(
                train_dataset,
                train_path=Path(args.train_path),
                cache_root=pna_cache_dir,
                verbose=True,
            )
        _barrier(dist_state)
        if not dist_state["is_main_process"]:
            prot_deg = get_or_compute_pna_degrees(
                train_dataset,
                train_path=Path(args.train_path),
                cache_root=pna_cache_dir,
                verbose=False,
            )
    else:
        prot_deg = get_or_compute_pna_degrees(
            train_dataset,
            train_path=Path(args.train_path),
            cache_root=pna_cache_dir,
            verbose=True,
        )
    model = KcatNet(
        prot_deg.to(device),
        mol_in_channels=model_config["mol_in_channels"],
        prot_in_channels=model_config["prot_in_channels"],
        prot_evo_channels=model_config["prot_evo_channels"],
        hidden_channels=model_config["hidden_channels"],
        pre_layers=model_config["pre_layers"],
        post_layers=model_config["post_layers"],
        aggregators=model_config["aggregators"],
        scalers=model_config["scalers"],
        total_layer=model_config["total_layer"],
        K=model_config["K"],
        heads=model_config["heads"],
        dropout=model_config["dropout"],
        dropout_attn_score=model_config["dropout_attn_score"],
        device=device,
    ).to(device)
    if dist_state["distributed"]:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        amsgrad=args.amsgrad,
    )
    scheduler = _build_scheduler(optimizer, args)
    mse_loss = torch.nn.MSELoss(reduction="mean")

    if precision_device_index is not None:
        gpu_name = torch.cuda.get_device_name(precision_device_index)
        major, minor = torch.cuda.get_device_capability(precision_device_index)
        _rank_print(dist_state, f"CUDA device: {gpu_name} | compute capability: {major}.{minor} | precision: {precision_mode}")
    else:
        _rank_print(dist_state, f"Device: {device} | precision: {precision_mode}")
    _rank_print(
        dist_state,
        f"DataLoader config: num_workers={args.num_workers} | pin_memory={pin_memory} | "
        f"persistent_workers={args.persistent_workers if args.num_workers > 0 else False} | "
        f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'n/a'}",
    )

    history = []
    best_val_rmse = float("inf")
    best_model_path = out_dir / "bestmodel.pth"
    last_ckpt_path = out_dir / "checkpoint_last.pt"
    started = time.time()
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    no_improve = 0

    epoch_progress = progress(range(1, args.epochs + 1), desc="Training", unit="epoch") if dist_state["is_main_process"] else range(1, args.epochs + 1)
    for epoch in epoch_progress:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_reg_loss = 0.0
        train_cluster_loss = 0.0

        batch_iterator = progress(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False) if dist_state["is_main_process"] else train_loader
        for batch in batch_iterator:
            optimizer.zero_grad()
            with _autocast_context(device, autocast_dtype=autocast_dtype):
                reg_pred, cluster_loss = _forward_model(model, batch, device)
                reg_pred = reg_pred.squeeze().reshape(-1)
                reg_y = batch.reg_y.to(device).squeeze().reshape(-1)

                reg_loss = mse_loss(reg_pred, reg_y)
                total_loss = args.regression_weight * reg_loss + args.cluster_weight * cluster_loss

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if args.clip_grad is not None and args.clip_grad > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            train_reg_loss += float(reg_loss.item())
            train_cluster_loss += float(cluster_loss.item())
            if dist_state["is_main_process"]:
                batch_iterator.set_postfix(loss=f"{float(total_loss.item()):.4f}")

        train_loss_tensor = torch.tensor([train_reg_loss, train_cluster_loss, len(train_loader)], dtype=torch.float64, device=device)
        if dist_state["distributed"]:
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        total_train_batches = max(1.0, float(train_loss_tensor[2].item()))
        train_reg_loss = float(train_loss_tensor[0].item() / total_train_batches)
        train_cluster_loss = float(train_loss_tensor[1].item() / total_train_batches)
        train_rmse = float(np.sqrt(train_reg_loss))

        should_validate = (epoch % max(1, args.val_every) == 0) or (epoch == args.epochs)
        val_metrics = {
            "rmse": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "r2_score": float("nan"),
            "mae": float("nan"),
            "mse": float("nan"),
        }
        if should_validate:
            _, _, val_metrics = evaluate_loader(
                model,
                val_loader,
                device,
                dist_state,
                autocast_dtype=autocast_dtype,
                desc="Validation",
                show_progress=False,
            )
        if dist_state["is_main_process"]:
            epoch_progress.set_postfix(loss=f"{train_reg_loss:.4f}")
        if scheduler is not None:
            if args.scheduler == "plateau":
                if should_validate:
                    scheduler.step(val_metrics["rmse"])
            else:
                scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        history_row = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            "train_regression_loss": train_reg_loss,
            "train_cluster_loss": train_cluster_loss,
            "val_rmse": val_metrics["rmse"],
            "val_pearson": val_metrics["pearson"],
            "val_spearman": val_metrics["spearman"],
            "val_r2_score": val_metrics["r2_score"],
            "val_mae": val_metrics["mae"],
            "lr": current_lr,
        }
        if dist_state["is_main_process"]:
            history.append(history_row)
            pd.DataFrame(history).to_csv(out_dir / "logfile.csv", index=False)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
            "scaler_state_dict": (scaler.state_dict() if scaler.is_enabled() else None),
            "model_config": model_config,
            "train_args": vars(args),
            "prot_deg": prot_deg.cpu(),
            "best_val_rmse": best_val_rmse,
            "precision_mode": precision_mode,
        }
        if dist_state["is_main_process"]:
            torch.save(checkpoint, last_ckpt_path)

        if should_validate:
            if val_metrics["rmse"] < best_val_rmse - args.min_delta:
                best_val_rmse = val_metrics["rmse"]
                checkpoint["best_val_rmse"] = best_val_rmse
                if dist_state["is_main_process"]:
                    torch.save(checkpoint, best_model_path)
                no_improve = 0
            else:
                no_improve += 1

        stop_flag = torch.tensor(
            [1 if should_validate and args.patience > 0 and no_improve >= args.patience else 0],
            dtype=torch.int64,
            device=device,
        )
        if dist_state["distributed"]:
            dist.broadcast(stop_flag, src=0)
        if args.patience > 0 and no_improve >= args.patience:
            break
        if stop_flag.item():
            break

    _barrier(dist_state)
    model, best_checkpoint = _load_model_from_checkpoint(best_model_path, device)
    val_pred, val_true, val_metrics = evaluate_loader(model, val_loader, device, dist_state, autocast_dtype=autocast_dtype, desc="Final validation")
    test_pred, test_true, test_metrics = evaluate_loader(model, test_loader, device, dist_state, autocast_dtype=autocast_dtype, desc="Final test")

    if dist_state["is_main_process"]:
        _write_metrics(out_dir / "results_val.csv", val_metrics)
        _write_metrics(out_dir / "results_test.csv", test_metrics)
        _write_metrics(
            out_dir / "final_results_val.csv",
            {key: val_metrics[key] for key in ["rmse", "pearson", "spearman", "r2_score", "mae", "mse"]},
        )
        _write_metrics(
            out_dir / "final_results_test.csv",
            {key: test_metrics[key] for key in ["rmse", "pearson", "spearman", "r2_score", "mae", "mse"]},
        )
        pd.DataFrame({"pred_log10_value": val_pred, "label_log10_value": val_true}).to_csv(
            out_dir / "pred_label_val.csv",
            index=False,
        )
        pd.DataFrame({"pred_log10_value": test_pred, "label_log10_value": test_true}).to_csv(
            out_dir / "pred_label_test.csv",
            index=False,
        )

    ended_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if dist_state["is_main_process"]:
        with open(out_dir / "time_running.dat", "w") as handle:
            handle.write(f"Start Time: {started_at}\n")
            handle.write(f"End Time: {ended_at}\n")
            handle.write(f"Elapsed Seconds: {time.time() - started:.2f}\n")

    run_summary = {
        "task_name": args.task_name,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "checkpoint": str(best_model_path),
        "checkpoint_last": str(last_ckpt_path),
        "best_val_rmse": float(best_checkpoint["best_val_rmse"]),
        "test_rmse": float(test_metrics["rmse"]),
        "device": str(device),
        "precision_mode": precision_mode,
    }
    if dist_state["is_main_process"]:
        _write_metrics(out_dir / "run_summary.csv", run_summary)
        save_json(
            out_dir / "run_state.json",
            {
                "task_name": args.task_name,
                "status": "completed",
                "checkpoint_best": str(best_model_path),
                "checkpoint_last": str(last_ckpt_path),
                "best_val_rmse": float(best_checkpoint["best_val_rmse"]),
                "train_args": vars(args),
            },
        )
    _cleanup_distributed(dist_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KcatNet with explicit train/val/test splits and shared embedding cache.")
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
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=256)
    parser.add_argument("--lazy_ligands", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp_backend", choices=["auto", "nccl", "gloo"], default="auto")

    args = parser.parse_args()
    has_explicit_paths = all([args.train_path, args.val_path, args.test_path])
    if not has_explicit_paths:
        if not args.split_group:
            parser.error("Provide either --train_path/--val_path/--test_path or --split_group with optional --threshold.")
        job = resolve_single_split_job(Path(args.base_dir), args.split_group, threshold=args.threshold)
        args.train_path = job["train_path"]
        args.val_path = job["val_path"]
        args.test_path = job["test_path"]
    main(args)
