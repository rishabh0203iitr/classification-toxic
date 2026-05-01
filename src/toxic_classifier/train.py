"""Training entrypoint.

  python -m toxic_classifier.train --config configs/base.yaml
  WANDB_MODE=disabled python -m toxic_classifier.train --config configs/smoke.yaml

For multi-GPU:
  torchrun --nproc_per_node=2 -m toxic_classifier.train --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from .data.dataset import collate_fn, ensure_tokenizer_exists, make_dataset_from_cfg
from .data.tokenizer import PAD_ID, load_tokenizer
from .model.classifier import ToxicClassifier
from .utils import (
    AvgMeter,
    apply_overrides,
    device_from_cfg,
    is_main_process,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logging,
)


def _setup_ddp() -> tuple[bool, int, int]:
    """Returns (is_ddp, rank, world_size). torchrun sets RANK/LOCAL_RANK/WORLD_SIZE."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count() if torch.cuda.is_available() else 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world
    return False, 0, 1


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with weight decay only on weight matrices (not biases / LayerNorm)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "norm" in name.lower() or "ln" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _maybe_init_wandb(cfg: dict) -> object | None:
    if not is_main_process():
        return None
    try:
        import wandb
    except ImportError:
        return None
    mode = cfg.get("wandb", {}).get("mode") or os.environ.get("WANDB_MODE")
    if mode == "disabled":
        return None
    project = cfg.get("wandb", {}).get("project", "toxic-classifier")
    run_name = cfg.get("run_name")
    return wandb.init(project=project, name=run_name, config=cfg, mode=mode)


def _make_loaders(cfg: dict, is_ddp: bool, rank: int, world: int) -> tuple[DataLoader, DataLoader]:
    train_ds = make_dataset_from_cfg(cfg, "train")
    val_ds = make_dataset_from_cfg(cfg, "val")
    bs = cfg["train"]["batch_size"]
    eval_bs = cfg["train"]["eval_batch_size"]
    nw = cfg["train"]["num_workers"]

    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
    elif cfg["train"].get("use_weighted_sampler", False):
        # Re-balance the rare positive class.
        labels = np.array([float(train_ds[i]["label"]) for i in range(len(train_ds))])
        n_pos = max(1, int(labels.sum()))
        n_neg = max(1, len(labels) - n_pos)
        w = np.where(labels > 0.5, 1.0 / n_pos, 1.0 / n_neg)
        train_sampler = WeightedRandomSampler(w.tolist(), num_samples=len(w), replacement=True)
        val_sampler = None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=(train_sampler is None and not is_ddp),
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    ys, scores = [], []
    for batch in loader:
        ids = batch["ids"].to(device, non_blocking=True)
        kpm = batch["key_padding_mask"].to(device, non_blocking=True)
        labels = batch["label"]
        logits = model(ids, key_padding_mask=kpm)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(labels.numpy())
        scores.append(probs)
    y = np.concatenate(ys) if ys else np.zeros(0)
    s = np.concatenate(scores) if scores else np.zeros(0)
    out = {"n": int(len(y))}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out["acc@0.5"] = float(((s > 0.5).astype(np.float32) == y).mean()) if len(y) else float("nan")
    return out


def _resolve_pos_weight(cfg: dict, train_loader: DataLoader, device: torch.device) -> torch.Tensor | None:
    pw = cfg["train"].get("pos_weight", None)
    if pw is None or pw is False:
        return None
    if pw == "auto":
        # Estimate on the (already-loaded) underlying dataset rather than draining the loader.
        ds = train_loader.dataset
        # Some datasets expose .labels directly; others compute per-row.
        if hasattr(ds, "labels"):
            labels = np.asarray(ds.labels, dtype=np.float32)
        else:
            labels = np.array([float(ds[i]["label"]) for i in range(len(ds))], dtype=np.float32)
        n_pos = float(labels.sum())
        n_neg = float(len(labels) - n_pos)
        if n_pos <= 0:
            return None
        w = n_neg / n_pos
        return torch.tensor([w], device=device, dtype=torch.float32)
    return torch.tensor([float(pw)], device=device, dtype=torch.float32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], dest="overrides")
    args = p.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.overrides)
    log = setup_logging()
    set_seed(cfg.get("seed", 42))

    is_ddp, rank, world = _setup_ddp()
    device = device_from_cfg(cfg) if not is_ddp else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if is_main_process():
        log.info("Device=%s, DDP=%s, world=%d", device, is_ddp, world)

    # Tokenizer (auto-train for raw mode if missing) ---------------------------
    if cfg["data"]["mode"] == "raw":
        ensure_tokenizer_exists(
            tokenizer_path=cfg["tokenizer"]["path"],
            train_csv=cfg["data"]["train_csv"],
            text_col=cfg["data"].get("text_col", "comment_text"),
            vocab_size=cfg["tokenizer"]["vocab_size"],
            lowercase=cfg["tokenizer"].get("lowercase", True),
        )
    tok = load_tokenizer(cfg["tokenizer"]["path"])
    vocab_size = tok.get_vocab_size()
    if is_main_process():
        log.info("Tokenizer: vocab=%d, max_len=%d", vocab_size, cfg["tokenizer"]["max_len"])

    # Data ---------------------------------------------------------------------
    train_loader, val_loader = _make_loaders(cfg, is_ddp, rank, world)
    if is_main_process():
        log.info("Train batches/epoch=%d, val batches=%d", len(train_loader), len(val_loader))

    # Model --------------------------------------------------------------------
    m = cfg["model"]
    model = ToxicClassifier(
        vocab_size=vocab_size,
        max_len=cfg["tokenizer"]["max_len"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        dim_ff=m["dim_ff"],
        dropout=m["dropout"],
        pool=m.get("pool", "cls"),
        pad_id=PAD_ID,
    ).to(device)
    if is_main_process():
        log.info("Model: %s params", f"{model.num_parameters():,}")

    if is_ddp:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )

    # Loss / optim / sched -----------------------------------------------------
    pos_weight = _resolve_pos_weight(cfg, train_loader, device)
    if is_main_process() and pos_weight is not None:
        log.info("BCE pos_weight=%.3f", float(pos_weight.item()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = _make_optimizer(
        model.module if is_ddp else model,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    total_steps = max(1, len(train_loader) * cfg["train"]["epochs"] // cfg["train"]["grad_accum_steps"])
    warmup = cfg["train"]["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: _lr_lambda(s, warmup, total_steps)
    )

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = GradScaler(device="cuda", enabled=use_amp)

    wb = _maybe_init_wandb(cfg)
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    best = -float("inf")
    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        if is_ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        meter = AvgMeter()
        t0 = time.time()
        for batch in train_loader:
            ids = batch["ids"].to(device, non_blocking=True)
            kpm = batch["key_padding_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(ids, key_padding_mask=kpm)
                loss = loss_fn(logits, labels)
                loss = loss / cfg["train"]["grad_accum_steps"]
            scaler.scale(loss).backward()
            if (global_step + 1) % cfg["train"]["grad_accum_steps"] == 0:
                if cfg["train"].get("grad_clip"):
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
            meter.update(float(loss.item()) * cfg["train"]["grad_accum_steps"], k=ids.size(0))
            global_step += 1

            if is_main_process() and global_step % cfg["train"]["log_every"] == 0:
                lr = sched.get_last_lr()[0]
                log.info(
                    "epoch=%d step=%d loss=%.4f lr=%.2e",
                    epoch, global_step, meter.avg, lr,
                )
                if wb is not None:
                    wb.log({"train/loss": meter.avg, "train/lr": lr, "step": global_step})

        # Per-epoch validation -------------------------------------------------
        if is_main_process():
            val = evaluate(model.module if is_ddp else model, val_loader, device)
            dt = time.time() - t0
            log.info(
                "epoch=%d done in %.1fs | train_loss=%.4f val_auc=%.4f val_pr_auc=%.4f n_val=%d",
                epoch, dt, meter.avg, val["auc"], val["pr_auc"], val["n"],
            )
            if wb is not None:
                wb.log(
                    {
                        f"val/{k}": v for k, v in val.items() if isinstance(v, (int, float))
                    } | {"epoch": epoch}
                )
            metric = val.get("auc", float("nan"))
            if not math.isnan(metric) and metric > best:
                best = metric
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    {
                        "model": (model.module if is_ddp else model).state_dict(),
                        "config": cfg,
                        "val": val,
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                )
                log.info("Saved best ckpt → %s (val_auc=%.4f)", ckpt_dir / "best.pt", best)
            save_checkpoint(
                ckpt_dir / "last.pt",
                {
                    "model": (model.module if is_ddp else model).state_dict(),
                    "config": cfg,
                    "val": val,
                    "epoch": epoch,
                    "global_step": global_step,
                },
            )
        if is_ddp:
            dist.barrier()

    if is_ddp:
        dist.destroy_process_group()
    if wb is not None:
        wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
