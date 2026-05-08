"""Evaluate a checkpoint on the test split with the Jigsaw bias-aware metric.

  python -m toxic_classifier.eval --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import collate_fn, ensure_tokenizer_exists, make_dataset_from_cfg
from .data.tokenizer import PAD_ID, load_tokenizer
from .metrics import compute_per_identity_aucs, jigsaw_bias_metric, power_mean
from .model.classifier import ToxicClassifier
from .utils import (
    apply_overrides,
    device_from_cfg,
    load_checkpoint,
    load_config,
    set_seed,
    setup_logging,
)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    ys, scores, idents = [], [], []
    for batch in loader:
        ids = batch["ids"].to(device, non_blocking=True)
        kpm = batch["key_padding_mask"].to(device, non_blocking=True)
        logits = model(ids, key_padding_mask=kpm)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch["label"].numpy())
        idents.append(batch["identities"].numpy())
    return (
        np.concatenate(ys) if ys else np.zeros(0),
        np.concatenate(scores) if scores else np.zeros(0),
        np.concatenate(idents, axis=0) if idents else np.zeros((0, 0)),
    )


def _bar_chart(per_identity, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [x.name for x in per_identity]
    sg = [x.subgroup_auc for x in per_identity]
    bpsn = [x.bpsn_auc for x in per_identity]
    bnsp = [x.bnsp_auc for x in per_identity]
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(names)), 4.5))
    ax.bar(x - w, sg, w, label="Subgroup")
    ax.bar(x, bpsn, w, label="BPSN")
    ax.bar(x + w, bnsp, w, label="BNSP")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("Per-identity bias AUCs")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], dest="overrides")
    p.add_argument("--ckpt", default=None, help="Override eval.ckpt from config")
    p.add_argument("--split", default="test", choices=["test", "val"])
    args = p.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.overrides)
    log = setup_logging()
    set_seed(cfg.get("seed", 42))
    device = device_from_cfg(cfg)

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

    ds = make_dataset_from_cfg(cfg, args.split)
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=True,
        collate_fn=collate_fn,
    )

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

    ckpt_path = args.ckpt or cfg["eval"]["ckpt"]
    log.info("Loading checkpoint %s", ckpt_path)
    state = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    y, s, idents = _predict(model, loader, device)
    log.info("Predicted %d examples", len(y))

    from sklearn.metrics import average_precision_score, roc_auc_score

    overall_auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    overall_prauc = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    acc = float(((s > cfg["eval"].get("threshold", 0.5)).astype(np.float32) == y).mean()) if len(y) else float("nan")

    identity_names = cfg["data"].get("identity_cols", [])
    per_id = compute_per_identity_aucs(y, s, idents, identity_names)
    final = jigsaw_bias_metric(overall_auc, per_id)

    out_dir = Path(cfg["eval"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": args.split,
        "n": int(len(y)),
        "overall_auc": overall_auc,
        "overall_pr_auc": overall_prauc,
        "accuracy@thr": acc,
        "threshold": cfg["eval"].get("threshold", 0.5),
        "jigsaw_bias_metric": final,
        "subgroup_auc_pmean": power_mean(np.array([x.subgroup_auc for x in per_id])),
        "bpsn_auc_pmean": power_mean(np.array([x.bpsn_auc for x in per_id])),
        "bnsp_auc_pmean": power_mean(np.array([x.bnsp_auc for x in per_id])),
        "per_identity": [x.__dict__ for x in per_id],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([x.__dict__ for x in per_id]).to_csv(out_dir / "per_identity.csv", index=False)
    _bar_chart(per_id, out_dir / "per_identity_aucs.png")
    log.info(
        "Overall AUC=%.4f | PR-AUC=%.4f | Jigsaw bias metric=%.4f | wrote %s",
        overall_auc, overall_prauc, final, out_dir,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_identity"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
