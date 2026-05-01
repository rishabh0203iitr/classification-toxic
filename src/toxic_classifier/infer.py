"""Batched inference on an arbitrary CSV with a `comment_text` column.

  python -m toxic_classifier.infer \
      --ckpt artifacts/base/ckpt/best.pt \
      --tokenizer data/processed/tokenizer.json \
      --input some_comments.csv \
      --output predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data.dataset import collate_fn
from .data.tokenizer import encode, load_tokenizer
from .model.classifier import ToxicClassifier
from .utils import load_checkpoint, setup_logging


class _OnTheFlyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len: int):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i):
        ids = encode(self.tok, self.texts[i], self.max_len)
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(0.0, dtype=torch.float32),
            "target_raw": torch.tensor(0.0, dtype=torch.float32),
            "identities": torch.zeros(0, dtype=torch.float32),
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--input", required=True, help="CSV with a comment_text column")
    p.add_argument("--output", required=True)
    p.add_argument("--text-col", default="comment_text")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    log = setup_logging()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    state = load_checkpoint(args.ckpt, map_location=device)
    cfg = state["config"]
    tok = load_tokenizer(args.tokenizer)
    vocab_size = tok.get_vocab_size()
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
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    df = pd.read_csv(args.input)
    texts = df[args.text_col].fillna("").astype(str).to_numpy()
    ds = _OnTheFlyDataset(texts, tok, cfg["tokenizer"]["max_len"])
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)

    scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer", dynamic_ncols=True):
            ids = batch["ids"].to(device, non_blocking=True)
            kpm = batch["key_padding_mask"].to(device, non_blocking=True)
            scores.append(torch.sigmoid(model(ids, key_padding_mask=kpm)).cpu().numpy())
    df["toxic_score"] = np.concatenate(scores) if scores else np.zeros(0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Wrote %d predictions to %s", len(df), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
