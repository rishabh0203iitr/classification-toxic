"""PyTorch Dataset / collate for Jigsaw toxic-comment classification.

Two modes:
  - `raw`    : reads a CSV into memory at construction, tokenises lazily in
               __getitem__. Convenient for the smoke test and small data.
  - `memmap` : reads token ids from a pre-tokenised flat binary (uint16) and
               offsets/lengths from a sidecar (int32). Per-epoch CPU work is
               near-zero, so the GPU is the bottleneck. Used for full training.

The collate function dynamically pads each batch to the longest sequence in
that batch (rather than always padding to model.max_len), saving compute on
short comments.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import PAD_ID, encode, load_tokenizer


class RawJigsawDataset(Dataset):
    """Reads a CSV into memory; tokenises on-the-fly."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer_path: str | Path,
        max_len: int,
        text_col: str = "comment_text",
        target_col: str | None = "target",
        identity_cols: Sequence[str] | None = None,
        toxic_threshold: float = 0.5,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.max_len = max_len
        self.text_col = text_col
        self.target_col = target_col
        self.identity_cols = list(identity_cols or [])
        self.toxic_threshold = toxic_threshold

        # Pre-extract numpy arrays for speed.
        self.texts = self.df[text_col].fillna("").astype(str).to_numpy()
        if target_col and target_col in self.df.columns:
            self.targets = self.df[target_col].fillna(0).to_numpy(dtype=np.float32)
        elif "toxicity" in self.df.columns:
            # test_*_expanded use 'toxicity' instead of 'target'
            self.targets = self.df["toxicity"].fillna(0).to_numpy(dtype=np.float32)
        else:
            self.targets = np.zeros(len(self.df), dtype=np.float32)
        self.identities = np.stack(
            [
                self.df[c].fillna(0).to_numpy(dtype=np.float32)
                if c in self.df.columns
                else np.zeros(len(self.df), dtype=np.float32)
                for c in self.identity_cols
            ],
            axis=1,
        ) if self.identity_cols else np.zeros((len(self.df), 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | np.ndarray]:
        ids = encode(self.tokenizer, self.texts[idx], self.max_len)
        y = float(self.targets[idx] >= self.toxic_threshold)
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.float32),
            "target_raw": torch.tensor(self.targets[idx], dtype=torch.float32),
            "identities": torch.from_numpy(self.identities[idx]),
        }


class MemmapJigsawDataset(Dataset):
    """Reads pre-tokenised ids from a flat uint16 file with int32 offsets+lengths.

    Layout (all little-endian):
      <split>.ids.bin   : uint16, concatenated token ids
      <split>.lens.bin  : int32,  one length per row (in tokens)
      <split>.labels.bin: float32, binary toxic label
      <split>.targets.bin: float32, raw target (continuous)
      <split>.idents.bin : float32 (n_rows, n_identities), row-major
    """

    def __init__(
        self,
        prefix: str | Path,
        n_identities: int,
        max_len: int,
    ) -> None:
        prefix = Path(prefix)
        self.lens = np.fromfile(prefix.with_suffix(".lens.bin"), dtype=np.int32)
        self.offsets = np.concatenate([[0], np.cumsum(self.lens, dtype=np.int64)])[:-1]
        total = int(self.offsets[-1] + self.lens[-1]) if len(self.lens) else 0
        self.ids = np.memmap(
            prefix.with_suffix(".ids.bin"), dtype=np.uint16, mode="r", shape=(total,)
        ) if total else np.zeros(0, dtype=np.uint16)
        self.labels = np.fromfile(prefix.with_suffix(".labels.bin"), dtype=np.float32)
        self.targets = np.fromfile(prefix.with_suffix(".targets.bin"), dtype=np.float32)
        if n_identities:
            self.idents = np.fromfile(
                prefix.with_suffix(".idents.bin"), dtype=np.float32
            ).reshape(-1, n_identities)
        else:
            self.idents = np.zeros((len(self.lens), 0), dtype=np.float32)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.lens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = int(self.offsets[idx])
        L = int(self.lens[idx])
        ids = np.asarray(self.ids[s : s + L], dtype=np.int64)  # cast to long here
        return {
            "ids": torch.from_numpy(ids),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "target_raw": torch.tensor(self.targets[idx], dtype=torch.float32),
            "identities": torch.from_numpy(self.idents[idx]),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Dynamic-pad collate. Pads with PAD_ID and returns a key_padding_mask
    where True == pad position (the convention for nn.MultiheadAttention)."""
    B = len(batch)
    lengths = torch.tensor([len(b["ids"]) for b in batch], dtype=torch.long)
    L = int(lengths.max().item()) if B else 0
    ids = torch.full((B, L), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = b["ids"]
    pad_mask = ids.eq(PAD_ID)
    # Don't mask the CLS at position 0 even for completely-empty rows.
    pad_mask[:, 0] = False
    labels = torch.stack([b["label"] for b in batch])
    target_raw = torch.stack([b["target_raw"] for b in batch])
    idents = torch.stack([b["identities"] for b in batch])
    return {
        "ids": ids,
        "key_padding_mask": pad_mask,
        "label": labels,
        "target_raw": target_raw,
        "identities": idents,
        "lengths": lengths,
    }


def ensure_tokenizer_exists(
    tokenizer_path: str | Path,
    train_csv: str | Path,
    text_col: str,
    vocab_size: int,
    lowercase: bool,
) -> None:
    """If tokenizer.json doesn't exist, train one from train_csv."""
    p = Path(tokenizer_path)
    if p.exists():
        return
    from .tokenizer import train_tokenizer

    df = pd.read_csv(train_csv, usecols=[text_col])
    texts = df[text_col].fillna("").astype(str).tolist()
    train_tokenizer(texts, p, vocab_size=vocab_size, lowercase=lowercase)


def make_dataset_from_cfg(cfg: dict, split: str) -> Dataset:
    """Factory: pick raw vs memmap based on cfg['data']['mode']."""
    d = cfg["data"]
    t = cfg["tokenizer"]
    if d["mode"] == "raw":
        csv_key = {"train": "train_csv", "val": "val_csv", "test": "test_csv"}[split]
        return RawJigsawDataset(
            csv_path=d[csv_key],
            tokenizer_path=t["path"],
            max_len=t["max_len"],
            text_col=d.get("text_col", "comment_text"),
            target_col=d.get("target_col", "target"),
            identity_cols=d.get("identity_cols", []),
            toxic_threshold=d.get("toxic_threshold", 0.5),
        )
    elif d["mode"] == "memmap":
        prefix = Path(d["processed_dir"]) / split
        return MemmapJigsawDataset(
            prefix=prefix,
            n_identities=len(d.get("identity_cols", [])),
            max_len=t["max_len"],
        )
    else:
        raise ValueError(f"unknown data.mode {d['mode']!r}")
