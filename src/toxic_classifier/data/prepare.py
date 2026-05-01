"""CLI: split + tokenise + dump memmap files for fast training.

Usage:
    python -m toxic_classifier.data.prepare --config configs/base.yaml

Produces under data/processed/:
    tokenizer.json
    train.{ids,lens,labels,targets,idents}.bin
    val.{ids,lens,labels,targets,idents}.bin
    test.{ids,lens,labels,targets,idents}.bin
And under data/splits/:
    train_idx.npy, val_idx.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import apply_overrides, load_config, set_seed, setup_logging
from .split import save_splits, stratified_train_val_split
from .tokenizer import encode, load_tokenizer, train_tokenizer


def _open_writers(out_prefix: Path, n_id: int):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    return {
        "ids": open(out_prefix.with_suffix(".ids.bin"), "wb"),
        "lens": open(out_prefix.with_suffix(".lens.bin"), "wb"),
        "labels": open(out_prefix.with_suffix(".labels.bin"), "wb"),
        "targets": open(out_prefix.with_suffix(".targets.bin"), "wb"),
        "idents": open(out_prefix.with_suffix(".idents.bin"), "wb"),
        "n_id": n_id,
    }


def _close_writers(w: dict) -> None:
    for v in w.values():
        if hasattr(v, "close"):
            v.close()


def _write_row(w: dict, ids: list[int], label: float, target: float, idents: np.ndarray) -> None:
    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(w["ids"])
    np.array([len(arr)], dtype=np.int32).tofile(w["lens"])
    np.array([label], dtype=np.float32).tofile(w["labels"])
    np.array([target], dtype=np.float32).tofile(w["targets"])
    np.asarray(idents, dtype=np.float32).tofile(w["idents"])


def _process_dataframe(
    df: pd.DataFrame,
    out_prefix: Path,
    tokenizer,
    *,
    text_col: str,
    target_col: str,
    identity_cols: list[str],
    toxic_threshold: float,
    max_len: int,
    desc: str,
) -> int:
    w = _open_writers(out_prefix, n_id=len(identity_cols))
    n = len(df)
    texts = df[text_col].fillna("").astype(str).to_numpy()

    # Some splits use 'toxicity' instead of 'target'
    if target_col in df.columns:
        targets = df[target_col].fillna(0).to_numpy(dtype=np.float32)
    elif "toxicity" in df.columns:
        targets = df["toxicity"].fillna(0).to_numpy(dtype=np.float32)
    else:
        targets = np.zeros(n, dtype=np.float32)
    labels = (targets >= toxic_threshold).astype(np.float32)
    idents = np.stack(
        [
            df[c].fillna(0).to_numpy(dtype=np.float32)
            if c in df.columns
            else np.zeros(n, dtype=np.float32)
            for c in identity_cols
        ],
        axis=1,
    ) if identity_cols else np.zeros((n, 0), dtype=np.float32)

    written = 0
    for i in tqdm(range(n), desc=desc, dynamic_ncols=True):
        ids = encode(tokenizer, texts[i], max_len)
        _write_row(w, ids, labels[i], targets[i], idents[i])
        written += 1
    _close_writers(w)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], dest="overrides")
    args = p.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.overrides)
    log = setup_logging()
    set_seed(cfg.get("seed", 42))

    d, t = cfg["data"], cfg["tokenizer"]
    proc_dir = Path(d["processed_dir"])
    splits_dir = Path(d["splits_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_csv = Path(d["train_csv"])
    test_pub = Path(d["test_public_csv"])
    test_priv = Path(d["test_private_csv"])

    # Tokenizer ----------------------------------------------------------------
    tok_path = Path(t["path"])
    if not tok_path.exists() or not t.get("train_if_missing", True):
        log.info("Loading existing tokenizer from %s", tok_path)
        tokenizer = load_tokenizer(tok_path)
    else:
        log.info("Training BPE tokenizer (vocab=%d) from %s", t["vocab_size"], train_csv)
        # Stream to avoid holding the full text column twice.
        chunks = pd.read_csv(train_csv, usecols=[d["text_col"]], chunksize=200_000)
        def _it():
            for ch in chunks:
                yield from ch[d["text_col"]].fillna("").astype(str)
        tokenizer = train_tokenizer(
            _it(),
            tok_path,
            vocab_size=t["vocab_size"],
            lowercase=t.get("lowercase", True),
        )
        log.info("Saved tokenizer to %s (vocab=%d)", tok_path, tokenizer.get_vocab_size())

    # Split --------------------------------------------------------------------
    log.info("Loading %s for train/val split", train_csv)
    df_train = pd.read_csv(train_csv)
    log.info("  %d rows; computing stratified split (val=%.2f%%)", len(df_train), 100 * d["val_fraction"])
    train_idx, val_idx = stratified_train_val_split(
        df_train,
        target_col=d["target_col"],
        identity_cols=d["identity_cols"],
        val_fraction=d["val_fraction"],
        toxic_threshold=d["toxic_threshold"],
        seed=cfg.get("seed", 42),
    )
    save_splits(splits_dir, train_idx, val_idx)
    log.info("Saved split indices: train=%d, val=%d", len(train_idx), len(val_idx))

    # Memmap dumps -------------------------------------------------------------
    _process_dataframe(
        df_train.iloc[train_idx],
        proc_dir / "train",
        tokenizer,
        text_col=d["text_col"],
        target_col=d["target_col"],
        identity_cols=d["identity_cols"],
        toxic_threshold=d["toxic_threshold"],
        max_len=t["max_len"],
        desc="encode-train",
    )
    _process_dataframe(
        df_train.iloc[val_idx],
        proc_dir / "val",
        tokenizer,
        text_col=d["text_col"],
        target_col=d["target_col"],
        identity_cols=d["identity_cols"],
        toxic_threshold=d["toxic_threshold"],
        max_len=t["max_len"],
        desc="encode-val",
    )
    del df_train

    log.info("Loading test (public + private expanded)")
    df_test_pub = pd.read_csv(test_pub)
    df_test_priv = pd.read_csv(test_priv)
    df_test = pd.concat([df_test_pub, df_test_priv], ignore_index=True)
    _process_dataframe(
        df_test,
        proc_dir / "test",
        tokenizer,
        text_col=d["text_col"],
        target_col=d["target_col"],
        identity_cols=d["identity_cols"],
        toxic_threshold=d["toxic_threshold"],
        max_len=t["max_len"],
        desc="encode-test",
    )
    log.info("Done. processed=%s", proc_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
