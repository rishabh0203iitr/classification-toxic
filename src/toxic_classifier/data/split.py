"""Stratified train/val splitting for Jigsaw.

The dataset has heavy class imbalance and a long-tail of identity columns.
A vanilla random split easily produces val sets where rare identities are
under- or over-represented, which then biases evaluation. We stratify on a
multi-label vector that combines:

  - the binary toxic label (target >= threshold)
  - a binary "identity-mentioned" indicator per identity column
    (id_col >= threshold)

so that the val split mirrors the train split's joint distribution over
toxicity × identity-presence. This is what `iterstrat` is built for.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_strat_labels(
    df: pd.DataFrame,
    target_col: str,
    identity_cols: list[str],
    toxic_threshold: float = 0.5,
) -> np.ndarray:
    y_tox = (df[target_col].fillna(0).to_numpy() >= toxic_threshold).astype(np.int8)
    cols = []
    for c in identity_cols:
        if c in df.columns:
            cols.append((df[c].fillna(0).to_numpy() >= toxic_threshold).astype(np.int8))
        else:
            cols.append(np.zeros(len(df), dtype=np.int8))
    Y = np.stack([y_tox, *cols], axis=1)
    return Y


def stratified_train_val_split(
    df: pd.DataFrame,
    target_col: str,
    identity_cols: list[str],
    val_fraction: float = 0.05,
    toxic_threshold: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (train_idx, val_idx) into df.

    Tries iterstrat (multi-label aware); falls back to a stratified split on
    the binary toxic label if iterstrat is unavailable.
    """
    Y = build_strat_labels(df, target_col, identity_cols, toxic_threshold)
    n = len(df)
    idx = np.arange(n)

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        mss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(mss.split(idx.reshape(-1, 1), Y))
    except Exception:  # noqa: BLE001 — fall through to single-label fallback
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            idx, test_size=val_fraction, random_state=seed, stratify=Y[:, 0]
        )
    return np.sort(train_idx), np.sort(val_idx)


def save_splits(out_dir: str | Path, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "train_idx.npy", train_idx)
    np.save(p / "val_idx.npy", val_idx)


def load_splits(out_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    p = Path(out_dir)
    return np.load(p / "train_idx.npy"), np.load(p / "val_idx.npy")
