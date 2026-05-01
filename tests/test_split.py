"""Split must be reproducible, with no leakage and reasonable stratification."""
import numpy as np
import pandas as pd

from toxic_classifier.data.split import stratified_train_val_split


def _toy_df(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(n),
        "comment_text": ["x"] * n,
        "target": rng.uniform(0, 1, size=n),
        "male": rng.uniform(0, 1, size=n),
        "female": rng.uniform(0, 1, size=n),
        "muslim": rng.uniform(0, 1, size=n),
    })


def test_split_no_leakage_and_size():
    df = _toy_df()
    tr, va = stratified_train_val_split(
        df, target_col="target", identity_cols=["male", "female", "muslim"],
        val_fraction=0.1, seed=0,
    )
    assert len(set(tr.tolist()) & set(va.tolist())) == 0
    assert len(tr) + len(va) == len(df)
    assert abs(len(va) / len(df) - 0.1) < 0.02


def test_split_reproducible():
    df = _toy_df(seed=1)
    a1, b1 = stratified_train_val_split(df, "target", ["male", "female", "muslim"], 0.1, seed=42)
    a2, b2 = stratified_train_val_split(df, "target", ["male", "female", "muslim"], 0.1, seed=42)
    assert (a1 == a2).all()
    assert (b1 == b2).all()
