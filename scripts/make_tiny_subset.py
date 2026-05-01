"""Produce a small, identity-rich subset of Jigsaw for the smoke test.

Sampling strategy:
- ~500 toxic + identity-mentioning rows (to make per-identity AUCs computable)
- ~500 non-toxic + identity-mentioning rows
- ~600 toxic / non-toxic rows without identity mentions
- 90/10/10 train/val/test split (~1500/250/250 rows total)

Repeatable via a fixed seed. Output: small CSVs with the same columns as
the original, so the same code paths exercise on smoke and full data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IDENTITY_COLS = [
    "male", "female", "transgender", "other_gender",
    "heterosexual", "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "physical_disability", "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness", "other_disability",
]


def _sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= n:
        return df
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=1500)
    p.add_argument("--n-val", type=int, default=250)
    p.add_argument("--n-test", type=int, default=250)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    src = Path(args.src) / "train.csv"
    print(f"Reading {src} ...")
    # Read the full file (~1.8M rows). We need it once; tolerate the few seconds.
    df = pd.read_csv(src)

    # Define groups
    has_id = (df[IDENTITY_COLS].fillna(0) >= 0.5).any(axis=1)
    is_tox = df["target"].fillna(0) >= 0.5

    pool_id_tox = df[has_id & is_tox]
    pool_id_nontox = df[has_id & ~is_tox]
    pool_noid_tox = df[~has_id & is_tox]
    pool_noid_nontox = df[~has_id & ~is_tox]

    print(
        f"Pools: id_tox={len(pool_id_tox)} id_nontox={len(pool_id_nontox)} "
        f"noid_tox={len(pool_noid_tox)} noid_nontox={len(pool_noid_nontox)}"
    )

    total = args.n_train + args.n_val + args.n_test
    # Quotas — ensure plenty of identity-mentioning examples for bias metrics.
    q_id_tox = total * 30 // 100
    q_id_nontox = total * 30 // 100
    q_noid_tox = total * 10 // 100
    q_noid_nontox = total - q_id_tox - q_id_nontox - q_noid_tox

    parts = [
        _sample(pool_id_tox, q_id_tox, rng),
        _sample(pool_id_nontox, q_id_nontox, rng),
        _sample(pool_noid_tox, q_noid_tox, rng),
        _sample(pool_noid_nontox, q_noid_nontox, rng),
    ]
    sub = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sub = sub.head(total)

    train = sub.iloc[: args.n_train]
    val = sub.iloc[args.n_train : args.n_train + args.n_val]
    test = sub.iloc[args.n_train + args.n_val :]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    def _frac_tox(d):
        return float((d["target"] >= 0.5).mean())

    print(
        f"Wrote {out}/{{train,val,test}}.csv: "
        f"train={len(train)} (tox={_frac_tox(train):.2%}), "
        f"val={len(val)} (tox={_frac_tox(val):.2%}), "
        f"test={len(test)} (tox={_frac_tox(test):.2%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
