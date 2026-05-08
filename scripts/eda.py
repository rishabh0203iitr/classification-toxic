"""One-shot EDA over the Jigsaw train + test files.

Prints a structured analysis (schema, target, sub-toxicity, identities,
co-mention, lengths, engagement, annotator counts, train-vs-test drift)
and saves a few plots into docs/results/eda/.

Run:
    python scripts/eda.py --train data/full/train.csv \
                          --test-public data/full/test_public_expanded.csv \
                          --test-private data/full/test_private_expanded.csv \
                          --out docs/results/eda
"""
from __future__ import annotations

import argparse
import json
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
SUBTOX_COLS = [
    "severe_toxicity", "obscene", "identity_attack",
    "insult", "threat", "sexual_explicit",
]
ENGAGE_COLS = ["funny", "wow", "sad", "likes", "disagree"]


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt_pct(x: float) -> str:
    return f"{100*x:6.2f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/full/train.csv")
    p.add_argument("--test-public", default="data/full/test_public_expanded.csv")
    p.add_argument("--test-private", default="data/full/test_private_expanded.csv")
    p.add_argument("--out", default="docs/results/eda")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    section("LOAD")
    print(f"train  : {args.train}")
    df = pd.read_csv(args.train)
    print(f"  rows : {len(df):,}")
    print(f"  cols : {len(df.columns)}")
    print(f"  dtypes: {df.dtypes.value_counts().to_dict()}")

    section("SCHEMA")
    for c in df.columns:
        nn = df[c].notna().sum()
        print(f"  {c:42s}  dtype={str(df[c].dtype):8s}  non-null={nn:>10,} ({nn/len(df):.1%})")

    section("TARGET (continuous toxicity in [0,1])")
    t = df["target"]
    print(f"  count   : {t.count():,}")
    print(f"  mean    : {t.mean():.4f}")
    print(f"  std     : {t.std():.4f}")
    print(f"  median  : {t.median():.4f}")
    qs = t.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    print("  quantiles: " + ", ".join(f"q{int(k*100)}={v:.4f}" for k, v in qs.items()))
    print(f"  exact 0 : {(t == 0).mean():.2%}")
    print(f"  >= 0.5  : {(t >= 0.5).sum():,} ({(t >= 0.5).mean():.2%})  ← binary positives")
    print(f"  >= 0.7  : {(t >= 0.7).mean():.2%}")
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]
    cuts = pd.cut(t, bins=bins, include_lowest=True, right=False)
    print("  bucket distribution:")
    for k, v in cuts.value_counts().sort_index().items():
        print(f"    {str(k):20s} {v:>10,} ({v/len(df):.2%})")

    section("SUB-TOXICITY LABELS (correlation with target)")
    for c in SUBTOX_COLS:
        if c not in df.columns:
            continue
        s = df[c]
        rho = s.corr(t)
        print(f"  {c:24s}  mean={s.mean():.4f}  >=0.5: {(s>=0.5).mean():.3%}  pearson_with_target={rho:.3f}")

    section("IDENTITY COLUMNS — coverage, mention rate, toxicity contrast")
    print(f"  Note: identities are only annotated on a subset (~{(df['identity_annotator_count']>0).mean():.1%}) of rows; the rest are NaN.")
    rows = []
    for c in IDENTITY_COLS:
        if c not in df.columns:
            continue
        s = df[c]
        nn = s.notna()
        n_anno = int(nn.sum())
        mention = (s >= 0.5)
        n_mention = int(mention.sum())
        tox_when_mentioned = float(t[mention].ge(0.5).mean()) if n_mention else float("nan")
        rows.append({
            "identity": c,
            "n_annotated": n_anno,
            "n_mentioned": n_mention,
            "mention_rate_among_annotated": n_mention / max(n_anno, 1),
            "tox_rate_when_mentioned": tox_when_mentioned,
            "tox_rate_overall": float((t >= 0.5).mean()),
            "tox_lift": (tox_when_mentioned / max((t >= 0.5).mean(), 1e-9)) if not np.isnan(tox_when_mentioned) else float("nan"),
        })
    id_df = pd.DataFrame(rows).sort_values("n_mentioned", ascending=False)
    print(id_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    id_df.to_csv(out / "identity_coverage.csv", index=False)

    section("IDENTITY × IDENTITY co-mention (top 10 pairs)")
    avail_ids = [c for c in IDENTITY_COLS if c in df.columns]
    sub = (df[avail_ids].fillna(0) >= 0.5).astype(np.int8)
    sub_arr = sub.to_numpy(dtype=np.int64, copy=True)
    co_mat = sub_arr.T @ sub_arr
    np.fill_diagonal(co_mat, 0)
    co = pd.DataFrame(co_mat, index=avail_ids, columns=avail_ids)
    pairs = []
    for i, a in enumerate(co.index):
        for j, b in enumerate(co.columns):
            if j <= i:
                continue
            pairs.append((a, b, int(co.iloc[i, j])))
    pairs.sort(key=lambda x: -x[2])
    for a, b, n in pairs[:10]:
        print(f"  {a:38s} & {b:38s} {n:>8,}")
    co.to_csv(out / "identity_comention.csv")

    section("COMMENT LENGTH (characters + whitespace tokens)")
    s = df["comment_text"].fillna("").astype(str)
    n_chars = s.str.len()
    n_words = s.str.split().str.len()
    print(f"  chars: mean={n_chars.mean():.1f}  median={int(n_chars.median())}  q90={int(n_chars.quantile(0.9))}  q95={int(n_chars.quantile(0.95))}  max={int(n_chars.max())}")
    print(f"  words: mean={n_words.mean():.1f}  median={int(n_words.median())}  q90={int(n_words.quantile(0.9))}  q95={int(n_words.quantile(0.95))}  max={int(n_words.max())}")
    print(f"  empty comments: {(n_chars == 0).sum()}")

    section("ENGAGEMENT FEATURES (descriptive only — not used as features by the model)")
    for c in ENGAGE_COLS:
        if c in df.columns:
            ss = df[c]
            print(f"  {c:10s}  mean={ss.mean():.3f}  q90={ss.quantile(0.9):.0f}  max={ss.max()}")

    section("ANNOTATOR COUNTS")
    for c in ["toxicity_annotator_count", "identity_annotator_count"]:
        if c in df.columns:
            ss = df[c]
            print(f"  {c:30s} mean={ss.mean():.2f}  median={int(ss.median())}  q90={int(ss.quantile(0.9))}  max={int(ss.max())}  zeros={(ss==0).mean():.2%}")

    section("TIME SPAN")
    if "created_date" in df.columns:
        d = pd.to_datetime(df["created_date"], errors="coerce")
        print(f"  earliest : {d.min()}")
        print(f"  latest   : {d.max()}")
        print(f"  span     : {(d.max() - d.min()).days} days")
        years = d.dt.year.value_counts().sort_index()
        print("  per year:")
        for y, n in years.items():
            print(f"    {int(y):4d}: {int(n):>9,}")

    section("TRAIN vs TEST distributions")
    pub = pd.read_csv(args.test_public)
    priv = pd.read_csv(args.test_private)
    test = pd.concat([pub, priv], ignore_index=True)
    test_target_col = "toxicity" if "toxicity" in test.columns else "target"
    pos_train = float((df["target"] >= 0.5).mean())
    pos_test = float((test[test_target_col] >= 0.5).mean())
    print(f"  rows train       : {len(df):,}")
    print(f"  rows test (union): {len(test):,}")
    print(f"  positive rate train : {pos_train:.4%}")
    print(f"  positive rate test  : {pos_test:.4%}")
    print(f"  test/train pos-rate ratio : {pos_test/pos_train:.3f}")
    # Identity mention rates
    print()
    print("  identity mention-rate drift (test vs train, among rows with identity annotations):")
    print(f"  {'identity':38s}  {'train_rate':>10s}  {'test_rate':>10s}  {'ratio':>6s}")
    drift_rows = []
    train_anno = df[df["identity_annotator_count"] > 0]
    test_anno = test[test["identity_annotator_count"] > 0] if "identity_annotator_count" in test.columns else test
    for c in IDENTITY_COLS:
        if c not in df.columns or c not in test.columns:
            continue
        tr = float((train_anno[c].fillna(0) >= 0.5).mean())
        te = float((test_anno[c].fillna(0) >= 0.5).mean())
        ratio = te / tr if tr > 0 else float("nan")
        drift_rows.append((c, tr, te, ratio))
    for c, tr, te, r in sorted(drift_rows, key=lambda x: -x[1]):
        print(f"  {c:38s}  {tr:>10.4%}  {te:>10.4%}  {r:>6.2f}")
    pd.DataFrame(drift_rows, columns=["identity", "train_rate", "test_rate", "ratio"]).to_csv(out / "identity_drift.csv", index=False)

    section("PLOTS")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable — skipping plots")
        return

    # Target histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["target"], bins=60)
    ax.set_xlabel("target (continuous toxicity score)")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("Distribution of target (log-y)")
    fig.tight_layout()
    fig.savefig(out / "target_hist.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'target_hist.png'}")

    # Length histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.clip(n_chars, 0, 2000), bins=60)
    ax.set_xlabel("comment length (chars, clipped at 2000)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of comment length")
    fig.tight_layout()
    fig.savefig(out / "length_hist.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'length_hist.png'}")

    # Identity coverage bar (annotated subset)
    id_show = id_df.sort_values("n_mentioned", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(id_show["identity"], id_show["n_mentioned"])
    ax.set_xscale("log")
    ax.set_xlabel("# rows mentioning identity (log scale, train)")
    ax.set_title("Identity mention frequency in train")
    fig.tight_layout()
    fig.savefig(out / "identity_mentions.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'identity_mentions.png'}")

    # Per-identity toxicity-when-mentioned vs base rate
    base = float((df["target"] >= 0.5).mean())
    show = id_df.sort_values("tox_rate_when_mentioned", ascending=True).dropna(subset=["tox_rate_when_mentioned"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(show["identity"], show["tox_rate_when_mentioned"], color="steelblue", label="P(toxic | identity mentioned)")
    ax.axvline(base, color="red", ls="--", label=f"overall base rate = {base:.3f}")
    ax.set_xlabel("P(toxic | identity mentioned)")
    ax.set_title("Toxicity rate by identity mention (train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "identity_tox_rate.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'identity_tox_rate.png'}")

    summary = {
        "n_train": int(len(df)),
        "n_test": int(len(test)),
        "n_columns": int(len(df.columns)),
        "pos_rate_train": pos_train,
        "pos_rate_test": pos_test,
        "median_chars": int(n_chars.median()),
        "q95_chars": int(n_chars.quantile(0.95)),
        "median_words": int(n_words.median()),
        "q95_words": int(n_words.quantile(0.95)),
        "n_identity_cols": len(IDENTITY_COLS),
        "rows_with_identity_annotations_train": int((df["identity_annotator_count"] > 0).sum()),
        "rows_with_identity_annotations_test": int((test["identity_annotator_count"] > 0).sum()) if "identity_annotator_count" in test.columns else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"Wrote summary → {out/'summary.json'}")


if __name__ == "__main__":
    main()
