# classification-toxic

A small **pre-LayerNorm Transformer encoder, built from scratch with stock
PyTorch layers**, trained for binary toxic-comment classification on the
[Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
dataset.

The repo demonstrates an end-to-end ML engineering workflow: a stratified
train/val/test split, an efficient data pipeline (with both an on-the-fly and
a pre-tokenised memmap path), the model architecture, a single- or multi-GPU
training loop, the official Jigsaw bias-aware evaluation, unit tests, and CI.

Verify the entire pipeline on a CPU by running the smoke pipeline against a tiny committed data subset
(`data/tiny/`) — see [Quick start](#quick-start) below.

**Full-training results** (single H100, 8 epochs, ~30 min training; checkpoint selection by Jigsaw bias metric on val):

| Metric                                                    | Value      |
|-----------------------------------------------------------|------------|
| Test overall ROC-AUC                                      | **0.9417** |
| Test PR-AUC                                               | 0.7020     |
| Test Accuracy @ 0.5                                       | 0.8807     |
| **Test Jigsaw bias metric** (headline, `min_subgroup_n=30`) | **0.8839** |
| Test Jigsaw bias metric (all 22 evaluable subgroups)      | 0.8473     |

Headline uses `min_subgroup_n=30` (matches val-time selection so we report
the metric we optimise). Per-identity breakdown including the rare tail —
which the headline filter excludes is in
[`docs/results/per_identity.csv`](docs/results/per_identity.csv) and the
chart there. Training curves are on W&B
([gvpatil-uw/toxic-classifier/runs/b8vq4xyx](https://wandb.ai/gvpatil-uw/toxic-classifier/runs/b8vq4xyx)).

The full design rationale and per-identity breakdown are in **[`report.md`](report.md)**.

---

## Repository layout

```
.
├── configs/                   # smoke.yaml (CPU) and base.yaml (single GPU)
├── data/
│   ├── tiny/                  # ~1500/250/250-row CSVs committed to the repo
│   ├── full/                  # the full Jigsaw CSVs (gitignored — see below)
│   └── README.md              # how to fetch the full dataset
├── scripts/make_tiny_subset.py
├── src/toxic_classifier/
│   ├── data/{tokenizer,split,dataset,prepare}.py
│   ├── model/{transformer,classifier}.py
│   ├── train.py  eval.py  infer.py  metrics.py  utils.py
├── tests/
├── pyproject.toml  requirements.txt  requirements-dev.txt
├── ci.yml.template            # drop into .github/workflows/ci.yml to enable CI
├── README.md
└── report.md
```

---

## Quick start (smoke test, CPU, ~75 s)

```bash
git clone https://github.com/rishabh0203iitr/classification-toxic.git
cd classification-toxic
python -m pip install -e .

# Train and evaluate on the committed data/tiny/ subset.
WANDB_MODE=disabled python -m toxic_classifier.train --config configs/smoke.yaml
WANDB_MODE=disabled python -m toxic_classifier.eval  --config configs/smoke.yaml
```

The smoke run trains a 333 k-parameter model for one epoch on the
committed `data/tiny/` CSVs, evaluates it on `data/tiny/test.csv`, and
writes `artifacts/smoke/eval/{metrics.json,per_identity.csv,per_identity_aucs.png}`.

It exercises the same code paths as the full pipeline — tokenizer
training, dataset, model forward, training loop, eval, and bias metric
— with a smaller config (4 k vocab, sequence length 64, one epoch).

---

## Setup (full training)

```bash
# 1. Install
python -m pip install -e ".[dev]"

# 2. Fetch the full Jigsaw competition data (~700 MB zip)
#    See data/README.md for instructions; place the CSVs in data/full/.

# 3. (Optional) sign in to Weights & Biases for experiment tracking
export WANDB_API_KEY="<your_key>"
export WANDB_PROJECT="toxic-classifier"
# To run without W&B: export WANDB_MODE=disabled

# 4. (Optional) reproduce the dataset analysis — writes CSVs + plots to docs/results/eda/
python scripts/eda.py
```

The committed EDA artefacts under [`docs/results/eda/`](docs/results/eda/)
already cover schema, target distribution, identity coverage + bias signal,
length, train/test drift, and 4 PNG plots — see `report.md` §2 for the
distilled findings that drove modelling decisions.

---

## Full training pipeline (single GPU)

```bash
# Pre-tokenise the full data once into data/processed/ (uint16 memmap).
python -m toxic_classifier.data.prepare --config configs/base.yaml

# Train (8 epochs, AMP, AdamW, warmup + cosine LR; checkpoint selection by val Jigsaw).
python -m toxic_classifier.train --config configs/base.yaml

# Evaluate the best checkpoint on the union of test_public_expanded + test_private_expanded.
python -m toxic_classifier.eval --config configs/base.yaml
```

Outputs land in `artifacts/base/`:
- `ckpt/{best.pt,last.pt}` — checkpoints (model weights + config + val metrics)
- `eval/metrics.json` — overall AUC, PR-AUC, per-identity table, Jigsaw bias metric
- `eval/per_identity_aucs.png` — bar chart of per-identity Subgroup / BPSN / BNSP AUCs

### Multi-GPU

The training loop supports `DistributedDataParallel`. Run with `torchrun`:

```bash
torchrun --nproc_per_node=2 -m toxic_classifier.train --config configs/base.yaml
```

The DataLoader sampler swaps to `DistributedSampler` automatically when
`RANK`/`WORLD_SIZE` env vars are set by `torchrun`.

---

## What's where

| Concern                    | File                                              |
|----------------------------|---------------------------------------------------|
| BPE tokenizer (trained from data, no pretrained weights) | `src/toxic_classifier/data/tokenizer.py` |
| Stratified train/val split (multi-label, identity-aware) | `src/toxic_classifier/data/split.py`     |
| Pre-tokenisation → memmap CLI                            | `src/toxic_classifier/data/prepare.py`   |
| Datasets (raw + memmap) + dynamic-pad collate            | `src/toxic_classifier/data/dataset.py`   |
| Pre-LN Transformer block                                  | `src/toxic_classifier/model/transformer.py` |
| Top-level classifier (embed → encoder → pool → head)      | `src/toxic_classifier/model/classifier.py`  |
| Training loop (AMP, DDP, warmup+cosine LR, ckpt, W&B)    | `src/toxic_classifier/train.py`          |
| Bias-aware metric (Subgroup / BPSN / BNSP / power-mean)  | `src/toxic_classifier/metrics.py`        |
| Eval driver                                               | `src/toxic_classifier/eval.py`           |
| Batched inference                                         | `src/toxic_classifier/infer.py`          |
| Unit + smoke tests                                        | `tests/`                                 |

---

## Testing & linting

```bash
pytest                          # all tests
pytest --ignore=tests/test_smoke.py   # fast subset (skip the smoke run)
ruff check .                    # lint
ruff format .                   # format in place
```

---

## Configuration

All entrypoints take `--config <path.yaml>` and `--set k.k.k=value` overrides.
Repeat `--set` for each override:

```bash
python -m toxic_classifier.train --config configs/base.yaml --set train.epochs=2 --set train.lr=1e-4
```

See `configs/{smoke,base}.yaml` for the full schema (data paths, tokenizer
options, model dims, optimisation, eval, W&B).

---

## License

MIT.
