# classification-toxic

A small **pre-LayerNorm Transformer encoder, built from scratch with stock
PyTorch layers**, trained for binary toxic-comment classification on the
[Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
dataset.

The repo demonstrates an end-to-end ML engineering workflow: a stratified
train/val/test split, an efficient data pipeline (with both an on-the-fly and
a pre-tokenised memmap path), the model architecture, a single- or multi-GPU
training loop, the official Jigsaw bias-aware evaluation, unit tests, and CI.

A reviewer can verify the entire pipeline in **under 2 minutes on a CPU** by
running `make smoke` against a tiny committed data subset (`data/tiny/`) — see
[Quick start](#quick-start) below.

The full design rationale is in **[`report.md`](report.md)**.

---

## Repository layout

```
.
├── configs/                   # smoke.yaml (CPU) and base.yaml (single GPU)
├── data/
│   ├── tiny/                  # ~1500/250/250-row CSVs committed to the repo
│   ├── full/                  # the full Jigsaw CSVs (gitignored — see below)
│   └── README.md              # how to fetch the full dataset
├── notebooks/                 # 01_eda.ipynb (data analysis)
├── scripts/make_tiny_subset.py
├── src/toxic_classifier/
│   ├── data/{tokenizer,split,dataset,prepare}.py
│   ├── model/{transformer,classifier}.py
│   ├── train.py  eval.py  infer.py  metrics.py  utils.py
├── tests/
├── Makefile
├── pyproject.toml  requirements.txt  requirements-dev.txt
├── .github/workflows/ci.yml
├── README.md
└── report.md
```

---

## Quick start (smoke test, CPU, ~75 s)

```bash
git clone https://github.com/rishabh0203iitr/classification-toxic.git
cd classification-toxic
python -m pip install -e .
make smoke
```

`make smoke` trains a 333 k-parameter model for one epoch on the committed
`data/tiny/` CSVs, evaluates it on `data/tiny/test.csv`, and writes
`artifacts/smoke/eval/{metrics.json,per_identity.csv,per_identity_aucs.png}`.

This is identical to the full pipeline — just with a smaller config, a 4 k
vocab, sequence length 64, and one epoch — so it exercises every module
(tokenizer training, dataset, model forward, training loop, eval, metrics).

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
```

---

## Full training pipeline (single GPU)

```bash
# Pre-tokenise the full data once into data/processed/ (uint16 memmap)
make prepare CONFIG=configs/base.yaml

# Train (4 epochs, AMP, AdamW, warmup + cosine LR)
make train CONFIG=configs/base.yaml

# Evaluate the best checkpoint on the union of test_public_expanded + test_private_expanded
make eval CONFIG=configs/base.yaml
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
make test            # pytest
make lint            # ruff check
make format          # ruff format
```

---

## Configuration

All entrypoints take `--config <path.yaml>` and `--set k.k.k=value` overrides.
For example:

```bash
python -m toxic_classifier.train --config configs/base.yaml --set train.epochs=2 train.lr=1e-4
```

See `configs/{smoke,base}.yaml` for the full schema (data paths, tokenizer
options, model dims, optimisation, eval, W&B).

---

## License

MIT.
