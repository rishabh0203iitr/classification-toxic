# Project report — Toxic-comment classification with a from-scratch pre-LN Transformer

## 1. Problem framing

The Jigsaw Unintended Bias in Toxicity Classification dataset is a corpus of
~1.8 M online comments labelled with continuous toxicity scores in `[0, 1]`,
plus per-comment scores for 24 identity dimensions (e.g. `male`, `muslim`,
`black`, `psychiatric_or_mental_illness`). The brief defines the binary
target as `toxic = (target >= 0.5)`.

The "Unintended Bias" angle is the key to the dataset: a naïve classifier
will score *any* comment that mentions a frequently-attacked identity
(e.g. "muslim", "gay") as toxic, because in the training data those tokens
co-occur with toxic content. The metric the competition uses, and the one we
implement here, decomposes into *Subgroup AUC*, *Background-Positive
Subgroup-Negative AUC*, and *Background-Negative Subgroup-Positive AUC* per
identity, then combines them via a generalized power mean (p = -5) so that
weakness on any single identity dominates the score. This pushes models away
from the "mention an identity → predict toxic" shortcut.

The architectural constraint is "**no pretrained weights**". We build a
small **pre-LayerNorm Transformer encoder** from scratch using
`nn.Linear`, `nn.MultiheadAttention`, `nn.LayerNorm`. (We do use the
HuggingFace `tokenizers` library to *train* a BPE vocabulary on the Jigsaw
training text — no pretrained vocabulary is loaded; the trainer is just a
fast implementation of standard BPE.)

## 2. Data analysis

Raw schema:
- `train.csv` (~1.80 M rows): `id, target, comment_text`, 5 toxicity
  sub-labels (`severe_toxicity`, `obscene`, `identity_attack`, `insult`,
  `threat`), 24 identity scores, plus engagement / publication metadata.
- `test.csv` (~97 k rows): unlabelled input only.
- `test_public_expanded.csv` and `test_private_expanded.csv` (~97 k each):
  the labelled test sets. We use **both** as held-out test, since
  `test.csv` has no labels.

Distributional facts that drove the design:

- **Class imbalance**: ~8% of the training rows have `target ≥ 0.5`.
  This makes BCE with `pos_weight ≈ 11.5` (or weighted sampling) a
  reasonable default. We expose both via config; the default for `base.yaml`
  is `pos_weight=auto` (computed from the training set), and a
  `WeightedRandomSampler` is available as an alternative.
- **Length distribution**: comment lengths are heavy-tailed but the
  95th percentile is ~120 BPE tokens with a 30 k vocab. Using `max_len=128`
  for full training keeps almost all comments intact, while saving ~3×
  compute relative to a 512-token cap.
- **Identity sparsity**: the identity columns are heavily skewed —
  e.g. `male`/`female`/`christian` are common, while
  `intellectual_or_learning_disability` appears in <0.5% of comments.
  This matters for stratification (see §3) and for the bias metric, where
  a tiny subgroup can swing the power-mean disproportionately.
- **Identity ↔ toxicity correlation**: this is the headline phenomenon —
  in the training set, the toxic rate among comments mentioning some
  identities is markedly higher than the base rate, *not because the model
  should learn that identity → toxic*, but because the data collection
  process surfaced more identity-attacking comments. A bias-blind model
  trivially internalises this.

## 3. Splitting strategy

We treat the test split as **frozen**: the union of
`test_public_expanded.csv` and `test_private_expanded.csv`. It is never
used during training or model selection.

Train/val is split 95/5 from `train.csv`, stratified using a multi-label
vector that combines:

  - the binary toxic label, and
  - one bit per identity column (mentioned vs. not, at the same 0.5 threshold).

We use `iterstrat.MultilabelStratifiedShuffleSplit` for this. We chose
multi-label-aware stratification rather than single-label because the val
set is ultimately *evaluated* per-identity — and single-label stratification
on the toxic label leaves rare-identity subgroup sizes to chance.

A unit test (`tests/test_split.py`) asserts no row-id overlap between
splits and that the split is reproducible from a seed.

## 4. Tokenisation

**Decision**: train a 30 k-vocabulary byte-level BPE on the Jigsaw training
text only, with NFKC normalisation and lowercasing. Special tokens are
`[PAD] (id=0)`, `[UNK] (id=1)`, `[CLS] (id=2)`. The CLS is prepended at
encode time; pooling defaults to taking the CLS hidden state.

We use the HuggingFace `tokenizers` library purely as a fast trainer —
no pretrained vocabulary is loaded. The output is a single
`tokenizer.json` we own.

**On-the-fly vs. pre-processed — the engineering decision**.

Two paths are implemented:

- `RawJigsawDataset` (mode `raw`): reads a CSV into memory at
  construction, tokenises lazily in `__getitem__`. Convenient for the
  smoke test and unit tests.
- `MemmapJigsawDataset` (mode `memmap`): reads pre-tokenised token ids
  from a uint16 memmap, with sidecar arrays for per-row lengths, labels,
  raw targets, and identities.

For full-scale training we pre-tokenise once (`make prepare`) and store
the result as memmap. Reasoning:

1. **CPU work per epoch goes to zero.** BPE on 1.8 M comments is
   roughly 30–60 s/epoch even with `tokenizers`'s native parallelism;
   doing it inside `DataLoader` workers means the GPU waits.
2. **Memmap is mmap'd; the OS page cache handles random access.**
   No I/O queues, no per-worker file handles. Random sampling is
   essentially memory access.
3. **Disk footprint is tiny.** 1.8 M comments × avg ~80 tokens × 2 bytes
   (uint16) ≈ 280 MB — much less than the source CSV.
4. **Determinism + reuse.** A given preprocessing run is reproducible
   from a config, so multi-run / multi-config training amortises the cost.

The trade-off is a separate prepare step in the workflow (one extra make
target). For the smoke test that doesn't matter — we use `mode=raw`
there so the reviewer doesn't need a multi-step setup.

The `collate_fn` dynamically pads each batch to the longest sequence in
that batch (rather than always padding to `max_len`). On Jigsaw this is
a 2–3× compute saving for short batches.

Class imbalance is handled by `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)`
by default; `WeightedRandomSampler` is also available.

## 5. Model architecture

`PreLNEncoderBlock` (`src/toxic_classifier/model/transformer.py`):

```
x →  x + Drop( Attn( LN(x) ) )
  →  x + FFN( LN(x) )
```

with `Attn = nn.MultiheadAttention(batch_first=True)` and
`FFN = Linear(d, dim_ff) → GELU → Dropout → Linear(dim_ff, d) → Dropout`.
Padding is handled via `key_padding_mask` (True == pad).

`ToxicClassifier` (`src/toxic_classifier/model/classifier.py`):

```
ids ─► nn.Embedding(vocab, d) (×√d) + nn.Embedding(max_len, d)
     ─► Dropout → N × PreLNEncoderBlock → final LayerNorm
     ─► CLS hidden state (or masked mean) → nn.Linear(d, 1)
```

**Why pre-LN** rather than post-LN? At small scale and with a
hand-rolled training loop, pre-LN is materially easier to train stably
without aggressive learning-rate warmup — the residual path has unit
norm at init, so gradient magnitudes don't blow up at deep layers
during the first few hundred steps. For a 4-layer model, post-LN can
work but tends to require careful warmup / temperature tuning.
(See Xiong et al., "On Layer Normalization in the Transformer
Architecture", ICML 2020.) For an interview project where reviewers
should be able to reproduce the run on a single GPU without surprises,
the robustness wins.

**Sizes:**

| Config | d_model | n_heads | n_layers | dim_ff | max_len | vocab  | params |
|--------|---------|---------|----------|--------|---------|--------|--------|
| smoke  |     64  |     2   |     2    |   128  |    64   | 4 096  | ~0.33 M |
| base   |    256  |     4   |     4    |  1024  |   128   | 30 000 | ~10 M   |

A 10 M-param model is small by NLP standards but appropriate when training
from scratch on ~1.8 M comments with no transfer learning.

**Initialisation**: `trunc_normal_(std=0.02)` on `Linear` weights and
embeddings, zeros on biases, ones on `LayerNorm` weights — a common
recipe that matches the GPT/BERT lineage.

## 6. Training

`src/toxic_classifier/train.py`:

- **Optimiser**: AdamW with `(β1, β2) = (0.9, 0.98)`. Weight decay 0.01
  on weight matrices, **0.0 on biases and LayerNorm scales** (this is
  the standard "no decay on 1-d parameters" trick — decaying LayerNorm
  scales pulls them towards zero and slows convergence).
- **Schedule**: linear warmup over `warmup_steps` → cosine decay to 0
  over the remaining steps. Default 1 000 warmup steps for `base.yaml`.
- **Mixed precision**: `torch.amp.autocast` + `GradScaler`, enabled
  automatically on CUDA (and disabled on CPU/smoke).
- **Gradient clipping**: ‖grad‖₂ ≤ 1.0.
- **Gradient accumulation**: configurable, defaults to 1 — useful when
  scaling to a smaller GPU.
- **Checkpointing**: `best.pt` (by val AUC) and `last.pt`, both
  containing the model state, the full config, and the val metrics that
  produced them. `infer.py` and `eval.py` resurrect the model from the
  checkpoint's `config["model"]` block — there is no separate
  "architecture file" the user must remember.
- **Logging**: Weights & Biases (`wandb`). Step-level loss and LR;
  per-epoch val AUC, PR-AUC, accuracy. The smoke test sets
  `WANDB_MODE=disabled` so reviewers don't need a W&B account.
- **Multi-GPU**: `torchrun --nproc_per_node=N -m toxic_classifier.train`
  enables `DistributedDataParallel` and a `DistributedSampler`. Single-GPU
  remains the default and `base.yaml` is sized for it.
- **Seed**: `random`, `numpy`, `torch`, `torch.cuda` all seeded from
  `cfg.seed`. Deterministic CUDNN is opt-in (it can hurt throughput
  significantly).

## 7. Evaluation

Headline numbers reported by `eval.py`:

- Overall ROC-AUC, PR-AUC, accuracy at the operating threshold.
- The **Jigsaw bias-aware metric**:
  - per identity: `Subgroup AUC`, `BPSN AUC`, `BNSP AUC`
  - per-identity-AUC arrays are reduced via the generalized power mean
    (p = -5 — this is a stiff penalty on the worst identity)
  - the four reduced quantities are weighted-averaged with `w = 0.25`
    on overall AUC.

Why this metric, in plain language:
- *Subgroup AUC* — does the model rank toxic vs. non-toxic *within* an
  identity? (Catches general weakness on a subgroup.)
- *BPSN AUC* — given non-toxic comments mentioning S and toxic comments
  not mentioning S, can the model rank them correctly? Tests for
  **false positives** on S — the "every comment mentioning gay is
  flagged" failure mode.
- *BNSP AUC* — symmetric, tests for false negatives on S.

Outputs land in `artifacts/<run>/eval/`:

- `metrics.json` — overall + per-identity table + the reduced metric.
- `per_identity.csv` — the table as a flat CSV.
- `per_identity_aucs.png` — bar chart of per-identity Subgroup / BPSN /
  BNSP AUCs. This plot is the single most informative output for this
  problem; it highlights which identities the model is silently failing on.

Smoke run example (1 epoch, ~1500 train rows, CPU, ~75 s wall-clock):

```json
{
  "n": 250,
  "overall_auc": 0.62,
  "overall_pr_auc": 0.47,
  "jigsaw_bias_metric": 0.55,
  "subgroup_auc_pmean": 0.52,
  "bpsn_auc_pmean": 0.52,
  "bnsp_auc_pmean": 0.54
}
```

These are intentionally weak — they prove the pipeline works
end-to-end on tiny data, not that the model is good.

### Full-training results (this repo)

Trained from scratch on a single H100 NVL GPU with the committed
`configs/base.yaml` (4 epochs, AMP, AdamW, ~10.9 M params, BPE vocab 30 k,
max_len 128). Total wall-clock from cold start to evaluated test metrics:

- prepare (BPE training + memmap encode of 1.8 M comments) — ~4.5 min
- training (4 epochs × ~217 s per epoch including val) — ~14.5 min
- evaluation (194 640 held-out test rows) — ~10 s

Per-epoch trajectory (logged to W&B as `train/epoch_{loss,acc}` and `val/{loss,auc,acc@0.5}`):

| Epoch | train loss | train acc | val loss | val AUC | val acc@0.5 |
|------:|-----------:|----------:|---------:|--------:|------------:|
|     0 |     0.6794 |    0.8545 |   0.6189 |  0.9320 |      0.8300 |
|     1 |     0.6024 |    0.8677 |   0.5887 |  0.9388 |      0.8709 |
|     2 |     0.5637 |    0.8750 |   0.5708 |  0.9416 |      0.8680 |
|     3 |     0.5300 |    0.8805 |   0.5954 |  0.9417 |      0.8861 |

Val loss bottoms out at epoch 2 and ticks up at epoch 3 while val AUC plateaus
— the classic "starting to overfit" signal. With more epochs we'd want early
stopping on `val/loss` (or on the bias metric) rather than blindly running to
the configured epoch count.

**Test set** — union of `test_public_expanded.csv` and `test_private_expanded.csv`,
n = 194,640:

| Metric | Value |
|---|---|
| Overall ROC-AUC | **0.9425** |
| Overall PR-AUC | 0.7073 |
| Accuracy @ 0.5 | 0.8875 |
| **Jigsaw bias metric** (the headline competition score) | **0.8652** |
| Subgroup-AUC power-mean (p = -5) | 0.7769 |
| BPSN-AUC power-mean (p = -5) | 0.8526 |
| BNSP-AUC power-mean (p = -5) | 0.8889 |

Per-identity Subgroup / BPSN / BNSP AUCs (the full breakdown is in
`docs/results/per_identity.csv`):

![Per-identity bias AUCs](docs/results/per_identity_aucs.png)

**Reading the plot.** The model is broadly strong (Subgroup AUC ≥ 0.80
on most identities), but a few subgroups stand out as weak:

- `other_religion` Subgroup AUC = 0.54 — only 29 examples; small-sample
  noise dominates this estimate.
- `homosexual_gay_or_lesbian` and `bisexual` Subgroup AUC ≈ 0.78 — the
  classic "identity-mention → toxic" failure mode that this dataset's
  bias metric is designed to surface; the BNSP at 0.95 confirms the
  model misses rather than over-predicts.
- `intellectual_or_learning_disability` BNSP = 0.65 — small subgroup
  (24 examples), and the model under-detects toxic content involving it.

These match the intuition that with no pretrained weights and just 4
epochs of training, the model's biggest gaps are on rare identities
where there isn't enough in-distribution signal to learn the
identity-vs.-attack distinction. They are the priority targets for
identity-aware re-weighting in §10.

**W&B run** (training curves, configs, hardware): [`gvpatil-uw/toxic-classifier/runs/29bqsxc5`](https://wandb.ai/gvpatil-uw/toxic-classifier/runs/29bqsxc5) (`base-v2`).

## 8. Inference

`src/toxic_classifier/infer.py` is a batched-inference CLI: takes a
checkpoint + tokenizer + a CSV with `comment_text`, writes a copy of the
CSV with an extra `toxic_score` column. Uses dynamic-pad batching, so
short comments don't waste GPU time. For larger workloads, `--num-workers`
controls CPU dataloading parallelism; the model itself is small enough
to run at >1 000 examples/s on a single modern GPU.

Future inference paths worth considering: TorchScript / ONNX export
(model is plain `nn.Module` with no graph quirks, so this is a one-liner),
INT8 dynamic quantisation (head + linear-heavy layers benefit), and a
small FastAPI wrapper for an HTTP endpoint.

## 9. Software / packaging

- **Python package** (`src/toxic_classifier/`) with a `pyproject.toml` and
  three console scripts (`toxic-train`, `toxic-eval`, `toxic-prepare`).
  Editable installs with `pip install -e .`.
- **Configs** are YAML, deliberately flat enough to be read at a glance.
  `--config configs/foo.yaml --set k.k=v` allows ad-hoc overrides without
  copy-pasting configs.
- **Makefile** is the canonical entrypoint. The two single commands a
  reviewer ever needs are `make smoke` and `make train`.
- **Tests** (`pytest`) cover model forward shapes, padding-mask
  invariance, split reproducibility / no-leakage, the bias-metric on a
  perfect predictor, and an end-to-end smoke run. `tests/test_smoke.py`
  is the same pipeline `make smoke` runs, exercised via the same Python
  API to guarantee they do not drift.
- **Lint**: ruff with a small ignore list (`E501`, `N806`) appropriate to
  ML code.
- **CI** (`.github/workflows/ci.yml`) runs lint + unit tests + the smoke
  pipeline on every push.
- **Secrets management**: the W&B API key is consumed via
  `WANDB_API_KEY`, never written into a config or committed. The git PAT
  used to push the repo lives only in `.git/config` and is recommended to
  be rotated.

## 10. Future work

In rough priority order:

- **Bigger model + longer training**. d_model=384, 6 layers, 8 heads,
  batch=512 with gradient accumulation, 6+ epochs would close most of
  the remaining gap to a fine-tuned BERT-base baseline (which typically
  scores ~0.94 on this benchmark) without violating the no-pretrained-
  weights rule.
- **Identity-aware re-weighting**. Reweight training examples so that
  the loss landscape directly mirrors the bias metric (down-weight
  bias-easy examples, up-weight identity-mentioning ones). Several
  Jigsaw winners used a variant of this.
- **Focal loss / loss shaping**. With ~8% positives, focal loss helps a
  small but consistent amount.
- **Knowledge distillation**. The constraint is "no pretrained weights",
  but it would be informative to compare against (and distil from) a
  full BERT-base fine-tune on the same data — both as a quality ceiling
  and to argue the parameter-efficiency story.
- **Quantisation + ONNX export + latency benchmarks**. The inference
  path is small enough that INT8 quantisation should be near-lossless.
- **Streaming / online inference**. Wrap the model in a tiny FastAPI
  service, add a `/predict` endpoint, and benchmark p50/p99 latency vs
  batch size.
- **Drift monitoring**. Online comment streams are non-stationary:
  vocabulary, slurs, and topic distribution shift over time. A simple
  cron job that runs the eval pipeline on a recent week's labelled
  sample and tracks subgroup-AUC time series would catch silent
  regressions early.
- **Dataset additions**. Jigsaw Toxic Comment Classification Challenge
  (the predecessor) and Civil Comments Toxicity provide complementary
  English-language toxic content; combining them improves robustness
  on the long tail of identities. Multi-lingual extension would be a
  larger project.

## 11. Limitations

- **Single English-language dataset**. The model has no chance of
  generalising to other languages, dialects, or platforms.
- **Label noise**. Toxicity is inherently subjective. The `target`
  scores are aggregated from multiple annotators; rows where annotators
  disagreed contribute noise that this model can't disentangle from
  signal.
- **Bias metric is a proxy, not a guarantee of fairness**. A high
  Jigsaw bias score does not certify that the model is "fair" on any
  particular real-world axis; it certifies that on the 24 identity
  columns in this dataset, the model's worst-subgroup AUC is decent.
  Real-world fairness review would also need disparate-impact analysis,
  user studies, and stakeholder-defined harm definitions.
- **No human evaluation**. AUC measures rank quality, not how the
  thresholded predictions feel to a content moderator. Production
  deployment would require threshold tuning per use case.
