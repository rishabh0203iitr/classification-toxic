# Project report — Toxic-comment classification

## 1. Problem framing

The Jigsaw Unintended Bias in Toxicity Classification dataset is a corpus of
~1.8 M online comments labelled with continuous toxicity scores in `[0, 1]`,
plus per-comment scores for 24 identity dimensions (e.g. `male`, `muslim`,
`black`, `psychiatric_or_mental_illness`). We define the binary
target as `toxic = (target >= 0.5)`.

The "Unintended Bias" angle is a important point to note: a naïve classifier
will score *any* comment that mentions a frequently-attacked identity
(e.g. "muslim", "gay") as toxic, because in the training data those tokens
co-occur with toxic content. The metric the competition uses, and the one we
implement here, decomposes into *Subgroup AUC*, *Background-Positive
Subgroup-Negative AUC*, and *Background-Negative Subgroup-Positive AUC* per
identity, then combines them via a generalized power mean (p = -5) so that
weakness on any single identity dominates the score. This helps detect a model that takes the
"mention an identity → predict toxic" shortcut.

We build a small **pre-LayerNorm Transformer encoder** from scratch using the 
HuggingFace `tokenizers` library to *train* a BPE vocabulary on the Jigsaw
training text.

## 2. Data analysis

Full reproducible analysis lives in [`scripts/eda.py`](scripts/eda.py); raw
outputs (CSV tables, full log, four plots) are committed under
[`docs/results/eda/`](docs/results/eda/). This section keeps only the
findings that drove a concrete modelling or engineering decision.

**Schema** — 1,804,874 rows × 45 columns. Five families: `id`, `comment_text`,
`target` + 6 sub-toxicity scores, **24 identity scores (annotated on only
22.4 % of rows)**, and engagement / publication metadata. Held-out test is
the union of `test_public_expanded.csv` + `test_private_expanded.csv`
(194,640 labelled rows); the bare `test.csv` is unlabelled and unused.

**Target.** Hyper-skewed: `mean=0.103, median=0, q95=0.6`. **70 %** of rows
are exactly 0; **8.0 %** are `target ≥ 0.5` (the binary positives); ~11.5 %
sit in the disagreement zone `[0.3, 0.7]` where annotators split. Decisions:
`pos_weight ≈ 11.5` in `BCEWithLogitsLoss`.

**Identity ↔ toxicity** — the bias problem in numbers. Toxic-rate among
comments *mentioning* each identity, vs. the 8 % base rate:

| Identity                  | P(toxic \| mentioned) | lift |
|---------------------------|----------------------:|-----:|
| black                     |                  31 % |  3.9× |
| homosexual_gay_or_lesbian |                  28 % |  3.5× |
| white                     |                  28 % |  3.5× |
| muslim                    |                  23 % |  2.8× |
| transgender               |                  21 % |  2.7× |

A bag-of-words classifier that uses on these tokens might look accurate but will
fail catastrophically.

![Toxicity rate by identity mention](docs/results/eda/identity_tox_rate.png)

**Identity sparsity & co-mention.** Only 22.4 % of rows are
identity-annotated; six identities have <100 mentions in train. Top
co-mention pairs include `male & female`, `black & white`,
`christian & muslim` — identity is multi-label, not categorical.
**Decision** → multi-label stratified train/val split on
(toxic × identity-presence) via `iterstrat` (`data/split.py`), so val gets
representative coverage of rare identities.

**Comment length.** Chars q95 = 953; whitespace-tokens q95 = 159; after BPE
the 95th-percentile compresses to ~120 tokens. **Decision** → `max_len=128`
in `configs/base.yaml`. Going to 256 helps only the longest 5 % at ~4×
attention compute.

**Sub-toxicity correlations with `target`** (Pearson):
`insult` 0.93 · `obscene` 0.49 · `identity_attack` 0.45 ·
`severe_toxicity` 0.39 · `threat` 0.29 · `sexual_explicit` 0.25.
**Decision** → `insult` is essentially a synonym; the rest are independent
axes that would make good auxiliary heads (future work).

**Train vs test drift.** Positive rates almost identical (8.00 % vs
7.94 %). Most identity mention rates within ±10 %; the only large outliers
are `intellectual_or_learning_disability` (2.4× more frequent in test) and
`atheist` (1.9×).

**Annotator counts.** Median 4 toxicity annotators per row

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
set is ultimately *evaluated* per-identity and single-label stratification
on the toxic label leaves rare-identity subgroup sizes to chance.

A unit test (`tests/test_split.py`) asserts no row-id overlap between
splits and that the split is reproducible from a seed.

## 4. Tokenisation

**Decision**: train a 30 k-vocabulary byte-level BPE on the Jigsaw training
text only, with NFKC normalisation and lowercasing. Special tokens are
`[PAD] (id=0)`, `[UNK] (id=1)`, `[CLS] (id=2)`. The CLS is prepended at
encode time; pooling defaults to taking the CLS hidden state.

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

1. **CPU and I/O work per epoch goes to zero.** BPE on 1.8 M comments is
   roughly 30–60 s/epoch even with `tokenizers`'s native parallelism;
   doing it inside `DataLoader` workers means the GPU waits.
3. **Disk footprint is tiny.** 1.8 M comments × avg ~80 tokens × 2 bytes
   (uint16) ≈ 280 MB — much less than the source CSV.
4. **Determinism + reuse.** A given preprocessing run is reproducible
   from a config, so multi-run / multi-config training amortises the cost.

The trade-off is a separate prepare step in the workflow (one extra make
target).

The `collate_fn` dynamically pads each batch to the longest sequence in
that batch (rather than always padding to `max_len`). On Jigsaw this is
a 2–3× compute saving for short batches.

Class imbalance is handled by `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)`
by default.

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
Architecture", ICML 2020.)

**Sizes:**

| Config | d_model | n_heads | n_layers | dim_ff | max_len | vocab  | params |
|--------|---------|---------|----------|--------|---------|--------|--------|
| smoke  |     64  |     2   |     2    |   128  |    64   | 4 096  | ~0.33 M |
| base   |    256  |     4   |     4    |  1024  |   128   | 30 000 | ~10 M   |

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
  `WANDB_MODE=disabled`.
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
- *BPSN AUC* — given non-toxic comments mentioning Subgroup and toxic comments
  not mentioning Subgroup, can the model rank them correctly? Tests for
  **false positives** on Subgroup.
- *BNSP AUC* — symmetric, tests for false negatives on Subgroup.

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

### Full-training results (this repo)

Trained from scratch with the committed `configs/base.yaml` (8 epochs, AMP, AdamW, ~10.9 M params, BPE vocab 30 k, max_len 128, single H100). **Checkpoint selection is by the Jigsaw bias-aware metric on val, not val AUC** (`cfg.train.best_metric = jigsaw`).

Per-epoch trajectory (W&B: [base-v3-jigsaw](https://wandb.ai/gvpatil-uw/toxic-classifier/runs/b8vq4xyx)):

| Epoch | train loss | train acc | val loss | val AUC | val acc@0.5 | **val Jigsaw** |
|------:|-----------:|----------:|---------:|--------:|------------:|----------------:|
|     0 |     0.6811 |    0.8541 |   0.6245 |  0.9319 |      0.8195 |          0.8706 |
|     1 |     0.6090 |    0.8678 |   0.6150 |  0.9362 |      0.8831 |          0.8789 |
|     2 |     0.5877 |    0.8705 |   0.6130 |  0.9365 |      0.8627 |          0.8797 |
|     3 |     0.5718 |    0.8739 |   0.6119 |  0.9386 |      0.8463 |          0.8814 |
|     4 |     0.5572 |    0.8766 |   0.6082 |  0.9383 |      0.8511 |          0.8790 |
| **5** | **0.5401** | **0.8803** | **0.6164** | **0.9407** | **0.8792** | **0.8834** ★ |
|     6 |     0.5231 |    0.8832 |   0.6323 |  0.9403 |      0.8833 |          0.8816 |
|     7 |     0.5103 |    0.8859 |   0.6364 |  0.9402 |      0.8836 |          0.8824 |

★ best epoch by val Jigsaw — saved to `best.pt`. Val loss bottoms out at
epoch 4 then climbs while val AUC plateaus, but val Jigsaw kept improving
to epoch 5: this is exactly the pattern bias-aware selection is supposed
to surface — a model that is no longer getting better at ranking overall
but is still getting fairer on the harder subgroups.

`val Jigsaw` here is computed with `train.val_min_subgroup_n = 30` so that
identities with fewer than 30 val examples don't crush the p=-5 power-mean
(the val set is 5 % of train; rare-identity AUCs are noisy at that size).
Test-time eval keeps the unfiltered metric — see the caveat at the end
of this section.

**Test set** — union of `test_public_expanded.csv` and `test_private_expanded.csv`,
n = 194,640:

| Metric | Value |
|---|---|
| Overall ROC-AUC | **0.9417** |
| Overall PR-AUC | 0.7020 |
| Accuracy @ 0.5 | 0.8807 |
| **Jigsaw bias metric** (the headline competition score) | **0.8473** |
| Subgroup-AUC power-mean (p = -5) | 0.7620 |
| BPSN-AUC power-mean (p = -5) | 0.8541 |
| BNSP-AUC power-mean (p = -5) | 0.8312 |

Per-identity Subgroup / BPSN / BNSP AUCs (the full breakdown is in
`docs/results/per_identity.csv`):

![Per-identity bias AUCs](docs/results/per_identity_aucs.png)

**Reading the plot.** The model is broadly strong (Subgroup AUC ≥ 0.80
on most identities), with weakness clustered on rare identities and on
the canonical "identity-mention → toxic" trap subgroups:

- `intellectual_or_learning_disability` BNSP = 0.53 (n=24) — the model
  badly under-detects toxic content directed at this group; the
  small-sample noise on this subgroup is the single biggest contributor
  to the test bias-metric's gap from the previous run.
- `other_religion` Subgroup = 0.61 (n=29), `heterosexual` = 0.65 (n=141)
  — small-sample noise dominates.
- `homosexual_gay_or_lesbian` (n=1065) and `bisexual` (n=34) Subgroup ≈
  0.77 — the classic shortcut failure: BNSP ≥ 0.95 (toxic content is
  detected when it mentions these identities), but BPSN ≈ 0.79 (false
  positives on identity-mentioning non-toxic comments). Same pattern on
  `black` (Subgroup 0.79, BPSN 0.78) and `white` (0.79 / 0.78).

**An honest caveat on the headline metric.** The test Jigsaw of 0.8473
is *slightly lower* than this repo's previous (4-epoch, val_auc-selected)
run that landed at 0.8652. Two factors:

1. The val-time Jigsaw uses `min_subgroup_n=30` for stability; the test
   Jigsaw is unfiltered. We are not directly optimizing the metric we
   report — we are optimizing a slightly more conservative version of it.
2. With only 24 test examples, `intellectual_or_learning_disability`
   BNSP is itself noisy (0.65 last run, 0.53 this run with the same
   architecture and seed).

The methodological point — that bias-aware selection picks epoch 5 over
epochs 6/7 even though their val AUCs are equivalent — stands. Closing
the val/test metric gap is the next step (use the same `min_subgroup_n`
on both, or weight subgroups by `sqrt(n)` instead of filtering); that's
on the future-work list.

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
