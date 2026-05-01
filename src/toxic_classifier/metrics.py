"""Metrics for the Jigsaw Unintended Bias task.

We implement the official competition metric: a generalized power mean
(p = -5) of:
  - the overall ROC-AUC, and
  - the per-identity Subgroup, BPSN, and BNSP AUCs.

Definitions (per identity subgroup S, with target y ∈ {0,1}):
  - Subgroup AUC = AUC restricted to examples mentioning S.
  - BPSN  AUC = AUC over (background-positive, subgroup-negative) — i.e.
                positive examples that DO NOT mention S, plus negative
                examples that DO mention S. Penalises false positives on S.
  - BNSP  AUC = symmetric: negatives outside S + positives inside S.
                Penalises false negatives on S.

Reference: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-9


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC; returns NaN if a class is missing (so the caller can drop it)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def power_mean(values: np.ndarray, p: float = -5.0) -> float:
    """Generalized mean used by the Jigsaw competition (p = -5)."""
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan")
    v = np.clip(v, EPS, None)
    return float(np.mean(v**p) ** (1.0 / p))


@dataclass
class IdentityAUCs:
    name: str
    n_in_subgroup: int
    subgroup_auc: float
    bpsn_auc: float
    bnsp_auc: float


def compute_per_identity_aucs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    identities: np.ndarray,
    identity_names: list[str],
    identity_threshold: float = 0.5,
    min_examples: int = 1,
) -> list[IdentityAUCs]:
    """Compute Subgroup/BPSN/BNSP for each identity column.

    `identities`: (N, K) float in [0,1]; thresholded for membership.
    """
    y_true = np.asarray(y_true).astype(np.int8).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    out: list[IdentityAUCs] = []
    for k, name in enumerate(identity_names):
        in_S = identities[:, k] >= identity_threshold
        n_in = int(in_S.sum())
        if n_in < min_examples:
            out.append(IdentityAUCs(name, n_in, float("nan"), float("nan"), float("nan")))
            continue
        # Subgroup
        subgroup = _safe_auc(y_true[in_S], y_score[in_S])
        # BPSN: positives outside S + negatives inside S
        bpsn_mask = ((~in_S) & (y_true == 1)) | (in_S & (y_true == 0))
        bpsn = _safe_auc(y_true[bpsn_mask], y_score[bpsn_mask])
        # BNSP: negatives outside S + positives inside S
        bnsp_mask = ((~in_S) & (y_true == 0)) | (in_S & (y_true == 1))
        bnsp = _safe_auc(y_true[bnsp_mask], y_score[bnsp_mask])
        out.append(IdentityAUCs(name, n_in, subgroup, bpsn, bnsp))
    return out


def jigsaw_bias_metric(
    overall_auc: float,
    per_identity: list[IdentityAUCs],
    p: float = -5.0,
    weight_overall: float = 0.25,
) -> float:
    """Final scalar: weighted (power-mean of [overall_auc, generalized means
    of subgroup/BPSN/BNSP])."""
    sg = np.array([x.subgroup_auc for x in per_identity], dtype=np.float64)
    bpsn = np.array([x.bpsn_auc for x in per_identity], dtype=np.float64)
    bnsp = np.array([x.bnsp_auc for x in per_identity], dtype=np.float64)
    parts = [
        overall_auc,
        power_mean(sg, p=p),
        power_mean(bpsn, p=p),
        power_mean(bnsp, p=p),
    ]
    parts = [x for x in parts if not (x is None or math.isnan(x))]
    if not parts:
        return float("nan")
    arr = np.array(parts, dtype=np.float64)
    weights = np.array([weight_overall] + [(1.0 - weight_overall) / max(len(arr) - 1, 1)] * (len(arr) - 1))
    weights = weights[: len(arr)]
    weights = weights / weights.sum()
    # Weighted arithmetic mean (per the official scoring, the four are
    # arithmetically averaged with overall weight = 0.25).
    return float(np.sum(arr * weights))
