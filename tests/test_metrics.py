"""Sanity tests for the bias-aware AUC metric."""
import math

import numpy as np

from toxic_classifier.metrics import (
    compute_per_identity_aucs,
    jigsaw_bias_metric,
    power_mean,
)


def test_perfect_predictor_gives_one():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = y.astype(float)  # exactly the labels
    idents = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    per = compute_per_identity_aucs(y, s, idents, ["A", "B"])
    for x in per:
        assert x.subgroup_auc == 1.0
        assert x.bpsn_auc == 1.0
        assert x.bnsp_auc == 1.0
    assert jigsaw_bias_metric(1.0, per) == 1.0


def test_random_predictor_around_half():
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.integers(0, 2, size=n)
    s = rng.uniform(0, 1, size=n)
    idents = (rng.uniform(0, 1, size=(n, 3)) > 0.5).astype(float)
    per = compute_per_identity_aucs(y, s, idents, ["A", "B", "C"])
    for x in per:
        assert 0.4 < x.subgroup_auc < 0.6
    val = jigsaw_bias_metric(0.5, per)
    assert 0.4 < val < 0.6


def test_power_mean_minus_five_penalises_min():
    """p=-5 power-mean should be much closer to the min than the arithmetic mean."""
    v = np.array([0.99, 0.99, 0.5])
    pm = power_mean(v, p=-5)
    am = float(v.mean())
    assert pm < am - 0.1


def test_compute_per_identity_aucs_filters_below_min_examples():
    """Subgroups with n_in_subgroup < min_examples should produce all-NaN
    per-identity rows so the downstream power-mean filter drops them."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    s = rng.uniform(0, 1, size=n)
    # Two identity columns: one common (~50% of rows), one rare (5 rows).
    common = (rng.uniform(0, 1, size=n) > 0.5).astype(float)
    rare = np.zeros(n)
    rare[:5] = 1.0
    idents = np.stack([common, rare], axis=1)

    from toxic_classifier.metrics import compute_per_identity_aucs

    per_id = compute_per_identity_aucs(y, s, idents, ["common", "rare"], min_examples=30)
    common_row, rare_row = per_id
    # common has plenty of examples → real numbers
    assert not math.isnan(common_row.subgroup_auc)
    # rare has 5 < 30 → filtered to NaN
    assert math.isnan(rare_row.subgroup_auc)
    assert math.isnan(rare_row.bpsn_auc)
    assert math.isnan(rare_row.bnsp_auc)
    assert rare_row.n_in_subgroup == 5


def test_jigsaw_metric_survives_all_nan_subgroup():
    """Degenerate case: no row in val mentions any tracked identity. The
    per-identity AUCs are all NaN; the jigsaw metric should not crash and
    should fall back to (or close to) the overall AUC."""
    from toxic_classifier.metrics import IdentityAUCs

    overall = 0.85
    per_id = [
        IdentityAUCs(name=f"id_{i}", n_in_subgroup=0,
                     subgroup_auc=float("nan"),
                     bpsn_auc=float("nan"),
                     bnsp_auc=float("nan"))
        for i in range(3)
    ]
    val = jigsaw_bias_metric(overall, per_id)
    # When all per-identity terms are NaN, only `overall` survives the
    # filter; the result equals overall.
    assert abs(val - overall) < 1e-9
