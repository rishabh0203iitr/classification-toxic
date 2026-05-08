"""Sanity tests for the bias-aware AUC metric."""
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
