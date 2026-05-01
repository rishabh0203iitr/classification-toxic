"""End-to-end smoke test using the committed tiny subset.

Skipped if data/tiny/{train,val,test}.csv aren't present. Runs train + eval,
asserts AUC is computed and metrics.json is written. Should complete in
under ~3 minutes on CPU.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "data" / "tiny"


@pytest.mark.skipif(
    not (TINY / "train.csv").exists(), reason="data/tiny/train.csv missing"
)
def test_smoke_runs(monkeypatch, tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    # Run training
    from toxic_classifier import eval as ev_mod
    from toxic_classifier import train as tr_mod

    cfg_path = ROOT / "configs" / "smoke.yaml"
    artifacts = tmp_path / "artifacts"

    overrides = [
        f"train.ckpt_dir={artifacts}/ckpt",
        f"eval.ckpt={artifacts}/ckpt/best.pt",
        f"eval.out_dir={artifacts}/eval",
        f"tokenizer.path={artifacts}/tokenizer.json",
        "train.epochs=1",
    ]

    rc = tr_mod.main(["--config", str(cfg_path)] + sum([["--set", o] for o in overrides], []))
    assert rc == 0
    best = artifacts / "ckpt" / "best.pt"
    assert best.exists(), f"best.pt missing at {best}"

    rc2 = ev_mod.main(["--config", str(cfg_path)] + sum([["--set", o] for o in overrides], []))
    assert rc2 == 0
    metrics_path = artifacts / "eval" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "overall_auc" in metrics
    assert metrics["n"] > 0
