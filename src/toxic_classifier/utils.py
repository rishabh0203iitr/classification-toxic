"""Utility helpers: config loading, seeding, logging, checkpoint I/O."""
from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("toxic_classifier")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["__config_path__"] = str(path)
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply --set k.k.k=v overrides into a nested dict, parsing v as YAML scalar."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be 'a.b=c', got {ov!r}")
        k, v = ov.split("=", 1)
        keys = k.split(".")
        d = cfg
        for kk in keys[:-1]:
            d = d.setdefault(kk, {})
        d[keys[-1]] = yaml.safe_load(v)
    return cfg


def device_from_cfg(cfg: dict[str, Any]) -> torch.device:
    name = cfg["train"].get("device", "cuda")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def save_checkpoint(path: str | os.PathLike, state: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | os.PathLike, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


@dataclass
class AvgMeter:
    n: int = 0
    s: float = 0.0
    items: list[float] = field(default_factory=list)

    def update(self, v: float, k: int = 1) -> None:
        self.n += k
        self.s += v * k
        self.items.append(v)

    @property
    def avg(self) -> float:
        return self.s / max(self.n, 1)


def is_main_process() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0
