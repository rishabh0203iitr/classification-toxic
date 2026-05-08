.PHONY: help install install-dev smoke prepare train eval eda test lint format clean tiny

PYTHON ?= python
CONFIG ?= configs/base.yaml

help:
	@echo "Targets:"
	@echo "  install       - install runtime deps + package"
	@echo "  install-dev   - install dev deps (pytest, ruff)"
	@echo "  smoke         - run end-to-end smoke pipeline on data/tiny/ (CPU, <5 min)"
	@echo "  tiny          - regenerate data/tiny/* from data/full/train.csv"
	@echo "  prepare       - tokenise full Jigsaw data into data/processed/"
	@echo "  train         - train with CONFIG=configs/base.yaml (default)"
	@echo "  eval          - evaluate latest checkpoint on the test set"
	@echo "  eda           - run scripts/eda.py over data/full/ → docs/results/eda/"
	@echo "  test          - run pytest"
	@echo "  lint          - run ruff check"
	@echo "  format        - run ruff format"
	@echo "  clean         - remove caches, processed data, artifacts"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

smoke:
	WANDB_MODE=disabled $(PYTHON) -m toxic_classifier.train --config configs/smoke.yaml
	WANDB_MODE=disabled $(PYTHON) -m toxic_classifier.eval  --config configs/smoke.yaml

tiny:
	$(PYTHON) scripts/make_tiny_subset.py --src data/full --out data/tiny

prepare:
	$(PYTHON) -m toxic_classifier.data.prepare --config $(CONFIG)

train:
	$(PYTHON) -m toxic_classifier.train --config $(CONFIG)

eval:
	$(PYTHON) -m toxic_classifier.eval --config $(CONFIG)

eda:
	$(PYTHON) scripts/eda.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

clean:
	rm -rf artifacts/ checkpoints/ runs/ wandb/ data/processed/ data/splits/ \
	       .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
