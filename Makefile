.PHONY: setup lock lock-serve test lint data train evaluate explain audit figures all clean \
        smoke backfill promote registry mlflow-ui export serve bench \
        docker-build docker-run docker-stop gate monitor

PY := .venv/bin/python
IMAGE := cfpb-classifier
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
export MLFLOW_DISABLE_AGENT_HINT := 1

setup:
	uv venv --python 3.14 .venv
	uv pip sync requirements.txt

# Re-resolve requirements.in into a fully pinned requirements.txt.
# Only run this when adding or bumping a dependency.
lock:
	uv pip compile requirements.in -o requirements.txt --python-version 3.14

# Serving image deps: CPU-only torch, pinned to the same versions as requirements.txt.
lock-serve:
	uv pip compile requirements-serve.in -c requirements.txt --torch-backend cpu \
		--python-version 3.14 --python-platform x86_64-manylinux_2_28 \
		--emit-index-url -o requirements-serve.txt

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

# Reproduces the exact 36,000 complaints pinned in data/splits.json.
# Add --resample to draw a fresh sample from a newer CFPB snapshot instead
# (this will change every downstream number).
data:
	$(PY) -m src.data_prep

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

explain:
	$(PY) -m src.explain.integrated_gradients
	$(PY) -m src.explain.shap_explainer
	$(PY) -m src.explain.attention_rollout

audit:
	$(PY) -m src.faithfulness
	$(PY) -m src.audit

figures:
	$(PY) -m src.report_figures

all: data train evaluate explain audit figures

clean:
	rm -rf data/processed/*.parquet results/explanations/*.json \
	       results/figures/* results/model

# ---- MLOps ------------------------------------------------------------------

# Real pipeline code on the 560-row fixture: data -> train (1 epoch) -> registry.
# Everything lands in .smoke/, never in data/ or results/.
smoke:
	rm -rf .smoke
	CFPB_SMOKE=1 $(PY) -m src.data_prep
	CFPB_SMOKE=1 $(PY) -m src.train
	CFPB_SMOKE=1 $(PY) -m src.smoke_check

# One-off: re-log the Day 2-4 runs from metrics.json and register the checkpoint.
backfill:
	$(PY) -m src.registry backfill

# make promote VERSION=3
promote:
	$(PY) -m src.registry promote --version $(VERSION)

registry:
	$(PY) -m src.registry show

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///$(CURDIR)/mlflow.db --port 5000

export:
	$(PY) -m src.registry export --dest build/model

# Host serving, model resolved from the registry by alias.
serve:
	$(PY) -m uvicorn src.serve:app --port 8000

bench:
	$(PY) -m src.loadtest --url http://127.0.0.1:8000

docker-build: export
	docker build --build-arg GIT_SHA=$(GIT_SHA) -t $(IMAGE):$(GIT_SHA) -t $(IMAGE):latest .

docker-run:
	docker compose up -d
	@echo "api    -> http://127.0.0.1:8000/docs"
	@echo "mlflow -> http://127.0.0.1:5000"

docker-stop:
	docker compose down

# Same checks the eval-gate workflow runs on every PR.
gate:
	$(PY) -m src.gate

monitor:
	$(PY) -m src.monitor
