.PHONY: setup lock test data train evaluate explain audit figures all clean

PY := .venv/bin/python

setup:
	uv venv --python 3.14 .venv
	uv pip sync requirements.txt

# Re-resolve requirements.in into a fully pinned requirements.txt.
# Only run this when adding or bumping a dependency.
lock:
	uv pip compile requirements.in -o requirements.txt --python-version 3.14

test:
	$(PY) -m pytest tests/ -q

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
