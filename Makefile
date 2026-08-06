.PHONY: setup lock data train explain audit all clean

PY := .venv/bin/python

setup:
	uv venv --python 3.14 .venv
	uv pip sync requirements.txt

# Re-resolve requirements.in into a fully pinned requirements.txt.
# Only run this when adding or bumping a dependency.
lock:
	uv pip compile requirements.in -o requirements.txt --python-version 3.14

data:
	$(PY) -m src.data_prep

train:
	$(PY) -m src.train

explain:
	$(PY) -m src.explain.integrated_gradients
	$(PY) -m src.explain.shap_explainer
	$(PY) -m src.explain.attention_rollout

audit:
	$(PY) -m src.faithfulness
	$(PY) -m src.audit

all: data train explain audit

clean:
	rm -rf data/processed/* results/explanations/* results/figures/*
