.PHONY: data train explain audit all clean

data:
	python -m src.data_prep

train:
	python -m src.train

explain:
	python -m src.explain.integrated_gradients
	python -m src.explain.shap_explainer
	python -m src.explain.attention_rollout

audit:
	python -m src.faithfulness
	python -m src.audit

all: data train explain audit

clean:
	rm -rf data/processed/* results/explanations/* results/figures/*
