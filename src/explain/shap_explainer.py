"""
Token-level attribution via SHAP (text explainer).

Contract:
    - SHAP is slow on text -- subsample the test set (~500 examples) rather
      than running on the full set. Must be the SAME 500 examples used for
      integrated_gradients.py and attention_rollout.py.
    - explain(model, tokenizer, text) -> list[(token, shap_value)]
    - Cache results to results/explanations/shap.json
"""

SAMPLE_SIZE = 500


def explain(model, tokenizer, text: str):
    """Return per-token SHAP values."""
    raise NotImplementedError


def main():
    """Entry point for the SHAP portion of `make explain`."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
