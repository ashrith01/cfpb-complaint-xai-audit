"""
Token-level attribution via Captum's Integrated Gradients.

Contract:
    - explain(model, tokenizer, text) -> list[(token, attribution_score)]
    - Cache results per example to results/explanations/integrated_gradients.json
    - Use the same fixed example set across all three explainer modules
      (see src/explain/shap_explainer.py and attention_rollout.py) so
      results are directly comparable.
"""


def explain(model, tokenizer, text: str):
    """Return per-token attribution scores via Integrated Gradients."""
    raise NotImplementedError


def main():
    """Entry point for the `explain` step of `make explain` (IG portion)."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
