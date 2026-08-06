"""
Faithfulness scoring for each explainer: does the explanation actually
reflect the model's decision process, or does it just look plausible?

Contract:
    - TOP_K_PCT: fraction of tokens treated as "top attributed" -- decide
      once (default 10%), do not tune after seeing results.
    - comprehensiveness(model, tokenizer, text, attributions) -> float
        Remove top-k% attributed tokens, re-run model, measure prediction
        confidence drop. Higher drop = more faithful.
    - sufficiency(model, tokenizer, text, attributions) -> float
        Keep ONLY top-k% attributed tokens, re-run model, measure whether
        prediction still matches original. Higher match rate = more faithful.
    - score_all_methods() -> aggregated table across IG / SHAP / attention,
      written to results/metrics.json under 'faithfulness'
"""

TOP_K_PCT = 0.10


def comprehensiveness(model, tokenizer, text: str, attributions) -> float:
    raise NotImplementedError


def sufficiency(model, tokenizer, text: str, attributions) -> float:
    raise NotImplementedError


def score_all_methods():
    """Entry point for the faithfulness portion of `make audit`."""
    raise NotImplementedError


if __name__ == "__main__":
    score_all_methods()
