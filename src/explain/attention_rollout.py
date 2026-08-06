"""
Token-level attribution via attention-weight rollout (the intentionally
weaker baseline explanation method -- see BRD for why this comparison matters).

Contract:
    - explain(model, tokenizer, text) -> list[(token, attention_score)]
    - Cache results to results/explanations/attention_rollout.json
    - Use the same fixed example set as the other two explainers.
"""


def explain(model, tokenizer, text: str):
    """Return per-token attention-rollout scores."""
    raise NotImplementedError


def main():
    """Entry point for the attention-rollout portion of `make explain`."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
