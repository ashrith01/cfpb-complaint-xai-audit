"""
Disagreement analysis across explanation methods -- this is where the
project's headline finding comes from.

Contract:
    - top_k_overlap(attr_a, attr_b, k=3) -> float
        % overlap of top-k tokens between two methods, per example.
    - find_confident_but_unfaithful(faithfulness_scores) -> list[example_id]
        Cases where a method assigns high attribution weight but scores
        low on faithfulness -- i.e. confidently wrong explanations.
    - summarize() -> writes results/figures/disagreement.png and a one-
      paragraph finding to be pasted into README.md's "Headline finding"
"""


def top_k_overlap(attr_a, attr_b, k: int = 3) -> float:
    raise NotImplementedError


def find_confident_but_unfaithful(faithfulness_scores):
    raise NotImplementedError


def summarize():
    """Entry point for `make audit` (final step, after faithfulness.py)."""
    raise NotImplementedError


if __name__ == "__main__":
    summarize()
