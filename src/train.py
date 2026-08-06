"""
Fine-tune a small transformer (DistilBERT baseline) for CFPB complaint
classification. Also run a TF-IDF + LogisticRegression baseline for comparison.

Contract:
    - run_baseline() -> dict: TF-IDF + LogReg, returns {"accuracy":, "macro_f1":}
    - train_config: small fixed grid (2-4 configs), not an open sweep
    - fine_tune(config) -> trained model + val metrics
    - select_best(results) -> best config by val macro-F1
    - evaluate_test(model) -> final test metrics + confusion matrix, saved to results/
"""

CONFIG_GRID = [
    # TODO: 2-4 configs max. Example shape:
    # {"lr": 2e-5, "epochs": 3, "max_len": 256, "batch_size": 16},
]


def run_baseline():
    """TF-IDF + LogisticRegression baseline. Write result to results/metrics.json under 'baseline'."""
    raise NotImplementedError


def fine_tune(config: dict):
    """Fine-tune DistilBERT with the given config. Return model + val metrics."""
    raise NotImplementedError


def evaluate_test(model, test_df):
    """Run once on held-out test set. Save accuracy, macro-F1, per-class F1, confusion matrix."""
    raise NotImplementedError


def main():
    """Entry point for `make train`."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
