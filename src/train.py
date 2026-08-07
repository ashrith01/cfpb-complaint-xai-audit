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

from __future__ import annotations

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.utils import load_label_map, load_split, update_metrics

SEED = 42

CONFIG_GRID = [
    # TODO: 2-4 configs max. Example shape:
    # {"lr": 2e-5, "epochs": 3, "max_len": 256, "batch_size": 16},
]

# Deliberately untuned. This is the floor the fine-tune has to clear, so tuning
# it would be tuning the goalpost -- a stronger baseline is only worth building
# if the fine-tune fails to beat this one.
BASELINE_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_features": 300_000,
    "sublinear_tf": True,
    "C": 1.0,
    "max_iter": 2000,
}


def run_baseline():
    """TF-IDF + LogisticRegression baseline. Write result to results/metrics.json under 'baseline'."""
    train_df, val_df = load_split("train"), load_split("val")
    _, id2label, short = load_label_map()

    cfg = BASELINE_CONFIG
    vec = TfidfVectorizer(
        ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"],
        max_features=cfg["max_features"],
        sublinear_tf=cfg["sublinear_tf"],
        strip_accents="unicode",
        lowercase=True,
    )

    t0 = time.time()
    x_train = vec.fit_transform(train_df["narrative"])
    x_val = vec.transform(val_df["narrative"])
    clf = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], random_state=SEED)
    clf.fit(x_train, train_df["label"])
    elapsed = time.time() - t0

    pred = clf.predict(x_val)
    y_val = val_df["label"]
    accuracy = float(accuracy_score(y_val, pred))
    macro_f1 = float(f1_score(y_val, pred, average="macro"))
    per_class = f1_score(y_val, pred, average=None, labels=sorted(id2label))
    per_class_f1 = {short[id2label[i]]: float(f) for i, f in enumerate(per_class)}

    payload = {
        "model": "tfidf+logreg",
        "config": {**cfg, "ngram_range": list(cfg["ngram_range"])},
        "seed": SEED,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_features": int(x_train.shape[1]),
        "fit_seconds": round(elapsed, 1),
        "val": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
        },
    }
    update_metrics("baseline", payload)

    print(f"\nBaseline: TF-IDF({cfg['ngram_range'][0]}-{cfg['ngram_range'][1]}gram, "
          f"{x_train.shape[1]:,} features) + LogisticRegression")
    print(f"  fit on {len(train_df):,} train docs in {elapsed:.1f}s")
    print(f"\n  val accuracy   {accuracy:.4f}")
    print(f"  val macro-F1   {macro_f1:.4f}   <- floor for the fine-tune to beat")
    print("\n  per-class F1 (val):")
    for name, f in sorted(per_class_f1.items(), key=lambda kv: kv[1]):
        print(f"    {name:<20} {f:.4f}")
    print(f"\n  written to results/metrics.json under 'baseline'")

    return {"accuracy": accuracy, "macro_f1": macro_f1}


def fine_tune(config: dict):
    """Fine-tune DistilBERT with the given config. Return model + val metrics."""
    raise NotImplementedError


def evaluate_test(model, test_df):
    """Run once on held-out test set. Save accuracy, macro-F1, per-class F1, confusion matrix."""
    raise NotImplementedError


def main():
    """Entry point for `make train`.

    Day 2 runs the baseline only; the fine-tune stage lands on Day 3-4 and will
    be appended here so `make train` produces both numbers in one pass.
    """
    run_baseline()


if __name__ == "__main__":
    main()
