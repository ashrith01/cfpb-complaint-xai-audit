"""
The fixed example set all three explainers run on.

Day 8's DoD requires IG, SHAP and attention rollout to produce attributions on
*the same* examples, so the selection lives here and each explainer imports it
rather than sampling independently.

Two constraints drive the design:

1. **NFR3 (<30 min for explanation generation).** Integrated Gradients was
   benchmarked at 1.31 s/example (n_steps=50) on this hardware, so the full
   5,400-example test set would take ~118 min. 500 examples takes ~11 min and
   leaves headroom for SHAP, which is slower still.

2. **Day 5's error analysis.** A random 500 would be ~85% correct predictions
   and would land mostly on easy cases where every method agrees and the
   comparison is uninformative. The interesting population is the 212
   confidently-wrong predictions -- where a plausible-looking explanation is
   most likely to be believed and most likely to mislead.

So errors are deliberately oversampled to half the set. That makes raw aggregate
faithfulness scores *unrepresentative of the test distribution*, which is why
every row carries its stratum and an inverse-sampling `weight`: Days 9-10 must
report per-stratum numbers, and any test-set-level claim must use the weights.
"""

from __future__ import annotations

import pandas as pd

from src.evaluate import predict_test
from src.train import SEED
from src.utils import RESULTS_DIR, load_label_map

EXAMPLE_SET_PATH = RESULTS_DIR / "explanations" / "example_set.parquet"
N_EXAMPLES = 500
CONFIDENT = 0.9

# stratum -> target count. Errors get half the budget despite being 15% of the
# test set; see module docstring.
ALLOCATION = {
    "confident_wrong": 150,
    "uncertain_wrong": 100,
    "confident_correct": 150,
    "uncertain_correct": 100,
}


def _stratum(row) -> str:
    conf = "confident" if row.confidence > CONFIDENT else "uncertain"
    return f"{conf}_{'correct' if row.correct else 'wrong'}"


def build_example_set(force: bool = False) -> pd.DataFrame:
    """Select and cache the stratified example set. Deterministic given SEED."""
    if EXAMPLE_SET_PATH.exists() and not force:
        return pd.read_parquet(EXAMPLE_SET_PATH)

    preds = predict_test()
    _, id2label, short = load_label_map()
    preds = preds.copy()
    preds["stratum"] = [_stratum(r) for r in preds.itertuples()]

    chunks = []
    for stratum, target in ALLOCATION.items():
        pool = preds[preds["stratum"] == stratum]
        if len(pool) <= target:
            picked = pool
        else:
            # Spread evenly across true classes inside the stratum so no single
            # class dominates, then top up at random if a class is short.
            per_class = target // len(id2label)
            picked = (
                pool.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(per_class, len(g)), random_state=SEED))
            )
            short_by = target - len(picked)
            if short_by > 0:
                rest = pool.drop(picked.index)
                picked = pd.concat(
                    [picked, rest.sample(n=min(short_by, len(rest)), random_state=SEED)]
                )
        # Inverse sampling weight: reweights this stratum back to its true share
        # of the test set, so aggregate claims can be made honestly.
        picked = picked.assign(weight=len(pool) / len(picked))
        chunks.append(picked)

    out = (
        pd.concat(chunks)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )
    out["true_label"] = out["product"].map(short)
    out["pred_label"] = out["pred"].map(lambda i: short[id2label[i]])

    cols = ["complaint_id", "narrative", "label", "product", "true_label",
            "pred", "pred_label", "confidence", "correct", "stratum", "weight"]
    out = out[cols]

    EXAMPLE_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(EXAMPLE_SET_PATH, index=False)
    return out


def main():
    df = build_example_set(force=True)
    print(f"example set: {len(df)} examples -> {EXAMPLE_SET_PATH}")
    print(f"\n{'stratum':<20} {'n':>4} {'weight':>7}")
    for s, g in df.groupby("stratum"):
        print(f"{s:<20} {len(g):>4} {g['weight'].iloc[0]:>7.2f}")
    print(f"\nerrors: {(~df['correct']).sum()} / {len(df)} "
          f"({(~df['correct']).mean():.0%}, vs 15% in the test set)")
    print(f"\ntrue-class coverage:\n{df['true_label'].value_counts().to_string()}")
    print("\ntop true->pred pairs among the sampled errors:")
    err = df[~df["correct"]]
    print(err.groupby(["true_label", "pred_label"]).size()
          .sort_values(ascending=False).head(6).to_string())


if __name__ == "__main__":
    main()
