"""
Error analysis on the held-out test set: which classes confuse with which, and
what the failures actually look like.

Contract:
    - predict_test() -> DataFrame with per-example prediction + full class
      probabilities, cached to results/predictions_test.parquet. Days 9-10 need
      the original prediction and confidence to measure how far an explanation
      moves them, so this is computed once here rather than re-run per explainer.
    - top_confused_pairs(cm, k) -> [(true, pred, count), ...] ranked by off-diagonal mass
    - sample_errors(...) -> N misclassified examples per pair, seeded
    - Output: results/error_examples.json, plus the taxonomy written by hand
      into notebooks/error_analysis.md after reading them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.train import MODEL_DIR, SEED, _device, _encode
from src.utils import RESULTS_DIR, load_label_map, load_split, update_metrics

PREDICTIONS_PATH = RESULTS_DIR / "predictions_test.parquet"
ERROR_EXAMPLES_PATH = RESULTS_DIR / "error_examples.json"
N_PAIRS = 4
N_PER_PAIR = 10
SNIPPET_CHARS = 700


def load_model(attn_implementation: str | None = None):
    """Load the fine-tuned checkpoint.

    `attn_implementation="eager"` is required by attention_rollout: the default
    SDPA kernel does not support `output_attentions=True` and returns None with
    only a warning rather than raising.
    """
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(f"{MODEL_DIR} missing -- run `make train` first")
    config = json.loads((MODEL_DIR / "train_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, **kwargs
    ).to(_device())
    return model, tokenizer, config


@torch.no_grad()
def predict_test(force: bool = False) -> pd.DataFrame:
    """Predictions + full class probabilities on the test set, cached to parquet."""
    if PREDICTIONS_PATH.exists() and not force:
        return pd.read_parquet(PREDICTIONS_PATH)

    model, tokenizer, config = load_model()
    _, id2label, short = load_label_map()
    test_df = load_split("test")
    device = _device()

    loader = DataLoader(_encode(tokenizer, test_df, config["max_len"]), batch_size=64)
    model.eval()
    probs = []
    for input_ids, mask, _ in loader:
        logits = model(
            input_ids=input_ids.to(device), attention_mask=mask.to(device)
        ).logits
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probs)

    out = test_df[["complaint_id", "narrative", "product", "label"]].copy()
    out["pred"] = probs.argmax(1)
    out["confidence"] = probs.max(1)
    out["correct"] = out["pred"] == out["label"]
    # Full distribution kept: faithfulness scoring needs the probability of the
    # originally-predicted class after tokens are removed, not just the argmax.
    for i in sorted(id2label):
        out[f"p_{short[id2label[i]]}"] = probs[:, i]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTIONS_PATH, index=False)
    return out


def top_confused_pairs(cm: np.ndarray, k: int = N_PAIRS) -> list[tuple[int, int, int]]:
    """(true, pred, count) for the k largest off-diagonal cells."""
    pairs = [
        (i, j, int(cm[i, j]))
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
        if i != j
    ]
    return sorted(pairs, key=lambda p: -p[2])[:k]


def sample_errors(preds: pd.DataFrame, pairs, id2label, short, n: int = N_PER_PAIR):
    """N misclassified examples per confused pair, seeded, highest-confidence first.

    Sorted by confidence descending on purpose: a wrong prediction the model was
    *sure* about is the case where a plausible-looking explanation does the most
    damage, and those are the ones Days 9-10 should be aimed at.
    """
    out = []
    for true_i, pred_j, count in pairs:
        subset = preds[(preds["label"] == true_i) & (preds["pred"] == pred_j)]
        subset = subset.sort_values("confidence", ascending=False).head(n)
        out.append({
            "true": short[id2label[true_i]],
            "pred": short[id2label[pred_j]],
            "count": count,
            "rate": round(count / int((preds["label"] == true_i).sum()), 4),
            "examples": [
                {
                    "complaint_id": int(r.complaint_id),
                    "confidence": round(float(r.confidence), 4),
                    "narrative": r.narrative[:SNIPPET_CHARS],
                    "truncated": len(r.narrative) > SNIPPET_CHARS,
                }
                for r in subset.itertuples()
            ],
        })
    return out


# Brand/product terms that name a money-service explicitly. Used to test whether
# the model is reasoning about the complaint or just detecting these tokens.
SERVICE_TERMS = (
    r"\b(?:zelle|venmo|paypal|cash ?app|coinbase|western union|moneygram|"
    r"remitly|wire transfer|crypto|bitcoin)\b"
)
BANK_TERMS = (
    r"\b(?:checking account|savings account|overdraft|debit card|"
    r"direct deposit|atm)\b"
)
MONEY_TRANSFER = "Money transfer, virtual currency, or money service"
CHECKING_SAVINGS = "Checking or savings account"


def shortcut_audit(preds: pd.DataFrame) -> dict:
    """Does accuracy on the money_transfer/checking_savings pair hinge on one lexical cue?

    These two classes account for the largest confusion in the matrix. If the
    model has learned "mentions Zelle -> money transfer" rather than reasoning
    about the complaint, accuracy should collapse when the cue is absent -- and
    that gives Days 6-10 a concrete, pre-registered hypothesis to test the
    attribution methods against.
    """
    has_service = preds["narrative"].str.contains(SERVICE_TERMS, case=False, regex=True)
    has_bank = preds["narrative"].str.contains(BANK_TERMS, case=False, regex=True)

    out = {}
    for cls in (MONEY_TRANSFER, CHECKING_SAVINGS):
        sel = preds["product"] == cls
        cells = {
            "service_term_present": preds[sel & has_service],
            "service_term_absent": preds[sel & ~has_service],
            "bank_terms_only": preds[sel & ~has_service & has_bank],
        }
        out[cls] = {
            k: {"n": int(len(v)), "accuracy": round(float(v["correct"].mean()), 4)}
            for k, v in cells.items()
            if len(v)
        }
    return out


def main():
    """Entry point for `make evaluate`."""
    np.random.seed(SEED)
    _, id2label, short = load_label_map()
    preds = predict_test()

    labels = sorted(id2label)
    cm = confusion_matrix(preds["label"], preds["pred"], labels=labels)
    pairs = top_confused_pairs(cm)
    report = sample_errors(preds, pairs, id2label, short)

    ERROR_EXAMPLES_PATH.write_text(json.dumps(report, indent=2) + "\n")

    n_wrong = int((~preds["correct"]).sum())
    print(f"test set: {len(preds):,} examples, {n_wrong:,} misclassified "
          f"({n_wrong / len(preds):.1%})")
    print(f"\nconfidence on correct   {preds.loc[preds['correct'], 'confidence'].mean():.4f}")
    print(f"confidence on incorrect {preds.loc[~preds['correct'], 'confidence'].mean():.4f}")

    hi_conf_wrong = preds[(~preds["correct"]) & (preds["confidence"] > 0.9)]
    print(f"\nconfidently wrong (p>0.9): {len(hi_conf_wrong):,} "
          f"({len(hi_conf_wrong) / n_wrong:.1%} of all errors)")

    print(f"\nTop {len(pairs)} confused pairs:")
    for p in report:
        print(f"  {p['true']:<18} -> {p['pred']:<18} {p['count']:>4} "
              f"({p['rate']:.1%} of true class)")

    shortcut = shortcut_audit(preds)
    print("\nLexical-shortcut audit (money-service terms):")
    for cls, cells in shortcut.items():
        print(f"  {cls}")
        for name, v in cells.items():
            print(f"    {name:<22} n={v['n']:>4}  accuracy={v['accuracy']:.3f}")

    update_metrics("error_analysis", {
        "n_test": int(len(preds)),
        "n_misclassified": n_wrong,
        "error_rate": round(n_wrong / len(preds), 4),
        "mean_confidence_correct": round(float(preds.loc[preds["correct"], "confidence"].mean()), 4),
        "mean_confidence_incorrect": round(float(preds.loc[~preds["correct"], "confidence"].mean()), 4),
        "n_confidently_wrong_p90": int(len(hi_conf_wrong)),
        "top_confused_pairs": [
            {k: p[k] for k in ("true", "pred", "count", "rate")} for p in report
        ],
        "lexical_shortcut": shortcut,
    })

    print(f"\nwrote {ERROR_EXAMPLES_PATH}")
    print(f"wrote {PREDICTIONS_PATH}")
    print("wrote results/metrics.json section 'error_analysis'")


if __name__ == "__main__":
    main()
