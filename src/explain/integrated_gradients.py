"""
Token-level attribution via Captum's Integrated Gradients.

Contract:
    - explain(model, tokenizer, text) -> list[(token, attribution_score)]
    - Cache results per example to results/explanations/integrated_gradients.json
    - Use the same fixed example set across all three explainer modules
      (see src/explain/shap_explainer.py and attention_rollout.py) so
      results are directly comparable.

Attribution is taken at the embedding layer (LayerIntegratedGradients), summed
over the hidden dimension to give one score per token.

The target is the model's **predicted** class, not the true label. The question
this project asks is "what drove the model's decision", which is well-posed even
when the decision is wrong -- and Day 5 established that roughly 40% of the
errors are mislabelled ground truth anyway, so attributing toward the gold label
would explain a decision the model never made.
"""

from __future__ import annotations

import time

import torch
from captum.attr import LayerIntegratedGradients

from src.evaluate import load_model
from src.explain.common import baseline_ids, encode, save_attributions
from src.explain.example_set import build_example_set

METHOD = "integrated_gradients"
# 50 steps costs ~1.31 s/example here vs 0.73 s at 20. The convergence delta
# (checked below) is what justifies the extra minutes, not convention.
N_STEPS = 50
INTERNAL_BATCH = 25


def _forward(model):
    def fn(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    return fn


def explain(model, tokenizer, text: str, max_len: int = 256, target: int | None = None):
    """Return per-token attribution scores via Integrated Gradients."""
    device = next(model.parameters()).device
    enc = encode(tokenizer, text, max_len)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    if target is None:
        with torch.no_grad():
            target = int(model(input_ids=ids, attention_mask=mask).logits.argmax(-1))

    lig = LayerIntegratedGradients(_forward(model), model.distilbert.embeddings)
    attr, delta = lig.attribute(
        ids,
        baselines=baseline_ids(tokenizer, ids).to(device),
        additional_forward_args=(mask,),
        target=target,
        n_steps=N_STEPS,
        internal_batch_size=INTERNAL_BATCH,
        return_convergence_delta=True,
    )
    # (1, seq, hidden) -> (seq,)
    scores = attr.sum(dim=-1).squeeze(0).detach().cpu()
    return (
        list(zip(enc["tokens"], scores[enc["positions"]].tolist())),
        enc,
        target,
        float(delta.abs().item()),
    )


def main():
    """Entry point for the `explain` step of `make explain` (IG portion)."""
    model, tokenizer, cfg = load_model()
    model.eval()
    examples = build_example_set()

    records, deltas = [], []
    t0 = time.time()
    for i, row in enumerate(examples.itertuples(), 1):
        pairs, enc, target, delta = explain(
            model, tokenizer, row.narrative, cfg["max_len"], target=int(row.pred)
        )
        deltas.append(delta)
        records.append({
            "complaint_id": int(row.complaint_id),
            "pred": int(row.pred),
            "confidence": float(row.confidence),
            "tokens": [t for t, _ in pairs],
            "token_ids": enc["token_ids"],
            "positions": enc["positions"],
            "attributions": [round(s, 6) for _, s in pairs],
        })
        if i % 25 == 0:
            rate = (time.time() - t0) / i
            print(f"\r[IG] {i}/{len(examples)}  {rate:.2f}s/ex  "
                  f"eta {rate * (len(examples) - i) / 60:.1f} min", end="", flush=True)

    path = save_attributions(METHOD, {
        "n_steps": N_STEPS,
        "max_len": cfg["max_len"],
        "target": "predicted_class",
        "baseline": "pad_with_specials_preserved",
        "layer": "distilbert.embeddings",
    }, records)

    # Convergence delta is IG's built-in self-check: completeness says the
    # attributions should sum to F(input) - F(baseline). A large delta means the
    # step count is too low and the attributions are not trustworthy.
    import statistics
    print(f"\r[IG] {len(records)} examples in {(time.time() - t0) / 60:.1f} min"
          f"  ({(time.time() - t0) / len(records):.2f}s/ex)")
    print(f"  convergence delta: mean {statistics.mean(deltas):.4f}  "
          f"median {statistics.median(deltas):.4f}  max {max(deltas):.4f}")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
