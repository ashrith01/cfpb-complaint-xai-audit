"""
Token-level attribution via attention-weight rollout (the intentionally
weaker baseline explanation method -- see BRD for why this comparison matters).

Contract:
    - explain(model, tokenizer, text) -> list[(token, attention_score)]
    - Cache results to results/explanations/attention_rollout.json
    - Use the same fixed example set as the other two explainers.

Rollout (Abnar & Zuidema, 2020) composes the per-layer attention matrices --
averaged over heads, with a residual term added and rows renormalised -- to
approximate how much each input token contributes to the final [CLS]
representation.

Why it is expected to be the weaker method, and why that is the point: attention
weights are computed **without reference to the class being predicted**. The same
rollout map explains every class equally, so it cannot distinguish evidence *for*
the prediction from evidence merely attended to. IG and SHAP are both
target-conditioned. If attention rollout nonetheless scores comparably on
faithfulness, that is evidence the faithfulness metrics are insensitive; if it
scores worse, it quantifies how much the class-conditioning is worth.

Attention is also strictly non-negative, so unlike IG and SHAP this method cannot
express "this token argued *against* the prediction". Day 10 should not read its
lack of negative attributions as disagreement.
"""

from __future__ import annotations

import time

import torch

from src.evaluate import load_model
from src.explain.common import encode, save_attributions
from src.explain.example_set import build_example_set

METHOD = "attention_rollout"


def rollout(attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Compose per-layer attention into a single input->output attribution map.

    attentions: tuple of (batch, heads, seq, seq), one per layer.
    Returns (seq,) -- the [CLS] row of the composed matrix.
    """
    result = None
    for layer in attentions:
        # Average over heads, then account for the residual stream: without the
        # identity term the composition decays toward a uniform map and the
        # rollout loses whatever signal the raw attention carried.
        a = layer.mean(dim=1).squeeze(0)
        a = a + torch.eye(a.size(-1), device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    return result[0]  # [CLS] row


def explain(model, tokenizer, text: str, max_len: int = 256):
    """Return per-token attention-rollout scores."""
    device = next(model.parameters()).device
    enc = encode(tokenizer, text, max_len)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_attentions=True)
    scores = rollout(out.attentions).detach().cpu()
    return list(zip(enc["tokens"], scores[enc["positions"]].tolist())), enc, out.logits


def main():
    """Entry point for the attention-rollout portion of `make explain`."""
    model, tokenizer, cfg = load_model()
    model.eval()
    examples = build_example_set()

    records = []
    t0 = time.time()
    for i, row in enumerate(examples.itertuples(), 1):
        pairs, enc, _ = explain(model, tokenizer, row.narrative, cfg["max_len"])
        records.append({
            "complaint_id": int(row.complaint_id),
            "pred": int(row.pred),
            "confidence": float(row.confidence),
            "tokens": [t for t, _ in pairs],
            "token_ids": enc["token_ids"],
            "positions": enc["positions"],
            "attributions": [round(s, 8) for _, s in pairs],
        })
        if i % 50 == 0:
            print(f"\r[rollout] {i}/{len(examples)}", end="", flush=True)

    path = save_attributions(METHOD, {
        "max_len": cfg["max_len"],
        "target": "none -- attention is not class-conditioned",
        "residual": "identity added, rows renormalised",
        "head_reduction": "mean",
    }, records)
    print(f"\r[rollout] {len(records)} examples in {time.time() - t0:.0f}s")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
