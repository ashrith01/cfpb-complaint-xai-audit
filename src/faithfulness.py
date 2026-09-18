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

Definitions follow DeYoung et al. (2020, ERASER), stated explicitly because the
sign conventions are easy to get backwards:

    p0   = p(predicted class | full input)
    comp = p0 - p(predicted class | input with the top-k% removed)   HIGHER is better
    suff = p0 - p(predicted class | only the top-k% kept)            LOWER  is better

A `random` pseudo-method is scored alongside the three real ones. Without it the
numbers are uninterpretable: deleting 10% of any text moves the prediction
somewhat, so a method only demonstrates faithfulness by beating what deleting
*arbitrary* tokens achieves. It is the control that makes the comparison mean
anything, and it is the first thing to check when a method looks good.

Perturbation happens in each method's own token space. SHAP tokenizes to word
spans and IG/rollout to wordpieces, but faithfulness asks "if we delete what
*this* method called important, does the prediction move?" -- which is well-posed
per method and needs no cross-method alignment. Alignment is only required for
Day 10's overlap metric, and is handled there.
"""

from __future__ import annotations

import json
import random

import numpy as np
import torch

from src.evaluate import load_model
from src.explain.common import load_attributions
from src.explain.example_set import build_example_set
from src.train import SEED
from src.utils import RESULTS_DIR, update_metrics

TOP_K_PCT = 0.10
METHODS = ("integrated_gradients", "shap", "attention_rollout")
BATCH = 32


def top_k_indices(scores, k_pct: float = TOP_K_PCT) -> list[int]:
    """Indices of the top k% tokens by signed attribution toward the prediction.

    Ranked by signed value, not absolute: a token with a large *negative*
    attribution argues against the predicted class, so removing it should push
    the prediction up, not down. Ranking by |score| would mix the two and blunt
    comprehensiveness for the methods that can express opposition (IG, SHAP).
    """
    n = max(1, int(round(len(scores) * k_pct)))
    return list(np.argsort(-np.asarray(scores))[:n])


@torch.no_grad()
def _probs(model, tokenizer, texts: list[str], max_len: int) -> np.ndarray:
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tokenizer(
            texts[i : i + BATCH],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        logits = model(
            input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)
        ).logits
        out.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(out)


def _rebuild(tokens: list[str], keep: set[int], tokenizer) -> str:
    """Reassemble text from a token subset.

    Wordpiece continuations ('##ing') are re-joined so that dropping a token
    yields a natural string rather than orphaned fragments.
    """
    parts = []
    for i, tok in enumerate(tokens):
        if i not in keep:
            continue
        if tok.startswith("##") and parts:
            parts[-1] += tok[2:]
        else:
            parts.append(tok)
    return " ".join(parts).strip()


def comprehensiveness(
    model,
    tokenizer,
    text: str,
    attributions,
    tokens=None,
    max_len: int = 256,
    target: int | None = None,
) -> float:
    """Drop in p(predicted class) after deleting the top-k% attributed tokens."""
    p0 = _probs(model, tokenizer, [text], max_len)[0]
    target = int(p0.argmax()) if target is None else target
    drop = set(top_k_indices(attributions))
    reduced = _rebuild(tokens, set(range(len(tokens))) - drop, tokenizer)
    return float(p0[target] - _probs(model, tokenizer, [reduced], max_len)[0][target])


def sufficiency(
    model,
    tokenizer,
    text: str,
    attributions,
    tokens=None,
    max_len: int = 256,
    target: int | None = None,
) -> float:
    """Drop in p(predicted class) when ONLY the top-k% attributed tokens are kept."""
    p0 = _probs(model, tokenizer, [text], max_len)[0]
    target = int(p0.argmax()) if target is None else target
    keep = set(top_k_indices(attributions))
    only = _rebuild(tokens, keep, tokenizer)
    return float(p0[target] - _probs(model, tokenizer, [only], max_len)[0][target])


def _score_method(model, tokenizer, records, examples, max_len, randomize=False):
    """Comprehensiveness + sufficiency for every example of one method."""
    rng = random.Random(SEED)
    by_id = {int(r.complaint_id): r for r in examples.itertuples()}

    comp_texts, suff_texts, full_texts, targets, ids = [], [], [], [], []
    for rec in records:
        tokens, scores = rec["tokens"], rec["attributions"]
        if not tokens:
            continue
        if randomize:
            scores = [rng.random() for _ in tokens]
        idx = set(top_k_indices(scores))
        allidx = set(range(len(tokens)))
        full_texts.append(_rebuild(tokens, allidx, tokenizer))
        comp_texts.append(_rebuild(tokens, allidx - idx, tokenizer))
        suff_texts.append(_rebuild(tokens, idx, tokenizer))
        targets.append(int(rec["pred"]))
        ids.append(int(rec["complaint_id"]))

    # p0 is measured on the *reassembled* full text, not the original narrative,
    # so the comprehensiveness delta isolates the effect of removing tokens
    # rather than also absorbing the tokenize/detokenize round-trip.
    p_full = _probs(model, tokenizer, full_texts, max_len)
    p_comp = _probs(model, tokenizer, comp_texts, max_len)
    p_suff = _probs(model, tokenizer, suff_texts, max_len)

    rows = []
    for j, cid in enumerate(ids):
        t = targets[j]
        rows.append(
            {
                "complaint_id": cid,
                "p0": float(p_full[j, t]),
                "comprehensiveness": float(p_full[j, t] - p_comp[j, t]),
                "sufficiency": float(p_full[j, t] - p_suff[j, t]),
                "pred_held_comp": bool(p_comp[j].argmax() == t),
                "pred_held_suff": bool(p_suff[j].argmax() == t),
                "stratum": by_id[cid].stratum,
                "weight": float(by_id[cid].weight),
                "correct": bool(by_id[cid].correct),
            }
        )
    return rows


def _aggregate(rows) -> dict:
    comp = np.array([r["comprehensiveness"] for r in rows])
    suff = np.array([r["sufficiency"] for r in rows])
    w = np.array([r["weight"] for r in rows])
    return {
        "n": len(rows),
        "comprehensiveness_mean": round(float(comp.mean()), 4),
        "comprehensiveness_median": round(float(np.median(comp)), 4),
        # Reweighted to the test distribution -- the example set deliberately
        # oversamples errors, so the unweighted mean is not a test-set estimate.
        "comprehensiveness_weighted": round(float((comp * w).sum() / w.sum()), 4),
        "sufficiency_mean": round(float(suff.mean()), 4),
        "sufficiency_median": round(float(np.median(suff)), 4),
        "sufficiency_weighted": round(float((suff * w).sum() / w.sum()), 4),
        "pred_flipped_after_removal": round(
            float(np.mean([not r["pred_held_comp"] for r in rows])), 4
        ),
        "pred_held_on_rationale_only": round(
            float(np.mean([r["pred_held_suff"] for r in rows])), 4
        ),
    }


def _by_stratum(rows) -> dict:
    out = {}
    for s in sorted({r["stratum"] for r in rows}):
        sub = [r for r in rows if r["stratum"] == s]
        out[s] = {
            "n": len(sub),
            "comprehensiveness_mean": round(
                float(np.mean([r["comprehensiveness"] for r in sub])), 4
            ),
            "sufficiency_mean": round(float(np.mean([r["sufficiency"] for r in sub])), 4),
        }
    return out


def score_all_methods():
    """Entry point for the faithfulness portion of `make audit`."""
    model, tokenizer, cfg = load_model()
    model.eval()
    examples = build_example_set()
    max_len = cfg["max_len"]

    results, per_example = {}, {}
    for method in METHODS:
        data = load_attributions(method)
        rows = _score_method(model, tokenizer, data["examples"], examples, max_len)
        results[method] = {**_aggregate(rows), "by_stratum": _by_stratum(rows)}
        per_example[method] = rows
        print(f"[{method}] scored {len(rows)}")

    # Control: the same pipeline with attribution scores replaced by noise.
    rows = _score_method(
        model,
        tokenizer,
        load_attributions(METHODS[0])["examples"],
        examples,
        max_len,
        randomize=True,
    )
    results["random"] = {**_aggregate(rows), "by_stratum": _by_stratum(rows)}
    per_example["random"] = rows
    print("[random] scored", len(rows))

    update_metrics("faithfulness", {"top_k_pct": TOP_K_PCT, "methods": results})
    (RESULTS_DIR / "faithfulness_per_example.json").write_text(
        json.dumps(per_example, separators=(",", ":")) + "\n"
    )

    order = list(METHODS) + ["random"]
    print(f"\ntop-k = {TOP_K_PCT:.0%} (committed before scoring)\n")
    print(f"{'method':<22} {'comp↑':>7} {'suff↓':>7} {'comp_wt':>8} {'flip%':>7} {'held%':>7}")
    for m in order:
        r = results[m]
        print(
            f"{m:<22} {r['comprehensiveness_mean']:>7.4f} "
            f"{r['sufficiency_mean']:>7.4f} {r['comprehensiveness_weighted']:>8.4f} "
            f"{r['pred_flipped_after_removal']:>7.1%} "
            f"{r['pred_held_on_rationale_only']:>7.1%}"
        )
    print("\ncomp = confidence lost when the top-10% is removed (higher = more faithful)")
    print("suff = confidence lost when ONLY the top-10% is kept (lower = more faithful)")
    return results


if __name__ == "__main__":
    score_all_methods()
