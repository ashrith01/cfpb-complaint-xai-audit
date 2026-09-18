"""
Token-level attribution via SHAP (text explainer).

Contract:
    - SHAP is slow on text -- subsample the test set (~500 examples) rather
      than running on the full set. Must be the SAME 500 examples used for
      integrated_gradients.py and attention_rollout.py.
    - explain(model, tokenizer, text) -> list[(token, shap_value)]
    - Cache results to results/explanations/shap.json

Uses shap.PartitionExplainer with the tokenizer as masker -- the standard route
for transformer text models. Unlike IG it is model-agnostic (forward passes
only, no gradients), which is exactly why it is worth comparing: if a
perturbation-based method and a gradient-based method disagree about what drove
the same prediction, at most one of them can be faithful.

SHAP returns values for every class; we keep the column for the model's
**predicted** class, matching the IG target so the two are comparable.
"""

from __future__ import annotations

import json
import time

import numpy as np
import shap
import torch

from src.evaluate import load_model
from src.explain.common import attribution_path, encode, save_attributions
from src.explain.example_set import build_example_set

METHOD = "shap"
SAMPLE_SIZE = 500
# Evaluation budget per example, chosen by measurement rather than convention.
# Against a 300-eval reference on sampled narratives:
#
#   max_evals=300   6.6 s/ex   (55 min for 500)   reference
#   max_evals=200   3.9 s/ex   (33 min)   spearman 0.930, top-5 overlap 0.93
#   max_evals=100   1.9 s/ex   (16 min)   spearman 0.829, top-5 overlap 0.67
#
# 200 keeps the ranking essentially intact at 40% of the cost. 100 does not:
# top-k overlap between methods is Day 10's headline metric, and a budget that
# only agrees with itself 67% of the time would inject more self-noise than the
# cross-method disagreement it is meant to measure.
#
# That 0.93 self-agreement is also the practical *ceiling* for Day 10 -- two
# methods cannot be shown to agree more than SHAP agrees with itself.
MAX_EVALS = 200
BATCH_SIZE = 32
CHECKPOINT_EVERY = 10


def _predict_fn(model, tokenizer, max_len: int):
    """Batch text -> class probabilities, for SHAP's model-agnostic interface."""
    device = next(model.parameters()).device

    def fn(texts):
        out = []
        texts = [str(t) for t in texts]
        for i in range(0, len(texts), BATCH_SIZE):
            enc = tokenizer(
                texts[i : i + BATCH_SIZE],
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out)

    return fn


def build_explainer(model, tokenizer, max_len: int):
    masker = shap.maskers.Text(tokenizer)
    return shap.Explainer(_predict_fn(model, tokenizer, max_len), masker)


def truncate_to_window(tokenizer, text: str, max_len: int) -> str:
    """Cut the narrative to exactly the span the model reads.

    SHAP's Text masker tokenizes the *raw* string, so without this it perturbs
    words the model never sees -- narratives here run to 2,214 tokens against a
    254-token window, and 35% of the example set is truncated by the model.

    The attributions out beyond the window are correctly near-zero (~1% of total
    mass), so this is not a correctness fix for those values. It is a precision
    fix for the ones that matter: the fixed 200-evaluation budget was being
    spread over up to 9x more tokens than the model consumes, making the estimate
    on the in-window tokens far noisier than it needed to be. It also puts SHAP
    on the same footing as IG and rollout, which only ever saw the window.
    """
    ids = tokenizer(text, truncation=True, max_length=max_len, add_special_tokens=False)[
        "input_ids"
    ]
    return tokenizer.decode(ids, skip_special_tokens=True)


def explain(
    model, tokenizer, text: str, max_len: int = 256, target: int | None = None, explainer=None
):
    """Return per-token SHAP values for the predicted class."""
    if explainer is None:
        explainer = build_explainer(model, tokenizer, max_len)
    if target is None:
        device = next(model.parameters()).device
        enc = encode(tokenizer, text, max_len)
        with torch.no_grad():
            target = int(
                model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).logits.argmax(-1)
            )

    sv = explainer([truncate_to_window(tokenizer, text, max_len)], max_evals=MAX_EVALS, silent=True)
    tokens = [t for t in sv.data[0]]
    values = sv.values[0][:, target]
    return list(zip(tokens, [float(v) for v in values])), target


def main():
    """Entry point for the SHAP portion of `make explain`.

    Checkpoints incrementally and resumes. This run takes ~16 min and spawns
    loky worker processes; on a machine already deep into swap the OS will kill
    it, and restarting from zero each time makes the step effectively
    uncompletable. Progress is flushed to a .partial file so a kill costs at
    most CHECKPOINT_EVERY examples.
    """
    model, tokenizer, cfg = load_model()
    model.eval()
    examples = build_example_set().head(SAMPLE_SIZE)
    explainer = build_explainer(model, tokenizer, cfg["max_len"])

    partial = attribution_path(METHOD).with_suffix(".partial.json")
    records, done = [], set()
    if partial.exists():
        records = json.loads(partial.read_text())
        done = {r["complaint_id"] for r in records}
        print(f"[SHAP] resuming: {len(done)} already done")

    todo = [r for r in examples.itertuples() if int(r.complaint_id) not in done]
    t0 = time.time()
    for i, row in enumerate(todo, 1):
        pairs, target = explain(
            model,
            tokenizer,
            row.narrative,
            cfg["max_len"],
            target=int(row.pred),
            explainer=explainer,
        )
        records.append(
            {
                "complaint_id": int(row.complaint_id),
                "pred": int(row.pred),
                "confidence": float(row.confidence),
                "tokens": [t for t, _ in pairs],
                "attributions": [round(s, 6) for _, s in pairs],
            }
        )
        if i % CHECKPOINT_EVERY == 0:
            partial.write_text(json.dumps(records, separators=(",", ":")))
            rate = (time.time() - t0) / i
            print(
                f"\r[SHAP] {len(records)}/{len(examples)}  {rate:.2f}s/ex  "
                f"eta {rate * (len(todo) - i) / 60:.1f} min",
                end="",
                flush=True,
            )
    partial.write_text(json.dumps(records, separators=(",", ":")))

    path = save_attributions(
        METHOD,
        {
            "max_evals": MAX_EVALS,
            "max_len": cfg["max_len"],
            "target": "predicted_class",
            "explainer": "shap.PartitionExplainer",
            "masker": "shap.maskers.Text(tokenizer)",
            "input": "truncated to the model's max_len window before explaining",
        },
        records,
    )
    print(
        f"\r[SHAP] {len(records)} examples in {(time.time() - t0) / 60:.1f} min"
        f"  ({(time.time() - t0) / len(records):.2f}s/ex)"
    )
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
