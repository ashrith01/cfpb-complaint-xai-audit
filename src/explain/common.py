"""
Shared plumbing for the three explainers.

All three must emit the *same* record shape over the *same* examples, or Day 9's
faithfulness scoring and Day 10's disagreement analysis have nothing to compare.
The schema is fixed here rather than in each explainer.

Record shape, per example:

    {
      "complaint_id": int,
      "pred": int,                 # class the model predicted (what we explain)
      "confidence": float,
      "tokens": [str, ...],        # real tokens only, specials stripped
      "token_ids": [int, ...],     # needed to rebuild masked inputs on Day 9
      "positions": [int, ...],     # index into the padded sequence
      "attributions": [float, ...] # signed, toward `pred`
    }

Attributions are stored **signed and unnormalised**. Faithfulness scoring needs
to know which tokens push *toward* the predicted class, and normalising per
example would destroy the cross-example comparisons Day 10 needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.utils import RESULTS_DIR

ATTRIBUTION_DIR = RESULTS_DIR / "explanations"


def attribution_path(method: str) -> Path:
    return ATTRIBUTION_DIR / f"{method}.json"


def encode(tokenizer, text: str, max_len: int):
    """Tokenize one narrative, returning tensors plus the non-special token positions.

    Special tokens ([CLS]/[SEP]/[PAD]) are excluded from the returned positions:
    they carry attribution mass but are not removable content, so ranking them
    alongside real tokens would corrupt the Day 9 top-k selection.
    """
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    special = enc["special_tokens_mask"][0].bool()
    attn = enc["attention_mask"][0].bool()
    keep = (attn & ~special).nonzero(as_tuple=True)[0]
    ids = enc["input_ids"][0]
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "positions": keep.tolist(),
        "tokens": tokenizer.convert_ids_to_tokens(ids[keep].tolist()),
        "token_ids": ids[keep].tolist(),
    }


def baseline_ids(tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    """All-[PAD] reference with [CLS]/[SEP] kept in place.

    A pure-zeros or pure-[PAD] baseline (specials included) puts the reference
    outside the distribution the model ever sees, which inflates attribution
    magnitude on the special tokens themselves.
    """
    base = torch.full_like(input_ids, tokenizer.pad_token_id)
    special = tokenizer.get_special_tokens_mask(
        input_ids[0].tolist(), already_has_special_tokens=True
    )
    for i, is_special in enumerate(special):
        if is_special:
            base[0, i] = input_ids[0, i]
    return base


def save_attributions(method: str, config: dict, records: list[dict]) -> Path:
    ATTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)
    path = attribution_path(method)
    path.write_text(
        json.dumps(
            {"method": method, "config": config, "n": len(records), "examples": records},
            indent=None,
            separators=(",", ":"),
        )
        + "\n"
    )
    return path


def load_attributions(method: str) -> dict:
    path = attribution_path(method)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make explain` first")
    return json.loads(path.read_text())
