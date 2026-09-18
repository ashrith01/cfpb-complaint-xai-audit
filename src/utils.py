"""
Shared plumbing: loading the processed splits, the label map, and merging into
results/metrics.json.

metrics.json accumulates across days -- 'baseline' (Day 2), 'finetune' (Day 3-4),
'faithfulness' (Day 9), 'disagreement' (Day 10) -- so every writer must merge
rather than overwrite, or the last step to run silently erases the others.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# CFPB_SMOKE=1 redirects every read and write under .smoke/ and shrinks the data
# and training grid, so CI can run the real pipeline code end to end on a small
# committed fixture without touching the pinned split or the published results.
SMOKE = os.environ.get("CFPB_SMOKE") == "1"
WORK_ROOT = ROOT / ".smoke" if SMOKE else ROOT

PROCESSED_DIR = WORK_ROOT / "data" / "processed"
SPLITS_PATH = WORK_ROOT / "data" / "splits.json"
RESULTS_DIR = WORK_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "metrics.json"
LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"


def load_split(name: str) -> pd.DataFrame:
    """Load data/processed/{name}.parquet ('train' | 'val' | 'test')."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make data` first")
    return pd.read_parquet(path)


def load_label_map() -> tuple[dict[str, int], dict[int, str], dict[str, str]]:
    """Return (label2id, id2label, short_labels) as written by data_prep."""
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"{LABEL_MAP_PATH} missing -- run `make data` first")
    m = json.loads(LABEL_MAP_PATH.read_text())
    return (
        m["label2id"],
        {int(k): v for k, v in m["id2label"].items()},
        m["short_labels"],
    )


def update_metrics(section: str, payload: dict) -> None:
    """Merge `payload` into results/metrics.json under `section`, preserving the rest."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    metrics[section] = payload
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def dataset_hash(path: Path = SPLITS_PATH) -> str:
    """sha256 over the pinned split IDs, in split order.

    Hashes the ID lists rather than the file bytes, so reformatting splits.json
    does not change the hash but reordering or swapping a single ID does. Order
    is included deliberately: it determines DataLoader batching (see data_prep).
    """
    splits = json.loads(Path(path).read_text())
    canonical = json.dumps(
        {k: [int(i) for i in splits[k]] for k in ("train", "val", "test")},
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
