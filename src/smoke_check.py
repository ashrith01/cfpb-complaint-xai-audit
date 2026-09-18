"""
Assert that a smoke run (`make smoke`) wrote every artifact the real pipeline
would, and that the registry holds a version carrying its lineage.

Checks existence and shape, never quality: one epoch on 336 rows is not meant to
learn anything. Quality is gated separately, against the committed full-scale
artifacts, by src/gate.py.
"""

from __future__ import annotations

import json
import sys

from mlflow import MlflowClient

from src import tracking
from src.utils import (
    FIGURES_DIR,
    LABEL_MAP_PATH,
    METRICS_PATH,
    PROCESSED_DIR,
    RESULTS_DIR,
    ROOT,
    SMOKE,
    SPLITS_PATH,
)

EXPECTED_FILES = [
    SPLITS_PATH,
    LABEL_MAP_PATH,
    *(PROCESSED_DIR / f"{s}.parquet" for s in ("train", "val", "test")),
    METRICS_PATH,
    FIGURES_DIR / "confusion_matrix.png",
    *(
        RESULTS_DIR / "model" / f
        for f in ("config.json", "model.safetensors", "tokenizer.json", "train_config.json")
    ),
]


def main() -> None:
    if not SMOKE:
        sys.exit("run with CFPB_SMOKE=1 -- this checks .smoke/, not the real results")

    failures = [f"missing {p.relative_to(ROOT)}" for p in EXPECTED_FILES if not p.exists()]

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        for section, key in (("baseline", "test"), ("finetune", "test")):
            if "macro_f1" not in metrics.get(section, {}).get(key, {}):
                failures.append(f"metrics.json has no {section}.{key}.macro_f1")

    exp_id = tracking.setup()
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{tracking.REGISTERED_MODEL}'")
    if not versions:
        failures.append("no registered model version")
    else:
        latest = max(versions, key=lambda v: int(v.version))
        for tag in ("git_sha", "dataset_hash"):
            if tag not in latest.tags:
                failures.append(f"registered v{latest.version} has no {tag} tag")
        runs = client.search_runs([exp_id], max_results=10)
        if len(runs) < 2:
            failures.append(f"expected baseline + fine-tune runs, found {len(runs)}")

    if failures:
        print("smoke check FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print(f"smoke check passed: {len(EXPECTED_FILES)} artifacts, registry v{latest.version}")


if __name__ == "__main__":
    main()
