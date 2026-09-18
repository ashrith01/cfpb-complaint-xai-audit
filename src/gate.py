"""
Regression gate: fail the build if the model, or the research finding it
supports, has regressed.

Runs against the committed artifacts -- never retrains (DistilBERT is a two-hour
job; CI runners would time out and the result would not be the shipped model).

    accuracy       macro-F1, recomputed from predictions_test.parquet, must be
                   >= MIN_MACRO_F1 and must beat the TF-IDF baseline
    ordering       comprehensiveness IG > SHAP > attention rollout -- the
                   ranking the audit's headline depends on
    control        every method beats the random-token control on both
                   comprehensiveness (higher) and sufficiency (lower)
    integrity      the pinned split hashes to the committed value, and the
                   predictions / faithfulness files cover exactly that split

Headline numbers are recomputed from the per-example files rather than read
from metrics.json, and the two are required to agree. Otherwise hand-editing
metrics.json would pass the gate while the evidence under it said otherwise.

    python -m src.gate [--json out.json] [--markdown out.md] [--base-metrics base.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from src.utils import METRICS_PATH, RESULTS_DIR, ROOT, SPLITS_PATH, dataset_hash, load_label_map

PREDICTIONS_PATH = RESULTS_DIR / "predictions_test.parquet"
FAITHFULNESS_PATH = RESULTS_DIR / "faithfulness_per_example.json"
SPLITS_HASH_PATH = ROOT / "data" / "splits.sha256"

# 0.84 was the first draft. The TF-IDF baseline scores 0.8411, so a 0.84 floor
# waves the baseline through -- a gate that cannot catch "we shipped the
# baseline" has no teeth. 0.845 sits between baseline (0.8411) and the shipped
# fine-tune (0.8495), and the beats-baseline check below backs it up.
MIN_MACRO_F1 = 0.845
FAITHFULNESS_ORDER = ("integrated_gradients", "shap", "attention_rollout")
CONTROL = "random"
TOLERANCE = 1e-4  # recomputed vs recorded; metrics.json stores 4 d.p. for faithfulness


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _comp_suff(faith: dict) -> dict[str, dict[str, float]]:
    out = {}
    for method, rows in faith.items():
        df = pd.DataFrame(rows)
        out[method] = {
            "comprehensiveness": float(df["comprehensiveness"].mean()),
            "sufficiency": float(df["sufficiency"].mean()),
        }
    return out


def run_checks() -> tuple[list[Check], dict]:
    checks: list[Check] = []
    metrics = json.loads(METRICS_PATH.read_text())
    preds = pd.read_parquet(PREDICTIONS_PATH)
    faith = json.loads(FAITHFULNESS_PATH.read_text())
    splits = json.loads(SPLITS_PATH.read_text())

    # -- accuracy ------------------------------------------------------------
    macro_f1 = float(f1_score(preds["label"], preds["pred"], average="macro"))
    _, id2label, short = load_label_map()
    labels = sorted(id2label)
    per_class = f1_score(preds["label"], preds["pred"], average=None, labels=labels)
    per_class_f1 = {short[id2label[i]]: float(f) for i, f in zip(labels, per_class)}
    recorded = metrics["finetune"]["test"]["macro_f1"]
    baseline = metrics["baseline"]["test"]["macro_f1"]
    checks.append(
        Check(
            "accuracy: recomputed macro-F1 matches metrics.json",
            abs(macro_f1 - recorded) < TOLERANCE,
            f"recomputed {macro_f1:.4f}, recorded {recorded:.4f}",
        )
    )
    checks.append(
        Check(
            f"accuracy: macro-F1 >= {MIN_MACRO_F1}",
            macro_f1 >= MIN_MACRO_F1,
            f"{macro_f1:.4f}",
        )
    )
    checks.append(
        Check(
            "accuracy: beats TF-IDF baseline",
            macro_f1 > baseline,
            f"{macro_f1:.4f} vs baseline {baseline:.4f} ({macro_f1 - baseline:+.4f})",
        )
    )

    # -- faithfulness ordering and control -----------------------------------
    scores = _comp_suff(faith)
    rec = metrics["faithfulness"]["methods"]
    drift = {
        m: abs(scores[m]["comprehensiveness"] - rec[m]["comprehensiveness_mean"]) for m in scores
    }
    checks.append(
        Check(
            "faithfulness: recomputed scores match metrics.json",
            max(drift.values()) < TOLERANCE,
            f"max |recomputed - recorded| = {max(drift.values()):.2e}",
        )
    )
    comp = [scores[m]["comprehensiveness"] for m in FAITHFULNESS_ORDER]
    checks.append(
        Check(
            "ordering: comprehensiveness IG > SHAP > attention rollout",
            all(a > b for a, b in zip(comp, comp[1:])),
            " > ".join(f"{m} {c:.3f}" for m, c in zip(FAITHFULNESS_ORDER, comp)),
        )
    )
    ctrl = scores[CONTROL]
    for m in FAITHFULNESS_ORDER:
        s = scores[m]
        ok = s["comprehensiveness"] > ctrl["comprehensiveness"] and (
            s["sufficiency"] < ctrl["sufficiency"]
        )
        checks.append(
            Check(
                f"control: {m} beats random tokens",
                ok,
                f"comp {s['comprehensiveness']:.3f} vs {ctrl['comprehensiveness']:.3f}, "
                f"suff {s['sufficiency']:.3f} vs {ctrl['sufficiency']:.3f}",
            )
        )

    # -- data integrity -------------------------------------------------------
    expected = SPLITS_HASH_PATH.read_text().strip()
    actual = dataset_hash(SPLITS_PATH)
    checks.append(
        Check(
            "integrity: splits.json hash matches data/splits.sha256",
            actual == expected,
            f"{actual[:12]} vs expected {expected[:12]}",
        )
    )
    test_ids = [int(i) for i in splits["test"]]
    checks.append(
        Check(
            "integrity: predictions cover exactly the pinned test split, in order",
            preds["complaint_id"].astype(int).tolist() == test_ids,
            f"{len(preds):,} predictions vs {len(test_ids):,} pinned test IDs",
        )
    )
    faith_ids = {int(r["complaint_id"]) for rows in faith.values() for r in rows}
    checks.append(
        Check(
            "integrity: faithfulness examples are all from the test split",
            faith_ids <= set(test_ids),
            f"{len(faith_ids - set(test_ids))} examples outside the test split",
        )
    )

    summary = {
        "macro_f1": macro_f1,
        "baseline_macro_f1": baseline,
        "per_class_f1": per_class_f1,
        "comprehensiveness": {m: v["comprehensiveness"] for m, v in scores.items()},
        "sufficiency": {m: v["sufficiency"] for m, v in scores.items()},
        "dataset_hash": actual,
    }
    return checks, summary


def _base_summary(path: Path) -> dict | None:
    """Headline numbers from the base branch's metrics.json, for the PR delta table."""
    if not path or not path.exists():
        return None
    m = json.loads(path.read_text())
    methods = m.get("faithfulness", {}).get("methods", {})
    return {
        "macro_f1": m["finetune"]["test"]["macro_f1"],
        "per_class_f1": m["finetune"]["test"]["per_class_f1"],
        "comprehensiveness": {k: v["comprehensiveness_mean"] for k, v in methods.items()},
        "sufficiency": {k: v["sufficiency_mean"] for k, v in methods.items()},
    }


def to_markdown(checks: list[Check], head: dict, base: dict | None) -> str:
    ok = all(c.passed for c in checks)
    lines = [
        "<!-- eval-gate -->",
        f"## Eval gate: {'✅ passed' if ok else '❌ failed'}",
        "",
        "| check | result | detail |",
        "|---|---|---|",
        *(f"| {c.name} | {'✅' if c.passed else '❌'} | {c.detail} |" for c in checks),
        "",
    ]

    def row(name, h, b):
        if b is None:
            return f"| {name} | {h:.4f} | – | – |"
        return f"| {name} | {h:.4f} | {b:.4f} | {h - b:+.4f} |"

    lines += [
        "### Metric deltas vs base branch",
        "",
        "| metric | PR | base | Δ |",
        "|---|---|---|---|",
    ]
    lines.append(row("test macro-F1", head["macro_f1"], base and base["macro_f1"]))
    for cls, v in sorted(head["per_class_f1"].items()):
        lines.append(row(f"F1 {cls}", v, base and base["per_class_f1"].get(cls)))
    for m in (*FAITHFULNESS_ORDER, CONTROL):
        lines.append(
            row(
                f"comprehensiveness {m}",
                head["comprehensiveness"][m],
                base and base["comprehensiveness"].get(m),
            )
        )
    if base is None:
        lines += ["", "_No base metrics supplied -- deltas omitted._"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.gate")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--base-metrics", type=Path)
    args = ap.parse_args(argv)

    checks, summary = run_checks()
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name:<{width}}  {c.detail}")
    passed = all(c.passed for c in checks)
    print(f"\neval gate {'PASSED' if passed else 'FAILED'}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {"passed": passed, "checks": [asdict(c) for c in checks], **summary}, indent=2
            )
        )
    if args.markdown:
        args.markdown.write_text(to_markdown(checks, summary, _base_summary(args.base_metrics)))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
