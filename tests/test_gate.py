"""
The eval gate must pass on the committed artifacts and fail on each kind of
regression it claims to catch. A gate that is only ever seen passing has not
been shown to work.
"""

import json

import pandas as pd
import pytest

from src import gate


def _failed(checks) -> set[str]:
    return {c.name.split(":")[0] for c in checks if not c.passed}


def test_passes_on_committed_artifacts():
    checks, summary = gate.run_checks()
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
    assert summary["macro_f1"] > gate.MIN_MACRO_F1


def test_threshold_rejects_the_baseline():
    """0.84 would have let the TF-IDF baseline (0.8411) through."""
    metrics = json.loads(gate.METRICS_PATH.read_text())
    assert metrics["baseline"]["test"]["macro_f1"] < gate.MIN_MACRO_F1


@pytest.fixture
def degraded_predictions(tmp_path, monkeypatch):
    preds = pd.read_parquet(gate.PREDICTIONS_PATH)
    # Flip 5% of correct predictions to a wrong class: ~0.80 macro-F1.
    correct = preds.index[preds["pred"] == preds["label"]]
    flip = correct[: int(0.05 * len(preds))]
    preds.loc[flip, "pred"] = (preds.loc[flip, "label"] + 1) % 8
    path = tmp_path / "predictions_test.parquet"
    preds.to_parquet(path)
    monkeypatch.setattr(gate, "PREDICTIONS_PATH", path)


def test_fails_on_accuracy_regression(degraded_predictions):
    checks, _ = gate.run_checks()
    failed = {c.name for c in checks if not c.passed}
    assert f"accuracy: macro-F1 >= {gate.MIN_MACRO_F1}" in failed
    assert "accuracy: beats TF-IDF baseline" in failed
    # metrics.json was not updated alongside, so the consistency check trips too.
    assert "accuracy: recomputed macro-F1 matches metrics.json" in failed


def _patched_faithfulness(tmp_path, monkeypatch, mutate):
    faith = json.loads(gate.FAITHFULNESS_PATH.read_text())
    mutate(faith)
    path = tmp_path / "faith.json"
    path.write_text(json.dumps(faith))
    monkeypatch.setattr(gate, "FAITHFULNESS_PATH", path)


def test_fails_when_ordering_breaks(tmp_path, monkeypatch):
    def swap(f):
        f["shap"], f["attention_rollout"] = f["attention_rollout"], f["shap"]

    _patched_faithfulness(tmp_path, monkeypatch, swap)
    assert "ordering" in _failed(gate.run_checks()[0])


def test_fails_when_a_method_drops_below_random(tmp_path, monkeypatch):
    def replace_with_random(f):
        f["attention_rollout"] = f["random"]

    _patched_faithfulness(tmp_path, monkeypatch, replace_with_random)
    failed = {c.name for c in gate.run_checks()[0] if not c.passed}
    assert "control: attention_rollout beats random tokens" in failed


def test_fails_when_split_changes(tmp_path, monkeypatch):
    splits = json.loads(gate.SPLITS_PATH.read_text())
    splits["test"][0], splits["test"][1] = splits["test"][1], splits["test"][0]
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(splits))
    monkeypatch.setattr(gate, "SPLITS_PATH", path)
    failed = {c.name for c in gate.run_checks()[0] if not c.passed}
    assert "integrity: splits.json hash matches data/splits.sha256" in failed
    assert "integrity: predictions cover exactly the pinned test split, in order" in failed


def test_markdown_reports_deltas(tmp_path):
    checks, summary = gate.run_checks()
    md = gate.to_markdown(checks, summary, gate._base_summary(gate.METRICS_PATH))
    assert "✅ passed" in md and "| test macro-F1 | 0.8495 | 0.8495 | +0.0000 |" in md
