"""
Serving contract, end to end through FastAPI, on a tiny randomly initialised
DistilBERT built in a temp dir -- no download, so it runs in CI. Checks shapes,
validation and logging, not predictions.
"""

import json

import pytest
from fastapi.testclient import TestClient
from transformers import (
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)

from src.utils import LABEL_MAP_PATH

TEXT = "my credit card was charged twice and the bank refused to refund the payment"
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *sorted(set(TEXT.split()))]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    (d / "vocab.txt").write_text("\n".join(VOCAB))
    BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
    label_map = json.loads(LABEL_MAP_PATH.read_text())
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    cfg = DistilBertConfig(
        vocab_size=len(VOCAB),
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        num_labels=8,
        id2label=id2label,
        label2id=label_map["label2id"],
    )
    DistilBertForSequenceClassification(cfg).save_pretrained(d)
    (d / "train_config.json").write_text(json.dumps({"max_len": 64}))
    (d / "registry.json").write_text(
        json.dumps({"name": "test-model", "version": "7", "alias": "production"})
    )

    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_URI", str(d))
    mp.setenv("DEVICE", "cpu")
    mp.setenv("REQUEST_LOG", str(d / "requests.jsonl"))
    from src.serve import app

    with TestClient(app) as c:
        c.log_path = d / "requests.jsonl"
        yield c
    mp.undo()


def test_health_reports_registry_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "7" and body["registry_alias"] == "production"


def test_predict_returns_all_class_probabilities(client):
    body = client.post("/predict", json={"narrative": TEXT}).json()
    assert len(body["probabilities"]) == 8
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert body["confidence"] == max(body["probabilities"].values())
    assert body["predicted_class"] in body["probabilities"]


def test_explain_returns_top_k_tokens(client):
    body = client.post("/explain", json={"narrative": TEXT, "top_k": 3, "n_steps": 5}).json()
    assert len(body["top_tokens"]) == 3
    scores = [t["attribution"] for t in body["top_tokens"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    "payload",
    [{}, {"narrative": "too short"}, {"narrative": "x" * 20_001}, {"narrative": TEXT, "top_k": 0}],
)
def test_invalid_input_is_rejected(client, payload):
    endpoint = "/explain" if "top_k" in payload else "/predict"
    assert client.post(endpoint, json=payload).status_code == 422


def test_requests_are_logged_without_text(client):
    client.post("/predict", json={"narrative": TEXT})
    last = json.loads(client.log_path.read_text().splitlines()[-1])
    assert {"ts", "latency_ms", "pred_class", "confidence", "input_chars"} <= last.keys()
    assert TEXT not in client.log_path.read_text()
