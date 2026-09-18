"""
FastAPI serving for the registered classifier.

    POST /predict   class, confidence and all 8 class probabilities
    POST /explain   top-k Integrated Gradients tokens toward the predicted class
    GET  /health    model version, registry alias, lineage, device

Model resolution (MODEL_URI):
    models:/<name>@<alias>  resolved through the MLflow registry at startup
                            (the default for `make serve` on the host)
    /path/to/checkpoint     a directory exported by `python -m src.registry export`,
                            with registry.json alongside (what the container uses)

Either way the model is loaded exactly once, in the lifespan hook, never per
request. The container takes the second path deliberately: `make docker-build`
resolves @production at build time, so the image is immutable and says which
registry version it holds, and it needs no access to the tracking store.

Every request emits one JSON log line: timestamp, endpoint, status, latency,
predicted class, confidence, input length. That log is the input to monitoring.
The narrative itself is never logged -- complaint text is personal data.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.explain.integrated_gradients import N_STEPS
from src.explain.integrated_gradients import explain as ig_explain
from src.utils import LABEL_MAP_PATH

DEFAULT_MODEL_URI = "models:/cfpb-complaint-classifier@production"
MAX_CHARS = 20_000  # ~10x the longest narrative in the training set; guards memory, not truncation
MIN_CHARS = 20

log = logging.getLogger("cfpb.serve")


def _setup_logging() -> None:
    log.setLevel(logging.INFO)
    log.propagate = False
    if log.handlers:
        return
    fmt = logging.Formatter("%(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    log.addHandler(stream)
    if path := os.environ.get("REQUEST_LOG", "logs/requests.jsonl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        log.addHandler(fh)


def _device() -> torch.device:
    if name := os.environ.get("DEVICE"):
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def resolve_model(uri: str) -> tuple[Path, dict]:
    """Return (local checkpoint dir, registry metadata) for a MODEL_URI."""
    if uri.startswith("models:/"):
        # Imported lazily: the serving image does not ship MLflow.
        import mlflow

        from src import registry, tracking

        mlflow.set_tracking_uri(tracking.tracking_uri())
        alias = uri.rsplit("@", 1)[1] if "@" in uri else tracking.PRODUCTION_ALIAS
        local = Path(mlflow.artifacts.download_artifacts(uri))
        return registry.hf_checkpoint_dir(local), registry.production_info(alias)
    path = Path(uri)
    meta_path = path / "registry.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"source": str(path)}
    return path, meta


class Model:
    """The loaded checkpoint plus everything /health reports about it."""

    def __init__(self, uri: str):
        t0 = time.perf_counter()
        path, self.meta = resolve_model(uri)
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()
        self.max_len = json.loads((path / "train_config.json").read_text())["max_len"]
        cfg = self.model.config
        self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
        # Exported next to the checkpoint for the container; the host falls back
        # to the committed copy in data/processed/.
        label_map = path / "label_map.json"
        if not label_map.exists():
            label_map = LABEL_MAP_PATH
        self.short = json.loads(label_map.read_text())["short_labels"]
        # One forward pass at a time. Concurrent torch calls from the threadpool
        # oversubscribe the CPU and are slower in aggregate than queueing them.
        self.lock = threading.Lock()
        self.load_seconds = time.perf_counter() - t0

    @torch.no_grad()
    def predict(self, text: str) -> tuple[int, list[float], int]:
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        with self.lock:
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return max(range(len(probs)), key=probs.__getitem__), probs, enc["input_ids"].shape[1]

    def explain(self, text: str, target: int, n_steps: int):
        with self.lock:
            return ig_explain(
                self.model, self.tokenizer, text, self.max_len, target=target, n_steps=n_steps
            )


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class PredictRequest(BaseModel):
    narrative: str = Field(min_length=MIN_CHARS, max_length=MAX_CHARS)


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


class ExplainRequest(PredictRequest):
    top_k: int = Field(default=10, ge=1, le=50)
    n_steps: int = Field(default=N_STEPS, ge=5, le=200)


class TokenAttribution(BaseModel):
    token: str
    position: int
    attribution: float


class ExplainResponse(BaseModel):
    predicted_class: str
    confidence: float
    method: str = "integrated_gradients"
    n_steps: int
    convergence_delta: float
    top_tokens: list[TokenAttribution]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_name: str | None
    model_version: str | None
    registry_alias: str | None
    model_git_sha: str | None
    dataset_hash: str | None
    serving_git_sha: str
    device: str
    load_seconds: float


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    app.state.model = Model(os.environ.get("MODEL_URI", DEFAULT_MODEL_URI))
    m = app.state.model
    log.info(
        json.dumps(
            {
                "event": "model_loaded",
                "version": str(m.meta.get("version")),
                "device": m.device.type,
                "load_seconds": round(m.load_seconds, 2),
            }
        )
    )
    yield


app = FastAPI(title="CFPB complaint classifier", lifespan=lifespan)


@app.middleware("http")
async def request_log(request: Request, call_next):
    t0 = time.perf_counter()
    request.state.log = {}
    response = await call_next(request)
    if request.url.path in ("/predict", "/explain"):
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "endpoint": request.url.path,
            "status": response.status_code,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            **request.state.log,
        }
        log.info(json.dumps(record))
    return response


def _version(m: Model) -> str:
    return str(m.meta.get("version", "unregistered"))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    m: Model = request.app.state.model
    pred, probs, n_tokens = m.predict(req.narrative)
    label = m.labels[pred]
    request.state.log.update(
        pred_class=m.short.get(label, label),
        confidence=round(probs[pred], 4),
        input_chars=len(req.narrative),
        input_tokens=n_tokens,
        model_version=_version(m),
    )
    return PredictResponse(
        predicted_class=m.short.get(label, label),
        predicted_label=label,
        confidence=probs[pred],
        probabilities={m.short.get(lbl, lbl): p for lbl, p in zip(m.labels, probs)},
        model_version=_version(m),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest, request: Request):
    m: Model = request.app.state.model
    target, probs, n_tokens = m.predict(req.narrative)
    pairs, enc, _, delta = m.explain(req.narrative, target, req.n_steps)
    ranked = sorted(zip(enc["positions"], pairs), key=lambda item: item[1][1], reverse=True)[
        : req.top_k
    ]
    label = m.labels[target]
    request.state.log.update(
        pred_class=m.short.get(label, label),
        confidence=round(probs[target], 4),
        input_chars=len(req.narrative),
        input_tokens=n_tokens,
        n_steps=req.n_steps,
        model_version=_version(m),
    )
    return ExplainResponse(
        predicted_class=m.short.get(label, label),
        confidence=probs[target],
        n_steps=req.n_steps,
        convergence_delta=delta,
        top_tokens=[
            TokenAttribution(token=tok, position=pos, attribution=score)
            for pos, (tok, score) in ranked
        ],
        model_version=_version(m),
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    m: Model = request.app.state.model
    return HealthResponse(
        status="ok",
        model_name=m.meta.get("name"),
        model_version=str(m.meta.get("version")) if "version" in m.meta else None,
        registry_alias=m.meta.get("alias"),
        model_git_sha=m.meta.get("git_sha"),
        dataset_hash=m.meta.get("dataset_hash"),
        serving_git_sha=os.environ.get("GIT_SHA", "local"),
        device=m.device.type,
        load_seconds=round(m.load_seconds, 3),
    )
