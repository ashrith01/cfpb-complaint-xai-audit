"""
Experiment tracking: MLflow setup and the lineage every run carries.

Backend is a local SQLite file (mlflow.db) with artifacts under mlruns/, both
gitignored. MLflow 3.x refuses the plain-directory file store without an opt-out
flag, and SQLite is equally local -- no server, nothing to host, nothing to pay.

Lineage is the point. Every run is tagged with:
    git_sha       -- the code that produced it (suffixed "-dirty" on an unclean tree)
    dataset_hash  -- sha256 of the pinned split IDs in data/splits.json

so "which model is in production and what data trained it" is answered by
reading two tags off the registered version, not by archaeology.
"""

from __future__ import annotations

import os
import subprocess

import mlflow

from src.utils import ROOT, SPLITS_PATH, WORK_ROOT, dataset_hash

EXPERIMENT = "cfpb-complaint-classifier"
REGISTERED_MODEL = "cfpb-complaint-classifier"
PRODUCTION_ALIAS = "production"

DB_PATH = WORK_ROOT / "mlflow.db"
ARTIFACT_ROOT = WORK_ROOT / "mlruns"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{DB_PATH}"


def setup() -> str:
    """Point MLflow at the local store and select the project experiment. Returns its id."""
    mlflow.set_tracking_uri(tracking_uri())
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        # Explicit artifact location: MLflow's default is ./mlruns relative to
        # whatever directory the process happens to start in.
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        exp_id = mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT.as_uri())
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def git_sha(short: bool = False) -> str:
    """HEAD commit, with "-dirty" if the tree has uncommitted changes to tracked files."""
    if sha := os.environ.get("GIT_SHA"):  # set at docker build time; no .git in the image
        return sha
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).returncode
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    sha = sha[:7] if short else sha
    return f"{sha}-dirty" if dirty else sha


def lineage_tags() -> dict[str, str]:
    tags = {"git_sha": git_sha()}
    if SPLITS_PATH.exists():
        tags["dataset_hash"] = dataset_hash()
    return tags


def log_scores(prefix: str, scores: dict, step: int | None = None) -> None:
    """Log accuracy, macro-F1 and per-class F1 from a train._score() dict."""
    metrics = {f"{prefix}_accuracy": scores["accuracy"], f"{prefix}_macro_f1": scores["macro_f1"]}
    metrics |= {f"{prefix}_f1.{k}": v for k, v in scores.get("per_class_f1", {}).items()}
    mlflow.log_metrics(metrics, step=step)
