"""
Model registry: log, register, promote, export -- and backfill the runs that
predate tracking.

    python -m src.registry backfill            # re-log the Day 2-4 runs from metrics.json
    python -m src.registry promote --version N # move the `production` alias to version N
    python -m src.registry export --dest DIR   # resolve @production and copy it out
    python -m src.registry show                # what is in production, and its lineage

Promotion uses a registry *alias*, not a stage. MLflow 3 deprecates the
Staging/Production stage field in favour of aliases; `@production` is the
equivalent, and it is what serving resolves -- no deploy hardcodes a path.

Why backfill rather than retrain: the grid took ~2.3 h on the M4 and its numbers
are already committed. Re-running it to populate a UI would spend two hours to
reproduce values that exist. The backfilled runs are tagged `source=backfill`
and stamped with the commit that produced them, not today's HEAD, so the lineage
they claim is the lineage they had.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import Model

from src import tracking
from src.utils import FIGURES_DIR, LABEL_MAP_PATH, METRICS_PATH, RESULTS_DIR, ROOT

MODEL_DIR = RESULTS_DIR / "model"
HF_ARTIFACT = "hf_model"  # name of the raw HF checkpoint inside the logged MLflow model
EXPORT_META = "registry.json"

# Commits that produced the pre-tracking results. The *code* commit is what a
# run's git_sha means everywhere else, so that is what these carry; the commit
# that recorded the numbers is kept alongside as results_commit.
BACKFILL_COMMITS = {
    "baseline": {"code": "5a2e998", "results": "5a2e998"},
    "finetune": {"code": "46a4211", "results": "11289d5"},
}


class ComplaintClassifier(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper so the registered model is loadable with plain MLflow tooling.

    Input: a DataFrame with a `narrative` column. Output: one probability column
    per class. Serving (src/serve.py) loads the raw checkpoint instead, because
    /explain needs gradient access the pyfunc interface does not expose.
    """

    def load_context(self, context):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = context.artifacts[HF_ARTIFACT]
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.max_len = json.loads((Path(path) / "train_config.json").read_text())["max_len"]

    def predict(self, context, model_input: pd.DataFrame, params=None):
        import torch

        enc = self.tokenizer(
            list(model_input["narrative"]),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1).numpy()
        labels = [self.model.config.id2label[i] for i in range(probs.shape[1])]
        return pd.DataFrame(probs, columns=labels)


def log_and_register(model_dir: Path = MODEL_DIR) -> str:
    """Log the checkpoint as an MLflow model under the active run and register it."""
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=ComplaintClassifier(),
        artifacts={HF_ARTIFACT: str(model_dir)},
        code_paths=[str(ROOT / "src")],
        # Explicit: MLflow's inference would snapshot the whole research venv.
        pip_requirements=["torch", "transformers", "pandas", "mlflow"],
    )
    mv = mlflow.register_model(info.model_uri, tracking.REGISTERED_MODEL)
    client = MlflowClient()
    tags = client.get_run(mlflow.active_run().info.run_id).data.tags
    for key in ("git_sha", "dataset_hash"):
        if key in tags:
            client.set_model_version_tag(tracking.REGISTERED_MODEL, mv.version, key, tags[key])
    return mv.version


def promote(version: str, alias: str = tracking.PRODUCTION_ALIAS) -> None:
    client = MlflowClient()
    previous = _alias_version(client, alias)
    client.set_registered_model_alias(tracking.REGISTERED_MODEL, alias, version)
    print(f"{tracking.REGISTERED_MODEL}@{alias}: v{previous or '-'} -> v{version}")


def _alias_version(client: MlflowClient, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(tracking.REGISTERED_MODEL, alias).version
    except mlflow.exceptions.MlflowException:
        return None


def production_info(alias: str = tracking.PRODUCTION_ALIAS) -> dict:
    """Name, version, run and lineage of the model currently behind `alias`."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(tracking.REGISTERED_MODEL, alias)
    run = client.get_run(mv.run_id)
    return {
        "name": tracking.REGISTERED_MODEL,
        "version": mv.version,
        "alias": alias,
        "run_id": mv.run_id,
        "run_name": run.info.run_name,
        "git_sha": run.data.tags.get("git_sha", "unknown"),
        "dataset_hash": run.data.tags.get("dataset_hash", "unknown"),
        "test_macro_f1": run.data.metrics.get("test_macro_f1"),
        "source": run.data.tags.get("source", "train"),
    }


def hf_checkpoint_dir(model_root: Path) -> Path:
    """Where the raw HF checkpoint sits inside a downloaded MLflow model.

    MLflow stores an artifact under its *source* directory name, not its key, so
    the path is read from the MLmodel file rather than assumed.
    """
    flavor = Model.load(model_root).flavors["python_function"]
    return model_root / flavor["artifacts"][HF_ARTIFACT]["path"]


def export(dest: Path, alias: str = tracking.PRODUCTION_ALIAS) -> dict:
    """Copy the HF checkpoint behind `alias` to `dest`, with its registry metadata.

    Used by `make docker-build`: the alias is resolved once, at build time, and
    the image carries an immutable copy plus registry.json saying exactly which
    version it is. The container never needs access to the registry itself.
    """
    info = production_info(alias)
    uri = f"models:/{tracking.REGISTERED_MODEL}@{alias}"
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(uri, dst_path=tmp))
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(hf_checkpoint_dir(local), dest)
    shutil.copy(LABEL_MAP_PATH, dest / "label_map.json")
    (dest / EXPORT_META).write_text(json.dumps(info, indent=2) + "\n")
    return info


def _full_sha(short: str) -> str:
    return subprocess.check_output(["git", "rev-parse", short], cwd=ROOT, text=True).strip()


def _commit_ms(sha: str) -> int:
    iso = subprocess.check_output(["git", "show", "-s", "--format=%cI", sha], cwd=ROOT, text=True)
    return int(datetime.fromisoformat(iso.strip()).timestamp() * 1000)


def backfill() -> None:
    """Re-log the baseline and the three-config grid from results/metrics.json.

    The selected grid run also gets the test metrics, the confusion matrix and
    the faithfulness figures as artifacts, and the checkpoint in results/model/
    is logged, registered and promoted to @production.
    """
    exp_id = tracking.setup()
    client = MlflowClient()
    existing = client.search_runs([exp_id], "tags.source = 'backfill'", max_results=1)
    if existing:
        print("backfill runs already present -- nothing to do")
        return
    if not (MODEL_DIR / "config.json").exists():
        sys.exit(f"{MODEL_DIR} missing -- the checkpoint is needed to register the model")

    m = json.loads(METRICS_PATH.read_text())
    data_hash = tracking.dataset_hash()

    def open_run(name: str, stage: str, family: str):
        code, results = BACKFILL_COMMITS[stage]["code"], BACKFILL_COMMITS[stage]["results"]
        run = client.create_run(
            exp_id,
            start_time=_commit_ms(results),
            run_name=name,
            tags={
                "source": "backfill",
                "model_family": family,
                "git_sha": _full_sha(code),
                "results_commit": _full_sha(results),
                "dataset_hash": data_hash,
                "mlflow.runName": name,
            },
        )
        return run.info.run_id

    # Baseline
    b = m["baseline"]
    run_id = open_run("baseline-tfidf-logreg", "baseline", "tfidf+logreg")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(b["config"])
        mlflow.log_metrics({"fit_seconds": b["fit_seconds"], "n_features": b["n_features"]})
        tracking.log_scores("val", b["val"])
        tracking.log_scores("test", b["test"])
    print(f"baseline           val {b['val']['macro_f1']:.4f}  test {b['test']['macro_f1']:.4f}")

    # Grid
    ft = m["finetune"]
    for i, entry in enumerate(ft["grid"]):
        cfg, val = entry["config"], entry["val"]
        name = f"distilbert-lr{cfg['lr']:g}-len{cfg['max_len']}"
        run_id = open_run(name, "finetune", "distilbert")
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("grid_index", i)
            mlflow.log_params({**cfg, "base_model": ft["model"], "seed": ft["seed"]})
            mlflow.log_params({"device": ft["device"]})
            mlflow.log_metrics({"best_epoch": val["epoch"], "train_seconds": val["train_seconds"]})
            tracking.log_scores("val", val)
            if i == ft["best_index"]:
                mlflow.set_tag("selected", "true")
                tracking.log_scores("test", ft["test"])
                _log_audit_artifacts(m)
                version = log_and_register(MODEL_DIR)
        print(
            f"{name:<18} val {val['macro_f1']:.4f}"
            + (
                f"  test {ft['test']['macro_f1']:.4f}  -> registered v{version}"
                if i == ft["best_index"]
                else ""
            )
        )

    promote(version)


def _log_audit_artifacts(m: dict) -> None:
    for fig in ("confusion_matrix.png", "disagreement.png", "example_walkthroughs.png"):
        if (FIGURES_DIR / fig).exists():
            mlflow.log_artifact(str(FIGURES_DIR / fig), "figures")
    faith = m.get("faithfulness", {}).get("methods", {})
    mlflow.log_metrics(
        {f"faith_comp.{k}": v["comprehensiveness_mean"] for k, v in faith.items()}
        | {f"faith_suff.{k}": v["sufficiency_mean"] for k, v in faith.items()}
    )
    mlflow.log_dict(m.get("faithfulness", {}), "audit/faithfulness.json")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backfill")
    p = sub.add_parser("promote")
    p.add_argument("--version", required=True)
    e = sub.add_parser("export")
    e.add_argument("--dest", type=Path, default=ROOT / "build" / "model")
    sub.add_parser("show")
    args = ap.parse_args(argv)

    tracking.setup()
    if args.cmd == "backfill":
        backfill()
    elif args.cmd == "promote":
        promote(args.version)
    elif args.cmd == "export":
        info = export(args.dest)
        print(f"exported {info['name']} v{info['version']} (@{info['alias']}) -> {args.dest}")
    elif args.cmd == "show":
        print(json.dumps(production_info(), indent=2))


if __name__ == "__main__":
    main()
