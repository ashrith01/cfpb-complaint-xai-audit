"""
Fine-tune a small transformer (DistilBERT baseline) for CFPB complaint
classification. Also run a TF-IDF + LogisticRegression baseline for comparison.

Contract:
    - run_baseline() -> dict: TF-IDF + LogReg, returns {"accuracy":, "macro_f1":}
    - train_config: small fixed grid (2-4 configs), not an open sweep
    - fine_tune(config) -> trained model + val metrics
    - select_best(results) -> best config by val macro-F1
    - evaluate_test(model) -> final test metrics + confusion matrix, saved to results/
"""

from __future__ import annotations

import json
import random
import time

import mlflow
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import tracking
from src.utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    SMOKE,
    load_label_map,
    load_split,
    update_metrics,
)

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = RESULTS_DIR / "model"

# Small fixed grid, not a sweep (BRD section 7 puts sweeps out of scope).
# Varies the two knobs that actually move a DistilBERT fine-tune -- learning rate
# and sequence length -- rather than sampling a large space thinly.
CONFIG_GRID = [
    {"lr": 5e-5, "epochs": 3, "max_len": 256, "batch_size": 16},
    {"lr": 2e-5, "epochs": 3, "max_len": 256, "batch_size": 16},
    {"lr": 5e-5, "epochs": 3, "max_len": 128, "batch_size": 32},
]
if SMOKE:
    # CI: one short config -- proves the loop runs end to end, not that it learns.
    CONFIG_GRID = [{"lr": 5e-5, "epochs": 1, "max_len": 64, "batch_size": 16}]

# Deliberately untuned. This is the floor the fine-tune has to clear, so tuning
# it would be tuning the goalpost -- a stronger baseline is only worth building
# if the fine-tune fails to beat this one.
BASELINE_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_features": 300_000,
    "sublinear_tf": True,
    "C": 1.0,
    "max_iter": 2000,
}


def run_baseline():
    """TF-IDF + LogisticRegression baseline. Write result to results/metrics.json under 'baseline'."""
    train_df, val_df, test_df = load_split("train"), load_split("val"), load_split("test")
    _, id2label, short = load_label_map()

    cfg = BASELINE_CONFIG
    vec = TfidfVectorizer(
        ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"],
        max_features=cfg["max_features"],
        sublinear_tf=cfg["sublinear_tf"],
        strip_accents="unicode",
        lowercase=True,
    )

    t0 = time.time()
    x_train = vec.fit_transform(train_df["narrative"])
    clf = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], random_state=SEED)
    clf.fit(x_train, train_df["label"])
    elapsed = time.time() - t0

    # Scored on val AND test. The fine-tune reports test, so a val-only baseline
    # would leave the headline "beats baseline" claim comparing two different
    # splits. Nothing is selected on test here -- the baseline config is fixed
    # and untuned -- so scoring it there costs no validity.
    val = _score(val_df["label"], clf.predict(vec.transform(val_df["narrative"])), id2label, short)
    test = _score(
        test_df["label"], clf.predict(vec.transform(test_df["narrative"])), id2label, short
    )

    payload = {
        "model": "tfidf+logreg",
        "config": {**cfg, "ngram_range": list(cfg["ngram_range"])},
        "seed": SEED,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_features": int(x_train.shape[1]),
        "fit_seconds": round(elapsed, 1),
        "val": val,
        "test": test,
    }
    update_metrics("baseline", payload)
    if mlflow.active_run():
        mlflow.log_params(payload["config"])
        mlflow.log_metrics({"fit_seconds": elapsed, "n_features": payload["n_features"]})
        tracking.log_scores("val", val)
        tracking.log_scores("test", test)

    print(
        f"\nBaseline: TF-IDF({cfg['ngram_range'][0]}-{cfg['ngram_range'][1]}gram, "
        f"{x_train.shape[1]:,} features) + LogisticRegression"
    )
    print(f"  fit on {len(train_df):,} train docs in {elapsed:.1f}s")
    print(f"\n  val  accuracy {val['accuracy']:.4f}   macro-F1 {val['macro_f1']:.4f}")
    print(
        f"  test accuracy {test['accuracy']:.4f}   macro-F1 {test['macro_f1']:.4f}"
        f"   <- floor for the fine-tune to beat"
    )
    print("\n  per-class F1 (test):")
    for name, f in sorted(test["per_class_f1"].items(), key=lambda kv: kv[1]):
        print(f"    {name:<20} {f:.4f}")
    print("\n  written to results/metrics.json under 'baseline'")

    return {"val": val, "test": test}


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _encode(tokenizer, df, max_len: int) -> TensorDataset:
    enc = tokenizer(
        list(df["narrative"]),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    return TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(df["label"].to_numpy(), dtype=torch.long),
    )


@torch.no_grad()
def _predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for input_ids, mask, _ in loader:
        logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def _score(y_true, y_pred, id2label, short) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=sorted(id2label))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_f1": {short[id2label[i]]: float(f) for i, f in enumerate(per_class)},
    }


def fine_tune(config: dict):
    """Fine-tune DistilBERT with the given config. Return model + val metrics."""
    _set_seed()
    device = _device()
    label2id, id2label, short = load_label_map()
    train_df, val_df = load_split("train"), load_split("val")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label={i: c for i, c in id2label.items()},
        label2id=label2id,
    ).to(device)

    train_ds = _encode(tokenizer, train_df, config["max_len"])
    val_ds = _encode(tokenizer, val_df, config["max_len"])
    # Seeded generator so shuffling is reproducible across runs (NFR4).
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=64)

    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * config["epochs"]
    warmup = int(0.1 * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: (
            s / max(1, warmup)
            if s < warmup
            else max(0.0, (total_steps - s) / max(1, total_steps - warmup))
        ),
    )

    t0 = time.time()
    best, best_state = {"macro_f1": -1.0}, None
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running = 0.0
        for step, (input_ids, mask, labels) in enumerate(train_loader, 1):
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=mask.to(device),
                labels=labels.to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

            running += out.loss.item()
            if step % 100 == 0:
                print(
                    f"\r  epoch {epoch}/{config['epochs']}  step {step}/"
                    f"{len(train_loader)}  loss {running / step:.4f}"
                    f"  [{time.time() - t0:.0f}s]",
                    end="",
                    flush=True,
                )

        metrics = _score(
            val_df["label"].to_numpy(), _predict(model, val_loader, device), id2label, short
        )
        print(
            f"\r  epoch {epoch}/{config['epochs']}  train_loss "
            f"{running / len(train_loader):.4f}  val_acc {metrics['accuracy']:.4f}"
            f"  val_macro_f1 {metrics['macro_f1']:.4f}  [{time.time() - t0:.0f}s]"
        )
        if mlflow.active_run():
            mlflow.log_metric("train_loss", running / len(train_loader), step=epoch)
            tracking.log_scores("val", metrics, step=epoch)

        # Keep the best epoch, not the last -- 3 epochs on 25k examples can
        # overfit, and selecting the last epoch would silently ship a worse model.
        if metrics["macro_f1"] > best["macro_f1"]:
            best = {**metrics, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    best["train_seconds"] = round(time.time() - t0, 1)
    return model, tokenizer, best


def select_best(results: list[dict]) -> int:
    """Index of the config with the highest val macro-F1."""
    return max(range(len(results)), key=lambda i: results[i]["val"]["macro_f1"])


def evaluate_test(model, tokenizer, test_df, max_len: int) -> dict:
    """Run once on held-out test set. Save accuracy, macro-F1, per-class F1, confusion matrix."""
    device = _device()
    _, id2label, short = load_label_map()
    loader = DataLoader(_encode(tokenizer, test_df, max_len), batch_size=64)

    y_true = test_df["label"].to_numpy()
    y_pred = _predict(model, loader, device)
    metrics = _score(y_true, y_pred, id2label, short)

    labels = sorted(id2label)
    names = [short[id2label[i]] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    _plot_confusion(cm, names)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = names
    return metrics


def _plot_confusion(cm, names) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Row-normalised: every class has the same support here, but rates are what
    # the Day 5 error analysis actually reads off the chart.
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("DistilBERT — test confusion matrix (row-normalised)")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    """Entry point for `make train`: baseline, then the fine-tune grid, then test.

    Every run is logged to MLflow (see src/tracking.py) with git SHA and dataset
    hash. The winner is logged as a model and registered, but not promoted:
    moving the `production` alias is a separate, deliberate step (`make promote`).
    """
    tracking.setup()
    lineage = tracking.lineage_tags()

    with mlflow.start_run(run_name="baseline-tfidf-logreg", tags=lineage):
        mlflow.set_tag("model_family", "tfidf+logreg")
        baseline = run_baseline()

    print(f"\n{'=' * 70}\nFine-tuning {MODEL_NAME} on {_device().type}\n{'=' * 70}")
    results, run_ids = [], []
    best_model = best_tokenizer = None
    for i, config in enumerate(CONFIG_GRID, 1):
        print(f"\n[{i}/{len(CONFIG_GRID)}] {config}")
        # Compare against the best *previous* config, before appending this one.
        prev_best = max((r["val"]["macro_f1"] for r in results), default=-1.0)
        name = f"distilbert-lr{config['lr']:g}-len{config['max_len']}"
        with mlflow.start_run(run_name=name, tags=lineage) as run:
            mlflow.set_tags({"model_family": "distilbert", "grid_index": i - 1})
            mlflow.log_params({**config, "base_model": MODEL_NAME, "seed": SEED})
            model, tokenizer, val_metrics = fine_tune(config)
            mlflow.log_metrics(
                {"best_epoch": val_metrics["epoch"], "train_seconds": val_metrics["train_seconds"]}
            )
        run_ids.append(run.info.run_id)
        results.append({"config": config, "val": val_metrics})
        if val_metrics["macro_f1"] > prev_best:
            best_model, best_tokenizer = model, tokenizer
        else:
            del model

    winner = select_best(results)
    config = results[winner]["config"]
    print(
        f"\nBest config [{winner + 1}/{len(CONFIG_GRID)}]: {config}"
        f"  val macro-F1 {results[winner]['val']['macro_f1']:.4f}"
    )

    # Held-out test set is touched exactly once, with the already-selected model.
    print("\nEvaluating on held-out test set (once)...")
    test_metrics = evaluate_test(best_model, best_tokenizer, load_split("test"), config["max_len"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(MODEL_DIR)
    best_tokenizer.save_pretrained(MODEL_DIR)
    (MODEL_DIR / "train_config.json").write_text(json.dumps(config, indent=2) + "\n")

    from src import registry  # deferred: registry imports this module

    with mlflow.start_run(run_id=run_ids[winner]):
        mlflow.set_tag("selected", "true")
        tracking.log_scores("test", test_metrics)
        mlflow.log_artifact(str(FIGURES_DIR / "confusion_matrix.png"), "figures")
        version = registry.log_and_register(MODEL_DIR)

    update_metrics(
        "finetune",
        {
            "model": MODEL_NAME,
            "seed": SEED,
            "device": _device().type,
            "grid": results,
            "best_config": config,
            "best_index": winner,
            "val": results[winner]["val"],
            "test": test_metrics,
        },
    )

    # Like-for-like: both numbers are on the held-out test set.
    delta = test_metrics["macro_f1"] - baseline["test"]["macro_f1"]
    print(f"\n{'=' * 70}")
    print(f"  test accuracy         {test_metrics['accuracy']:.4f}")
    print(f"  test macro-F1         {test_metrics['macro_f1']:.4f}")
    print(f"  baseline test macro-F1 {baseline['test']['macro_f1']:.4f}")
    print(
        f"  delta                 {delta:+.4f}"
        f"   {'BEATS baseline' if delta > 0 else 'DOES NOT beat baseline'}"
    )
    print("\n  per-class F1 (test):")
    for name, f in sorted(test_metrics["per_class_f1"].items(), key=lambda kv: kv[1]):
        print(f"    {name:<20} {f:.4f}")
    print(f"\n  model      -> {MODEL_DIR}")
    print(f"  registry   -> {tracking.REGISTERED_MODEL} v{version} (promote with `make promote`)")
    print(f"  confusion  -> {FIGURES_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
