"""
Drift monitoring: Population Stability Index against a reference window, set
against realised accuracy on the same windows.

    python -m src.monitor                          # drift report from local data
    python -m src.monitor --refresh-metadata ZIP   # rebuild the date/volume files from a snapshot
    python -m src.monitor --from-log logs/requests.jsonl   # PSI over live serving traffic

Why this is not the plan's "pull a newer snapshot" design
---------------------------------------------------------
The CFPB stopped publishing complaint narratives on 14 Aug 2026. The current
bulk file has no narrative column, and the public API no longer returns one --
not even for complaints that had a narrative in the 6 Aug snapshot this project
was built on. There is therefore no post-training *text* to score.

What survives is metadata: every complaint's `Date received` and `Product`. That
supports three honest measurements instead:

1. Temporal drift inside the data we have. Dates are joined onto the pinned
   split, and the held-out test set is cut by quarter. Each quarter is compared
   with the training reference on narrative length, top-token distribution,
   predicted-class distribution and confidence, alongside realised macro-F1.
   The production model was trained on a sample spanning *every* quarter, so it
   is never out of time here; this measures drift magnitude, not decay.

2. An out-of-time backtest. To ask the real question -- does PSI warn before
   accuracy drops? -- a model must be scored on periods it never saw. Retraining
   DistilBERT per window is hours of compute; the TF-IDF baseline refits in
   seconds. It is fit on 2024 training rows only and scored on every later test
   quarter. Its absolute accuracy is not the production model's, but the
   relationship between its PSI and its accuracy loss is a real, out-of-time
   measurement.

3. Class-prior drift on genuinely new traffic. Complaints received after the
   training snapshot (7 Aug 2026 onward) exist as metadata. Their product mix
   is compared with the training window's -- the one form of post-training
   drift that is still observable.

PSI thresholds are the conventional 0.1 / 0.25. Those assume large samples, and
a quarter here is only 300-900 complaints, so every PSI is reported next to a
bootstrap noise floor: the 95th percentile of PSI between the reference and
same-sized random draws *from the reference itself*. A PSI below that floor is
indistinguishable from sampling noise, whatever its verdict band says.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.utils import FIGURES_DIR, RESULTS_DIR, ROOT, load_label_map, load_split

DATES_PATH = ROOT / "data" / "complaint_dates.csv"
VOLUME_PATH = ROOT / "data" / "product_volume_by_month.csv"
PREDICTIONS_PATH = RESULTS_DIR / "predictions_test.parquet"
REPORT_PATH = RESULTS_DIR / "drift_report.json"
FIGURE_PATH = FIGURES_DIR / "drift.png"

SEED = 42
TOP_N_TOKENS = 500
LENGTH_BINS = 10
EPS = 1e-4
N_BOOT = 200
STABLE, MODERATE = 0.10, 0.25
TRAINING_SNAPSHOT = "2026-08-06"  # data/raw/SOURCE.txt
BACKTEST_CUTOFF = "2025-01-01"  # backtest model sees only complaints received before this
MIN_WINDOW = 150  # quarters smaller than this are merged into the previous one
_TOKEN = re.compile(r"[a-z]+")


# --------------------------------------------------------------------------
# PSI
# --------------------------------------------------------------------------


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    """PSI between two count (or proportion) vectors over the same bins."""
    p = np.clip(np.asarray(ref, float) / np.sum(ref), EPS, None)
    q = np.clip(np.asarray(cur, float) / np.sum(cur), EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def verdict(value: float) -> str:
    if value < STABLE:
        return "stable"
    return "moderate" if value < MODERATE else "significant"


def quantile_edges(ref: np.ndarray, bins: int = LENGTH_BINS) -> np.ndarray:
    """Bin edges at reference deciles, open at both ends so no value falls outside."""
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def binned(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=edges)[0]


def categorical(values, categories) -> np.ndarray:
    c = Counter(values)
    return np.array([c.get(k, 0) for k in categories])


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TokenBins:
    """Top-N reference tokens plus an OTHER bin, so vocabulary shift shows up as mass
    moving into OTHER rather than being silently dropped."""

    def __init__(self, ref_texts, n: int = TOP_N_TOKENS):
        counts = Counter(t for text in ref_texts for t in tokens(text))
        self.vocab = [w for w, _ in counts.most_common(n)]
        self.index = {w: i for i, w in enumerate(self.vocab)}

    def counts(self, texts) -> np.ndarray:
        out = np.zeros(len(self.vocab) + 1)
        for text in texts:
            for t in tokens(text):
                out[self.index.get(t, len(self.vocab))] += 1
        return out


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


class Reference:
    """Everything needed to score a window of documents against the reference."""

    def __init__(self, texts: pd.Series, preds: np.ndarray, conf: np.ndarray, n_classes: int):
        self.texts = texts.reset_index(drop=True)
        self.lengths = texts.str.len().to_numpy()
        self.preds, self.conf = np.asarray(preds), np.asarray(conf)
        self.classes = list(range(n_classes))
        self.len_edges = quantile_edges(self.lengths)
        self.conf_edges = quantile_edges(self.conf)
        self.tok = TokenBins(self.texts)
        self.ref = {
            "narrative_length": binned(self.lengths, self.len_edges),
            "token_distribution": self.tok.counts(self.texts),
            "predicted_class": categorical(self.preds, self.classes),
            "confidence": binned(self.conf, self.conf_edges),
        }

    def window(self, texts, preds, conf) -> dict[str, np.ndarray]:
        texts = pd.Series(list(texts))
        return {
            "narrative_length": binned(texts.str.len().to_numpy(), self.len_edges),
            "token_distribution": self.tok.counts(texts),
            "predicted_class": categorical(preds, self.classes),
            "confidence": binned(np.asarray(conf), self.conf_edges),
        }

    def score(self, texts, preds, conf) -> dict[str, float]:
        cur = self.window(texts, preds, conf)
        return {k: psi(self.ref[k], cur[k]) for k in self.ref}

    def noise_floor(self, n: int, rng: np.random.Generator) -> dict[str, float]:
        """95th-percentile PSI of size-n draws from the reference against itself.

        Text and prediction references can come from different sets (training
        text, held-out predictions), so they are drawn independently.
        """
        draws = {k: [] for k in self.ref}
        for _ in range(N_BOOT):
            it = rng.choice(len(self.texts), size=n, replace=False)
            ip = rng.choice(len(self.preds), size=min(n, len(self.preds)), replace=False)
            for k, v in self.score(self.texts.iloc[it], self.preds[ip], self.conf[ip]).items():
                draws[k].append(v)
        return {k: float(np.percentile(v, 95)) for k, v in draws.items()}


def _quarters(dates: pd.Series) -> pd.Series:
    return pd.PeriodIndex(pd.to_datetime(dates), freq="Q").astype(str)


def _merge_small(q: pd.Series) -> pd.Series:
    """Fold quarters under MIN_WINDOW rows into the preceding quarter."""
    counts = q.value_counts().sort_index()
    mapping, last = {}, None
    for quarter, n in counts.items():
        if n < MIN_WINDOW and last is not None:
            mapping[quarter] = last
        else:
            mapping[quarter] = last = quarter
    merged = q.map(mapping)
    # Relabel folded windows so the report says what they contain.
    tail = {v: v for v in merged.unique()}
    for quarter, target in mapping.items():
        if quarter != target:
            tail[target] = f"{target}+{quarter[-2:]}"
    return merged.map(tail)


def _macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro"))


def _window_rows(ref: Reference, windows: pd.DataFrame, rng) -> list[dict]:
    rows = []
    for q, g in windows.groupby("window", sort=True):
        s = ref.score(g["narrative"], g["pred"].to_numpy(), g["confidence"].to_numpy())
        floor = ref.noise_floor(len(g), rng)
        rows.append(
            {
                "window": q,
                "n": int(len(g)),
                "macro_f1": _macro_f1(g["label"], g["pred"]),
                "accuracy": float((g["label"] == g["pred"]).mean()),
                "psi": {k: round(v, 4) for k, v in s.items()},
                "noise_floor_p95": {k: round(v, 4) for k, v in floor.items()},
                "verdict": {k: verdict(v) for k, v in s.items()},
                "above_noise": {k: bool(s[k] > floor[k]) for k in s},
            }
        )
    return rows


# --------------------------------------------------------------------------
# 1. Production model, quarterly
# --------------------------------------------------------------------------


def production_drift(dates: pd.DataFrame, rng) -> dict:
    """Test-set quarters vs the training reference, for the shipped DistilBERT."""
    test = pd.read_parquet(PREDICTIONS_PATH).merge(dates, on="complaint_id", how="inner")
    train = load_split("train")
    # Reference predictions: the model's own output on held-out data, pooled
    # over all quarters. Training-set predictions would be overconfident.
    ref = Reference(
        train["narrative"],
        preds=test["pred"].to_numpy(),
        conf=test["confidence"].to_numpy(),
        n_classes=len(load_label_map()[1]),
    )
    test["window"] = _merge_small(_quarters(test["date_received"]))
    rows = _window_rows(ref, test, rng)
    return {
        "model": "distilbert (production, registry @production)",
        "reference": "training split (text); pooled held-out predictions (pred/conf)",
        "overall_macro_f1": _macro_f1(test["label"], test["pred"]),
        "windows": rows,
    }


# --------------------------------------------------------------------------
# 2. Out-of-time backtest
# --------------------------------------------------------------------------


def backtest(dates: pd.DataFrame, rng) -> dict:
    """TF-IDF + LogReg fit on pre-cutoff training rows, scored on every test quarter."""
    from src.train import BASELINE_CONFIG  # the same untuned config as the Day 2 baseline

    train = load_split("train").merge(dates, on="complaint_id")
    test = load_split("test").merge(dates, on="complaint_id")
    fit = train[train["date_received"] < BACKTEST_CUTOFF]

    cfg = BASELINE_CONFIG
    vec = TfidfVectorizer(
        ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"],
        max_features=cfg["max_features"],
        sublinear_tf=cfg["sublinear_tf"],
        strip_accents="unicode",
    )
    clf = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], random_state=SEED)
    clf.fit(vec.fit_transform(fit["narrative"]), fit["label"])
    proba = clf.predict_proba(vec.transform(test["narrative"]))
    test["pred"], test["confidence"] = proba.argmax(1), proba.max(1)

    in_time = test[test["date_received"] < BACKTEST_CUTOFF]
    ref = Reference(
        fit["narrative"],
        preds=in_time["pred"].to_numpy(),
        conf=in_time["confidence"].to_numpy(),
        n_classes=proba.shape[1],
    )
    test["window"] = _merge_small(_quarters(test["date_received"]))
    rows = _window_rows(ref, test, rng)
    in_f1 = _macro_f1(in_time["label"], in_time["pred"])
    for r in rows:
        r["out_of_time"] = r["window"][:4] >= BACKTEST_CUTOFF[:4]
        r["f1_drop_vs_in_time"] = round(in_f1 - r["macro_f1"], 4)

    oot = [r for r in rows if r["out_of_time"]]
    corr = {}
    for k in ref.ref:
        x = [r["psi"][k] for r in oot]
        y = [r["f1_drop_vs_in_time"] for r in oot]
        corr[k] = round(float(pd.Series(x).corr(pd.Series(y), method="spearman")), 3)
    return {
        "model": "tfidf+logreg (Day 2 config), refit on training rows received before cutoff",
        "cutoff": BACKTEST_CUTOFF,
        "n_fit": int(len(fit)),
        "in_time_macro_f1": round(in_f1, 4),
        "windows": rows,
        "spearman_psi_vs_f1_drop": corr,
    }


# --------------------------------------------------------------------------
# 3. Class prior on post-snapshot traffic
# --------------------------------------------------------------------------


def class_prior_drift() -> dict:
    """Product mix of complaints received after the training snapshot vs before it."""
    vol = pd.read_csv(VOLUME_PATH)
    label2id, id2label, short = load_label_map()
    cats = [id2label[i] for i in sorted(id2label)]
    before = vol[(vol["month"] >= "2024-01") & (vol["month"] < TRAINING_SNAPSHOT[:7])]
    after = vol[vol["month"] >= TRAINING_SNAPSHOT[:7]]
    # The snapshot month is split across both windows; it is counted as "after"
    # only if the day-level file says so, which monthly counts cannot. Exclude it.
    after = after[after["month"] > TRAINING_SNAPSHOT[:7]]
    ref = before.groupby("product")["n"].sum().reindex(cats, fill_value=0).to_numpy()
    cur = after.groupby("product")["n"].sum().reindex(cats, fill_value=0).to_numpy()
    balanced = np.full(len(cats), 1.0 / len(cats))
    value = psi(ref, cur)
    return {
        "reference": f"all complaints in the 8 classes received 2024-01 to {TRAINING_SNAPSHOT[:7]} (excl.)",
        "current": f"complaints received after {TRAINING_SNAPSHOT[:7]} (post-snapshot, no narratives)",
        "n_reference": int(ref.sum()),
        "n_current": int(cur.sum()),
        "reference_share": {short[c]: round(float(v), 4) for c, v in zip(cats, ref / ref.sum())},
        "current_share": {short[c]: round(float(v), 4) for c, v in zip(cats, cur / cur.sum())},
        "psi_current_vs_reference": round(value, 4),
        "verdict": verdict(value),
        # The model was trained on a balanced sample, so the gap between its
        # training prior and real traffic is itself a (static) shift worth stating.
        "psi_traffic_vs_balanced_training_prior": round(psi(balanced, ref), 4),
    }


# --------------------------------------------------------------------------
# Serving-log drift
# --------------------------------------------------------------------------


def log_drift(log_path: Path) -> dict:
    """PSI of logged /predict traffic against the held-out prediction reference.

    Uses only what the request log records -- predicted class, confidence and
    input length -- since the narrative itself is never logged.
    """
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("endpoint") == "/predict" and r.get("status") == 200]
    if not rows:
        raise SystemExit(f"no successful /predict records in {log_path}")
    ref = pd.read_parquet(PREDICTIONS_PATH)
    _, id2label, short = load_label_map()
    classes = [short[id2label[i]] for i in sorted(id2label)]
    ref_pred = [classes[i] for i in ref["pred"]]
    cur = pd.DataFrame(rows)
    len_edges = quantile_edges(ref["narrative"].str.len().to_numpy())
    conf_edges = quantile_edges(ref["confidence"].to_numpy())
    scores = {
        "predicted_class": psi(
            categorical(ref_pred, classes), categorical(cur["pred_class"], classes)
        ),
        "confidence": psi(
            binned(ref["confidence"].to_numpy(), conf_edges),
            binned(cur["confidence"].to_numpy(), conf_edges),
        ),
        "narrative_length": psi(
            binned(ref["narrative"].str.len().to_numpy(), len_edges),
            binned(cur["input_chars"].to_numpy(), len_edges),
        ),
    }
    return {
        "log": str(log_path),
        "n_requests": len(cur),
        "psi": {k: round(v, 4) for k, v in scores.items()},
        "verdict": {k: verdict(v) for k, v in scores.items()},
    }


# --------------------------------------------------------------------------
# Metadata refresh
# --------------------------------------------------------------------------


def refresh_metadata(zip_path: Path) -> None:
    """Rebuild data/complaint_dates.csv and data/product_volume_by_month.csv from a snapshot.

    Only metadata columns are read, so this works on post-Aug-2026 snapshots
    that no longer carry narratives.
    """
    splits = json.loads((ROOT / "data" / "splits.json").read_text())
    pinned = {int(i) for k in ("train", "val", "test") for i in splits[k]}
    _, id2label, _ = load_label_map()
    cats = set(id2label.values())
    dates, volume = [], Counter()
    with zipfile.ZipFile(zip_path) as zf:
        (member,) = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        with zf.open(member) as fh:
            for ch in pd.read_csv(
                fh,
                usecols=["Date received", "Product", "Complaint ID"],
                dtype=str,
                chunksize=1_000_000,
            ):
                ids = ch["Complaint ID"].astype(int)
                hit = ch[ids.isin(pinned)]
                dates.append(
                    pd.DataFrame(
                        {
                            "complaint_id": hit["Complaint ID"].astype(int),
                            "date_received": hit["Date received"].str[:10],
                        }
                    )
                )
                in_cat = ch[ch["Product"].isin(cats) & (ch["Date received"] >= "2024-01-01")]
                volume.update(zip(in_cat["Date received"].str[:7], in_cat["Product"]))
    d = pd.concat(dates).sort_values("complaint_id")
    d.to_csv(DATES_PATH, index=False)
    v = pd.DataFrame(
        [(m, p, n) for (m, p), n in sorted(volume.items())], columns=["month", "product", "n"]
    )
    v.to_csv(VOLUME_PATH, index=False)
    print(f"wrote {DATES_PATH.relative_to(ROOT)} ({len(d):,} of {len(pinned):,} pinned IDs dated)")
    print(f"wrote {VOLUME_PATH.relative_to(ROOT)} ({len(v):,} month x product rows)")


# --------------------------------------------------------------------------
# Figure + entry point
# --------------------------------------------------------------------------


def _plot(report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = ["token_distribution", "narrative_length", "predicted_class", "confidence"]
    colors = dict(zip(feats, ["#2a6fdb", "#e07b39", "#3a9d5d", "#8b5fbf"]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex="col")
    for col, key, title in (
        (0, "production", "Production DistilBERT (trained on all quarters)"),
        (1, "backtest", f"Out-of-time backtest: TF-IDF fit before {BACKTEST_CUTOFF}"),
    ):
        rows = report[key]["windows"]
        x = np.arange(len(rows))
        labels = [r["window"] for r in rows]
        ax = axes[0, col]
        for f in feats:
            ax.plot(x, [r["psi"][f] for r in rows], marker="o", color=colors[f], label=f)
            ax.plot(
                x,
                [r["noise_floor_p95"][f] for r in rows],
                ls=":",
                color=colors[f],
                alpha=0.6,
            )
        ax.axhline(STABLE, color="grey", ls="--", lw=0.8)
        ax.axhline(MODERATE, color="grey", ls="-", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("PSI vs reference  (dotted = noise floor p95)")
        ax.set_yscale("log")
        if col == 0:
            ax.legend(fontsize=8, loc="upper left")
        ax2 = axes[1, col]
        ax2.plot(x, [r["macro_f1"] for r in rows], marker="o", color="black")
        ax2.set_ylabel("macro-F1 on window")
        ax2.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        if key == "backtest":
            cut = next(i for i, r in enumerate(rows) if r["out_of_time"])
            for a in (ax, ax2):
                a.axvline(cut - 0.5, color="red", lw=1)
            ax2.text(cut - 0.4, ax2.get_ylim()[0], " out of time →", color="red", fontsize=8)
    fig.suptitle("Drift (PSI) against realised accuracy, by quarter received", fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.monitor")
    ap.add_argument("--refresh-metadata", type=Path, metavar="ZIP")
    ap.add_argument("--from-log", type=Path, metavar="JSONL")
    args = ap.parse_args(argv)

    if args.refresh_metadata:
        refresh_metadata(args.refresh_metadata)
        return
    if args.from_log:
        print(json.dumps(log_drift(args.from_log), indent=2))
        return

    rng = np.random.default_rng(SEED)
    dates = pd.read_csv(DATES_PATH)
    report = {
        "psi_thresholds": {"stable": f"< {STABLE}", "moderate": f"{STABLE}-{MODERATE}"},
        "production": production_drift(dates, rng),
        "backtest": backtest(dates, rng),
        "class_prior": class_prior_drift(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    _plot(report)
    _print(report)
    print(f"\nwrote {REPORT_PATH.relative_to(ROOT)} and {FIGURE_PATH.relative_to(ROOT)}")


def _print(report: dict) -> None:
    feats = ["narrative_length", "token_distribution", "predicted_class", "confidence"]
    for key in ("production", "backtest"):
        print(f"\n{key}: {report[key]['model']}")
        print(f"  {'window':<11}{'n':>5}{'F1':>8}   " + "  ".join(f"{f[:12]:>12}" for f in feats))
        for r in report[key]["windows"]:
            cells = "  ".join(
                f"{r['psi'][f]:>7.3f}{'*' if r['above_noise'][f] else ' '}    " for f in feats
            )
            print(f"  {r['window']:<11}{r['n']:>5}{r['macro_f1']:>8.4f}   {cells}")
    print("  (* = above the bootstrap noise floor)")
    print(f"\nbacktest Spearman(PSI, F1 drop): {report['backtest']['spearman_psi_vs_f1_drop']}")
    cp = report["class_prior"]
    print(f"\nclass prior after snapshot: PSI {cp['psi_current_vs_reference']} ({cp['verdict']})")


if __name__ == "__main__":
    main()
