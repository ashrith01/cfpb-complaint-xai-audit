"""
Download, clean, and split the CFPB Consumer Complaint Database subsample.

Source: https://www.consumerfinance.gov/data-research/consumer-complaints/

Contract:
    - CATEGORIES: hardcoded list of 6-8 product categories to keep (decide once, don't change mid-project)
    - clean_narrative(text) -> str: strip redaction placeholders (e.g. "XXXX"), boilerplate, empty rows
    - stratified_split(df, seed=42) -> (train_df, val_df, test_df): saved indices to data/splits.json
    - Output: data/processed/{train,val,test}.parquet, ~30-40k narratives total after subsampling

Why DATE_MIN exists: the `Product` column carries 21 distinct values accumulated
across several taxonomy revisions -- "Credit reporting or other personal consumer
reports", "Credit reporting, credit repair services, or other personal consumer
reports" and "Credit reporting" are the same concept under three different names,
as are "Credit card" and "Credit card or prepaid card". Restricting to complaints
received on or after 2024-01-01 pins the data to a single taxonomy revision, which
gives 8 clean, well-populated classes with no merge table to maintain.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORIES: list[str] = [
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Checking or savings account",
    "Credit card",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Vehicle loan or lease",
    "Student loan",
]

# Compact names for figure axes -- the raw product strings are far too long to
# label a confusion matrix or a per-class faithfulness chart with.
SHORT_LABELS: dict[str, str] = {
    "Credit reporting or other personal consumer reports": "credit_reporting",
    "Debt collection": "debt_collection",
    "Checking or savings account": "checking_savings",
    "Credit card": "credit_card",
    "Money transfer, virtual currency, or money service": "money_transfer",
    "Mortgage": "mortgage",
    "Vehicle loan or lease": "vehicle_loan",
    "Student loan": "student_loan",
}

SEED = 42
DATE_MIN = "2024-01-01"  # freezes the product taxonomy to one revision (see module docstring)
PER_CLASS = 4_500  # 8 x 4,500 = 36,000, inside the BRD's 30-40k band
POOL_PER_CLASS = 40_000  # reservoir size per class; bounds memory during the 12 GB scan
MIN_CHARS = 100  # post-cleaning length floor; drops stubs
SPLIT = (0.70, 0.15, 0.15)
CHUNKSIZE = 200_000

BULK_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ZIP_PATH = RAW_DIR / "complaints.csv.zip"
SPLITS_PATH = ROOT / "data" / "splits.json"
LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"
SOURCE_PATH = RAW_DIR / "SOURCE.txt"

COL_DATE = "Date received"
COL_PRODUCT = "Product"
COL_NARRATIVE = "Consumer complaint narrative"
COL_ID = "Complaint ID"
USECOLS = [COL_DATE, COL_PRODUCT, COL_NARRATIVE, COL_ID]


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def download(force: bool = False) -> Path:
    """Fetch the bulk complaint archive into data/raw/, skipping if already present.

    The archive is ~1.4 GB, so re-downloading on every `make data` is not an
    option. Provenance (byte count + the server's Last-Modified) is recorded to
    data/raw/SOURCE.txt because the database grows daily -- results are only
    reproducible against a known snapshot.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and not force:
        print(f"[download] using cached {ZIP_PATH.name} "
              f"({ZIP_PATH.stat().st_size / 1e9:.2f} GB) -- pass force=True to refresh")
        return ZIP_PATH

    print(f"[download] fetching {BULK_URL}")
    with urllib.request.urlopen(BULK_URL) as resp:
        total = int(resp.headers.get("content-length", 0))
        last_modified = resp.headers.get("last-modified", "unknown")
        tmp = ZIP_PATH.with_suffix(".zip.part")
        done = 0
        with open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r[download] {done / 1e9:.2f}/{total / 1e9:.2f} GB "
                          f"({pct:.1f}%)", end="", flush=True)
        print()
        tmp.replace(ZIP_PATH)

    SOURCE_PATH.write_text(
        f"url: {BULK_URL}\n"
        f"bytes: {ZIP_PATH.stat().st_size}\n"
        f"last_modified: {last_modified}\n"
    )
    return ZIP_PATH


def _csv_member(path: Path) -> str:
    """Name of the single CSV inside the archive, so we can stream without extracting."""
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if len(members) != 1:
        raise RuntimeError(f"expected exactly one CSV in {path.name}, found {members}")
    return members[0]


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

# CFPB scrubs PII before publishing, leaving placeholder runs behind:
#   "on XX/XX/2024 I called"      -> redacted date
#   "account XXXX XXXX XXXX 1234" -> redacted account number
#   "they charged me {$1000.00}"  -> currency mask
#   "they charged me {$XX.XX}"    -> redacted currency mask
_RE_REDACTED_DATE = re.compile(r"\bX{1,4}\s*/\s*X{1,4}\s*/\s*(?:X{2,4}|\d{2,4})\b", re.I)
_RE_CURRENCY_MASK = re.compile(r"\{\s*\$?\s*([^}]*?)\s*\}")
# Not \bX{2,}\b: redaction runs are frequently glued to digits ("{$2500.00XXXX",
# "XXXX1234") where there is no word boundary. Letters on *both* sides are
# excluded instead, so real names survive -- EXXON and TJ MAXX must not become
# "E ON" and "TJ M".
_RE_REDACTED_RUN = re.compile(r"(?<![A-Za-z])X{2,}(?![A-Za-z])")
_RE_STRAY_BRACE = re.compile(r"[{}]")
# Unwrapping a nested mask ("${ {$3600.00 } }") leaves a run of dollar signs.
_RE_DUP_DOLLAR = re.compile(r"\$(?:\s*\$)+")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def clean_narrative(text: str) -> str:
    """Strip CFPB redaction placeholders and boilerplate. Return cleaned string.

    Redaction runs are *deleted* rather than replaced with a sentinel token.
    A high-frequency sentinel like ``[REDACTED]`` would itself become a feature
    the classifier can key on -- and since Days 9-10 audit exactly what the model
    keys on, we should not manufacture a shortcut for it to find.

    Returns "" for anything shorter than MIN_CHARS once cleaned; callers drop those.
    """
    if not isinstance(text, str):
        return ""

    text = _RE_REDACTED_DATE.sub(" ", text)
    # "{$1000.00}" -> "$1000.00", but drop the mask entirely when the amount
    # itself was redacted ("{$XX.XX}"). Applied to a fixed point because the
    # source contains nested masks like "${ {$3600.00 } }".
    for _ in range(3):
        text, n = _RE_CURRENCY_MASK.subn(
            lambda m: " " if "X" in m.group(1).upper() else f" ${m.group(1)} ", text
        )
        if not n:
            break
    text = _RE_REDACTED_RUN.sub(" ", text)
    # Unterminated masks ("{$2500.00XXXX") leave an orphan brace behind.
    text = _RE_STRAY_BRACE.sub(" ", text)
    text = _RE_DUP_DOLLAR.sub("$", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text if len(text) >= MIN_CHARS else ""


def _dedupe_key(text: str) -> str:
    """Hash of the normalized narrative, used to drop mass-filed duplicates.

    Credit-repair template letters are filed in bulk and differ only in the
    fields CFPB redacts. Because clean_narrative() removes those placeholders
    first, the templates collapse to identical strings here and get caught.
    """
    norm = _RE_WHITESPACE.sub(" ", _RE_NON_ALNUM.sub(" ", text.lower())).strip()
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


# --------------------------------------------------------------------------
# Scan / filter / sample
# --------------------------------------------------------------------------

def load_and_filter(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Stream the archive, keeping a bounded uniform sample of eligible rows per class.

    Only four of the sixteen columns are read: materializing 17M rows x 16 cols
    would blow the 16 GB budget (NFR1). Cleaning and deduping happen inside the
    chunk loop, and each class is held in a reservoir of at most POOL_PER_CLASS
    rows so peak memory stays flat regardless of how large the class is upstream.
    Reservoir sampling (rather than "first N seen") keeps the pool uniform over
    the whole two-year window instead of clustering on whichever dates the file
    happens to start with.
    """
    member = _csv_member(path)
    rng = random.Random(SEED)

    stats = dict(scanned=0, has_narrative=0, in_window=0, in_category=0,
                 cleaned=0, deduped=0)
    seen: set[str] = set()
    pools: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    eligible: dict[str, int] = {c: 0 for c in CATEGORIES}
    categories = set(CATEGORIES)

    with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
        reader = pd.read_csv(
            fh,
            usecols=USECOLS,
            chunksize=CHUNKSIZE,
            # Dates stay as strings: the bulk file and the API export disagree on
            # whether the column carries a time+timezone, and ISO-8601 compares
            # correctly either way lexicographically.
            dtype={COL_PRODUCT: "string", COL_NARRATIVE: "string", COL_DATE: "string"},
        )
        for i, chunk in enumerate(reader):
            if i == 0:
                missing = [c for c in USECOLS if c not in chunk.columns]
                if missing:
                    raise RuntimeError(
                        f"bulk CSV is missing expected columns {missing}; "
                        f"got {list(chunk.columns)}"
                    )

            stats["scanned"] += len(chunk)

            chunk = chunk[chunk[COL_NARRATIVE].notna()]
            stats["has_narrative"] += len(chunk)

            # fillna: a missing date would leave pd.NA in the mask, and boolean
            # indexing on a mask containing NA raises rather than dropping the row.
            chunk = chunk[chunk[COL_DATE].fillna("") >= DATE_MIN]
            stats["in_window"] += len(chunk)

            chunk = chunk[chunk[COL_PRODUCT].isin(categories)]
            stats["in_category"] += len(chunk)

            for cid, product, narrative in zip(
                chunk[COL_ID], chunk[COL_PRODUCT], chunk[COL_NARRATIVE]
            ):
                cleaned = clean_narrative(narrative)
                if not cleaned:
                    continue
                stats["cleaned"] += 1

                key = _dedupe_key(cleaned)
                if key in seen:
                    continue
                seen.add(key)
                stats["deduped"] += 1

                product = str(product)
                eligible[product] += 1
                pool = pools[product]
                row = {"complaint_id": int(cid), "narrative": cleaned, "product": product}
                if len(pool) < POOL_PER_CLASS:
                    pool.append(row)
                else:
                    # Reservoir: replace a random slot with probability k/n.
                    j = rng.randrange(eligible[product])
                    if j < POOL_PER_CLASS:
                        pool[j] = row

            if (i + 1) % 10 == 0:
                print(f"\r[scan] {stats['scanned'] / 1e6:.1f}M rows read, "
                      f"{stats['deduped']:,} kept", end="", flush=True)

    print(f"\r[scan] {stats['scanned'] / 1e6:.1f}M rows read, "
          f"{stats['deduped']:,} kept")

    df = pd.DataFrame([r for pool in pools.values() for r in pool])
    return df, {**stats, **{f"eligible::{k}": v for k, v in eligible.items()}}


def balanced_sample(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Take exactly PER_CLASS rows per category. Fails loudly if a class falls short."""
    short = {
        c: n for c in CATEGORIES
        if (n := int((df["product"] == c).sum())) < PER_CLASS
    }
    if short:
        raise RuntimeError(
            f"not enough narratives after cleaning/dedupe for {short} "
            f"(need {PER_CLASS} each); widen DATE_MIN or lower PER_CLASS"
        )
    out = pd.concat(
        [df[df["product"] == c].sample(n=PER_CLASS, random_state=seed) for c in CATEGORIES],
        ignore_index=True,
    )
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_split(df, seed: int = SEED):
    """Stratified train/val/test split by category. Save split indices to data/splits.json.

    Splits are recorded by Complaint ID rather than row position: IDs survive a
    refreshed snapshot and a reshuffled DataFrame, positional indices do not,
    which would quietly break reproducibility (NFR4) the first time the data is
    re-downloaded.
    """
    train_frac, val_frac, test_frac = SPLIT

    train_df, rest_df = train_test_split(
        df, train_size=train_frac, random_state=seed, stratify=df["product"]
    )
    val_df, test_df = train_test_split(
        rest_df,
        train_size=val_frac / (val_frac + test_frac),
        random_state=seed,
        stratify=rest_df["product"],
    )

    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(
        {
            "seed": seed,
            "date_min": DATE_MIN,
            "per_class": PER_CLASS,
            "split": list(SPLIT),
            "categories": CATEGORIES,
            "train": sorted(int(i) for i in train_df["complaint_id"]),
            "val": sorted(int(i) for i in val_df["complaint_id"]),
            "test": sorted(int(i) for i in test_df["complaint_id"]),
        },
        indent=2,
    ) + "\n")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _write_label_map() -> dict[str, int]:
    """Single source of truth for label ids, consumed by train.py and every explainer."""
    label2id = {c: i for i, c in enumerate(CATEGORIES)}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_MAP_PATH.write_text(json.dumps(
        {
            "label2id": label2id,
            "id2label": {str(i): c for c, i in label2id.items()},
            "short_labels": SHORT_LABELS,
        },
        indent=2,
    ) + "\n")
    return label2id


def _report(stats: dict[str, int], splits: dict[str, pd.DataFrame]) -> None:
    print("\nFilter funnel")
    print(f"  rows scanned            {stats['scanned']:>12,}")
    print(f"  with a narrative        {stats['has_narrative']:>12,}")
    print(f"  received >= {DATE_MIN}  {stats['in_window']:>12,}")
    print(f"  in one of {len(CATEGORIES)} categories {stats['in_category']:>12,}")
    print(f"  survived cleaning       {stats['cleaned']:>12,}"
          f"   (-{stats['in_category'] - stats['cleaned']:,} too short)")
    print(f"  after dedupe            {stats['deduped']:>12,}"
          f"   (-{stats['cleaned'] - stats['deduped']:,} duplicates)")

    print("\nClass balance")
    width = max(len(c) for c in CATEGORIES)
    print(f"  {'product'.ljust(width)}  {'eligible':>9}  {'train':>6}  {'val':>5}  "
          f"{'test':>5}  {'total':>6}")
    for c in CATEGORIES:
        counts = [int((splits[s]["product"] == c).sum()) for s in ("train", "val", "test")]
        print(f"  {c.ljust(width)}  {stats[f'eligible::{c}']:>9,}  "
              f"{counts[0]:>6,}  {counts[1]:>5,}  {counts[2]:>5,}  {sum(counts):>6,}")
    totals = [len(splits[s]) for s in ("train", "val", "test")]
    print(f"  {'TOTAL'.ljust(width)}  {'':>9}  {totals[0]:>6,}  {totals[1]:>5,}  "
          f"{totals[2]:>5,}  {sum(totals):>6,}")


def main():
    """Entry point for `make data`."""
    path = download()

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < 2:
        print(f"[warn] only {free_gb:.1f} GB free", file=sys.stderr)

    df, stats = load_and_filter(path)
    df = balanced_sample(df)

    label2id = _write_label_map()
    df["label"] = df["product"].map(label2id).astype("int16")

    train_df, val_df, test_df = stratified_split(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, part in splits.items():
        out = PROCESSED_DIR / f"{name}.parquet"
        part[["complaint_id", "narrative", "product", "label"]].to_parquet(out, index=False)
        print(f"[write] {out.relative_to(ROOT)}  ({len(part):,} rows)")
    print(f"[write] {LABEL_MAP_PATH.relative_to(ROOT)}")
    print(f"[write] {SPLITS_PATH.relative_to(ROOT)}")

    _report(stats, splits)


if __name__ == "__main__":
    main()
