# Running Technical Report

Working log for `faithful-xai-cfpb`, updated as each day of `PLAN.md` completes.
Records what was built, what the numbers were, and — most importantly — the
decisions and surprises, since those are what a reader can't reconstruct from
the code afterwards.

- **Requirements:** [`BRD.md`](./BRD.md)
- **Day-by-day checklist:** [`PLAN.md`](./PLAN.md)
- **Final polished report:** [`README.md`](./README.md) (written Day 12)

**Status:** Days 1–4 complete. Day 5 (error analysis) next.

| Day | Work | Status | Key number |
|---|---|---|---|
| 1 | Data prep | ✅ | 36,000 narratives, 8 classes × 4,500 |
| 2 | TF-IDF + LogReg baseline | ✅ | test macro-F1 **0.8411** |
| 3–4 | Fine-tune DistilBERT | ✅ | test macro-F1 **0.8495** (+0.0084) |
| 5 | Error analysis | ⬜ | |
| 6–7 | Integrated Gradients + SHAP | ⬜ | |
| 8 | Attention rollout | ⬜ | |
| 9 | Faithfulness scoring | ⬜ | |
| 10 | Disagreement / audit | ⬜ | |
| 11 | Packaging | ⬜ | |
| 12 | Report | ⬜ | |

---

## Environment

MacBook M4 Pro, 16 GB unified memory. Python 3.14.4, **torch 2.13.0 with MPS
available**, transformers 5.14.1, pandas 3.0.5, scikit-learn 1.9.0, shap 0.52.0,
captum 0.9.0.

Dependencies are managed with [uv](https://docs.astral.sh/uv/): `requirements.in`
holds the direct dependencies, `requirements.txt` is a fully pinned lock produced
by `make lock`. The lock stays pip-installable, so `pip install -r requirements.txt`
works for anyone who doesn't use uv. `make setup` creates the venv and syncs.

Two library versions are a major release above what the original scaffolding
assumed (`transformers>=4.44` resolved to 5.14.1, `pandas>=2.2` to 3.0.5), which
is why the lock was pinned on Day 1 rather than deferred to Day 11.

---

## Day 1 — Data prep ✅

**Output:** `data/processed/{train,val,test}.parquet` — 36,000 narratives across
8 product classes, 4,500 each, split 25,200 / 5,400 / 5,400.

### Filter funnel

```
rows scanned              16,935,987
with a narrative           3,831,109
received >= 2024-01-01     2,137,523
in one of 8 categories     2,107,321
survived cleaning          2,039,645   (-67,676 too short)
after dedupe                 983,364   (-1,056,281 duplicates)
```

### Decisions

**Source: full bulk CSV archive rather than the filtered API.** The archive is
1.4 GB compressed / 9.06 GB expanded. `data_prep.py` streams the CSV directly out
of the zip and reads only the 4 columns it needs, so the 9 GB is never written to
disk and peak memory stays flat across a 16.9M-row scan.

**Restricted to complaints received on or after 2024-01-01.** The `Product`
column carries **21 distinct values accumulated across several taxonomy
revisions** — "Credit reporting or other personal consumer reports", "Credit
reporting, credit repair services, or other personal consumer reports" and
"Credit reporting" are the same concept under three names, as are "Credit card"
and "Credit card or prepaid card". The date filter pins the data to one revision,
yielding 8 clean classes with no merge table to write or maintain.

**Balanced at 4,500 per class rather than natural frequency.** Credit reporting
is 76% of the real distribution; left alone it would dominate macro-F1 and leave
tail classes with ~35 test examples each. Balancing makes the macro-F1 target
meaningful and gives the Day 9–10 explanation audit equal support per class.
Every class had ≥18,919 candidates, so no upsampling was needed.

**Per-class sampling uses a seeded reservoir** rather than "first N seen", so the
sample is uniform over the whole two-year window instead of clustering on
whichever dates the file happens to start with.

### Surprise: dedupe removed 52% of eligible narratives

`-1,056,281` of 2,039,645. Credit reporting alone fell from ~1.5M to 535k.

Mass-filed credit-repair template letters differ only in the fields CFPB redacts,
so once `clean_narrative()` strips the `XXXX` placeholders they collapse to
byte-identical strings and the hash catches them. Had they stayed in, the 4,500
credit-reporting examples would have been substantially the same letter repeated
— leaking across the train/test split and inflating both the test metrics and the
faithfulness audit those metrics feed. This was the highest-leverage decision in
the step.

### Cleaning

Redaction runs are **deleted, not replaced with a sentinel token**. A
high-frequency `[REDACTED]` marker would itself become a feature the classifier
could key on, and Days 9–10 audit precisely what the model keys on — manufacturing
a shortcut for the audit to find would be self-defeating.

The redaction regex needed two iterations. The obvious `\bX{2,}\b` silently misses
runs glued to digits (`$2500.00XXXX`, `XXXX1234`) because there's no word
boundary between a digit and an `X`. But simply dropping the boundaries mangles
real company names — `EXXON` → `E ON`, `TJ MAXX` → `TJ M`. The shipped pattern
`(?<![A-Za-z])X{2,}(?![A-Za-z])` excludes letters on *both* sides, handling both.
This matters here specifically because company names are exactly the kind of token
the Day 9–10 audit will ask whether the model relies on.

Also handled: nested currency masks (`${ {$3600.00 } }`), unterminated masks
(`{$2500.00XXXX`), and fully-redacted amounts (`{$XX.XX}`, dropped entirely).

### Verification

- No complaint-ID leakage across splits; no duplicate narrative text across splits
- Test set stratified at exactly 675/class across 8 classes
- `splits.json` IDs agree with the parquet files; `label` column agrees with `label_map.json`
- Zero remaining cleaning artifacts (no `XX` runs, braces, doubled `$`, newlines, or sub-100-char rows)
- **Two consecutive runs produced byte-identical `splits.json` and `train.parquet`** — NFR4 demonstrated, not asserted

Splits are recorded by **Complaint ID, not row position**: IDs survive a refreshed
snapshot and a reshuffled DataFrame, positional indices don't. Provenance for the
exact snapshot used is in `data/raw/SOURCE.txt`.

---

## Day 2 — Baseline ✅

TF-IDF (1–2 gram, 185,763 features, `min_df=3`, sublinear) + LogisticRegression,
fit on 25,200 documents in 11.6s.

| | val |
|---|---|
| accuracy | 0.8437 |
| **macro-F1** | **0.8445** |

The baseline is **deliberately untuned**. It is the floor the fine-tune is
measured against, so tuning it would be tuning the goalpost — worth revisiting
only if the fine-tune fails to beat it.

### The acceptance criterion needed restating

BRD §9 sets a ≥0.75 macro-F1 target, but bag-of-words clears it by nine points.
That row also says *"establish baseline first, beat it"* — so the rule was right
and only the number was a guess made before any baseline existed.

**The operative Day 3–4 target is >0.8445, not 0.75.** A DistilBERT run landing at
0.78 would technically satisfy the BRD while actually losing to TF-IDF.

### Per-class F1 splits the task in two

| near-solved | | confusable core | |
|---|---|---|---|
| mortgage | 0.949 | checking_savings | 0.784 |
| student_loan | 0.940 | credit_reporting | 0.791 |
| vehicle_loan | 0.858 | debt_collection | 0.799 |
| money_transfer | 0.825 | credit_card | 0.811 |

Mortgage and student loan are close to keyword detection. The confusion among
checking/savings, credit reporting, debt collection and credit card is where the
fine-tune has room to gain — and where the Day 9–10 faithfulness audit should be
most informative, since those are the cases where the model *cannot* be leaning on
a single obvious token.

### Structural note

Added `src/utils.py` holding split loading and a **merging** `update_metrics()`.
`results/metrics.json` accumulates `baseline` → `finetune` → `faithfulness` →
`disagreement` across four separate days; a writer that overwrote rather than
merged would silently erase earlier sections. Cheaper to get right now than to
debug on Day 10.

---

## Day 3–4 — Fine-tune ✅

DistilBERT-base-uncased with a sequence-classification head, plain PyTorch loop
on MPS. Total grid runtime ~2h10m on the M4 Pro.

### Grid (selected on val macro-F1)

| # | lr | max_len | batch | best val macro-F1 | best epoch | time |
|---|---|---|---|---|---|---|
| **1** | **5e-5** | **256** | **16** | **0.8527** | **2** | 56 min |
| 2 | 2e-5 | 256 | 16 | 0.8497 | 3 | 56 min |
| 3 | 5e-5 | 128 | 32 | 0.8430 | 3 | 26 min |

### Held-out test result (touched once, with the already-selected model)

| | TF-IDF + LogReg | DistilBERT | delta |
|---|---|---|---|
| accuracy | 0.8400 | 0.8496 | +0.0096 |
| **macro-F1** | **0.8411** | **0.8495** | **+0.0084** |

**A fine-tuned transformer beats a bag-of-words baseline by 0.8 points of
macro-F1** on this task — after ~2 hours of training versus 11 seconds.

### Correction made during this step

The first run printed `delta +0.0050 BEATS baseline`, comparing DistilBERT's
**test** score against the baseline's **val** score — two different splits. The
baseline is now scored on both, and the comparison above is like-for-like. This
costs no validity: the baseline config is fixed and untuned, so nothing is being
selected on test.

The corrected gap (+0.0084) is slightly *larger* than the incorrect one, because
both models lose ground from val to test. But the error could just as easily have
run the other way and manufactured a win, and this comparison is the number the
whole project rests on.

### Decisions

**Best epoch, not last.** Config 1 peaked at epoch 2 (0.8527) and *declined* at
epoch 3 (0.8504) while train loss kept falling 0.42 → 0.28 — textbook overfitting.
Keeping the last epoch would have shipped a measurably worse model. This also
makes `epochs` redundant as a grid axis: a 3-epoch run yields the 1- and 2-epoch
results for free, so the grid spends its slots on learning rate and sequence
length instead.

**Plain PyTorch loop rather than HF `Trainer`.** transformers 5.14.1 is a major
version above what the scaffolding assumed, and `Trainer` is where that churn
would bite. The Day 6–8 explainers need direct access to the embedding layer for
Integrated Gradients regardless.

**Benchmarked before launching.** ~46 min per 256-token config measured on 40
steps, so the 2-hour cost was known upfront rather than discovered at hour three.

### Sequence length matters more than learning rate

Config 3 (128 tokens) scored **0.8430 — below the 0.8445 val baseline**. Halving
the context turned the transformer into a *worse-than-bag-of-words* model, while
halving the learning rate cost only 0.003. Median narrative length is ~898
characters, so a 128-token window truncates away roughly half of a typical
complaint.

### Where the gain actually landed

| class | baseline | DistilBERT | delta |
|---|---|---|---|
| vehicle_loan | 0.8785 | 0.8978 | +0.0192 |
| credit_reporting | 0.7684 | 0.7840 | +0.0157 |
| mortgage | 0.9227 | 0.9381 | +0.0154 |
| money_transfer | 0.8227 | 0.8324 | +0.0097 |
| debt_collection | 0.7958 | 0.7988 | +0.0030 |
| credit_card | 0.8090 | 0.8114 | +0.0025 |
| checking_savings | 0.7949 | 0.7972 | +0.0023 |
| student_loan | 0.9367 | 0.9362 | −0.0005 |

The gain is not uniform, and it is *not* concentrated where Day 2 predicted. The
confusable core (checking_savings, debt_collection, credit_card) barely moved —
+0.003 or less. DistilBERT's advantage came from credit_reporting and
vehicle_loan instead. **The classes that are hard for bag-of-words are, for the
most part, hard for the transformer too**, which suggests the difficulty is
genuine label ambiguity rather than a modelling shortfall.

### Confusion structure (`results/figures/confusion_matrix.png`)

Off-diagonal mass is concentrated in three places:

- **money_transfer → checking_savings, 0.16** — the single largest confusion. A
  complaint about a transfer from a bank account is genuinely both.
- **credit_reporting ↔ debt_collection, 0.10 / 0.09** — near-symmetric, the
  signature of genuine overlap rather than a one-way bias. A collections account
  appearing on a credit report is one event filed under two products.
- **credit_card → checking_savings, 0.08**

Best-classified: student_loan 0.96, mortgage 0.93, vehicle_loan 0.90 — each has a
distinctive vocabulary.

This structure is the input to Day 5, and it sets up the real question for Days
9–10: **on the ~20% of cases the model gets wrong, and on the confusable pairs
above, do the explanation methods agree about what drove the decision?** Those
are exactly the predictions where a faithful explanation would be worth having —
and where an unfaithful one would be most misleading to a compliance reviewer.

### Artifacts

- `results/model/` — best checkpoint + tokenizer + `train_config.json` (255 MB, gitignored)
- `results/figures/confusion_matrix.png`
- `results/metrics.json` — `baseline` and `finetune` sections, full grid recorded

---

## Day 5 — Error analysis 🚧

Next.
