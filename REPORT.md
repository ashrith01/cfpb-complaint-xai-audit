# Running Technical Report

Working log for `faithful-xai-cfpb`, updated as each day of `PLAN.md` completes.
Records what was built, what the numbers were, and — most importantly — the
decisions and surprises, since those are what a reader can't reconstruct from
the code afterwards.

- **Requirements:** [`BRD.md`](./BRD.md)
- **Day-by-day checklist:** [`PLAN.md`](./PLAN.md)
- **Final polished report:** [`README.md`](./README.md) (written Day 12)

**Status:** Days 1–10 complete. Days 11–12 (packaging + report) next.

| Day | Work | Status | Key number |
|---|---|---|---|
| 1 | Data prep | ✅ | 36,000 narratives, 8 classes × 4,500 |
| 2 | TF-IDF + LogReg baseline | ✅ | test macro-F1 **0.8411** |
| 3–4 | Fine-tune DistilBERT | ✅ | test macro-F1 **0.8495** (+0.0084) |
| 5 | Error analysis | ✅ | 15.0% error rate; **lexical shortcut found** |
| 6–7 | Integrated Gradients + SHAP | ✅ | 500 examples, same set |
| 8 | Attention rollout | ✅ | 9s, all 3 aligned |
| 9 | Faithfulness scoring | ✅ | IG **0.455** > SHAP 0.294 > attn 0.263 > random 0.031 |
| 10 | Disagreement / audit | ✅ | **19.3%** attn unfaithful where IG faithful |
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

### …but the margin is statistically marginal

| test | result |
|---|---|
| McNemar (paired, exact) | 282 baseline-wrong/BERT-right vs 230 the other way, **p = 0.0241** |
| macro-F1 delta, bootstrap 95% CI (2,000 resamples) | **[+0.0002, +0.0164]** |
| resamples where DistilBERT wins | 97.7% |

The gain is significant at α=0.05, but **the lower bound of the interval is
+0.0002 — all but touching zero.** The honest statement is "DistilBERT beats
TF-IDF, but by an amount this test set can only just resolve," not "DistilBERT is
better." Any README claim must carry the interval, not the point estimate.

Note also the paired counts: DistilBERT fixes 282 of the baseline's errors *and
introduces 230 new ones of its own*. The net gain is small because the two models
fail on largely different examples, not because the transformer is uniformly
better.

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

## Day 5 — Error analysis ✅

Implemented `src/evaluate.py` (`make evaluate`) — the module the BRD architecture
listed but the scaffolding never created. It caches per-example test predictions
with **full class probabilities** to `results/predictions_test.parquet`, because
Day 9 faithfulness scoring needs the probability of the originally-predicted class
after tokens are removed, not just the argmax. Computing it once here avoids
re-running the model inside each of the three explainers.

Full write-up: [`notebooks/error_analysis.md`](./notebooks/error_analysis.md).

**812 of 5,400 misclassified (15.0%).** Mean confidence 0.9221 when correct vs
0.7162 when wrong — but **212 errors (26.1%) still land above p>0.9**. That
confidently-wrong population is the Day 9–10 target.

### The four largest confusions are two symmetric pairs

| true | predicted | count | % of true class |
|---|---|---|---|
| money_transfer | checking_savings | 106 | 15.7% |
| credit_reporting | debt_collection | 68 | 10.1% |
| checking_savings | money_transfer | 61 | 9.0% |
| debt_collection | credit_reporting | 60 | 8.9% |

Symmetry is diagnostic. A one-way bias would mean the model favours a class;
confusion flowing equally both ways means the boundary itself is ill-defined.

### Headline: the model has learned a lexical shortcut

Accuracy on the money_transfer / checking_savings pair, split by whether the
narrative names a money service explicitly (`zelle|venmo|paypal|cash app|
coinbase|western union|moneygram|remitly|wire transfer|crypto|bitcoin`):

| true class | cue | n | accuracy |
|---|---|---|---|
| money_transfer | term present | 448 | **0.955** |
| money_transfer | term absent | 227 | **0.533** |
| money_transfer | bank terms only | 34 | **0.118** |
| checking_savings | term absent | 632 | **0.878** |
| checking_savings | term present | 43 | **0.442** |

**The model appears to be detecting brand names rather than reasoning about the
complaint.** With a service named it is 95.5% accurate; without, 53.3%; and on
the 34 money-transfer complaints using *only* bank vocabulary it collapses to
**11.8% — below the 12.5% random-guess floor for 8 balanced classes.**

This is established *behaviourally*, with no attribution method involved, which
makes it the most valuable thing found so far. It converts Days 6–10 from "do
these highlights look sensible?" into a falsifiable test:

> A faithful attribution method should place its mass on the brand token in
> precisely these cases. One that instead highlights surrounding complaint
> language is describing reasoning the model is not doing.

Explainability work rarely has any ground truth to check against. This is a
partial one, obtained for free.

### Failure taxonomy (hand review of all 40 sampled errors)

1. **Mislabeled ground truth (~40%)** — the CFPB product field is chosen by the
   *consumer* at filing time, not derived from the text. Repeatedly the model's
   prediction fits the narrative better than the gold label: a complaint labelled
   `checking_savings` that is entirely about Zelle's design (p=0.990), a Coinbase
   account takeover labelled `checking_savings`, an FCRA §605 dispute letter to
   TransUnion labelled `debt_collection`.
2. **Genuinely dual-nature events (~30%)** — a collections account on a credit
   report is one event with two valid labels; several narratives cite FDCPA and
   FCRA in the same paragraph. This is an irreducible ceiling, and the most likely
   reason Day 3–4's fine-tune gained ≤0.003 F1 on exactly these classes.
3. **Discriminating evidence absent from the input (~20%)** — either redaction
   removed the deciding brand token, or the label depends on the institution's
   registration (Chime and Relay Financial are money-services businesses, so
   ordinary checking-account complaints about them are filed as money_transfer).
   Neither fact is in the text.
4. **No product signal at all (~10%)** — generic statute citations and demand
   letters. One error is an FCRA template with placeholders left unfilled.

### Consequences for the audit

**Model error ≠ explanation error.** Categories 1 and 3 mean a large share of
"wrong" predictions are wrong for reasons no explanation could repair.
Faithfulness asks whether an explanation reflects *the model's* decision process —
well-posed even when the decision is wrong. Conflating the two would score label
noise as unfaithfulness.

**The Day 6–8 example set must be stratified, not random** — it needs the
confidently-wrong cases (n=212) and both directions of the symmetric pairs, or
the comparison runs only on easy cases where every method agrees.

---

## Days 6–8 — Attribution methods ✅

All three explainers run on the **same 500 stratified examples**, targeting the
model's **predicted** class (not the true label — the question is what drove the
decision, which stays well-posed when the decision is wrong, and Day 5 found ~40%
of errors are mislabelled ground truth anyway).

| method | runtime | notes |
|---|---|---|
| Integrated Gradients (Captum) | 10.5 min | 50 steps, convergence delta mean 0.055 |
| SHAP (PartitionExplainer) | 16.2 min | max_evals=200, chosen by measurement |
| attention rollout | 9 s | not class-conditioned — the intentional weak baseline |

### Two bugs that would have silently corrupted the audit

**Attention rollout produced nothing.** The checkpoint loads with PyTorch's SDPA
attention kernel, which does not support `output_attentions=True` and returns
`None` with only a *warning*. Had a downstream index not tripped, the third method
could have shipped empty attributions. Fixed with an explicit eager load and a
guard that raises a clear error.

**SHAP was explaining text the model cannot read.** Its Text masker tokenizes the
*raw* string, so it perturbed up to **2,214 tokens against the model's 254-token
window**, with 35.4% of the example set truncated by the model. The attribution
values out there were correctly ~0 (1.1% of mass), so this was not wrong in the
obvious way — the damage was precision: a fixed 200-evaluation budget spread over
up to 9× more tokens than the model consumes, worst on exactly the long
narratives where attribution is hardest. Now truncated to the model window first,
verified prediction-preserving to 0.00000 mean probability change.

The second is the kind of bug that would have quietly weakened SHAP and produced
a confident, wrong headline about which method to trust.

### Operational note

The machine was deep into swap (10.7 GB of 12 GB) and the OS killed the SHAP run
twice. SHAP now checkpoints every 10 examples and resumes, so a kill costs 10
examples rather than the whole run — the difference between a step that
eventually finishes and one that never does.

---

## Day 9 — Faithfulness ✅

`TOP_K_PCT = 0.10`, committed in code before any score existed. Definitions follow
DeYoung et al. (2020): comprehensiveness = confidence lost when the top-10% is
removed (**higher is better**); sufficiency = confidence lost when *only* the
top-10% is kept (**lower is better**).

| method | comp ↑ | suff ↓ | comp (reweighted) | pred flips on removal | rationale alone holds |
|---|---|---|---|---|---|
| **Integrated Gradients** | **0.455** | **0.003** | 0.384 | 52.8% | **94.4%** |
| SHAP | 0.294 | −0.043 | 0.267 | 46.8% | 89.4% |
| attention rollout | 0.263 | 0.117 | 0.224 | 31.8% | 82.2% |
| *random (control)* | *0.031* | *0.467* | *0.024* | *6.2%* | *46.2%* |

**The random control is what makes this readable.** Deleting 10% of any text moves
a prediction somewhat; a method earns credit only by beating arbitrary deletion.
All three clear it comfortably — so all three carry real signal — but the spread
between them is large, and attention rollout recovers only **55%** of the
faithfulness IG does over that baseline.

SHAP's **negative sufficiency (−0.043)** is a real effect, not noise: keeping only
its top-10% *raises* confidence above the full text, because the discarded 90%
contains evidence arguing against the prediction. A rationale can be more
persuasive to the model than the document it came from.

Ranking uses **signed** attribution, not absolute value. A large negative score
means the token argues *against* the prediction, so removing it should push
confidence up. Ranking by `|score|` would blend supporting and opposing evidence
and systematically blunt comprehensiveness for the two methods that can express
opposition — biasing the comparison toward the method expected to be worst.

---

## Day 10 — Disagreement audit ✅

Cross-method comparison is done in **word space** (wordpieces merged, attributions
summed), because IG/rollout attribute to wordpieces and SHAP to word spans;
comparing raw token lists would score tokenisation artefacts as disagreement.

### Pairwise top-3 overlap — against a ~0.93 ceiling, not 1.0

| pair | mean overlap | share with **zero** shared tokens |
|---|---|---|
| IG vs SHAP | 0.287 | 39.4% |
| IG vs attention | 0.295 | 28.4% |
| SHAP vs attention | **0.133** | **66.0%** |

SHAP and attention rollout share **no top-3 token at all on two-thirds of
examples** — about the same prediction, of the same model.

### Confidently wrong *and* unfaithful (of 150 confident-wrong examples)

| method | count | rate |
|---|---|---|
| Integrated Gradients | 13 | 8.7% |
| SHAP | 36 | 24.0% |
| attention rollout | 38 | 25.3% |
| *random* | *135* | *90.0%* |

**In 19.3% of confidently-wrong predictions, attention rollout's explanation was
unfaithful while Integrated Gradients' explanation of the same example was not.**
That is the decision-relevant number: not "both methods are imperfect" but "one
was right and one was not, on the cases that matter most."

### Independent check against the Day 5 behavioural prior

Day 5 established without any attribution method that the model keys on
money-service brand names (accuracy 0.955 with one named, 0.118 without). Asked to
explain those same predictions:

| method | brand token in top-3 |
|---|---|
| Integrated Gradients | **94.6%** (35/37) |
| SHAP | 73.0% |
| attention rollout | 73.0% |

IG almost always recovers the cue the model demonstrably relies on. This is the
rarest thing in explainability work — a partial ground truth — and it agrees with
the faithfulness ranking derived independently.

### A claim I had to withdraw

My first version of the disagreement figure was titled *"Disagreement is worst
where it matters most."* The data say otherwise: overlap is essentially **flat**
across strata (0.12–0.33), and IG-vs-attention is in fact *highest* on
confidently-wrong examples (0.329). The title asserted a trend the chart itself
contradicted.

The honest reading is less tidy and more damning: disagreement **does not improve**
on the confident predictions a reviewer is most likely to trust. Corrected in both
the figure and `results/headline_finding.md`.

---

## Day 11–12 — Packaging & report 🚧

Next.
