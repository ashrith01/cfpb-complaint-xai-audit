# Execution Plan

Derived from `BRD.md`. Check off as you go. Each day's "Definition of done"
is the gate — don't start the next day until it's met.

## Day 1 — Data prep ✅
- [x] Download CFPB Consumer Complaint Database (bulk CSV or API)
- [x] Filter to 6–8 chosen product categories (decide and hardcode the list in `src/data_prep.py`)
- [x] Clean: strip boilerplate/redaction placeholders (`XXXX`), drop empty narratives, dedupe
- [x] Stratified train/val/test split, seeded, saved to `data/splits.json`
- [x] Subsample to ~30–40k narratives total
- **DoD:** ✅ `data/processed/` populated (36,000 rows, 8 classes × 4,500), class balance
  table printed by `make data`

**Outcome:** 16.9M complaints → 3.83M with narratives → 2.11M in-window/in-category →
983,364 after cleaning + dedupe → 36,000 sampled (25,200 / 5,400 / 5,400).
Dedupe removed **52% of eligible narratives** — mass-filed credit-repair template letters
that become identical once redaction placeholders are stripped. Credit reporting alone
fell from ~1.5M to 535k. Without this the majority class would have been near-duplicate
boilerplate, leaking across the split and inflating both test metrics and the
faithfulness audit.

Scope decision: restricted to complaints received on/after **2024-01-01**, which pins the
`Product` column to a single taxonomy revision. The full history carries 21 product values
across several revisions (three different names for "credit reporting" alone); the date
filter yields 8 clean classes with no merge table to maintain.

## Day 2 — Baseline ✅
- [x] TF-IDF + Logistic Regression baseline
- [x] Report accuracy + macro-F1 on val set
- **DoD:** ✅ baseline written to `results/metrics.json` under `baseline`

**Val accuracy 0.8437 · macro-F1 0.8445** (1–2 gram TF-IDF, 185,763 features,
untuned LogReg, 11.6s fit).

⚠️ **This raises the bar.** BRD §9 lists a ≥0.75 macro-F1 target, but the same row
says "establish baseline first, beat it" — and a bag-of-words model already clears
0.75 comfortably. **The operative target for Day 3–4 is >0.8445, not 0.75.** Hitting
0.75 with DistilBERT would be a regression dressed up as success.

Per-class F1 splits cleanly into easy and hard classes, which sets up Day 5:
mortgage (0.949) and student_loan (0.940) are near-solved by keyword presence,
while checking_savings (0.784), credit_reporting (0.791) and debt_collection
(0.799) are the confusable core. Expect the fine-tune's gain — and the most
interesting explanation-faithfulness behaviour — to concentrate there.

## Day 3–4 — Fine-tune ✅
- [x] `src/train.py`: DistilBERT sequence classification head
- [x] Try 2–4 configs (learning rate, epochs, max_len) — small grid, not a sweep
- [x] Select best by val macro-F1
- [x] Run on held-out test set once, report final metrics + confusion matrix
- **DoD:** ✅ `results/metrics.json` has test accuracy, macro-F1, per-class F1;
  confusion matrix at `results/figures/confusion_matrix.png`

**Test macro-F1 0.8495 vs baseline 0.8411 — a +0.0084 gain** (both on the held-out
test set). ⚠️ **Statistically marginal:** McNemar p=0.0241, bootstrap 95% CI
[+0.0002, +0.0164]. DistilBERT fixes 282 baseline errors and introduces 230 new
ones. Any claim must carry the interval, not the point estimate. Best config: lr 5e-5, max_len 256, batch 16, best at **epoch 2 of 3**
(epoch 3 overfit: val macro-F1 fell to 0.8504 while train loss kept dropping).

Two things worth carrying into the explainability work:

1. **max_len 128 scored 0.8430 — below the baseline.** Sequence length dominates
   learning rate here; a 128-token window truncates ~half a typical narrative.
2. **The gain did not land where Day 2 predicted.** The confusable core
   (checking_savings, debt_collection, credit_card) moved ≤+0.003. The gain came
   from credit_reporting (+0.016) and vehicle_loan (+0.019). What's hard for
   bag-of-words is mostly hard for the transformer too — suggesting genuine label
   ambiguity, not a modelling shortfall.

Dominant confusions: money_transfer → checking_savings (0.16), and
credit_reporting ↔ debt_collection (0.10 / 0.09, near-symmetric).

## Day 5 — Error analysis ✅
- [x] Which classes confuse with which? Pull 10 misclassified examples per top-confused pair
- [x] Write a short failure taxonomy (ambiguous narrative, mislabeled ground truth, genuinely hard case)
- **DoD:** ✅ `results/figures/confusion_matrix.png` + [`notebooks/error_analysis.md`](notebooks/error_analysis.md)

**812/5,400 wrong (15.0%); 212 of those at p>0.9.** Four largest confusions are two
symmetric pairs (money_transfer↔checking_savings, credit_reporting↔debt_collection).

🔑 **Headline finding candidate — the model learned a lexical shortcut.** On
money_transfer, accuracy is 0.955 when the narrative names a money service
(zelle/venmo/paypal/…), 0.533 when it doesn't, and **0.118 on the 34 cases using
only bank vocabulary — below the 0.125 random-guess floor.** Established
behaviourally, with no attribution method involved.

This gives Days 6–10 a falsifiable test instead of a plausibility judgement: a
faithful method should put its mass on the brand token in exactly these cases.

Taxonomy: mislabeled ground truth ~40% (the CFPB product field is consumer-chosen,
not derived from text), dual-nature events ~30%, evidence absent from input ~20%
(redacted brand, or institution-type labels), no signal ~10%.

⚠️ Two constraints this puts on later days:
- Model error ≠ explanation error — don't score label noise as unfaithfulness
- The Day 6–8 example set must be **stratified** (confidently-wrong + both
  directions of each symmetric pair), not random

## Day 6–7 — Attribution methods (IG + SHAP) ✅
- [x] `src/explain/integrated_gradients.py` via Captum
- [x] `src/explain/shap_explainer.py` — same 500-example stratified set
- [x] Cache per-example token attributions to `results/explanations/`
- **DoD:** ✅ both cached on the identical example set

IG 10.5 min (50 steps, convergence delta 0.055); SHAP 16.2 min (max_evals=200).
⚠️ NFR3 (<30 min, full test set) is unachievable — IG alone is ~118 min for 5,400.
Ran 500 stratified examples in ~27 min total and reported that honestly.

## Day 8 — Attention rollout ✅
- [x] `src/explain/attention_rollout.py` — weaker baseline explanation method
- **DoD:** ✅ all 3 methods on the same fixed example set, verified identical IDs + tokens

🐛 SDPA attention silently returns `None` for `output_attentions=True` (warning only,
no exception) — would have shipped empty attributions. Fixed with eager load + guard.

## Day 9 — Faithfulness scoring ✅
- [x] `src/faithfulness.py`: comprehensiveness + sufficiency
- [x] top-k = 10%, committed in code before any score existed
- **DoD:** ✅ per-method, per-example, aggregated + per-stratum

| method | comp ↑ | suff ↓ | rationale alone holds |
|---|---|---|---|
| **Integrated Gradients** | **0.455** | **0.003** | **94.4%** |
| SHAP | 0.294 | −0.043 | 89.4% |
| attention rollout | 0.263 | 0.117 | 82.2% |
| *random (control)* | *0.031* | *0.467* | *46.2%* |

Added a **random-attribution control** not in the original plan — without it the
numbers are uninterpretable, since deleting any 10% moves the prediction somewhat.

## Day 10 — Disagreement / audit ✅
- [x] `src/audit.py`: top-3 overlap across methods, per example (in word space)
- [x] Confidently-wrong + unfaithful cases identified
- [x] Headline finding written → `results/headline_finding.md`
- **DoD:** ✅ `results/figures/disagreement.png` + failure modes

🔑 **HEADLINE:** methods agree on only **13–29%** of top-3 tokens (ceiling ~93%);
SHAP vs attention share *zero* tokens on **66%** of examples. Faithfulness breaks
the tie: **in 19.3% of confidently-wrong predictions, attention rollout was
unfaithful while IG on the same example was not.**

Independent check — IG puts the behaviourally-established brand token in its top-3
**94.6%** of the time vs 73% for the others.

⚠️ Withdrew a claim: my first chart title said "disagreement is worst where it
matters most". It isn't — overlap is flat across strata (0.12–0.33). The honest
reading is worse: it does **not improve** on confident predictions.

## Day 11 — Packaging ✅
- [x] `Makefile` targets: `setup`, `lock`, `test`, `data`, `train`, `evaluate`, `explain`, `audit`, `figures`, `all`
- [x] `requirements.txt` pinned (uv lock from `requirements.in`)
- [x] Fresh-clone test
- **DoD:** ✅ clean clone → deps sync, 32 tests pass, `make data` reproduces
  train/val/test **byte-identically**

Added 32 tests covering `clean_narrative` (frozen into the committed dataset) and
the faithfulness/overlap primitives the headline rests on.

🔑 New measurement — attention rollout ranks **pure punctuation** in its top-3
**23.8%** of the time (67.6% of explanations contain ≥1), vs 2.4% for IG. That is
the *mechanism* behind its low comprehensiveness, not a restatement of it.

## Day 12 — Report ✅
- [x] README rewritten as a technical report, headline finding in paragraph 1
- [x] Figures: confusion matrix, disagreement + faithfulness (3 panels), 2 example walkthroughs
- [x] Resume bullet with real numbers
- **DoD:** ✅ finding stated with a number in the first paragraph

> Fine-tuned DistilBERT for CFPB complaint routing (8 classes, macro-F1 0.85) and
> built a faithfulness audit of its explainability layer (Integrated Gradients,
> SHAP, attention rollout) — found the three methods agreed on only 13–29% of
> their top-attributed tokens, and that attention-based explanations were
> unfaithful in 19.3% of high-confidence errors where gradient-based explanations
> were not.

## MLOps wrap ✅ (2026-09-18)
- [x] Multi-stage `Dockerfile` (uv builder, CPU-only torch, non-root runtime), allowlist `.dockerignore`, `docker-compose.yml` (api + MLflow UI), `make docker-build` / `docker-run`
- [x] MLflow tracking (local SQLite) with git SHA + dataset hash on every run; Day 2–4 runs backfilled; winner registered and promoted to `@production`; serving resolves by alias
- [x] `ci.yml` (ruff, pytest, smoke pipeline on a 560-row fixture, docker build) and `eval-gate.yml` (accuracy, faithfulness ordering, random control, split integrity; PR comment with deltas)
- [x] FastAPI `/predict`, `/explain`, `/health`; JSON request log; latency/QPS/cold-start benchmark
- [x] `src/monitor.py`: PSI with a bootstrap noise floor vs realised accuracy, out-of-time backtest, post-snapshot class prior
- [ ] Push `demo/ship-baseline`, open the PR, capture the failing gate
- **DoD:** ✅ gate passes on main and fails on the shipped-baseline branch; container serves `@production` v1

Changed from the plan: macro-F1 floor 0.845 (0.84 passes the baseline); SQLite
instead of the MLflow file store (refused by MLflow 3.16); drift measured
within the original window, because the CFPB stopped publishing narratives on
14 Aug 2026.

## Guardrails (re-read if scope creep starts)
- No explaining generative output — classification only
- No model comparison beyond baseline + 1 fine-tune variant
- If time is short, cut disagreement-analysis *depth* before cutting faithfulness scoring — the audit is the differentiator
