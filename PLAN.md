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

## Day 3–4 — Fine-tune
- [ ] `src/train.py`: DistilBERT sequence classification head
- [ ] Try 2–4 configs (learning rate, epochs, max_len) — small grid, not a sweep
- [ ] Select best by val macro-F1
- [ ] Run on held-out test set once, report final metrics + confusion matrix
- **DoD:** `results/metrics.json` has test accuracy, macro-F1, per-class F1, confusion matrix saved as figure

## Day 5 — Error analysis
- [ ] Which classes confuse with which? Pull 10 misclassified examples per top-confused pair
- [ ] Write a short failure taxonomy (ambiguous narrative, mislabeled ground truth, genuinely hard case)
- **DoD:** `results/figures/confusion_matrix.png` + a short markdown note in `notebooks/`

## Day 6–7 — Attribution methods (IG + SHAP)
- [ ] `src/explain/integrated_gradients.py` via Captum
- [ ] `src/explain/shap_explainer.py` — subsample test set (~500 examples, SHAP is slow on text)
- [ ] Cache per-example token attributions to `results/explanations/`
- **DoD:** attributions generated and cached for both methods on the same example set

## Day 8 — Attention rollout
- [ ] `src/explain/attention_rollout.py` — weaker baseline explanation method
- **DoD:** all 3 methods produce attributions on the same fixed example set

## Day 9 — Faithfulness scoring
- [ ] `src/faithfulness.py`: comprehensiveness (remove top-k% attributed tokens, measure prediction change)
- [ ] sufficiency (keep only top-k% tokens, measure if prediction holds)
- [ ] Decide top-k threshold upfront (start at 10%), don't tune it post-hoc to flatter results
- **DoD:** faithfulness score per method, per example, aggregated into a table

## Day 10 — Disagreement / audit
- [ ] `src/audit.py`: % overlap of top-3 tokens across methods, per example
- [ ] Identify cases where a method is confidently wrong (high attribution weight, low faithfulness)
- [ ] This is the headline finding — write it down in one sentence with a number
- **DoD:** disagreement chart + failure-mode examples pulled out

## Day 11 — Packaging
- [ ] `Makefile` targets: `data`, `train`, `explain`, `audit`, `all`
- [ ] `requirements.txt` pinned
- [ ] Fresh-clone test: does `make all` actually run end-to-end?
- **DoD:** clean clone → `make all` succeeds without manual intervention

## Day 12 — Report
- [ ] Fill in README "Headline finding" section
- [ ] 4–5 final figures (confusion matrix, faithfulness comparison, disagreement chart, 2 example walkthroughs)
- [ ] Resume bullet finalized with real numbers (see BRD §13)
- **DoD:** a stranger can read the README in <5 min and understand the finding

## Guardrails (re-read if scope creep starts)
- No explaining generative output — classification only
- No model comparison beyond baseline + 1 fine-tune variant
- If time is short, cut disagreement-analysis *depth* before cutting faithfulness scoring — the audit is the differentiator
