# Execution Plan

Derived from `BRD.md`. Check off as you go. Each day's "Definition of done"
is the gate — don't start the next day until it's met.

## Day 1 — Data prep
- [ ] Download CFPB Consumer Complaint Database (bulk CSV or API)
- [ ] Filter to 6–8 chosen product categories (decide and hardcode the list in `src/data_prep.py`)
- [ ] Clean: strip boilerplate/redaction placeholders (`XXXX`), drop empty narratives, dedupe
- [ ] Stratified train/val/test split, seeded, saved to `data/splits.json`
- [ ] Subsample to ~30–40k narratives total
- **DoD:** `data/processed/` populated, class balance table in a notebook cell or printed to console

## Day 2 — Baseline
- [ ] TF-IDF + Logistic Regression baseline
- [ ] Report accuracy + macro-F1 on val set
- **DoD:** baseline number written down in `results/metrics.json` — this is the floor to beat

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
