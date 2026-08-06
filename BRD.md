# Business Requirements Document
## Project: Explainable Fine-Tuned Classifier for High-Stakes Text Decisions

**Owner:** Ashrith Vadde
**Target roles:** Applied AI Engineer / ML Engineer (product-facing)
**Duration:** 12 working days (~2 weeks)
**Environment:** MacBook M4 Pro, 16GB unified memory, MLX / PyTorch-MPS

---

## 1. Problem Statement

Product teams increasingly deploy fine-tuned models for consequential text
classification (loan complaint routing, content moderation, support-ticket
triage). Regulators and internal risk teams require these decisions to be
**explainable** — but saliency-based explanations are often presented without
verifying they actually reflect the model's decision process. This project
builds a fine-tuned classifier for a real high-stakes task, then builds and
**validates** an explainability layer for it, rather than assuming the
explanations are trustworthy.

## 2. Business Objective

Demonstrate — with evidence, not assertion — that:
1. A small model can be fine-tuned to a production-usable accuracy on a
   real, high-stakes classification task.
2. Standard explainability methods (attention, gradient-based attribution,
   SHAP) can be generated for that model.
3. Those explanations can be **quantitatively audited for faithfulness**,
   and the audit itself produces a decision-relevant finding (e.g. which
   method to trust, and when explanations mislead).

## 3. Target Task & Dataset

**Task:** Multi-class classification of consumer complaint narratives into
product category (e.g., credit reporting, debt collection, mortgage, credit
card, student loan).

**Dataset:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — public domain, US government data, no licensing risk, real stakes (regulatory/financial), free-text narratives with labeled product categories.

**Why this task:**
- Auto-gradable (hard labels, not judge-dependent)
- High-stakes domain → explainability isn't decorative, it's the point
- Public, license-clean, large enough to subsample cleanly
- Distinct from your existing Samsung PRISM work (tabular/image/text
  classical ML) — this is the LLM-era extension of that same skill

**Scope cut for time:** subsample to 6–8 product categories, ~30–40k
narratives (stratified), max sequence length 256 tokens.

## 4. Stakeholders (framed as if this were a real product)

| Stakeholder | Concern this project addresses |
|---|---|
| Compliance/Risk team | Can we justify why a complaint was routed here? |
| ML engineering team | Which explanation method is worth shipping? |
| End user (support agent) | Does the highlighted text make sense? |
| Hiring manager (real audience) | Did the candidate validate their claims or just visualize them? |

## 5. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Fine-tune a small transformer (DistilBERT baseline; Qwen3-1.7B stretch) on the classification task |
| FR2 | Report accuracy, macro-F1, per-class F1, confusion matrix on a held-out test set |
| FR3 | Generate token-level attributions via ≥2 methods: Integrated Gradients (Captum), SHAP, attention-weight rollout |
| FR4 | Compute faithfulness metrics per method: comprehensiveness and sufficiency |
| FR5 | Quantify disagreement between explanation methods on the same predictions |
| FR6 | Surface a failure-mode set: cases where the top-confidence explanation is unfaithful |
| FR7 | Ship a reproducible CLI / notebook: fine-tune → predict → explain → audit, runnable end-to-end |
| FR8 | Publish results as a technical report (README) with charts, not raw logs |

## 6. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR1 | Must run fully on M4 Pro, 16GB unified memory — no required cloud step |
| NFR2 | Fine-tuning run completes in <2 hours on local hardware |
| NFR3 | Explanation generation for the full test set completes in <30 min |
| NFR4 | All runs seeded and reproducible (fixed seeds, versioned data split) |
| NFR5 | Code structured as an installable package, not notebook-only, with `make` targets |
| NFR6 | No PII beyond what's already public/scrubbed in the CFPB dataset |

## 7. Out of Scope (explicit — protects the timeline)

- Explaining *generative* LLM outputs (mechanistic interpretability, SAEs, activation patching)
- Multi-model comparison beyond 1 base model + 1 fine-tune variant
- Human-subject evaluation of explanation usefulness
- Production deployment / serving infrastructure
- Hyperparameter sweep beyond a small, fixed grid (≤4 configs)

## 8. Technical Architecture

```
data/                  # raw + processed CFPB subsample, versioned split
├── raw/
├── processed/
└── splits.json        # seeded train/val/test indices

src/
├── data_prep.py        # download, clean, stratified subsample, split
├── train.py             # fine-tune DistilBERT (+ optional LoRA on Qwen3)
├── evaluate.py           # accuracy, F1, confusion matrix
├── explain/
│   ├── integrated_gradients.py
│   ├── shap_explainer.py
│   └── attention_rollout.py
├── faithfulness.py       # comprehensiveness, sufficiency scoring
└── audit.py               # disagreement analysis, failure-mode mining

notebooks/
└── report.ipynb           # final charts, human-readable walkthrough

results/
├── metrics.json
├── explanations/          # per-example attributions, cached
└── figures/

README.md                  # technical report — the actual deliverable
Makefile                   # make data / make train / make explain / make audit
```

**Stack:** Python, HuggingFace Transformers, PEFT (if LoRA variant), Captum,
SHAP, scikit-learn, matplotlib/plotly.

## 9. Success Metrics (Acceptance Criteria)

| Metric | Target |
|---|---|
| Test macro-F1 | ≥0.75 (task-dependent; establish baseline first, beat it) |
| Faithfulness: comprehensiveness drop | Reported per method, no fixed threshold — the *comparison* is the result |
| Explanation disagreement | Quantified (e.g., % of top-3 tokens overlapping between methods) |
| Reproducibility | `make all` runs end-to-end from clean clone on a fresh machine |
| Deliverable quality | README readable in <5 min, states one clear finding in the first paragraph |

## 10. Timeline (12 working days)

| Day | Work | Output |
|---|---|---|
| 1 | Data prep: download, clean, stratify, seed splits | `data/processed/`, `splits.json` |
| 2 | Baseline: zero-shot/TF-IDF+LogReg baseline for comparison | baseline metrics |
| 3–4 | Fine-tune DistilBERT, tune 2–4 configs, pick best | `results/metrics.json`, confusion matrix |
| 5 | Error analysis on test set — which classes/patterns fail | failure taxonomy v1 |
| 6–7 | Implement Integrated Gradients + SHAP explainers | per-example attributions cached |
| 8 | Implement attention-rollout as third (weaker) method | attribution set complete |
| 9 | Faithfulness scoring: comprehensiveness + sufficiency, all methods | `faithfulness.py` output |
| 10 | Disagreement analysis: where methods diverge, why, on which classes | audit charts |
| 11 | Package as reproducible CLI + `Makefile`, verify clean-clone run | working `make all` |
| 12 | Write README as technical report, finalize 4–5 figures | shippable repo |

Buffer: none built in — if you slip, cut FR5/FR6 depth before cutting FR3/FR4 (the faithfulness audit is the differentiator; keep it even if narrower).

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| SHAP is slow on text at scale | Subsample test set for SHAP specifically (e.g. 500 examples), full set for IG/attention |
| Fine-tune underperforms baseline | Budget Day 2 baseline explicitly so you have a floor to beat, not just a hope |
| Faithfulness metrics show all methods are bad | That's still a finding — report it honestly, don't cherry-pick |
| Scope creep into generative explainability | Out-of-scope list above; re-read it if tempted |
| CFPB text is messy (redactions, boilerplate) | Budget Day 1 fully for cleaning; don't start training on dirty data |

## 12. Deliverables Checklist

- [ ] Public GitHub repo, clean commit history
- [ ] README with: problem, approach, one headline finding, 4–5 figures, reproduction steps
- [ ] `requirements.txt` / `pyproject.toml`, pinned versions
- [ ] `Makefile` with `data`, `train`, `explain`, `audit`, `all` targets
- [ ] Cached results committed (or linked) so reviewers don't need to rerun everything
- [ ] One clear resume bullet drawn directly from the headline finding

## 13. Resume Bullet (target)

> Fine-tuned DistilBERT for CFPB complaint classification (macro-F1: X.XX);
> built and audited a token-attribution explainability layer (Integrated
> Gradients, SHAP, attention) using faithfulness metrics — found
> attention-based explanations were unfaithful in X% of high-confidence
> predictions where gradient-based methods were correct.
