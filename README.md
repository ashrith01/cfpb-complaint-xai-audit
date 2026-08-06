# Explainable Fine-Tuned Classifier for High-Stakes Text Decisions

Fine-tunes a small transformer on CFPB consumer complaint classification,
then builds and **audits** an explainability layer for it — measuring
whether the explanations are actually faithful to the model's decisions,
rather than assuming they are.

Full requirements: see [`BRD.md`](./BRD.md).
Execution plan / day-by-day: see [`PLAN.md`](./PLAN.md).

## Status

🚧 Scaffolded, not yet implemented. Continuing in Claude Code — see `PLAN.md`
for the current step.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
make data       # download + clean + split CFPB subsample
make train      # fine-tune baseline classifier
make explain    # generate attributions (IG, SHAP, attention)
make audit      # faithfulness scoring + disagreement analysis
```

## Repo layout

```
data/                  # raw + processed CFPB subsample, versioned split
src/                    # data prep, training, explainability, audit code
notebooks/              # final report notebook
results/                # metrics, cached explanations, figures
BRD.md                  # business requirements doc
PLAN.md                  # day-by-day execution plan + acceptance criteria
```

## Headline finding

_(fill in once the audit step runs — this is the first paragraph a reviewer
will read, keep it to 1–2 sentences with a number in it)_
