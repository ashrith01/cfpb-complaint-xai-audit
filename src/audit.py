"""
Disagreement analysis across explanation methods -- this is where the
project's headline finding comes from.

Contract:
    - top_k_overlap(attr_a, attr_b, k=3) -> float
        % overlap of top-k tokens between two methods, per example.
    - find_confident_but_unfaithful(faithfulness_scores) -> list[example_id]
        Cases where a method assigns high attribution weight but scores
        low on faithfulness -- i.e. confidently wrong explanations.
    - summarize() -> writes results/figures/disagreement.png and a one-
      paragraph finding to be pasted into README.md's "Headline finding"

Cross-method comparison happens in **word space**. IG and attention rollout
attribute to wordpieces ('car', '##ing') while SHAP attributes to word spans, so
comparing raw token lists would score a mismatch wherever a word happens to split
-- an artefact of tokenisation, not a disagreement about the model. Wordpieces
are merged back into words with their attributions summed before any overlap is
computed.

Overlap is also read against a ceiling, not against 1.0: SHAP at max_evals=200
agrees with its own 300-eval reference only 0.93 of the time (see
shap_explainer.MAX_EVALS), so ~0.93 is the practical maximum any two methods
could show, and 1.0 is not the right reference point.
"""

from __future__ import annotations

import json
import re
import string
from itertools import combinations

import numpy as np
import pandas as pd

from src.evaluate import SERVICE_TERMS
from src.explain.common import load_attributions
from src.explain.example_set import build_example_set
from src.explain.shap_explainer import MAX_EVALS
from src.faithfulness import METHODS
from src.utils import FIGURES_DIR, RESULTS_DIR, update_metrics

TOP_K = 3
SELF_AGREEMENT_CEILING = 0.93  # SHAP @200 evals vs its own @300 reference
FINDING_PATH = RESULTS_DIR / "headline_finding.md"


def merge_wordpieces(tokens: list[str], scores: list[float]) -> dict[str, float]:
    """Collapse wordpieces to whole words, summing attribution. Word -> score."""
    words, vals = [], []
    for tok, s in zip(tokens, scores):
        if tok.startswith("##") and words:
            words[-1] += tok[2:]
            vals[-1] += s
        else:
            words.append(tok.strip())
            vals.append(s)
    out: dict[str, float] = {}
    for w, v in zip(words, vals):
        w = w.strip().lower()
        if w:
            # A word occurring twice keeps its strongest contribution rather than
            # being double-counted into an artificially high rank.
            out[w] = max(out.get(w, float("-inf")), v)
    return out


def top_k_words(word_scores: dict[str, float], k: int = TOP_K) -> set[str]:
    return {w for w, _ in sorted(word_scores.items(), key=lambda kv: -kv[1])[:k]}


def top_k_overlap(attr_a: dict[str, float], attr_b: dict[str, float],
                  k: int = TOP_K) -> float:
    """Fraction of the top-k words shared between two methods for one example."""
    a, b = top_k_words(attr_a, k), top_k_words(attr_b, k)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _load_word_attributions() -> dict[str, dict[int, dict[str, float]]]:
    out = {}
    for method in METHODS:
        data = load_attributions(method)
        out[method] = {
            e["complaint_id"]: merge_wordpieces(e["tokens"], e["attributions"])
            for e in data["examples"]
        }
    return out


def find_confident_but_unfaithful(per_example: dict, threshold: float = 0.05):
    """Confidently-wrong predictions whose explanation barely moves the model.

    These are the dangerous cases for a compliance reviewer: the model is sure,
    the model is wrong, and the highlighted rationale turns out not to be what
    the decision rested on -- so the explanation invites agreement with an error.
    """
    out = {}
    for method, rows in per_example.items():
        out[method] = [
            r["complaint_id"] for r in rows
            if r["stratum"] == "confident_wrong" and r["comprehensiveness"] < threshold
        ]
    return out


def brand_token_test(word_attr, examples) -> dict:
    """Do the methods attribute to the brand tokens the model demonstrably keys on?

    Day 5 established *behaviourally* -- with no attribution method involved --
    that accuracy on money_transfer collapses from 0.955 to 0.118 when no money
    service is named. So on examples where such a term is present and the model
    predicted money_transfer, a faithful method should rank that token highly.
    This is the closest thing to ground truth available here.
    """
    pat = re.compile(SERVICE_TERMS, re.I)
    results = {m: {"hits": 0, "n": 0} for m in METHODS}

    for row in examples.itertuples():
        terms = {t.lower() for t in pat.findall(row.narrative)}
        if not terms or row.pred_label != "money_transfer":
            continue
        # The brand may be split across wordpieces; match on prefix too.
        for method in METHODS:
            attr = word_attr[method].get(int(row.complaint_id))
            if not attr:
                continue
            results[method]["n"] += 1
            top = top_k_words(attr, TOP_K)
            if any(any(t.startswith(w[:4]) or w.startswith(t[:4]) for w in top)
                   for t in terms):
                results[method]["hits"] += 1

    for m in results:
        n = results[m]["n"]
        results[m]["rate"] = round(results[m]["hits"] / n, 4) if n else None
    return results


# Function words and punctuation cannot be evidence for a product category. If a
# method ranks them top-3 it is not identifying rationale, and removing them will
# barely move the model -- which is the mechanism behind a low comprehensiveness
# score rather than a restatement of it.
_STOPWORDS = set(
    "the a an and or of to in is was for i my me it that this on at be with have "
    "has had they them you we as not but if then so all any are were do did does "
    "from by".split()
)


def _is_junk(word: str) -> bool:
    return all(c in string.punctuation for c in word) or word in _STOPWORDS


def junk_token_audit(word_attr) -> dict:
    """Share of each method's top-3 that is punctuation or a function word."""
    out = {}
    for method in METHODS:
        junk, punct = [], []
        for attr in word_attr[method].values():
            top = top_k_words(attr, TOP_K)
            if not top:
                continue
            junk.append(sum(_is_junk(w) for w in top) / len(top))
            punct.append(
                sum(all(c in string.punctuation for c in w) for w in top) / len(top)
            )
        out[method] = {
            "top3_junk_share": round(float(np.mean(junk)), 4),
            "top3_punctuation_share": round(float(np.mean(punct)), 4),
            "examples_with_any_punctuation": round(
                float(np.mean([p > 0 for p in punct])), 4),
        }
    return out


def _plot(pairwise, per_stratum, faith) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    short = {"integrated_gradients": "IG", "shap": "SHAP",
             "attention_rollout": "attention", "random": "random"}

    ax = axes[0]
    labels = [f"{short[a]}\nvs {short[b]}" for a, b in pairwise]
    vals = [pairwise[p]["mean"] for p in pairwise]
    bars = ax.bar(labels, vals, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.axhline(SELF_AGREEMENT_CEILING, ls="--", c="#666", lw=1)
    ax.text(2.4, SELF_AGREEMENT_CEILING + .02, "measurement ceiling (0.93)",
            ha="right", fontsize=8, color="#666")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"mean top-{TOP_K} word overlap")
    ax.set_title(f"Methods disagree about the same predictions")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .02, f"{v:.2f}",
                ha="center", fontsize=10, fontweight="bold")

    ax = axes[1]
    strata = list(next(iter(per_stratum.values())).keys())
    x = np.arange(len(strata))
    for i, (pair, by_s) in enumerate(per_stratum.items()):
        ax.plot(x, [by_s[s] for s in strata], marker="o",
                label=f"{short[pair[0]]} vs {short[pair[1]]}")
    ax.set_xticks(x, [s.replace("_", "\n") for s in strata], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"mean top-{TOP_K} overlap")
    # Deliberately not "worst where it matters most" -- the data show the
    # opposite of a trend, and the flatness is itself the point: disagreement
    # does not resolve on the confident predictions a reviewer would trust.
    ax.set_title("Disagreement is uniform — it does not\nimprove on confident predictions")
    ax.legend(fontsize=8)

    ax = axes[2]
    order = list(METHODS) + ["random"]
    comp = [faith[m]["comprehensiveness_mean"] for m in order]
    suff = [faith[m]["sufficiency_mean"] for m in order]
    x = np.arange(len(order))
    ax.bar(x - .2, comp, .4, label="comprehensiveness ↑", color="#4C78A8")
    ax.bar(x + .2, suff, .4, label="sufficiency ↓", color="#E45756")
    ax.axhline(0, c="#333", lw=.8)
    ax.set_xticks(x, [short[m] for m in order])
    ax.set_ylabel("confidence delta")
    ax.set_title("Faithfulness: only IG is both\ncomprehensive and sufficient")
    ax.legend(fontsize=8)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "disagreement.png", dpi=150)
    plt.close(fig)


def summarize():
    """Entry point for `make audit` (final step, after faithfulness.py)."""
    examples = build_example_set()
    word_attr = _load_word_attributions()
    per_example = json.loads((RESULTS_DIR / "faithfulness_per_example.json").read_text())
    faith = json.load(open(RESULTS_DIR / "metrics.json"))["faithfulness"]["methods"]

    stratum_of = {int(r.complaint_id): r.stratum for r in examples.itertuples()}
    ids = sorted(word_attr[METHODS[0]])

    pairwise, per_stratum = {}, {}
    for a, b in combinations(METHODS, 2):
        ov = {i: top_k_overlap(word_attr[a][i], word_attr[b][i]) for i in ids}
        vals = np.array(list(ov.values()))
        pairwise[(a, b)] = {
            "mean": round(float(vals.mean()), 4),
            "median": round(float(np.median(vals)), 4),
            "zero_overlap_rate": round(float((vals == 0).mean()), 4),
            "full_agreement_rate": round(float((vals == 1).mean()), 4),
        }
        by_s = {}
        for s in ("confident_correct", "confident_wrong",
                  "uncertain_correct", "uncertain_wrong"):
            sel = [ov[i] for i in ids if stratum_of[i] == s]
            by_s[s] = round(float(np.mean(sel)), 4)
        per_stratum[(a, b)] = by_s

    unfaithful = find_confident_but_unfaithful(per_example)
    brand = brand_token_test(word_attr, examples)
    junk = junk_token_audit(word_attr)
    _plot(pairwise, per_stratum, faith)

    # The precise claim: attention fails where IG succeeds, on the same example.
    # "Both were unfaithful" is a much weaker statement than "one was right".
    by_m = {m: {r["complaint_id"]: r for r in rows} for m, rows in per_example.items()}
    cw_ids = [i for i in ids if stratum_of[i] == "confident_wrong"]
    ar_bad_ig_ok = sum(
        1 for i in cw_ids
        if by_m["attention_rollout"][i]["comprehensiveness"] < 0.05
        and by_m["integrated_gradients"][i]["comprehensiveness"] >= 0.05
    )

    payload = {
        "top_k": TOP_K,
        "self_agreement_ceiling": SELF_AGREEMENT_CEILING,
        "shap_max_evals": MAX_EVALS,
        "pairwise_overlap": {f"{a}|{b}": v for (a, b), v in pairwise.items()},
        "overlap_by_stratum": {f"{a}|{b}": v for (a, b), v in per_stratum.items()},
        "confident_but_unfaithful_counts": {m: len(v) for m, v in unfaithful.items()},
        "n_confident_wrong": len(cw_ids),
        "attention_unfaithful_where_ig_faithful": ar_bad_ig_ok,
        "brand_token_test": brand,
        "junk_token_audit": junk,
    }
    update_metrics("disagreement", payload)

    print(f"Pairwise top-{TOP_K} word overlap (ceiling ~{SELF_AGREEMENT_CEILING}):")
    for (a, b), v in pairwise.items():
        print(f"  {a:<21} vs {b:<19} mean {v['mean']:.3f}  "
              f"zero-overlap {v['zero_overlap_rate']:.1%}")

    print(f"\nOverlap by stratum:")
    strata = list(next(iter(per_stratum.values())).keys())
    print(f"  {'pair':<40}" + "".join(f"{s:>20}" for s in strata))
    for (a, b), by_s in per_stratum.items():
        print(f"  {a[:18]+' vs '+b[:18]:<40}" + "".join(f"{by_s[s]:>20.3f}" for s in strata))

    n_cw = sum(1 for i in ids if stratum_of[i] == "confident_wrong")
    print(f"\nConfidently wrong AND unfaithful (comprehensiveness < 0.05), "
          f"of {n_cw} confident-wrong examples:")
    for m, v in unfaithful.items():
        print(f"  {m:<22} {len(v):>3}  ({len(v) / n_cw:.1%})")

    print(f"\nBrand-token test -- does the top-{TOP_K} contain the term the model "
          f"demonstrably keys on?")
    for m, v in brand.items():
        if v["n"]:
            print(f"  {m:<22} {v['hits']:>3}/{v['n']}  ({v['rate']:.1%})")

    print(f"\nattention unfaithful WHERE IG faithful (same example): "
          f"{ar_bad_ig_ok}/{n_cw} ({ar_bad_ig_ok / n_cw:.1%})")

    print(f"\nWhat is in the top-{TOP_K}? (punctuation and function words cannot be "
          f"evidence for a product category)")
    print(f"  {'method':<22}{'stopword/punct':>16}{'punctuation':>14}"
          f"{'>=1 punct':>12}")
    for m, v in junk.items():
        print(f"  {m:<22}{v['top3_junk_share']:>15.1%}{v['top3_punctuation_share']:>14.1%}"
              f"{v['examples_with_any_punctuation']:>12.1%}")

    _write_finding(pairwise, per_stratum, faith, unfaithful, brand, n_cw, ar_bad_ig_ok)
    print(f"\nwrote {FIGURES_DIR / 'disagreement.png'}")
    print(f"wrote {FINDING_PATH}")


def _write_finding(pairwise, per_stratum, faith, unfaithful, brand, n_cw,
                   ar_bad_ig_ok: int) -> None:
    lo = min(v["mean"] for v in pairwise.values())
    hi = max(v["mean"] for v in pairwise.values())
    worst_zero = max(v["zero_overlap_rate"] for v in pairwise.values())
    strata_vals = [v for by_s in per_stratum.values() for v in by_s.values()]

    text = f"""# Headline finding

**On 500 held-out CFPB complaint predictions, three standard explanation methods
agree on only {lo:.0%}–{hi:.0%} of their top-{TOP_K} tokens — and faithfulness scoring shows
this is not a tie between equally valid views.**

Integrated Gradients, SHAP and attention rollout were run on the *same* 500
predictions of the *same* fine-tuned DistilBERT. SHAP and attention rollout share
no top-{TOP_K} token at all on {worst_zero:.0%} of examples. The reference point is not
100%: SHAP at the evaluation budget used here reproduces its own higher-budget
ranking {SELF_AGREEMENT_CEILING:.0%} of the time, so ~{SELF_AGREEMENT_CEILING:.0%} is the practical ceiling.

Deleting the top 10% of tokens each method identifies costs the model:

| method | confidence lost (comprehensiveness ↑) | rationale alone preserves prediction |
|---|---|---|
| Integrated Gradients | {faith['integrated_gradients']['comprehensiveness_mean']:.3f} | {faith['integrated_gradients']['pred_held_on_rationale_only']:.1%} |
| SHAP | {faith['shap']['comprehensiveness_mean']:.3f} | {faith['shap']['pred_held_on_rationale_only']:.1%} |
| attention rollout | {faith['attention_rollout']['comprehensiveness_mean']:.3f} | {faith['attention_rollout']['pred_held_on_rationale_only']:.1%} |
| random tokens (control) | {faith['random']['comprehensiveness_mean']:.3f} | {faith['random']['pred_held_on_rationale_only']:.1%} |

All three beat random, so all three carry real signal. But over that random
baseline, attention rollout recovers only
{(faith['attention_rollout']['comprehensiveness_mean'] - faith['random']['comprehensiveness_mean']) / (faith['integrated_gradients']['comprehensiveness_mean'] - faith['random']['comprehensiveness_mean']):.0%} of the faithfulness Integrated
Gradients does.

**The decision-relevant number:** among the {n_cw} predictions the model got wrong
while highly confident — the cases where a plausible explanation is most likely
to be believed and most likely to mislead — attention rollout produced an
explanation that barely moves the model (comprehensiveness < 0.05)
{len(unfaithful['attention_rollout']) / n_cw:.1%} of the time, versus
{len(unfaithful['integrated_gradients']) / n_cw:.1%} for Integrated Gradients. In
**{ar_bad_ig_ok / n_cw:.1%}** of those cases attention rollout was unfaithful while
Integrated Gradients on the same example was not.

**An independent check.** Error analysis established behaviourally — with no
attribution method involved — that the model leans on money-service brand names:
accuracy on money-transfer complaints is 0.955 when one is named and 0.118 when
only bank vocabulary appears. Asked to explain those predictions, Integrated
Gradients puts the brand token in its top-{TOP_K} {brand['integrated_gradients']['rate']:.0%} of the time,
against {brand['attention_rollout']['rate']:.0%} for attention rollout and {brand['shap']['rate']:.0%} for SHAP.

**What this does not show.** Disagreement is roughly flat across confident and
uncertain predictions ({min(strata_vals):.2f}–{max(strata_vals):.2f} across strata), so it is not
concentrated in hard cases — which is worse news than if it were, because it does
not resolve on the confident predictions a reviewer is most likely to trust. And
faithfulness is a claim about whether an explanation reflects the model's
computation, not about whether the model is right.

_Generated by `make audit`; numbers in results/metrics.json section `disagreement`._
"""
    FINDING_PATH.write_text(text)


if __name__ == "__main__":
    summarize()
