"""
Example walkthroughs for the report: two individual predictions, shown with all
three methods' token attributions side by side.

The aggregate charts show *that* the methods disagree. These show *how* -- which
is what makes the finding legible to someone who will not read a faithfulness
table. Both examples are confidently-wrong predictions where Integrated
Gradients scored faithful and attention rollout did not, so they illustrate the
headline rather than decorating it.

Selected by criteria, not by eye: see WALKTHROUGH_IDS.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.audit import merge_wordpieces  # noqa: E402
from src.explain.common import load_attributions  # noqa: E402
from src.explain.example_set import build_example_set  # noqa: E402
from src.faithfulness import METHODS  # noqa: E402
from src.utils import FIGURES_DIR  # noqa: E402

# Both are confident_wrong with IG comprehensiveness >= 0.30 and attention < 0.05.
#   15634858 -- a mortgage complaint predicted vehicle_loan. IG finds "car";
#               attention highlights mortgage vocabulary and a full stop, i.e. it
#               tells a story that contradicts the model's own output.
#   16282350 -- a short vehicle-loan complaint predicted student_loan. Two of
#               attention's top three tokens are punctuation.
WALKTHROUGH_IDS = [15634858, 16282350]
N_SHOW = 14
SHORT = {
    "integrated_gradients": "Integrated Gradients",
    "shap": "SHAP",
    "attention_rollout": "attention rollout",
}


def _plot_example(axes, cid, examples, word_attr):
    row = examples[examples["complaint_id"] == cid].iloc[0]
    for ax, method in zip(axes, METHODS):
        attr = word_attr[method][cid]
        items = sorted(attr.items(), key=lambda kv: -kv[1])[:N_SHOW][::-1]
        words = [w for w, _ in items]
        vals = np.array([v for _, v in items])
        norm = vals / (np.abs(vals).max() or 1)
        colors = [
            "#C44E52" if w in {".", ",", ";", ":", "!", "?", "-", "'"} or len(w) <= 2 else "#4C78A8"
            for w in words
        ]
        ax.barh(range(len(words)), norm, color=colors)
        ax.set_yticks(range(len(words)), words, fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([])
        ax.set_title(SHORT[method], fontsize=10, fontweight="bold")
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
    return row


def main():
    examples = build_example_set()
    word_attr = {
        m: {
            e["complaint_id"]: merge_wordpieces(e["tokens"], e["attributions"])
            for e in load_attributions(m)["examples"]
        }
        for m in METHODS
    }

    fig, axes = plt.subplots(
        len(WALKTHROUGH_IDS), len(METHODS), figsize=(13, 4.4 * len(WALKTHROUGH_IDS))
    )
    for r, cid in enumerate(WALKTHROUGH_IDS):
        row = _plot_example(axes[r], cid, examples, word_attr)
        axes[r][0].set_ylabel(
            f"true: {row.true_label}\npredicted: {row.pred_label} (p={row.confidence:.2f})",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        "Same prediction, same model, three explanations\n"
        "Top-14 words by attribution. Red = punctuation or a function word, which "
        "cannot be evidence for a product category.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "example_walkthroughs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    for cid in WALKTHROUGH_IDS:
        row = examples[examples["complaint_id"] == cid].iloc[0]
        print(f"\n--- {cid}: true={row.true_label} pred={row.pred_label} p={row.confidence:.3f}")
        print("   ", row.narrative[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
