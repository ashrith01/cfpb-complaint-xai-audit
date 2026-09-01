# Day 5 — Error analysis & failure taxonomy

DistilBERT, held-out test set: **5,400 examples, 812 misclassified (15.0%)**.

Reproduce with `make evaluate`. Raw examples: `results/error_examples.json`.
Confusion matrix: `results/figures/confusion_matrix.png`.

| | mean confidence |
|---|---|
| correct predictions | 0.9221 |
| incorrect predictions | 0.7162 |

The model is meaningfully less confident when wrong — but **212 errors (26.1% of
all errors) still come in above p>0.9**. That confidently-wrong population is the
target for the Day 9–10 audit: those are the predictions where a plausible-looking
explanation would be most likely to be believed and most likely to mislead.

## Top confused pairs

| true | predicted | count | % of true class |
|---|---|---|---|
| money_transfer | checking_savings | 106 | 15.7% |
| credit_reporting | debt_collection | 68 | 10.1% |
| checking_savings | money_transfer | 61 | 9.0% |
| debt_collection | credit_reporting | 60 | 8.9% |

The four largest confusions are **two symmetric pairs**, not four independent
failures. Symmetry is diagnostic: a one-way bias would suggest the model favours
one class, whereas confusion flowing equally in both directions suggests the
boundary itself is ill-defined.

---

## The headline: the model has learned a lexical shortcut

Testing whether accuracy on the money_transfer / checking_savings pair depends on
one surface cue — whether the narrative names a money service explicitly
(`zelle|venmo|paypal|cash app|coinbase|western union|moneygram|remitly|wire
transfer|crypto|bitcoin`):

| true class | cue | n | accuracy |
|---|---|---|---|
| money_transfer | term present | 448 | **0.955** |
| money_transfer | term absent | 227 | **0.533** |
| money_transfer | bank terms only, no service term | 34 | **0.118** |
| checking_savings | term absent | 632 | **0.878** |
| checking_savings | term present | 43 | **0.442** |

**The model appears to be detecting brand names rather than reasoning about the
complaint.** Where a money-service is named, it is 95.5% accurate. Where it isn't,
accuracy falls to 53.3% — and on the 34 money-transfer complaints that use *only*
bank vocabulary, it collapses to **11.8%**, far below the 12.5% random-guess rate
for 8 balanced classes. Symmetrically, checking/savings complaints that happen to
mention a money service get misclassified 55.8% of the time.

This is a shortcut that works, because the correlation is real and strong. It is
also exactly the kind of decision rule a compliance reviewer would want to know
about, and it gives Days 6–10 a **concrete, pre-registered hypothesis**:

> If the attribution methods are faithful, they should place their mass on the
> brand token (`zelle`, `venmo`, …) in precisely these cases. A method that
> instead highlights the surrounding complaint language is describing reasoning
> the model is not doing.

That is a far sharper test than "do the highlights look sensible to a human",
because we now have an independent, behaviourally-established claim about what
the model is keying on.

---

## Failure taxonomy

Based on a hand review of all 40 sampled errors (10 per top pair, highest
confidence first). Proportions are approximate — a 40-example read, not a
measurement.

Institutions named in individual complaints are described by type below
("a credit bureau", "a peer-to-peer payment service") rather than by name. CFPB
complaints are unverified consumer allegations, and nothing here is a claim about
any company; the classification analysis does not depend on which one it was. The
money-service *term list* in the shortcut test above is unchanged — it defines the
experiment rather than describing any one complaint.

### 1. Mislabeled ground truth — the label contradicts the narrative (~40%)

The CFPB product field is **chosen by the consumer when filing**, not derived
from the text. In a substantial share of cases the model's prediction fits the
narrative better than the gold label does.

- *true `checking_savings`, predicted `money_transfer`, p=0.990* — the narrative
  is entirely about a peer-to-peer payment service's identity-verification
  design and fraudsters linking victim tokens. Nothing about a bank account.
- *true `checking_savings`, predicted `money_transfer`, p=0.948* — a
  cryptocurrency-exchange account takeover. Cryptocurrency is explicitly inside
  the money_transfer class definition.
- *true `debt_collection`, predicted `credit_reporting`, p=0.953* — a formal FCRA
  §605 dispute letter addressed to a credit bureau's dispute department.

Consequence for the audit: **an unfaithful-looking explanation of a "wrong"
prediction may be a faithful explanation of a right one.** Days 9–10 must not
treat model error and explanation error as the same thing, or label noise will be
scored as unfaithfulness.

### 2. Genuinely dual-nature events (~30%)

One event that legitimately belongs to two products. No single label is correct.

- A collections account appearing on a credit report is one event filed under
  either `debt_collection` or `credit_reporting`; several narratives cite FDCPA
  **and** FCRA in the same paragraph.
- A fraudulent wire sent via ACH through a retail bank is simultaneously a
  transfer complaint and a bank-account complaint.

This is an irreducible ceiling. It is the most likely explanation for why the
Day 3–4 fine-tune gained ≤0.003 F1 on exactly these classes — the remaining
headroom isn't modelling capacity, it's that the question has two right answers.

### 3. The discriminating evidence isn't in the input (~20%)

Two distinct mechanisms, same consequence:

**Redaction removed the deciding token.** CFPB scrubs brand names inconsistently.
Several money_transfer narratives read as pure checking-account complaints
because the service name was redacted out — *"their decision to block my access
to ▮ … I had previously been using ▮ through my ▮ savings account"* is almost
certainly a peer-to-peer payment service, and with that word removed the
remaining text supports the model's `checking_savings` prediction.

**The label depends on institution type, not content.** Several consumer fintech
apps are registered as money-services businesses rather than banks, so complaints
about them are filed under money_transfer even when the narrative describes an
ordinary checking-account problem (a refused mobile deposit, a delayed direct
deposit). That fact lives in the company's registration, not in the text.

Consequence for the audit: **no attribution method can be faithful about evidence
that is absent.** These cases put a hard floor under achievable faithfulness, and
should be reported as such rather than counted against any particular method.

### 4. Genuinely hard — no product signal at all (~10%)

Generic legal boilerplate that could attach to any product: statute citations,
identity-theft assertions, records requests, demand letters. One error is a
verbatim FCRA template with the account details left as unfilled placeholders
(*"[ List specific accounts … ]"*). There is nothing in the text to classify.

---

## What this changes downstream

1. **The example set for Days 6–8 should be stratified, not random.** It needs to
   include confidently-wrong predictions (n=212) and both directions of the
   symmetric pairs — otherwise the audit measures explanation quality only on
   easy cases where every method agrees and the comparison is uninformative.

2. **Days 9–10 must separate label error from explanation error.** Categories 1
   and 3 above mean a meaningful share of "wrong" predictions are wrong for
   reasons no explanation method could repair. Faithfulness measures whether the
   explanation reflects *the model's* decision process — which is a question that
   remains well-posed even when the model's decision is wrong, and the audit
   should be framed that way.

3. **There is now a testable prior about model behaviour.** The lexical-shortcut
   result is an independent, behavioural claim about what the model keys on,
   established without any attribution method. That makes it a rare thing in
   explainability work: a partial ground truth to check the explainers against,
   rather than a plausibility judgement.
