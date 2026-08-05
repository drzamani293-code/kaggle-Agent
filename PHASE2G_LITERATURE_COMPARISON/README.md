# PHASE2G_LITERATURE_COMPARISON

Comparison of our Rule 30 corpus against three primary sources:
**Rowland (2006)**, **Kopra (arXiv:2202.13809 / TCS 2022)**, and
**Jen (Physica D 45, 1990)**.

---

## Read `RETRIEVAL_LOG.md` first

**Task 1 of the brief — read Rowland and Kopra completely — was not completed.
None of the three papers was retrieved; not one page was read.** Every fetch
returns **HTTP 403 at the proxy's CONNECT stage**, for every host, including
Wikipedia. No copies were supplied on disk. Only `WebSearch` worked, returning
titles, URLs and model-generated summaries of snippets.

Every statement attributed to a paper in this package is therefore
**UNVERIFIED — SECONDARY** or **UNVERIFIED — FIGURE CAPTION**, never a reading.

## What was salvaged, and it matters

One attributed statement was numerical and testable: a 48-term sequence of the
eventual period lengths of rule 30's left diagonals.

```
ours      : 1 1 1 2 1 2 2 1 4 1 4 4 4 4 4 4 4 4 4 4 4 4 4 2 4 4 4 4 1 8 1 8 8 8 8 8 8 8 8 8 8 8 8 8 8 4 8 8
attributed: 1 1 1 2 1 2 2 1 4 1 4 4 4 4 4 4 4 4 4 4 4 4 4 2 4 4 4 4 1 8 1 8 8 8 8 8 8 8 8 8 8 8 8 8 8 4 8 8
```

**Exact match on all 48 terms**, by two independent period computations
(`verify_against_literature.py`). Consequences:

1. **Our edge-aligned coordinate `w_t(k)` is Rowland's `k`-th left diagonal.**
2. **The corpus's largest unverified assumption is retired.** Phase 2F flagged
   the rule-numbering convention as resting on a 14-bit prefix from a secondary
   source; a 48-term agreement with a published sequence is a far stronger
   anchor.
3. Several results must be **relabelled as known**.

Separately: our frontier gives an implied order/chaos boundary speed of
**0.25368** against the known **≈ 0.252** — a 0.67 % difference. Our
`T(K)/K ≈ 1.34` is `1/(1 − 0.252)`; it is a known constant, not our observation.

## The main correction

Rowland's **Proposition 2**, quoted second-hand — *each period doubling occurs
exactly when a diagonal becomes a white stripe and the diagonal to its left has
an odd number of black cells in each repeating block* — appears to be **exactly**
our **T-14 + the DOUBLING half of T-10**, including the direction of "to its
left".

So **T-07, T-08, T-10 (doubling half), T-11 and T-14 are relabelled KNOWN
(Rowland 2006)**, and Phase 2F's assessment *"T-13 – T-19 … POSSIBLY NEW as a
block"* is **withdrawn**.

**Brief task 4 answered:** yes — Rowland has `P(K) ∈ {P(K-1), 2P(K-1)}`, at
least implicitly and probably explicitly. Confidence high; verification pending.

## What survives as plausibly new

The **preperiod** material, because every attributed statement about Rowland
concerns *periods* only:

* **T-18** — the exact recurrence `D(K)=0 ⟹ T(K)=T(K-1)`, `D(K)=1 ⟹ T(K)=τ(K)+1`
* **T-19** — `T(K) = ρ(K)+1` at every resetting level
* **T-16** — the reset calculus and the constant `Φ_K`
* **T-21.5** — `D(K) = w_{T(K-1)}(K) XOR Φ_K`, one new transient bit per level

**T-18 is nominated as the strongest plausibly-new result** — with the explicit
caveat that a 2006 paper about exactly this structure is the most likely place
in the literature for such a recurrence to be hiding, and we did not read it.

**T-02 / T-03** (column periodicity) move to **UNCERTAIN**: Jen's proposition
and Kopra's Theorem 3.5 occupy that territory and neither could be read.

## What the comparison shows we are missing

**Rowland's `2^n` nested-restart mechanism** and the **conditional
time-reversibility** behind it (rule 30 is invertible backwards when the right
half of each row is white). We have no analogue; our whole corpus works on the
left because we had no handle on the right. This is the most useful thing this
phase found, and it is a gap in our work, not an overlap.

## Files

```
README.md                       this file
RETRIEVAL_LOG.md                what was attempted, what failed, source-status labels
ROWLAND_COMPARISON.md           correspondence table; the 48-term test; tasks 4 and 5
KOPRA_COMPARISON.md             Theorem 3.5 and the T-02/T-03 question (unresolved)
JEN_COMPARISON.md               the 1990 proposition and what must be checked
REVISED_NOVELTY_LEDGER.md       downgrades, survivors, and the strongest new candidate
REVISED_PAPER_OUTLINE.md        replaces PAPER_DRAFT_V1.md; new title, demoted sections
verify_against_literature.py    the 48-term check and the boundary-slope check
prefix_lab.py, rule30_lab.py    libraries (from Phase 2C/2E, unchanged)
results/literature_check.json   raw output
results/literature_check.log    stdout
```

Also corrected in place, per brief task 6:
`RULE30_VERIFIED_THEOREMS_V1/NOVELTY_STATUS.md` and
`RULE30_VERIFIED_THEOREMS_V1/PAPER_DRAFT_V1.md` now carry banners pointing here.

## Controls

* No paper was read; nothing is presented as a reading of one.
* Where evidence points toward a result being known, it is relabelled known.
  No discovery is claimed for any structure that may already exist.
* Every attributed statement carries a source-status label.
* The one verified thing here is **our own computation**, and it is labelled as
  such.
* Copyright: only short fragments necessary for identification are quoted
  (one clause of Rowland's Proposition 2 and two figure captions); everything
  else is paraphrase.
* Problem 1 is untouched, and no new computational search was run.
