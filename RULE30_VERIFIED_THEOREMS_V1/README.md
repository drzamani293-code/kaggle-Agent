# RULE30_VERIFIED_THEOREMS_V1

Consolidation of six phases of work on Wolfram's Rule 30 Prize Problem 1:
every genuinely proved theorem extracted, formalised, audited and catalogued,
with every correction recorded and every unproved claim demoted to its correct
category.

**This package does not solve Problem 1 and makes no partial claim on it.**
No new computational search was run; no numerical range was extended.

---

## Start here

```bash
cd RULE30_VERIFIED_THEOREMS_V1
python3 run_verified_theorems_tests.py            # expect "71 passed, 0 FAILED", ~40 s
python3 run_verified_theorems_tests.py --fast     # ~1 s (skips the factor recount)
python3 run_verified_theorems_tests.py --no-artifacts   # if extracted alone
```

Read in this order:
`DEFINITIONS_AND_NOTATION.md` → `VERIFIED_THEOREMS.md` →
`PROOF_AUDIT_REPORT.md` → `MASTER_CLAIM_LEDGER.md` →
`THEOREM_DEPENDENCY_GRAPH.md`.

## What is in the corpus

| category | count | where |
|---|---|---|
| **A** formally proved theorems | 23 (≈35 sub-results) | `VERIFIED_THEOREMS.md` |
| **B** conditional theorems | 6 | ledger §B |
| **C** finite certificates | 12 | `COMPUTATIONAL_CERTIFICATES.md` |
| **D** bounded observations | 12 | ledger §D |
| **E** conjectures | 7 | ledger §E |
| **F** refuted claims | 7 | ledger §F |
| **G** vacuous / equivalent-to-target | 5 | ledger §G |
| **H** open bridge | 1 | ledger §H |

**No claim is category A because it passed tests.** Category A requires a
complete proof, written without reference to code, that was reconstructed
independently in the audit.

## The three headline theorems

* **T-03.** For every finitely supported seed, **at most one** column of the
  space-time diagram is eventually periodic. (Strengthens the adjacent-columns
  result T-02.)
* **T-08 – T-11.** In light-cone coordinates the rule is one-sided, so every
  prefix is autonomous and eventually periodic; the tower is a skew product
  with one-bit fibres and a period-doubling hierarchy, `P(K) ∈ {P(K-1),2P(K-1)}`.
* **T-18.** An **exact** recurrence for the preperiod, driven by one bit:
  `D(K)=0 ⟹ T(K)=T(K-1)`; `D(K)=1 ⟹ T(K)=τ(K)+1`.

## The one thing that is missing

> **BR-01 (the Bridge, OPEN).** If the centre column is eventually periodic
> then some **second** column is eventually periodic.

With T-03 this settles Problem 1 immediately. Five routes across Phases 1–2E
all reduce back to it; two apparent routes turned out to be degenerate — one is
"(H) ⟹ (H)", the other is Problem 1 restated. `THEOREM_DEPENDENCY_GRAPH.md` §4.

## What the audit changed

Four corrections were found in **shipped, tested packages**:

| | |
|---|---|
| **C-08** | EA2's hypothesis "`t ≥ K/2`" is spurious — prefixes are autonomous from `t = 0` |
| **C-09** | R3's `max` is redundant: since `τ(K) ≥ T(K-1)`, the bound is just `T(K) ≤ τ(K)+1` |
| **C-12** | "no finite-state description exists" restricted to the class actually ruled out |
| **C-15** | F5 is not "refuted"; its conclusion is impossible outright, so it **is** Problem 1 restated |

Plus **C-16**: the audit made an indexing error on T-15 (level vs coordinate),
caught it, and recorded it. Eighteen corrections in total, with original and
corrected wording side by side, in `CORRECTIONS_AND_RETRACTIONS.md`.

## Files

```
README.md                          this file
SOURCE_MANIFEST.md                 every imported result, origin, and final status
MASTER_CLAIM_LEDGER.md             every claim in exactly one of categories A-H
DEFINITIONS_AND_NOTATION.md        one notation; conversion table; index conventions
VERIFIED_THEOREMS.md               T1-T23 with full proofs, no code references
PROOF_AUDIT_REPORT.md              independent reconstruction; findings A-01..A-10
THEOREM_DEPENDENCY_GRAPH.md        mermaid + text tree + assumptions table
CORRECTIONS_AND_RETRACTIONS.md     C-01..C-18, original vs corrected wording
COMPUTATIONAL_CERTIFICATES.md      programs, ranges, solvers, artifacts, hashes
NOVELTY_STATUS.md                  novelty UNRESOLVED; search terms; who to ask
PAPER_DRAFT_V1.md                  paper draft (claims no solution to Problem 1)
EXPERT_REVIEW_REQUEST.md           concise request for a CA researcher
run_verified_theorems_tests.py     71 checks incl. banned-overclaim scan
FINAL_VALIDATION_REPORT.md         results of the clean-extraction run
results/final_validation.log       stdout of that run
```

## Controls

* Theorem status is never conferred by testing.
* Every all-`K` theorem is separated from every bounded observation, and every
  bounded observation names its range.
* Failed conjectures, refuted routes and superseded methods are preserved.
* "No witness found" is never described as UNSAT; the suite scans for it.
* No inference is drawn from growing complexity.
* Novelty is **unresolved** — no primary source was reachable (HTTP 403), and
  nothing here is claimed to be new or claimed to be known.
