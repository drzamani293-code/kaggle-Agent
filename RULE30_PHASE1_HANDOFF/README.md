# RULE30_PHASE1_HANDOFF

Phase 1 audit package for **Wolfram Rule 30 Prize Problem 1**: is the centre
column of rule 30, started from a single black cell, eventually periodic?

**The problem is open. Nothing in this package resolves it, and nothing here
should be read as progress towards a computational resolution — no finite
computation can prove non-periodicity.** What the package contains is a
verified implementation, a self-contained proof of the strongest *known*
partial result, an explicit statement of the smallest missing lemma, and the
finite computations that constrain (but do not exclude) a hypothetical period.

Source repository: `drzamani293-code/kaggle-agent`, branch
`claude/rule30-prize-problem-1-qr1omb`.
Phase 1 commit: **`b42ce46d480d750c4e779f48f415dde6f9b292c9`**.

---

## Start here

```bash
cd RULE30_PHASE1_HANDOFF
pip install -r requirements.txt      # numpy only
python3 run_all_tests.py             # ~5 min, expect "51 passed, 0 FAILED"
```

Then read, in this order:

1. **`CLAIM_LEDGER.md`** — every substantive claim, its type (theorem /
   computation / observation / conjecture), where it is supported, what it
   assumes, how it was independently verified, and its known weaknesses.
   Section H lists claims deliberately **not** made, and the two that were
   **retracted**.
2. **`PHASE1_AUDIT.md`** — the experimental audit: settings, commands,
   results, anomalies, and §9, the list of conclusions that cannot be drawn.
3. **`FORMAL_PROOF_AT_MOST_ONE_COLUMN.md`** — fully quantified proof of the
   main theorem, including an explicit accounting of what is *not* used
   (no compactness, no reversibility, no bi-infinite time).
4. **`FACTOR_COMPLEXITY_BOUND.md`** — the lemma, proof, parameters, checksums
   and three independent recomputations behind `T + p ≥ 998140`.
5. **`CONJECTURE_REGISTRY.md`** — formal statements, evidence, possible
   counterexamples, falsification tests, and what each would yield if proved.
6. **`PROOF_DEPENDENCY_GRAPH.md`** — the dependency graphs.
7. **`REPRODUCE_ALL.md`** — exact commands from a clean machine.
8. **`MANIFEST.md`** — complete file list with sizes and SHA-256.

---

## The three results worth checking hardest

| Result | Type | Where |
|---|---|---|
| **At most one column of the space-time diagram is eventually periodic** (Theorem W2′) | proved theorem | `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md` |
| If the centre column is eventually periodic, then `T + p ≥ 998140`, and any `p ≤ 20000` forces `T > 979998` | finite computation + proved lemma | `FACTOR_COMPLEXITY_BOUND.md`, `PHASE1_AUDIT.md` §5.7–5.8 |
| **MB1-loc(`a`) is false** for `a = 0, 4, 8, 12, 16, 20, 24, 28` — the local form of the bridging lemma, refuted by explicit witnesses | refutation | `PHASE1_AUDIT.md` §8 |

The gap between Theorem W2′ and Problem 1 is a single statement: *if the centre
column is eventually periodic, some other column is too.* The smallest explicit
sufficient version is **MB1** (`CONJECTURE_REGISTRY.md` C3). MB1 is
**vacuously true if Problem 1 has the expected answer**, so it can never be
tested computationally — only proved. That logical point is the one most likely
to be lost in summary.

---

## Provenance warning

The brief that started this work referred to pre-existing files
(`research_notes.md`, `rule30_lab.py`, `rule30_experiments.ipynb`). **None of
them existed** — the host repository is an unrelated Kaggle agent, and a search
over every commit on every branch found zero cellular-automaton files.
Everything here was written from scratch in one session, and no pre-existing
repository file was modified. Details and the exact search commands are in
`PHASE1_AUDIT.md` §0.

Consequence for the audit: correctness could not be established by comparison
with prior work. It rests instead on four independent implementations, an
exhaustive check of the local rule against Wolfram's rule *number*, four
hand-computed rows, a mirror-automaton check (rule 86) that catches
orientation errors, and a single weak external anchor — 14 bits matched against
a web-search report of OEIS A051023, because `oeis.org` is blocked by the
network proxy in this environment. That external anchor is the weakest link in
the package and is flagged as such in the ledger (B6).

---

## Layout

```
RULE30_PHASE1_HANDOFF/
├── README.md                            this file
├── MANIFEST.md                          complete file list + SHA-256
├── REPRODUCE_ALL.md                     exact commands from a clean machine
├── CLAIM_LEDGER.md                      one row per claim
├── PROOF_DEPENDENCY_GRAPH.md            logical dependencies
├── PHASE1_AUDIT.md                      the experimental audit
├── WIDTH2_PROOF_RECONSTRUCTION.md       expository proof + gap analysis
├── FORMAL_PROOF_AT_MOST_ONE_COLUMN.md   fully quantified proof of W2′
├── FACTOR_COMPLEXITY_BOUND.md           the T + p ≥ 998140 bound in full
├── CONJECTURE_REGISTRY.md               C1–C9, with falsification tests
├── research_notes.md                    session log, incl. mistakes made
├── requirements.txt                     numpy
│
├── rule30_lab.py                        engines, verification suite, statistics
├── gen_center_column.py                 sequence generation + SHA-256
├── verify_sequence_independent.py       mirror-engine + numpy regeneration
├── verify_factor_bound.py               three independent factor counts
├── run_phase1.py                        main experiment driver
├── run_anomaly_followup.py              10⁶-bit independent-sample re-tests
├── run_bridge_probe.py                  transfer-profile experiments
├── find_mb1loc_witnesses.py             MB1-loc counterexample search
├── run_all_tests.py                     the whole battery, clean process
│
└── phase1_results/                      all artifacts (JSON, TXT, logs)
```

---

## Failed and negative results are included, not omitted

* Both candidate anomalies died on replication and are reported in full
  (`PHASE1_AUDIT.md` §6): a "missing" length-11 factor that was a sampling
  artefact, and a `+3.311σ` mod-3 bias that fell to `+1.408σ` on a disjoint
  sample.
* The powers-of-two experiment found nothing (§5.6, smallest binomial
  `p = 0.238`) and is reported as nothing.
* Two claims made in earlier drafts were **retracted** and are recorded as
  such (`CLAIM_LEDGER.md` H5, H9), together with the measurement that
  contradicted each.
* An analysis episode where two probes appeared to disagree — and turned out
  to be answering different questions — is recorded in §6.3 rather than
  quietly resolved.
* `phase1_results/` retains every log, including the ones from runs that were
  superseded.
