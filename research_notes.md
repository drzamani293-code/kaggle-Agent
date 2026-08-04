# Research Notes — Rule 30 Prize Problem 1

Running log. Newest entries at the bottom. This file did not exist before
2026-08-04; see `PHASE1_AUDIT.md` §0.

---

## 2026-08-04 — Session 1 (Phase 1)

### Starting state

The brief referred to `README.md`, `research_notes.md`, `rule30_lab.py` and
`rule30_experiments.ipynb`. **Only `README.md` exists**, and it documents an
unrelated Kaggle agent. A search over every commit on every branch found zero
Rule 30 / cellular-automaton files. There was no prior work to audit, compare
against, or preserve. Everything below was built from scratch; no pre-existing
file was modified.

### What was built

* `rule30_lab.py` — four independent implementations (list+table, numpy+table,
  big-integer bit-parallel, rule-86 mirror), a 21-check verification suite,
  all Phase 1 statistics, and the bridging-lemma probes.
* `run_phase1.py`, `run_anomaly_followup.py`, `run_bridge_probe.py`,
  `find_mb1loc_witnesses.py` — experiment drivers writing to `phase1_results/`.
* `PHASE1_AUDIT.md`, `WIDTH2_PROOF_RECONSTRUCTION.md`,
  `CONJECTURE_REGISTRY.md`.

### Verification

21/21 checks pass. The load-bearing ones: exhaustive check of the rule table
against the rule *number*; four hand-computed rows; three implementations
agreeing on complete rows; the rule-86 mirror check (the only test that would
catch a left/right orientation error); and mechanical verification of all three
lemmas used in the width-2 proof.

External corroboration is weak: `oeis.org` is blocked by the proxy (403), so
A051023 could only be checked against a web-search snippet — our first 14 bits
match it, and match the hand computation.

### Main results

* Rigorous, from finite computation: if the center column is eventually
  periodic with preperiod `T` and period `p`, then `T + p ≥ 998140`
  (factor-counting), and any `p ≤ 20000` forces `T > 979998`.
* Statistics: nothing. Frequency `0.5008` at `N = 10^6`, linear complexity
  `≈ N/2`, factor complexity saturating the sample bound, flat autocorrelation.
  Both candidate anomalies died: a "missing" length-11 factor was a sampling
  artefact, and a `+3.3σ` mod-3 bias failed to replicate on a disjoint sample
  (`+1.4σ` on 4× the data).
* Powers of two: no signal (smallest binomial `p` = 0.238 across three
  families of 18 points).

### Theory

Reconstructed the width-2 argument from first principles, importing nothing
from the literature, and then **strengthened it**: the strip between two
columns is a finite-state machine driven by its two boundary columns, so
eventual periodicity of any two columns propagates inward. Hence

> **at most one column of the whole space-time diagram is eventually periodic.**

The gap to Problem 1 is therefore exactly: rule out the single exceptional
column. Proposition 6 makes this concrete — if `col_0` is `p`-periodic then
`col_{-1}` is `p`-periodic **iff** `col_1` agrees at lag `p` on the zero set of
`col_0`. That is the nominated smallest missing lemma (MB1, Registry C3).

### The one genuinely new experimental finding

MB1 itself is untestable (vacuous if Problem 1 has the expected answer), so we
tested its finitary shadow MB1-loc(`a`): "`col_0` agreeing at lag `p` over
`[t-a, t]` forces `col_1` to agree at `t`". **Refuted with explicit,
re-verified witnesses for `a = 0, 4, 8, 12, 16, 20, 24, 28.**

The associated conditional probability depends only on the *past* of `col_0`
(0.50 → 0.76 as `a` goes 0 → 16) and not at all on its future (flat 0.500) —
which is exactly what the light cone predicts, since `a_t(1)` depends on
`a_{t-1}(0), a_{t-2}(0), …` and on no cell later than `t`.

### Open, and the thing to do next

The transfer probability keeps climbing at the deepest look-backs
(0.7627 / 0.7829 / 0.8082 / 0.9583 at `a = 16/20/24/28`) but on rapidly
shrinking samples (7.6e4 / 4.7e3 / 292 / 24). Whether it tends to a limit
below 1 or to 1 is **undetermined**, and the two answers imply opposite
research programmes (Registry C5):

* limit `< 1` → MB1 needs a global proof; local-window strategies are dead.
* limit `= 1` → MB1-loc might hold at some finite depth, which would be a
  direct route to Problem 1.

Deciding this needs `N ≈ 10^7` bits and lags to `10^6`, which needs a better
generator than the current `O(N^2/64)` bit-parallel one (light-cone-restricted
evolution).

### Mistakes made and corrected during the session

* Two bridging probes reported 0.654 and 0.500 for what looked like the same
  quantity. Neither was buggy — they conditioned on two-sided vs forward-only
  windows. Resolved only by writing a third, deliberately naive list-based
  recomputation. Recorded in `PHASE1_AUDIT.md` §6.3.
* A first draft of `WIDTH2_PROOF_RECONSTRUCTION.md` §12 asserted the transfer
  probability "saturates around 0.755, not at 1". The larger run showed it
  still rising. Corrected; the claim is now explicitly listed as undetermined.
* The first left-region symmetry search only scanned `τ ≤ 4` and reported a
  bounded band; the true minimal diagonal period is 16.
