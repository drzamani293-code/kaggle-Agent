# PHASE2A_SAT_MB1LOC

SAT/SMT formulations of the finite statement **MB1-loc(a)**, from the Rule 30
Prize Problem 1 project. Phase 2A builds and validates the formulations; it
does **not** attempt a proof, and none is claimed. Phase 2B is not started.

Prerequisite reading: the Phase 1 package (`RULE30_PHASE1_HANDOFF`), in
particular `WIDTH2_PROOF_RECONSTRUCTION.md` §11–12 where MB1 and MB1-loc are
defined.

---

## Start here

```bash
cd PHASE2A_SAT_MB1LOC
pip install -r requirements.txt          # numpy, z3-solver, python-sat
python3 run_phase2a_tests.py             # ~13 min, expect "37 passed, 0 FAILED"
python3 run_phase2a_tests.py --fast      # ~2 min, skips the deep witnesses
```

To regenerate everything from scratch (~40 min):

```bash
python3 run_phase2a_search.py            # writes results/phase2a_results.json
python3 build_witness_registry.py        # writes WITNESS_REGISTRY.json
```

Then read `PHASE2A_RESULTS.md`.

---

## The statement

> **MB1-loc(`a`)** — for all `t > a` and all `p ≥ 1`:
> if `col_0(s) = col_0(s+p)` for every `s ∈ [t-a, t]`, and `col_0(t) = 0`,
> then `col_1(t) = col_1(t+p)`.

A counterexample is a triple `(a, t, p)` satisfying the hypotheses but not the
conclusion, **in the actual rule-30 orbit of the single-cell seed**.

## The one thing to understand before reading any result

The faithful encoding of the real orbit (**MODEL B**) has **zero free
variables** — the seed determines every cell — so a SAT solver applied to it is
a *certificate checker*, not a search engine. Genuine search requires dropping
the seed (**MODEL A**), and a satisfying MODEL A patch is therefore **not** a
counterexample to MB1-loc.

| | free vars | SAT means | UNSAT means |
|---|---|---|---|
| **MODEL A** (seed-free) | base row | a patch of the *relaxation*; **not** a counterexample | **THEOREM**: MB1-loc holds for that `(a,p)` and every `t` |
| **MODEL B** (full light cone to time 0) | none | **CERTIFICATE**: a real counterexample | that triple is not a counterexample (complete for the triple) |
| **MODEL B′** (two windows on verified orbit rows) | none | same as B, relative to the base rows | same as B |

Every certificate file states its model on line 1. `WITNESS_REGISTRY.json`
keeps `model = "real_orbit"` and `model = "A_relaxation"` strictly apart.

## Results in one table

| Result | Label |
|---|---|
| MODEL A satisfiable on **all 832** instances (`a ≤ 48`, `p ≤ 64`, all `t`) — no UNSAT anywhere | COMPUTATIONAL CERTIFICATE |
| Hence **no seed-free local argument** at depth `≤ 48`, lag `≤ 64` can prove MB1-loc | THEOREM |
| All **8** Phase 1 witnesses re-certified (4 configurations + orbit re-simulation) | COMPUTATIONAL CERTIFICATE |
| The `a=0` witness re-certified by **full MODEL B**, base row recovered as the seed | COMPUTATIONAL CERTIFICATE |
| Exhaustive MODEL B sweep `t ≤ 48, p ≤ 24`: **122** counterexamples, matching simulation exactly | CERTIFICATE + BOUNDED UNSAT |
| Proposition 6 control **UNSAT** exactly as the hand proof predicts, 4-constraint core | THEOREM + encoding validation |
| No new witness for `a ≥ 32` — but only **1** eligible position exists at `a = 32` and **0** at `a ≥ 36` | BOUNDED UNSAT |

**Not claimed:** that MB1-loc(`a`) holds for any `a`. Absence of a witness
under finite bounds is reported as BOUNDED UNSAT with the bounds and the
eligibility counts, never as a universal statement.

## Files

```
README.md                      this file
PHASE2A_METHOD.md              design rationale, bounds, protocols, threats to validity
PHASE2A_RESULTS.md             all verdicts, with labels
SAT_MODEL_SPECIFICATION.md     formal model definitions + soundness proofs
WITNESS_REGISTRY.json          every witness, real and relaxation, kept apart
requirements.txt               numpy, z3-solver, python-sat

mb1loc_sat.py                  models, encodings, solvers, checkers, enumeration
run_phase2a_search.py          the systematic search (8 sections)
build_witness_registry.py      assembles WITNESS_REGISTRY.json
run_phase2a_tests.py           the test battery
rule30_lab.py                  Phase 1 library (verified rule-30 engines)
mb1loc_witnesses.json          Phase 1 witness data, input to certification

results/
  phase2a_results.json         raw output of the search
  phase2a_run.log              stdout of the search
  model_a_patches/             26 MODEL A certificates (NOT counterexamples)
  model_bt_certificates/       8 MODEL B' witness certificates
  unsat_cores/                 Proposition 6 control cores
  modelB_witness_a00_t17_p1.txt  the full-light-cone certificate
```

## Independence

Every verdict is produced more than once:

* **encodings** — ENC1 (8 direct clauses/transition), ENC2 (Tseitin with
  auxiliaries), Z3-native (no shared CNF code);
* **solvers** — CaDiCaL 1.5.3, Glucose 4.2, MiniSat 2.2, Z3 5.0.0;
* **a fourth, solver-free procedure** — exhaustive enumeration of all `2^w`
  base rows, evolved with the Phase 1 rule-30 table;
* **orbit re-simulation** — every claimed real counterexample is re-checked by
  simulating the single-cell orbit from time 0.

Across 832 + 3168 + 18 + 12 instances there were **zero disagreements**.

## Known limitations, stated up front

* **No machine-checked UNSAT proof.** DRUP export is wired up, but the control
  instances are refuted by unit propagation so the emitted proof is empty, and
  no DRAT checker is installed in this environment
  (`PHASE2A_RESULTS.md` §5.1).
* **MODEL B′ assumes the simulator** for its base rows. MODEL B assumes only
  the seed and is run wherever affordable — which is only the `a=0` witness at
  full size.
* **`p ≤ 64`** in the MODEL A sweep; nothing is claimed beyond.
* **The `a ≥ 32` search is data-limited**, not method-limited
  (`PHASE2A_RESULTS.md` §8).
* One reporting defect was found by the test battery and fixed; it is recorded
  rather than silently corrected (`PHASE2A_RESULTS.md` §5.2).
