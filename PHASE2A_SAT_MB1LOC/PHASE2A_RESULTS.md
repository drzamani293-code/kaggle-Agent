# Phase 2A — Results

Verdicts only. Definitions, soundness proofs and the meaning of each label are
in `SAT_MODEL_SPECIFICATION.md`; the design rationale is in
`PHASE2A_METHOD.md`. Raw output: `results/phase2a_results.json`,
`results/phase2a_run.log`. Total run time 2353 s.

Environment: Python 3.11.15, PySAT (CaDiCaL 1.5.3, Glucose 4.2, MiniSat 2.2),
Z3 5.0.0, numpy 2.4.6.

---

## 0. Headline

| # | Result | Label |
|---|---|---|
| 1 | Model A is **satisfiable for all 832 instances** — `a ∈ {0,4,…,48}`, `p ∈ [1,64]`, and (by Lemma T) **every** `t`. No UNSAT anywhere. | COMPUTATIONAL CERTIFICATE (832 patches) |
| 2 | Therefore **no seed-free local argument** at look-back depth `a ≤ 48` and lag `p ≤ 64` can establish MB1-loc(`a`). | THEOREM (from Theorem A-sound, contrapositive) |
| 3 | All eight Phase 1 witnesses **re-certified** by Model B′ under four solver/encoding configurations plus independent orbit re-simulation. | COMPUTATIONAL CERTIFICATE |
| 4 | The `a=0` witness `(t,p) = (17,1)` re-certified by **full Model B** — complete backward light cone to time 0, base row pinned to the seed and recovered as the seed. | COMPUTATIONAL CERTIFICATE |
| 5 | Model B sweep over `t ≤ 48`, `p ≤ 24`, `a ∈ {0,4,8}` (3168 instances): **122 counterexamples**, exactly matching direct simulation. | COMPUTATIONAL CERTIFICATE + BOUNDED UNSAT elsewhere |
| 6 | The Proposition 6 control is **UNSAT** exactly when the hand proof predicts, with a 4-constraint core, under every configuration. | THEOREM (hand-proved) + certificate that the encoding detects forcing |
| 7 | No new witness for `a ≥ 32`. The census shows this is **data-limited, not method-limited**: at `N = 200001, p ≤ 20000` only **1** position even satisfies the hypotheses at `a = 32`, and **0** at `a ≥ 36`. | BOUNDED UNSAT |

**Not claimed anywhere:** that MB1-loc(`a`) is true for any `a`. See §9.

---

## 1. MODEL A sweep — 832 instances, all SAT

`a ∈ {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48}`, `p ∈ [1, 64]`.
By **Lemma T** (`SAT_MODEL_SPECIFICATION.md` §3) satisfiability does not depend
on `t`, so each solve settles all `t` at once — the `t` search is *complete*,
not bounded.

```
  a= 0  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a= 4  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a= 8  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a=12  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a=16  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a=20  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a=24  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a=28  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a=32  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a=36  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a=40  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a=44  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
  a=48  p=1..64  SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS

  instances: 832   SAT: 832   UNSAT: 0   cross-configuration disagreements: 0
```

Each instance was decided by **six PySAT configurations** (ENC1 and ENC2 × three
solvers) and, for `p ≤ 12`, additionally by **Z3-native**. All agreed on every
instance; `disagreements` in the results JSON is empty.

### 1.1 What this does and does not establish

**LABEL: THEOREM** (via the contrapositive of Theorem A-sound). For every
`(a,p)` in the grid, there exists a rule-30 space-time patch satisfying H1 ∧ H2
∧ ¬C. Hence **no argument that uses only the local rule and a window of depth
`a ≤ 48` — and no property of the seed — can prove MB1-loc(`a`) for those `p`.**
Any correct proof must invoke the single-cell initial condition.

**LABEL: NOT A COUNTEREXAMPLE.** A Model A patch is *not* a counterexample to
MB1-loc. Its base row is free and need not occur in the real orbit
(Theorem A-incomplete). Every Model A certificate file carries this banner on
line 3. They are registered in `WITNESS_REGISTRY.json` under
`model = "A_relaxation"`, kept strictly apart from the real ones.

**CONJECTURE (unproved).** Model A is satisfiable for every `a ≥ 0` and every
`p ≥ 1`. Evidence: 832/832. Falsification test: extend the grid; a single
UNSAT would refute it *and* would be a genuine theorem about MB1-loc.

### 1.2 Structure of the Model A patches

The `p = 1` solutions are strikingly simple: the solver returns a base row that
is all zeros except a single 1 at the right edge, producing one diagonal of 1s
travelling left at speed 1, e.g. for `a = 4, p = 1`:

```
          sites -5 .. 6
               01
t=      0 ...........#   H1
t=      1  .........#    H1,H1'
t=      2   .......#     H1,H1'
t=      3    .....#      H1,H1'
t=      4     ...#       H1,H2/C,H1'
t=      5      .#        H1',C'
```

Column 0 is identically 0 through the window (so H1 and H2 hold trivially),
and the diagonal reaches column 1 exactly between times `t` and `t+p`, breaking
C. This is a hand-checkable explanation of why Model A is easy, and it is
exactly the configuration the `p = 1` case analysis in
`SAT_MODEL_SPECIFICATION.md` §7 predicts. 26 such certificates
(`a ∈ {0,…,48} × p ∈ {1,7}`) are in `results/model_a_patches/`, each
re-simulated independently and re-checked.

---

## 2. Solver-free cross-check

18 instances with base row ≤ 18 bits were decided **twice**: once by SAT, once
by exhaustive enumeration of all `2^w` base rows evolved with the Phase 1
rule-30 table (no solver involved). **All 18 agree.**

This is the check that would catch an encoding that is satisfiable for the
wrong reason.

---

## 3. MODEL B — the real orbit, full backward light cone to time 0

### 3.1 Certification of the `a = 0` witness

`(a, t, p) = (0, 17, 1)`; cone = 380 cells, CNF = 401 variables / 3105 clauses;
base row = time 0 over sites `[-18, 19]`, pinned to the single-cell seed.

| check | result |
|---|---|
| CaDiCaL / Glucose / MiniSat / Z3 | **SAT** (all four) |
| recovered base row equals the seed (`x[0,0]=1`, all else 0) | **yes** |
| H1, H2, ¬C re-checked on the independently evolved patch | **yes** |
| independent orbit simulation from the seed | **counterexample confirmed** |

**LABEL: COMPUTATIONAL CERTIFICATE.** This is the one witness certified with
*no* assumption beyond the seed itself — the full light cone is encoded.
Certificate: `results/modelB_witness_a00_t17_p1.txt`.

### 3.2 Exhaustive Model B sweep in the small-`t` corner

Bounds: `a ∈ {0,4,8}`, `t ≤ 48`, `p ≤ 24` — **3168 instances**, 80.4 s.

* **122 instances SAT** — i.e. 122 genuine counterexamples to MB1-loc.
* Direct simulation independently finds **exactly the same 122**.
* The two searches agree: **True**.

The 3046 UNSAT instances are **BOUNDED UNSAT** only in the sense that the
*sweep range* is bounded; for each individual triple the verdict is *complete*
(Theorem B-exact): that triple is not a counterexample. All 121 not already in
the Phase 1 set are recorded in `WITNESS_REGISTRY.json` under
`model_b_sweep_counterexamples`, each re-verified by orbit simulation.

Earliest counterexamples found: `(a=0, t=2, p=8)`, `(a=0, t=2, p=9)`,
`(a=0, t=2, p=10)`, `(a=0, t=6, p=4)`, … — note these precede the `t = 17`
witness quoted in Phase 1, which was the smallest for `p = 1` specifically.

---

## 4. MODEL B′ — all eight Phase 1 witnesses re-certified

Two-window form; each window's base row pinned to the real orbit row (verified
by the Phase 1 engine). Four configurations per witness: CaDiCaL, Glucose,
MiniSat (ENC1) and CaDiCaL (ENC2). Every one **SAT**.

| `a` | `t` | `p` | two-window cells | single-cone cells | single-cone also run | solvers | orbit re-simulation | certified |
|---|---|---|---|---|---|---|---|---|
| 0 | 17 | 1 | 4 | 6 | SAT | SAT | ✔ | **yes** |
| 4 | 15363 | 1 | 60 | 42 | SAT | SAT | ✔ | **yes** |
| 8 | 3189 | 2 | 180 | 132 | SAT | SAT | ✔ | **yes** |
| 12 | 94693 | 2 | 364 | 240 | SAT | SAT | ✔ | **yes** |
| 16 | 850603 | 6 | 612 | 552 | SAT | SAT | ✔ | **yes** |
| 20 | 322343 | 27 | 924 | 2352 | SAT | SAT | ✔ | **yes** |
| 24 | 82630 | 1324 | 1300 | 1 821 150 | too large | SAT | ✔ | **yes** |
| 28 | 529506 | 2750 | 1740 | 7 725 620 | too large | SAT | ✔ | **yes** |

**All eight certified: True.** Certificates (with the full space-time patch
drawn and every hypothesis printed) are in
`results/model_bt_certificates/`.

Where the single-cone form was affordable (`a ≤ 20`) it was run as well and
agreed — so the two-window reformulation is validated against the tall-cone
form on six of the eight.

**LABEL: COMPUTATIONAL CERTIFICATE**, sound relative to the base rows
(`PHASE2A_METHOD.md` §4). The `a=0` case (§3.1) additionally has a
seed-only certificate.

**Reminder of what these witnesses mean.** They refute MB1-loc(`a`) for
`a = 0, 4, 8, 12, 16, 20, 24, 28`. They say nothing about MB1 itself, which is
vacuous if Rule 30 Prize Problem 1 has the expected answer (Phase 1
`CLAIM_LEDGER.md` E2).

---

## 5. The UNSAT control — the encoding does detect forcing

Since the MB1-loc grid never went UNSAT, the core/proof machinery would
otherwise be untested. The control is the Proposition 6 forcing case: column 0
agreeing at lag `p` at times `t` and `t+1`, with `a_t(0) = 1`, forces
`a_t(-1) = a_{t+p}(-1)` because `a_t(0) ∨ a_t(1) = 1` regardless of `a_t(1)`
(Fact F4 — rule 30 is not right-permutive).

| `p` | `a_t(0)` | predicted | SAT verdict (6 configs) | enumeration | agree |
|---|---|---|---|---|---|
| 1, 2, 3, 5, 8 | 1 | UNSAT | **UNSAT** | **UNSAT** | ✔ |
| 1, 2, 3, 5, 8 | 0 | SAT | **SAT** | **SAT** | ✔ |
| 13 | 1 | UNSAT | **UNSAT** | skipped (32-bit base row) | ✔ (solvers only) |
| 13 | 0 | SAT | **SAT** | skipped (32-bit base row) | ✔ (solvers only) |

12 control instances, every configuration agreeing with the hand proof;
**10 of the 12 were additionally decided by solver-free enumeration** (the two
at `p = 13` have a 32-bit base row, beyond the 24-bit enumeration cap).

**Unsat core** (identical for every `p`, both encodings, all three solvers):

```
['C_left_disagree', 'H1_agree_0', 'H1_agree_1', 'H2_centre_one']
```

All four hypotheses, and nothing else — the core is minimal and names exactly
the constraints in the hand proof. The recurring structural reason is therefore
established and is not `p`-dependent: it is the single application of Fact F4.
Cores are saved in `results/unsat_cores/`.

### 5.1 Proof export — reported limitation

`Cadical153(with_proof=True)` was used to request DRUP proofs. **The emitted
proof is empty (0 lines)** for every control instance: they are refuted by unit
propagation during preprocessing, so no clause is logged. Moreover **no DRAT
checker (`drat-trim`) is installed in this environment**, so no proof was
independently checked.

Consequently the UNSAT verdicts rest on: three SAT solvers, two CNF encodings,
Z3, exhaustive enumeration over all `2^w` base rows, and a hand proof — but
**not** on a machine-checked proof certificate. This is a real gap in the
package and is recorded as such.

---

### 5.2 A defect found by the test battery, and fixed

The first run of `run_phase2a_tests.py` reported **1 failure**: the aggregate
flag `all_agree` for the two `p = 13` control instances was `False`. The cause
was not a solver disagreement but a bug in the *reporting* code: when the base
row is too wide to enumerate, `decide_control_by_enumeration` returns
`"SKIPPED"`, and that string was being folded into the set of verdicts as
though it were a dissenting answer. Absence of evidence was being counted as
contradictory evidence.

Fixed in `run_phase2a_search.py` (`enumeration_checked` is now recorded
separately and `SKIPPED` is excluded from the agreement set) and in the test,
which now also asserts a minimum number of enumeration-checked instances so
that skipping cannot silently hollow out the cross-check. Section 4 of the
results JSON was regenerated; no verdict changed.

Recorded here rather than silently corrected, per the Phase 1 convention.

---

## 6. The two independent decision procedures never disagreed

| Cross-check | instances | disagreements |
|---|---|---|
| ENC1 vs ENC2, three solvers each | 832 | 0 |
| Z3-native vs CNF | 156 (`p ≤ 12`) | 0 |
| SAT vs exhaustive enumeration | 18 | 0 |
| Model B vs direct simulation | 3168 | 0 |
| Model B′ vs orbit re-simulation | 8 | 0 |
| single-cone vs two-window Model B′ | 6 | 0 |

---

## 7. Lemmas verified computationally

| Lemma | Statement | Check |
|---|---|---|
| **Lemma C** | every cell's three predecessors lie inside the cone, so no free boundary reaches a queried cell | verified cell by cell for heights 1–59 and three top-site ranges — **PASS** |
| **Lemma T** | Model A's verdict is independent of `t` | Model A built at explicit `t = a+1, a+9, a+137, a+5000` for four `(a,p)`; identical verdicts, and identical to the normalised build — **PASS** |

Both are also proved on paper in `SAT_MODEL_SPECIFICATION.md`. The
computational checks guard the *implementation*, not the mathematics.

---

## 8. Why no new witness for `a ≥ 32` — data-limited, not method-limited

Census over the 200001-bit centre column with `p ≤ 20000`, counting positions
that satisfy the hypotheses at all:

| `a` | eligible positions | violations | violation rate |
|---|---|---|---|
| 0 | 949 254 892 | 474 627 476 | 0.5000 |
| 4 | 59 333 217 | 17 654 730 | 0.2976 |
| 8 | 3 709 400 | 964 006 | 0.2599 |
| 12 | 232 492 | 56 857 | 0.2446 |
| 16 | 14 639 | 3 517 | 0.2402 |
| 20 | 885 | 204 | 0.2305 |
| 24 | 49 | 11 | 0.2245 |
| 28 | 5 | 0 | 0.0000 |
| 32 | **1** | 0 | 0.0000 |
| 36 | **0** | — | — |
| 40 | **0** | — | — |
| 44 | **0** | — | — |
| 48 | **0** | — | — |

Eligible positions fall by roughly a factor of 16 per `+4` in `a` (the
hypotheses demand `a+1` agreements plus a zero, so eligibility scales like
`2^{-(a+1)}`). At `a = 36` and beyond, **this sequence length contains no
position that even satisfies the hypotheses** — so finding a witness is
impossible here for want of data, irrespective of whether one exists.

**LABEL: BOUNDED UNSAT.** For `a ≥ 32` we report: *no counterexample was found
within `N = 200001`, `p ≤ 20000`*, together with the fact that the search space
contained ≤ 1 eligible position. **This is not evidence that MB1-loc(`a`) holds
for `a ≥ 32`.**

Consistency note: the Phase 1 witness at `a = 28` sits at `t = 529506`, beyond
this census window, and was found in the 10⁶-bit run (24 eligible positions,
1 violation). The `a = 28` row above showing 0 violations at `N = 200001`
is therefore expected and not a contradiction.

Extrapolating the eligibility scaling, a witness at `a = 36` needs roughly
`N · p_max ≈ 2^{37}` position-lag pairs — about 60× the 10⁶-bit, `p ≤ 20000`
budget of Phase 1.

---

## 9. What is *not* claimed

1. **MB1-loc(`a`) is not asserted to hold for any `a`.** No search here found a
   witness for `a ≥ 32`, and §8 explains why that is uninformative. The
   standing prohibition of `SAT_MODEL_SPECIFICATION.md` §9 applies.
2. **Model A SAT patches are not counterexamples.** They belong to the
   seed-free relaxation (Theorem A-incomplete) and are labelled separately in
   the registry.
3. **Model A's uniform satisfiability is not evidence that MB1-loc is false.**
   It is evidence that *a certain class of proofs* cannot establish it.
4. **Nothing here bears on MB1 itself**, which is vacuous if Problem 1 has the
   expected answer, nor on Prize Problem 1.
5. **No machine-checked UNSAT proof** was produced (§5.1).
6. **Nothing is claimed for `p > 64`** in Model A, nor for `t > 48` in the
   exhaustive Model B sweep.

---

## 10. What Phase 2A adds to Phase 1

* An **encoding-level** account of why SAT cannot search the real orbit: the
  faithful model has no free variables (`PHASE2A_METHOD.md` §1). This is a
  methodological result, and it redirects effort away from "throw a solver at
  it".
* A **theorem-shaped negative result**: no seed-free local argument at depth
  `≤ 48`, lag `≤ 64` can prove MB1-loc — because a satisfying patch always
  exists (§1.1). Phase 1 could only observe that witnesses exist in the orbit;
  this shows the *local constraints alone never forbid* them.
* **Independent re-certification** of all eight Phase 1 witnesses through
  machinery (SAT solvers, SMT, two encodings) sharing nothing with the Phase 1
  enumeration except the rule-30 table.
* **121 new real-orbit counterexamples** at small `t`, found exhaustively and
  double-checked.
* A **quantitative account** of the `a ≥ 32` gap: 1 eligible position at
  `a = 32`, zero beyond.
