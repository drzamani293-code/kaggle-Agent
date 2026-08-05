# PHASE2E_TRANSIENT_FRONTIER

The transient frontier of the edge-aligned prefix tower of rule 30: an exact
recurrence for the preperiod `T(K)`, the reset events that create it, and what
the moving diagonal reads before anything settles. Phase 2E only — **no final
proof is attempted and none is claimed.** Phase 2F is not started.

Prior packages: `RULE30_PHASE1_HANDOFF`, `PHASE2A_SAT_MB1LOC`,
`PHASE2B_GLOBAL_DEFECT`, `PHASE2C_EDGE_PREFIX`, `PHASE2D_COLLAPSE_CHAIN`.

---

## Start here

```bash
cd PHASE2E_TRANSIENT_FRONTIER
pip install -r requirements.txt          # numpy only
python3 run_phase2e_tests.py             # expect "55 passed, 0 FAILED", ~2 s
python3 run_phase2e_tests.py --fast      # ~1 s
python3 run_phase2e.py                   # regenerate everything, ~18 s
python3 run_phase2e.py --quick           # K <= 3000
```

Read in this order: `TRANSIENT_DEFINITIONS.md` → `TRANSIENT_RECURRENCE.md` →
`RESET_SCHEDULE_THEORY.md` → `FRONTIER_DYNAMICS.md` →
`TRANSIENT_LEMMA_REGISTRY.md`.

---

## 1. The headline result

Phase 2D proved the **inequality** `T(K) ≤ max(T(K-1), τ_K + 1)`. Phase 2E turns
it into an **exact dichotomy with a one-bit criterion.**

Define the **defect bit** of level `K`:

```
    D(K)  =  w_{T(K-1)}(K)  XOR  w_{T(K-1)+P(K)}(K)
```

**THEOREM B.** Every level is exactly one of

| class | condition | preperiod |
|---|---|---|
| **INHERITING** | `D(K) = 0`, or the level is DOUBLING or NEUTRAL | `T(K) = T(K-1)` |
| **RESETTING** | `D(K) = 1` (forces COLLAPSING) | `T(K) = τ_K + 1`, exactly |

Proved for all `K` (Theorems B1–B4, from a defect-propagation lemma);
**verified at 29 999 / 29 999 levels with 0 mismatches**, `K ≤ 30000`.

Two immediate consequences, both theorems:

* `T(K) = ρ(K) + 1` at **every** RESETTING level — the brief's question answered
  exactly, not statistically.
* `T(K) = T(K_0) + Σ_{RESETTING K'} δ(K')`, so

  ```
      T(K)/K  =  (density of RESETTING levels) × (mean reset increment)
  ```

  Measured at `K = 30000`: `0.546485 × 2.451873 = 1.339911` against
  `T(K)/K = 1.339900`.

**This factorises Phase 2B's open target G5.** `liminf T(K)/K > 1` is now
*exactly equivalent* to a statement about how often levels collapse late
(registry item **F1**), with a strictly weaker sub-target **F2** (positive
density of `D(K) = 1`). It is a reformulation, **not** a reduction: neither
factor has a proved lower bound, and no route from F1 to a contradiction exists.

## 2. Results table

| Result | Label |
|---|---|
| **A** `T(K) = max_{k ≤ K} r_{P(K)}(k)` — `T` decomposes into coordinate preperiods | **THEOREM**, 30001 levels, 0 mismatches |
| **Lemma 2.1** a collapse erases the lag-`p` *discrepancy*, not just the value | **THEOREM**, 0 violations on the orbit |
| **B1–B4** the RESETTING / INHERITING dichotomy and the exact recurrence | **THEOREM**, 29999/29999 |
| **B5** `T(K)/K =` density × mean increment | **THEOREM** (identity) + measurement |
| **3.1** `T(K) = ρ(K)+1` at every RESETTING level | **THEOREM**, 16394/16394 |
| **3.2** `T(K) = ρ(K)+1` ⟺ `w_{T(K)-1}(K-1) = 1`; on INHERITING levels this is a coincidence (47.6 %) | **THEOREM** + measurement |
| **3.4 / 4.2** `T(K+1) - T(K) ≤ P(K)` when coordinate `K` is not eventually zero | **THEOREM**, 0 violations, max 15 vs bound 16 |
| **4.6** frontier speed `≥ 1/P`; measured `≈ 1/1.34` | **THEOREM** + a **12×** gap to the measurement |
| **5.1** the co-moving strip **is** the original diagram, `q_t(r) = x_t(r)` | **THEOREM**, 0 mismatches |
| **5.4** the strip is **not** autonomous — explicit witness at every radius 1…12 | **THEOREM** + 12 finite counterexamples |
| **8.4** under (H), no column other than the centre is eventually periodic | **THEOREM** (conditional, from W2′) |
| Hash-consed proof DAG with the periodicity rewrite: reduction factor **1.39–1.46**, size `≈ 0.35 t²` | **NEGATIVE RESULT**, `t ≤ 3000` |
| 69 % of the diagonal's backward cone never settles; cone entirely transient from `s ≈ 0.81 t` | BOUNDED OBSERVATION, `t ≤ 3000` |
| Phase 2C Theorem D1 is **vacuous** on the real orbit for every `p ≥ 7` | **NEGATIVE RESULT** |

## 3. What Phase 2E did NOT do

* **It did not touch the Bridge** (Phase 1 C2 / Phase 2B G1). Nor did 2A, 2B,
  2C or 2D. Phase 2E ends at the same wall from a fifth direction, and
  `TRANSIENT_PERIODICITY_CONSEQUENCES.md` §3 says exactly why each of the four
  candidate routes (A–D) fails — Route B fails because its bridge is *false*,
  not merely unproved.
* **It did not prove `T(K) > K`.** That remains a BOUNDED OBSERVATION for
  `18 ≤ K ≤ 30000`, with `K ∈ {0,…,17}` as genuine counterexamples, kept.
* **It found no contradiction with the periodicity hypothesis, and claims none.**
* **It closed one route of its own.** Symbolic simplification of the diagonal
  with the full strength of the proved periodicity structure buys a constant
  factor of 1.45 and leaves a quadratic DAG. Recorded as a negative result
  alongside Phase 2D's ANF finding.
* **It found the co-moving strip to be a change of viewpoint with zero
  information gain** — Theorem 5.1 identifies it with the original columns.
  That is a negative result about the brief's §5, stated up front rather than
  dressed up.

## 4. Corrections to earlier claims

| earlier statement | correction |
|---|---|
| Phase 2B/2C: "`T(K) ≈ 1.29 K`" / "`≈ 1.34 K`" | `T(K)/K` is **not monotone** — `1.3400` at `K=100`, `1.2525` at `K=400`, `1.3399` at `K=30000`. No limit is claimed to exist and no constant is fitted. |
| Phase 2D R3 stated as `T(K) ≤ max(T(K-1), τ_K+1)` | since `τ_K ≥ T(K-1)` always, the max is redundant: the bound is just `T(K) ≤ τ_K + 1`, and Theorem B says when it is tight. |
| "T(K) = ρ(K)+1 usually (76 %)" | the aggregate mixes a **theorem** (100 % on RESETTING levels) with a **coincidence** (47.6 % on INHERITING levels). The aggregate figure is not a meaningful quantity. |

## 5. Files

```
README.md                              this file
TRANSIENT_DEFINITIONS.md               §1  T, R, tau, A, K_per -- which coincide
TRANSIENT_RECURRENCE.md                §2  Lemma 2.1, Theorems B1-B5
RESET_SCHEDULE_THEORY.md               §3  rho, increments, gaps
FRONTIER_DYNAMICS.md                   §4  bounds on dT and dK_per
COMOVING_STRIP_DYNAMICS.md             §5  the strip is the original diagram
DIAGONAL_RESET_SKELETON.md             §6  reset ancestry of the diagonal
DIAGONAL_PROOF_DAG.md                  §7  hash-consed DAG, a negative result
TRANSIENT_PERIODICITY_CONSEQUENCES.md  §8  conditional consequences of (H)
TRANSIENT_LEMMA_REGISTRY.md            §9  F1-F5
TRANSIENT_CLASSIFICATION.csv               29999 rows, every level classified
RESET_SCHEDULE_RESULTS.csv                 29999 rows, tau / rho / increments
requirements.txt

frontier_lab.py                        all derivations, checks and measurements
run_phase2e.py                         measurement driver (8 sections)
run_phase2e_tests.py                   test battery (55 checks)
prefix_lab.py                          Phase 2C library (exact T, P)
collapse_lab.py                        Phase 2D library (fibre classification)
rule30_lab.py                          Phase 1 library (verified rule-30 engines)

results/
  phase2e_results.json                 raw output
  phase2e_run.log                      stdout of the measurement run
  phase2e_tests.log                    stdout of the test battery
```

## 6. Scientific controls applied

* **Two or three independent implementations of every computed quantity.**
  `T(K), P(K)`: first-repeat sweep, hash reference, incremental skew-product,
  plus full periodicity+minimality verification at sampled `K`.
  `r_Q(k)`: backward `unseen`-mask sweep, explicit suffix-OR array swept
  forward, and a per-coordinate scan straight from the definition.
  `τ_K`: direct minimum scan and Phase 2D's cycle-phase index.
  `K_per`: running maximum and bucket-count.
* **All-`K` theorems are separated from bounded observations everywhere.**
  Theorems B1–B4 are proved for all `K` precisely so that the classification
  does not rest on the (bounded) facts that NEUTRAL never occurs and that every
  level past 400 is COLLAPSING.
* **Failed and negative results preserved**: Routes A–D in §8, the DAG's
  constant-factor saving, the vacuity of Theorem D1, the strip's identification
  with the original diagram, the five levels where `ρ` is undefined, and the 18
  levels with `T(K) ≤ K`.
* **No bounded search is described as UNSAT.** "No NEUTRAL level for
  `K ≤ 30000`" and "34 strip words not observed at `R = 11`" mean exactly that.
* **No asymptotic constant is fitted.** Every ratio quoted is a value measured
  at a stated `K` or `t`.
* **No inference from growing complexity.** The DAG's quadratic growth and the
  strip's word counts are reported as measurements and are never offered as
  evidence of aperiodicity.
* **No randomness tests were run** (standing constraint since Phase 2B).
