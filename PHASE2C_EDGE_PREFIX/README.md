# PHASE2C_EDGE_PREFIX

The autonomous prefix dynamics of rule 30 in edge-aligned coordinates: exact
preperiods `T(K)`, periods `P(K)`, and the diagonal bit `w_K(K)`.
Phase 2C only — **no final proof is attempted and none is claimed.**

Prior packages: `RULE30_PHASE1_HANDOFF`, `PHASE2A_SAT_MB1LOC`,
`PHASE2B_GLOBAL_DEFECT`.

---

## Start here

```bash
cd PHASE2C_EDGE_PREFIX
pip install -r requirements.txt      # numpy only
python3 run_phase2c_tests.py         # expect "35 passed, 0 FAILED", ~1 s
python3 run_phase2c_tests.py --fast  # ~1 s
python3 run_phase2c.py --k-max 30000 # regenerate everything, ~60 s
```

Read: `PREFIX_MAP_THEORY.md` → `SKEW_PRODUCT_ANALYSIS.md` →
`PREFIX_PERIOD_RESULTS.md` → `DIAGONAL_BRIDGE_ATTEMPT.md`.

## The object

```
    w_{t+1}(k) = w_t(k-2) XOR ( w_t(k-1) OR w_t(k) ),   w_0 = (1,0,0,...)
```

is rule 30 in edge-aligned coordinates `w_t(k) = x_t(-t+k)`. It is **one-sided**,
so each prefix `W_t^K = (w_t(0..K))` evolves autonomously under a map `F_K`.
`T(K)`, `P(K)` are the exact tail and cycle length of that orbit. The centre
column is the **diagonal** `x_t(0) = w_t(t)`.

## Results

| Result | Label |
|---|---|
| `pi_K ∘ F_{K+1} = F_K ∘ pi_K`; coordinate `K+1` is a **one-bit fibre** driven by coordinates `K-1, K` | **THEOREM** P1, P2 (exhaustively verified) |
| Fibre trichotomy: `c=1` ⟹ constant map; `c=0` ⟹ bijection | **THEOREM** (Lemma F) |
| `P(K) ∈ {P(K-1), 2P(K-1)}` — so **E1** and **E2** are theorems | **THEOREM** S1 |
| `T(K) <= T(K-1) + P(K-1)` | **THEOREM** S2 |
| `T(K-1) <= T(K)` — so **E3** is a theorem | **THEOREM** S3 |
| `T(K) > floor(K/2) - P(K)` whenever coordinate `K` is ever 1 | **THEOREM** S4 |
| **`w_t(7) = 0` for every `t`** — the unique permanently-zero coordinate | **THEOREM** (2-cycle exhibited) |
| Exact `T(K), P(K)` for **all `K <= 30000`**, three independent algorithms agreeing | **EXACT COMPUTATION** |
| **E4 is FALSE as stated** — counterexamples exactly `K = 0..17` | **REFUTED**; `E4′` (`K >= 18`) is a bounded observation |
| Only **four** period doublings up to 30000: `K = 3, 8, 29, 400`; `P = 16` throughout `[400, 30000]` | **EXACT** |
| In the real orbit the fibre class is **COLLAPSING 4995×, DOUBLING 4×, NEUTRAL 0×** | COMPUTATIONAL OBSERVATION |
| `x_{t+P(K)}(0) = x_t(P(K))` for `T(K) <= t <= K - P(K)` | **THEOREM** D1 (verified at all 16 pairs where it has content) |
| If `T(K) <= K - P` eventually, the centre column is **not** eventually periodic | **THEOREM D2 (conditional)** |
| **(HYP-D) is false for every measured `K`** — the D2 bridge is blocked | **NEGATIVE RESULT** |

## The correction to Phase 2B

Phase 2B's registry item **G5** framed `T(K)/K > 1` as the structurally
interesting direction and suggested `T(K) <= K` would look like a positive
answer to the periodicity question. **Phase 2C shows the opposite.** Theorem D2
proves that `T(K) <= K - P` eventually would *establish aperiodicity*. The
measured `T(K) ≈ 1.34K` therefore **blocks the only bridge found**, and is
evidence for neither answer. `DIAGONAL_BRIDGE_ATTEMPT.md` §3 states the
retraction.

## Files

```
README.md                       this file
PREFIX_MAP_THEORY.md            F_K in coordinates; P1 projection; P2 skew form
SKEW_PRODUCT_ANALYSIS.md        Lemma F/C; theorems S1-S4; E1-E3 proved
PREFIX_PERIOD_RESULTS.md        exact T,P to K=30000; E1-E5 verdicts
FORCED_BIT_CLASSIFICATION.md    the three fibre classes and the real-orbit census
DIAGONAL_BRIDGE_ATTEMPT.md      D1, D2, and why the route is blocked
PREFIX_PERIOD_TABLE.csv         K, T_K, P_K, T_minus_K, T_over_K  (30001 rows)
requirements.txt

prefix_lab.py                   all definitions, three exact algorithms, checks
run_phase2c.py                  measurement driver (6 sections)
run_phase2c_tests.py            test battery
rule30_lab.py                   Phase 1 library

results/phase2c_results.json    raw output
results/phase2c_run.log         stdout of the measurement run
results/phase2c_tests.log       stdout of the test battery
```

## Scientific controls applied

* **Nothing is inferred for all `K` from `K <= 30000`.** The all-`K` statements
  (S1–S4, P1, P2, D1, D2) are proved; everything else is labelled bounded.
* **`T(K) > K` is not used as evidence of centre-column aperiodicity** — §3 of
  the bridge document explains why it is the opposite.
* Every exact `T(K), P(K)` is produced by **three independent algorithms**
  (first-repeat scan, hash reference, incremental skew-product) and additionally
  checked for **minimality of both `T` and `P`** at sampled `K`.
* No `2^{K+1}` state space is ever stored; only `T_max+1` row integers.
* Counterexamples and failures preserved: E4's 18 counterexamples, the
  permanently-zero coordinate 7 (which breaks S4's unconditional form), the
  blocked D2 bridge, the three failed D2 variants, and the discarded
  suffix-minimum algorithm that a cross-check caught
  (`PREFIX_PERIOD_RESULTS.md` §2).
