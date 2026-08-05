# Computational Certificate Catalog

Finite computations, kept strictly apart from `VERIFIED_THEOREMS.md`. Each
entry states what it certifies **and what it does not**.

All paths are relative to the repository root. Hashes were computed on the
delivered artifacts; `run_verified_theorems_tests.py` re-computes them and
fails on any mismatch.

Environment for all runs: Python 3.11.15 on Linux; numpy 2.4.6; for Phase 2A
additionally PySAT with CaDiCaL 1.5.3, Glucose 4.2, MiniSat 2.2, and Z3 5.0.0.
No randomness is used anywhere in the corpus (no `random`, no `np.random`), so
every run is bit-for-bit reproducible.

---

## CC-01 — the centre-column sequences

| field | value |
|---|---|
| **program** | `RULE30_PHASE1_HANDOFF/gen_center_column.py` |
| **input range** | `t < 1 000 001` and `t < 200 001` |
| **output artifacts** | `RULE30_PHASE1_HANDOFF/phase1_results/center_column_1000001.txt`, `center_column_200001.txt` |
| **SHA-256** | `0bb02e4ed6c3d80bd832eed0b6991cc99f6de01a5f32fa795281d3efb2e9104e` (10⁶); `cf018b09db932412d5f42c3d03076181a34d806173fbee749e30c871fe5d994a` (200001) |
| **independent implementations** | four engines: list+rule-table, numpy+rule-table, big-integer bit-parallel `row = ((row<<1) ^ (row | (row>>1))) & mask`, and the **rule-86 mirror** with reversed shifts. Verified by `verify_sequence_independent.py`; the numpy cross-check covers a prefix only (it is `O(N²)` in cells) |
| **certifies** | these specific bits are the centre column of the single-cell rule-30 orbit, under the rule-numbering convention fixed in `DEFINITIONS_AND_NOTATION.md` §1 |
| **does NOT certify** | that the convention is Wolfram's. The only external anchor is a 14-bit prefix match to a *secondary* transcription of OEIS A051023 (`oeis.org` returned HTTP 403). This is the largest unverified assumption in the corpus |

## CC-02 — the factor count `998140`

| field | value |
|---|---|
| **program** | `RULE30_PHASE1_HANDOFF/verify_factor_bound.py` |
| **input** | `center_column_1000001.txt`, word length `n = 28` |
| **output artifact** | `phase1_results/verify_factor_1M.log` |
| **result** | **998140** distinct length-28 factors, against a ceiling of `N − n + 1 = 999973`; fraction attained `0.998166` |
| **independent implementations** | three algorithms — Python hash-set, numpy sort-unique on packed integers, and a string-set pass |
| **certifies** | the numeral in **CT-01**. With the factor-counting lemma (T4) it yields `T + p ≥ 998140` |
| **does NOT certify** | anything about aperiodicity. The method's ceiling is the prefix length: with `N` bits it can never give better than `T + p ≲ N`, so **no computation of this kind can ever prove the centre column aperiodic** |

## CC-03 — period scans

| field | value |
|---|---|
| **programs** | `run_phase1.py` (`p ≤ 10000`, `N = 200001`), `run_anomaly_followup.py` (`p ≤ 20000`, `N = 10⁶`) |
| **output artifacts** | `phase1_results/phase1_results.json: period_scan`; `anomaly_followup.json: period_scan` |
| **result** | every candidate period fails; largest first-mismatch position over all `p ≤ 10000` is **11** (at `p = 9988`). Consequence: if the centre column is eventually periodic with `p ≤ 20000` then `T > 979998` (**CT-02**) |
| **independent implementations** | fast big-integer scan, cross-checked against a naive `O(Np)` scan for `p ≤ 64` |
| **does NOT certify** | anything for `p > 20000` |

## CC-04 — real-orbit MB1-loc witnesses

| field | value |
|---|---|
| **program** | `RULE30_PHASE1_HANDOFF/find_mb1loc_witnesses.py` |
| **output artifact** | `phase1_results/mb1loc_witnesses.json`; logs `mb1loc.log`, `mb1loc_1M.log` |
| **result** | explicit witnesses refuting MB1-loc(`a`) for `a = 0, 4, 8, 12, 16, 20, 24, 28` |
| **independent implementations** | re-simulated twice — inside the search script, and again by `run_all_tests.py` step 5 in a separate process, which **re-derives** rather than reading the JSON |
| **certifies** | MB1-loc is false at those depths |
| **does NOT certify** | that MB1-loc is false for every `a`. `a ≥ 32` is untested and the scan is **data-limited**: at `N = 200001, p ≤ 20000` only **1** position satisfies the hypotheses at `a = 32`, **0** at `a ≥ 36` (`BO-01`). This is **not** UNSAT |
| **correction attached** | witness polarity is not uniform — see C-02 |

## CC-05 — Model A, 832 instances

| field | value |
|---|---|
| **program** | `PHASE2A_SAT_MB1LOC/run_phase2a_search.py`, encodings in `mb1loc_sat.py` |
| **input range** | `a ∈ {0,4,…,48}` × `p ∈ [1,64]` = 832 instances |
| **solvers** | six PySAT configurations (ENC1 direct and ENC2 Tseitin × CaDiCaL 1.5.3, Glucose 4.2, MiniSat 2.2), plus **Z3 5.0.0** natively for `p ≤ 12`, plus **solver-free exhaustive enumeration** on 18 instances with base row ≤ 18 bits |
| **output artifacts** | `PHASE2A_SAT_MB1LOC/results/phase2a_results.json` (SHA-256 `29bc3bbed880faae1a55e94c61f30112dc9fc09fed76b279bc86f120d3a37e57`), 26 patch certificates in `results/model_a_patches/`, unsat cores in `results/unsat_cores/` |
| **result** | **832 SAT, 0 UNSAT, 0 cross-configuration disagreements** |
| **certifies** | with **T21.11** (translation invariance ⇒ the `t` search is complete, not bounded) and **T21.12** (soundness), that **no seed-free local argument at depth `a ≤ 48` and lag `p ≤ 64` can prove MB1-loc** |
| **does NOT certify** | that MB1-loc is false. A Model A patch has a **free base row** that need not occur in the real orbit; it is not a counterexample (Theorem A-incomplete). Every patch file carries this banner on line 3, and the registry keeps them under `model = "A_relaxation"`, apart from the real witnesses |

## CC-06 — zero-wall automaton

| field | value |
|---|---|
| **program** | `PHASE2B_GLOBAL_DEFECT/defect_lab.py: wall_automaton` |
| **input range** | radius `r = 1..5` (`2^{4r+1}` states) |
| **result** | total wall states `32, 512, 8192, 131072, …`; maximal invariant set `16, 128, 992, 7616, 59136` — **non-empty at every radius computed** |
| **output artifact** | `PHASE2B_GLOBAL_DEFECT/results/phase2b_results.json` (SHA-256 `81725e5ae6837cd7222bed02cc9dcda3e536c579b72d8a54a6a0e2b4bfe36731`) |
| **certifies** | no bounded impossibility is obtained from the local wall relaxation at `r ≤ 5` |
| **does NOT certify** | anything at `r ≥ 6` — those are **NOT COMPUTED** (`3.4×10⁷` to `8.6×10⁹` states), never "no result". And "locally admissible but globally unreachable" is not decided: at `r = 4`, only 1 763 of 7 616 invariant states were observed in the real orbit, which is a **gap, not a proof of unreachability** |
| **superseded in strength by** | **T21.8**, which proves the analogous statement for the co-moving strip automaton **for every radius** and finds the recurrent core is the *entire* state space |

## CC-07 — exact `T(K)`, `P(K)`

| field | value |
|---|---|
| **program** | `PHASE2C_EDGE_PREFIX/run_phase2c.py`, algorithms in `prefix_lab.py` |
| **input range** | `K ≤ 30000`, `T_max = 50000` |
| **output artifacts** | `PHASE2C_EDGE_PREFIX/PREFIX_PERIOD_TABLE.csv` (30001 rows, SHA-256 `2494997051f5c5cd521ae83276740f6ec152f520263e64aa96aa3b2d5a1bd959`); `results/phase2c_results.json` (SHA-256 `18229b7c9041ee81918e087d440cde515918be1b5cf0c3264cc4d7684e3b9563`) |
| **independent implementations** | **three**: first-repeat two-pointer sweep; hash-based reference (first repeated state) on `K ≤ 400`; incremental skew-product on `K ≤ 1500`. Plus full periodicity **and minimality** verification at 13 sampled `K` |
| **certifies** | `T(K)` and `P(K)` are the **exact** preperiod and minimal period, not bounds |
| **does NOT certify** | anything for `K > 30000`. Every claim about "all `K`" that quotes this table is a bounded observation |
| **correction attached** | the first algorithm attempted (suffix minima) was wrong and was replaced — see C-17 |

## CC-08 — collapse-chain classification

| field | value |
|---|---|
| **program** | `PHASE2D_COLLAPSE_CHAIN/run_phase2d.py`, `collapse_lab.py` |
| **input range** | `K ≤ 30000` (formulas verified on 2995 levels; chain analysis on 2598) |
| **output artifact** | `PHASE2D_COLLAPSE_CHAIN/results/phase2d_results.json` (SHA-256 `18b4b8d6dd86b3e41b7f97bb08a9d26c3550427d4601c3e6396d95b25806ec21`) |
| **result** | R1/R2/R3 exact — **0 mismatches / 2995 levels**; CH1 unique — **2598/2598**; eventually-zero coordinates `{2,7,28,399}`; every level in `[401,30000]` COLLAPSING |
| **certifies** | the formulas of **T-16**, **T-17** hold on the computed range (they are separately **proved** for all `K`) |
| **does NOT certify** | completeness of `{2,7,28,399}` beyond `K = 30000` (`BO-04`, C-10), nor that NEUTRAL never occurs — by **T-14** there were exactly **four opportunities** |

## CC-09 / CC-11 — Phase 2E transient classification

| field | value |
|---|---|
| **program** | `PHASE2E_TRANSIENT_FRONTIER/run_phase2e.py` §§2, 9, 10 |
| **input range** | `K ≤ 30000` (29999 levels), `T_max = 46000` |
| **output artifacts** | `TRANSIENT_CLASSIFICATION.csv` (30000 lines, 15 columns, SHA-256 `db21ed34a9fdda73e9d7d47198f915852b8b266f9640ae533cb5d98e41b76aa9`); `RESET_SCHEDULE_RESULTS.csv` (30000 lines, 8 columns, SHA-256 `f3e3b07233fdb85f40baca09c5aa4f5b48690e480817c039651dcdb97664aaa1`); `results/phase2e_results.json` (SHA-256 `67a42a319f8d49c17e2fd4a8603a10d30e16617c471676e33d8c13092dae2349`) |
| **result** | two-way: RESETTING 16394 / INHERITING 13605. Five-way: INHERITED 13601, RESET_IMMEDIATE 4386, PHASE_DELAY 12008, PERIOD_DOUBLING 4, ZERO_PREDECESSOR 0. **0 prediction failures, 0 disagreements** between two independent derivations per level |
| **certifies** | **T-18** and **T-19** hold at every computed level; the reset times `σ, τ, ρ` are pairwise distinct in the measured proportions |
| **does NOT certify** | that the classification pattern continues; nor the density `0.546485` or mean increment `2.451873` as limits (`BO-10`) |

## CC-10 — strip lookahead counterexamples

| field | value |
|---|---|
| **program** | `PHASE2E_TRANSIENT_FRONTIER/transient_lab.py: strip_lookahead` |
| **input range** | `R = 1,2,3,4,6`; `t ≤ 20000` |
| **result** | for each `R`, an explicit pair of times with the **same** radius-`R` strip state and **different** `q_{t+R+1}(0)`; and no such pair for `n ≤ R` |
| **certifies** | the negative half of **T21.9**: the bound `n ≤ R` is exactly attained |
| **does NOT certify** | anything about larger radii |

## CC-12 — the centre column is not `m`-step Markov

| field | value |
|---|---|
| **program** | `transient_lab.py: centre_history_determines_next` |
| **input range** | `m ∈ {1,2,4,8,12,16,20}`, `t ≤ 50000` |
| **result** | for each `m`, an explicit pair of times with the same last `m` centre bits and a different next bit (e.g. `t = 227` and `t = 1994` at `m = 20`) |
| **certifies** | the centre column is not generated by a memory-`m` automaton reading only its own past, for `m ≤ 20` |
| **does NOT certify** | anything about aperiodicity. An eventually periodic sequence with a long preperiod also fails such tests at small `m` |

---

## What the whole catalog does not do

1. **No certificate proves non-periodicity.** CT-01 and CT-02 are lower bounds
   on `T + p`; they scale linearly with the computation and can never close.
2. **No bounded scan is UNSAT.** The word UNSAT appears in this corpus only
   where a formal finite formula was solved and a solver returned
   unsatisfiable — the Proposition 6 control (where a hand proof *predicts*
   UNSAT, and the 4-constraint core matches) and the Model B sweep outside its
   122 counterexamples. "No witness found" is always written as such.
3. **Ranges are never extrapolated.** Every table caption names its `K` or `t`.
4. **Independent implementation ≠ independent formalisation.** Four engines
   agreeing rules out coding slips; it does **not** rule out a shared
   misreading of the rule-numbering convention (CC-01's caveat).
