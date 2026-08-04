# PHASE2B_GLOBAL_DEFECT

The temporal defect field of Rule 30 and the "zero wall" that eventual
periodicity of the centre column would impose. Phase 2B-GLOBAL only: **no final
proof is attempted and none is claimed.** Phase 2C is not started.

Prior packages: `RULE30_PHASE1_HANDOFF` (foundations, Theorems W2/W2′, the
MB1 gap analysis) and `PHASE2A_SAT_MB1LOC` (SAT formulations of MB1-loc).

---

## Start here

```bash
cd PHASE2B_GLOBAL_DEFECT
pip install -r requirements.txt        # numpy only
python3 run_phase2b_tests.py           # ~3 min, expect "60 passed, 0 FAILED"
python3 run_phase2b_tests.py --fast    # ~1 s
```

Regenerate all measurements (~6 min):

```bash
python3 run_phase2b.py                 # writes results/phase2b_results.json
```

Read in this order: `DEFECT_DYNAMICS.md` → `ZERO_WALL_LEMMAS.md` →
`EDGE_ALIGNED_DYNAMICS.md` → `GLOBAL_LEMMA_REGISTRY.md`.

---

## The setting

For a lag `p`, the temporal defect field is `d_t(j) = x_{t+p}(j) XOR x_t(j)`.
If the centre column were eventually `p`-periodic after `T`, then `d_t(0) = 0`
for all `t >= T` — an infinite **zero wall** at column 0. Phase 2B asks whether
the real defect field can be compatible with such a wall.

**It does not answer that.** What it produces is the exact algebra of the
defect field, the exact consequences of a wall, three structural theorems in
edge-aligned coordinates, and the closing-off of two proposed routes.

## Results

| Result | Label |
|---|---|
| `d_{t+1}(j) = d_t(j-1) + d_t(j) + d_t(j+1) + x_t(j)d_t(j+1) + x_t(j+1)d_t(j) + d_t(j)d_t(j+1)` over GF(2) | **THEOREM**, exhaustive over all 64 assignments |
| A black centre cell with no defect **masks** the defect to its right | **THEOREM** (D3/D4) — the mechanism behind every wall lemma |
| Wall equation `d_t(-1) = d_t(1)·(1 XOR x_t(0))` | **THEOREM** W1 (= Phase 1 Prop. 6 in defect variables) |
| A run of `k` ones in the centre column forces a **zero triangle** of height `k` reaching `k` columns left | **THEOREM** W3 |
| A wall makes the right half-plane's defect dynamics **closed**, and the left half-plane **slaved** to it | **THEOREM** W2, W5 |
| `d_t(±(t+p)) = 1` always — the defect field is **never empty**, for any lag, at any time | **THEOREM** D7 |
| Edge-aligned coordinates turn rule 30 into a **one-sided** rule `w_{t+1}(k) = w_t(k-2) XOR (w_t(k-1) OR w_t(k))` | **THEOREM** EA1 |
| Every fixed-width prefix therefore evolves **autonomously** and is **eventually periodic** | **THEOREM** EA2 |
| The centre column is the **diagonal** `x_t(0) = w_t(t)` | **THEOREM** EA3 |
| Measured: preperiod `T(K) ≈ 1.29 K`, periods `1,2,4,8,16` — so the diagonal **never leaves the transient** | COMPUTATIONAL OBSERVATION, `K ≤ 1280` |
| Zero-wall automaton, exhaustive `r ≤ 5`: maximal invariant set **non-empty at every radius** | **no bounded impossibility obtained** |
| Longest real wall over 83 lags × 30000 steps: **18** (at `p = 59`) | COMPUTATIONAL OBSERVATION |

## Two routes closed, and kept

* **G2** (support-edge defects obstruct the wall): the edge defects move
  *outward*, away from column 0. **Refuted as a route**, retained in the
  registry.
* **G4** (a wall forces a closed finite-state cycle incompatible with the seed):
  the automaton is nondeterministic so "cycle" is the wrong object; and it is
  seed-blind by construction, which Phase 1 §10.3 proves cannot work.
  **Partially refuted**, retained.

## One new target

**G5** — in edge-aligned coordinates, `liminf T(K)/K > 1`. It is the only
registry item that is *intrinsically seed-dependent* (all of G1–G4 are false or
vacuous without the seed, and Phase 1 proved seed-blindness is fatal), the only
one with half of it already a theorem (EA2), and the only one with a **live,
cheap falsification test**: a single `K` with `T(K) ≤ K` refutes it.

It is a reformulation, **not** a reduction — no route from G5 to a
contradiction has been constructed.

## Known limitations, stated up front

* **No literature was reconstructed.** Every primary source (Kopra arXiv:2202.13809,
  the TCS version, Jen 1990, arXiv:2604.00165) returned **HTTP 403** through
  this environment's proxy. `LITERATURE_GAP_MAP.md` records the failure and
  labels every attributed statement **UNVERIFIED — SNIPPET ONLY**. Nothing in
  Phase 2B depends on any of it. **This part of the brief was not completed.**
* **Automaton radii 6, 7, 8 were not computed** — `2^(4r+1)` states puts them at
  `3.4×10^7` to `8.6×10^9`. Recorded as NOT COMPUTED, not as "no result".
* **"Locally admissible but globally unreachable" is not decidable here.** Such
  states are reported as *not observed in the sampled range*.
* Every consequence of (H-WALL) is **vacuous if Problem 1 has the expected
  answer** and cannot be tested computationally.
* `p = 1` makes W3's hypothesis provably vacuous; `p = 2` never realised it with
  run ≥ 2. Preserved as negative results.

## Files

```
README.md                      this file
DEFECT_DYNAMICS.md             the exact defect equations (XOR/OR, ANF, case table)
ZERO_WALL_LEMMAS.md            W1-W7, consequences of the wall
DEFECT_GEOMETRY_RESULTS.md     real-orbit geometry over 83 lags
ZERO_WALL_AUTOMATON.md         finite-state wall analysis, r <= 5
EDGE_ALIGNED_DYNAMICS.md       EA1-EA3 and the transient/diagonal picture
LITERATURE_GAP_MAP.md          what could not be retrieved, and what would be needed
GLOBAL_LEMMA_REGISTRY.md       G1-G5 with evidence, counterexamples, tests
requirements.txt

defect_lab.py                  all derivations, checks and measurements
run_phase2b.py                 measurement driver (7 sections)
run_phase2b_tests.py           test battery
rule30_lab.py                  Phase 1 library (verified rule-30 engines)

results/
  phase2b_results.json         raw output
  phase2b_run.log              stdout of the measurement run
  phase2b_tests.log            stdout of the test battery
  defect_rows_p{1,7,27,1324,2750}.json   per-row geometry tables
```

## Scientific controls applied

* Every Boolean identity verified **exhaustively**, never by sampling, and then
  re-checked cell-by-cell against the real orbit.
* Real-orbit results and seed-free relaxations are kept in separate sections and
  separately labelled; the automaton section states on every line that it is a
  relaxation.
* **No bounded search is called UNSAT.** The automaton's non-empty invariant
  sets are reported as "no impossibility obtained"; `r ≥ 6` as "not computed".
* Negative and failed results preserved: G2, G4, the `p = 1` vacuity, the
  centre-word table with no signal, and the literature access failure.
* No statistical-regularity claim is used as evidence, and **no randomness
  tests were run** (per the brief).
