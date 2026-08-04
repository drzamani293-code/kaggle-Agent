# Defect Geometry in the Real Orbit

Direct simulation of the single-cell orbit, forming `d_t(j) = x_{t+p}(j) XOR
x_t(j)` and measuring its geometry. **No randomness tests are used anywhere**
(per the brief); everything reported is a structural count or a proved
invariant.

Settings: `t_max = 1500` for per-row geometry (301 rows sampled per lag),
`t_max = 30000` for wall-run statistics, **83 lags** —
`p ∈ {1,…,64}` ∪ primes up to 1021 ∪ powers of two up to 2048 ∪ the six
distinct witness lags `{1, 2, 6, 27, 1324, 2750}` of the eight Phase 1 MB1-loc
witnesses.

Machine-readable output: `results/phase2b_results.json` (key `geometry`,
`wall_runs`, `crossing_events`) and per-row tables
`results/defect_rows_p{1,7,27,1324,2750}.json`.

---

## 1. The support edges — **THEOREM**, verified for every lag

> `d_t(+(t+p)) = 1`, `d_t(-(t+p)) = 1`, and `d_t(j) = 0` for `|j| > t+p`.

Proof in `DEFECT_DYNAMICS.md` (Corollary D7). Checked at every sampled row of
every one of the 83 lags: `support_edge_theorem_holds_for_all_lags = true`.

**Consequences that are certain, and one that is not.**

* The defect field is **never empty**, at any time, for any lag. The leftmost
  and rightmost defects are at exactly `∓(t+p)` — not merely bounded by them.
* The defect support is exactly the interval `[-(t+p), t+p]`, so it grows at
  speed 1 in both directions.
* **What this does *not* give:** the edge defects move *outward*, away from
  column 0. They do not approach the wall, and D7 alone therefore poses no
  obstruction to (H-WALL). Registry item **G2** was formulated on the hope that
  it would; §5 below records that hope as **not realised**.

## 2. Per-row geometry

| `p` | mean defect density | fraction of rows with `d_t(0)=0` | runs per row | runs covering column 0 |
|---|---|---|---|---|
| 1 | 0.50358 | 0.56146 | 375.5 | 0.439 |
| 2 | 0.50364 | 0.45515 | 375.9 | 0.545 |
| 3 | 0.50026 | 0.53821 | 374.7 | 0.462 |
| 7 | 0.49966 | 0.57475 | 378.4 | 0.425 |
| 16 | 0.49219 | 0.48173 | 374.3 | 0.518 |
| 27 | 0.49787 | 0.46512 | 386.8 | 0.535 |
| 59 | 0.49961 | 0.43854 | 405.5 | 0.561 |
| 64 | 0.49649 | 0.47176 | 404.9 | 0.528 |
| 127 | 0.50060 | 0.55482 | 438.4 | 0.445 |
| 256 | 0.50056 | 0.48837 | 510.6 | 0.512 |
| 1024 | 0.49992 | 0.48505 | 888.5 | 0.515 |
| 1324 | 0.49998 | 0.50498 | 1036.7 | 0.495 |
| 2048 | 0.50055 | 0.49502 | 1398.9 | 0.505 |
| 2750 | 0.50097 | 0.51827 | 1754.8 | 0.482 |

**COMPUTATIONAL OBSERVATION.** Defect density sits at `0.500 ± 0.005` for every
lag examined. The number of maximal defect runs per row grows with `p` simply
because the support width is `2(t+p)+1`; per unit width the run count is
essentially constant.

**COMPUTATIONAL OBSERVATION.** "Runs covering column 0" per row is `≈ 0.5` for
every lag — i.e. the defect component that contains column 0 exists roughly
half the time, which is just the restatement of `d_t(0) = 1` about half the
time. There is no lag at which column 0 is unusually protected.

## 3. Connected components and crossing

Components are taken as maximal runs of consecutive defect sites within a row
(the natural notion for a radius-1 rule, since D5 gives speed-≤1 propagation).

* **Column-0 crossing.** A run "crosses" column 0 if its interval contains 0.
  Across all 83 lags the crossing rate per row is `0.425`–`0.561`, tightly
  centred on `0.5`.
* **Lifetimes.** A component touching column −1, 0 or 1 persists, in the sense
  of `d_t(0) = 1` holding, for a geometrically distributed number of steps; the
  complementary quantity (the *wall* runs) is tabulated in §4 and is the more
  informative one for this project.

## 4. Wall runs — how long the real orbit keeps `d_t(0) = 0`

A *wall run* is a maximal interval of consecutive `t` with `d_t(0) = 0`: a
finite instance of exactly the object (H-WALL) posits to be infinite.
`t_max = 30000`, all 83 lags.

| `p` | number of wall runs | longest | mean |
|---|---|---|---|
| 1 | 7472 | 15 | 2.001 |
| 2 | 7436 | 14 | 2.025 |
| 7 | 7505 | 11 | 1.999 |
| 16 | 7580 | 12 | 1.977 |
| 27 | 7471 | 14 | 2.006 |
| 64 | 7521 | 18 | 2.018 |
| 1024 | 7505 | 16 | 1.990 |
| 1324 | 7553 | 12 | 1.993 |
| 2750 | 7554 | 15 | 1.978 |

Length histogram at `p = 7`: 3782 runs of length 1, 1833 of 2, 911 of 3, 499 of
4, 266 of 5, 106 of 6, 46 of 7, 34 of 8, 18 of 9, 7 of 10, 3 of 11.

**Headline numbers.** Over all 83 lags: **longest wall = 18**, attained at
`p = 59`; mean of the per-lag longest wall = 14.07; mean wall length = 1.999
for every lag.

**COMPUTATIONAL OBSERVATION, and the warning that goes with it.** The observed
wall lengths halve with each additional step, exactly as an unbiased
independent model would predict, and the longest observed wall in `30000`
steps × 83 lags is 18. **This is not evidence that no infinite wall exists.**
It is the same situation as Phase 1 §9.1: a sequence with an eventual period
larger than the sample would show precisely this. The numbers bound nothing.

## 5. G2 assessed — the support-edge idea does not work

Registry item **G2** proposed that the deterministic support-edge defects (D7)
might be incompatible with an infinite wall. The measurements settle the shape
of the question negatively:

* the edge defects are at `±(t+p)` and move **outward** at speed 1;
* the defect field near column 0 has density `≈ 0.5` and shows no interaction
  with the edges at the times sampled;
* a wall at column 0 constrains a region the edges are receding from.

**Conclusion: G2 as stated is not a viable route.** It is retained in the
registry with status *refuted as a strategy* (not as a mathematical statement —
it was never a theorem). Preserving it is the point: this is a failed idea, and
the brief requires failed results to be kept.

## 6. Centre words and crossing events

For `p = 7`, `k = 5`: the conditional fraction of `t` with `d_t(0) = 1` given
the centre word `x_t(0)…x_{t+4}(0)`, over all 32 words. Extremes:

| centre word | wall (`d=0`) | defect (`d=1`) | defect fraction |
|---|---|---|---|
| `10011` | 352 | 275 | 0.4386 |
| `01001` | 324 | 272 | 0.4564 |
| … | | | |
| `10110` | 316 | 361 | 0.5332 |
| `00100` | 256 | 317 | 0.5532 |

**COMPUTATIONAL OBSERVATION.** The spread is `0.44`–`0.55` across all 32 words,
with sample sizes of a few hundred each. There is no word that strongly
predicts a wall. **No inference is drawn from this** — with 32 words examined
and no correction, a spread of this size is unremarkable, and by the brief's
own rule statistical regularity would prove nothing even if it were larger.
The table is reported because it is a negative result and negative results are
kept.

## 7. Summary of what §§1–6 establish

| Claim | Label |
|---|---|
| `d_t(±(t+p)) = 1` and support `= [-(t+p), t+p]`, all lags | **THEOREM**, verified |
| Defect density `≈ 0.5`, uniformly in `p` | COMPUTATIONAL OBSERVATION |
| Column-0 crossing rate `≈ 0.5`, uniformly in `p` | COMPUTATIONAL OBSERVATION |
| Longest real wall over 83 lags × 30000 steps is 18 | COMPUTATIONAL OBSERVATION |
| No centre word strongly predicts a wall | COMPUTATIONAL OBSERVATION (negative) |
| The support-edge route (G2) does not obstruct a wall | strategy assessment, negative |
| Anything about whether an infinite wall exists | **NOTHING** |
