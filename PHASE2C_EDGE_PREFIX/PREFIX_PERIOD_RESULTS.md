# Exact Prefix Periods — Results

Exact `T(K)` and `P(K)` for **every `K` from 0 to 30000** (the brief asked for
at least 5000). Full table: `PREFIX_PERIOD_TABLE.csv` (30001 rows).
Raw output: `results/phase2c_results.json`, `results/phase2c_run.log`.
Total run time **58.6 s**.

---

## 1. Why these values are exact, not estimates

`F_K` is a **deterministic** map on the finite set `{0,1}^{K+1}`. For a
deterministic orbit, a single coincidence `W_T^K = W_{T+Q}^K` implies
periodicity forever, and the *first* such coincidence gives precisely the tail
`rho` and cycle length `lambda`. So

```
    P(K) = min { Q : exists t with W_t^K = W_{t+Q}^K },
    T(K) = min { t : W_t^K = W_{t+P(K)}^K },
```

are the true values, provided the simulation reaches the first repeat — which
it does for every `K <= 30000` (`undetermined K: 0`). No `2^{K+1}` state space
is ever stored: the algorithm keeps only the `T_max+1` row integers.

## 2. Three independent computations, cross-checked

| algorithm | principle | range | agrees |
|---|---|---|---|
| **first-repeat scan** | lowest-set-bit of `W_{t+Q} XOR W_t`, two-pointer sweep over `K`; `Q` over powers of two (justified by **S1**) | `K <= 30000` | — (primary) |
| **hash reference** | store every prefix state in a dict, stop at the first repeat | `K <= 400` | **yes** |
| **incremental skew-product** | uses **S1** (`P(K) ∈ {P, 2P}`) and **S2** (`T(K) <= T(K-1)+P(K-1)`) to examine only an `O(P)` window per `K` | `K <= 1500` | **yes** |

Plus a fourth, direct check at 13 sampled `K` (`0, 1, 5, 17, 18, 50, 400, 401,
1000, 2500, 5000, 10000, 12000`) verifying all three of: periodicity from `T`
over the entire simulated range; **minimality of `T`**; **minimality of `P`**
(no proper power-of-two divisor works). **All pass.**

A methodological note kept for the record: the first version of the scan used a
*suffix-minimum* criterion ("`W_{t+Q} = W_t` for all `t >= T`") with a sentinel
for the empty suffix. That sentinel made short ranges succeed spuriously and
produced `T(K) = T_max, P(K) = 1` for `K >= 4`. It was caught immediately by
disagreement with the hash reference, and replaced by the first-repeat
criterion, which is both exact and cheaper. **Preserved as a failed method.**

## 3. The E-statements

| | statement | verdict |
|---|---|---|
| **E1** | `P(K)` is always a power of two | **THEOREM** (S1) — 0 counterexamples over `K <= 30000`, as required |
| **E2** | `P(K)` divides `P(K+1)`, and `P(K+1) <= 2 P(K)` | **THEOREM** (S1 + P1a) — 0 counterexamples |
| **E3** | `T(K)` is nondecreasing | **THEOREM** (P1a) — 0 counterexamples |
| **E4** | `T(K) > K` | **FALSE as stated.** Exactly **18 counterexamples**: `K = 0, 1, ..., 17`. Holds for every `18 <= K <= 30000`. Weakest provable form: `T(K) > floor(K/2) - P(K)` (**THEOREM S4**) |
| **E5** | `T(K) - K` is unbounded | **CONJECTURE**, supported: `T(K)-K` ranges from **34** (min over `K >= 100`) to **10202**, and equals **10197** at `K = 30000`, growing like `0.34 K` |

### 3.1 E4's counterexamples, in full

```
K :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
T :  0  1  2  2  2  2  2  2  2  5  6  8  8 10 11 13 16 16
```

`T(K) <= K` for all of these. The first `K` with `T(K) > K` is **18**
(`T(18) = 20`). The corrected statement — **E4′: `T(K) > K` for all
`18 <= K <= 30000`** — is a **BOUNDED OBSERVATION**, not a theorem, and the
gap to the provable bound `K/2 - P(K)` is a factor of about 2.7.

### 3.2 Measured values

| `K` | `T(K)` | `P(K)` | `T(K)/K` |
|---|---|---|---|
| 10 | 6 | 4 | 0.600 |
| 18 | 20 | 4 | 1.111 |
| 100 | 134 | 8 | 1.340 |
| 400 | 501 | 16 | 1.2525 |
| 1000 | 1275 | 16 | 1.2750 |
| 5000 | 6646 | 16 | 1.3292 |
| 10000 | 13315 | 16 | 1.3315 |
| 20000 | 26839 | 16 | 1.3419 |
| 30000 | 40197 | 16 | 1.3399 |

`T(K)/K` drifts slowly upward and sits near **1.34** at the top of the range.
This is consistent with Phase 1's independent measurement in the *original*
coordinates (regular-region band width `≈ 0.74 t`, i.e. `K/T ≈ 0.74`,
`T/K ≈ 1.35`) and with Phase 2B's `1.29` over the much shorter range
`K <= 1280`. Three independent measurements, two coordinate systems.

## 4. The period hierarchy

**Only four doublings occur up to `K = 30000`:**

| `P(K)` | first `K` with this period |
|---|---|
| 1 | 0 |
| 2 | **3** |
| 4 | **8** |
| 8 | **29** |
| 16 | **400** |

`P(K) = 16` for every `400 <= K <= 30000`. The doubling points
`3, 8, 29, 400` grow with ratios `2.7, 3.6, 13.8` — accelerating, with no
apparent pattern from four data points. **No fifth doubling was found below
K = 30000**, and nothing here predicts where one occurs, or whether `P` is
bounded.

> **CONJECTURE (open, weakly supported).** `P(K) -> infinity`. Four data points
> is not evidence; the alternative `P(K) = 16` for all `K >= 400` is equally
> consistent with everything measured. **Falsification test:** extend the table;
> a fifth doubling point would support unboundedness, its absence at, say,
> `K = 10^6` would make boundedness look plausible. Neither is currently known.

`S1` guarantees the hierarchy is exactly a doubling one — no other jumps are
possible. That part is a theorem.

## 5. Reproduce

```bash
python3 run_phase2c.py --k-max 30000     # ~60 s, writes the CSV and JSON
python3 run_phase2c_tests.py             # re-derives and cross-checks
```

`PREFIX_PERIOD_TABLE.csv` columns: `K, T_K, P_K, T_minus_K, T_over_K`.

## 6. What these numbers do *not* show

* **Nothing is claimed for `K > 30000`.** E4′ and E5 are bounded observations.
  S1–S4 are the only all-`K` statements, and they are proved.
* **`T(K) > K` is not evidence that the centre column is aperiodic.** In fact
  it is the direction that *blocks* the only bridge Phase 2C found — see
  `DIAGONAL_BRIDGE_ATTEMPT.md` §4. This corrects the framing in Phase 2B's G5.
* The gap between the proved lower bound `K/2 - P(K)` and the observed `1.34 K`
  is unexplained.
