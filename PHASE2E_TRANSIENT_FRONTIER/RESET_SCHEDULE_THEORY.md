# Reset Schedule Theory — Phase 2E §3

The brief asks: with `ρ(K) = max{ t < T(K) : w_t(K-1) = 1 }`, is `T(K)` usually
exactly `ρ(K) + 1`?

**Answer: on RESETTING levels it is *always* exactly `ρ(K)+1`, and that is a
theorem, not a tendency. On INHERITING levels it holds about 47.6 % of the time
and is a coincidence with no mechanism behind it.** Aggregating the two classes
into one "usually" figure (76.2 %) is misleading, and this section keeps them
apart.

---

## 1. The theorem

### THEOREM 3.1

> If level `K` is RESETTING then `ρ(K)` is defined and `T(K) = ρ(K) + 1 = τ_K + 1`.

*Proof.* By Theorem B1, `T(K) = τ_K + 1` and `w_{τ_K}(K-1) = 1` with
`τ_K < T(K)`. So `τ_K` belongs to the set `{ t < T(K) : w_t(K-1) = 1 }`, whence
`ρ(K) ≥ τ_K`. Conversely `ρ(K) ≤ T(K) - 1 = τ_K`. ∎

### PROPOSITION 3.2 (general identity, both classes)

> `T(K) = ρ(K) + 1` **iff** `w_{T(K)-1}(K-1) = 1`.

*Proof.* Immediate from the definition of `ρ` as a maximum. ∎

So the question "is `T(K) = ρ(K)+1`?" is exactly the question "is the last step
before settling a forcing step?". Theorem 3.1 says: on RESETTING levels, yes by
construction. Proposition 3.2 says: on INHERITING levels the question is about
an unrelated coordinate value at an unrelated time.

### PROPOSITION 3.3 (when `ρ` is undefined)

> `ρ(K)` is undefined iff `w_t(K-1) = 0` for every `t < T(K)`.

Measured for `K ≤ 30000`: exactly `K ∈ {4, 5, 6, 7, 8}`. Two mechanisms, both
checkable by hand:

* `K = 4, 5, 6`: `T(K) = 2` and the support bound `w_t(k) = 0` for `k > 2t`
  makes coordinate `K-1` identically zero before time `2`.
* `K = 8`: coordinate `7` is **permanently** zero (Phase 2D, a theorem).
* `K = 7`: `T(7) = 2` and coordinate `6` is zero at `t = 0, 1`.

These five are reported, not discarded.

---

## 2. Measurements

`run_phase2e.py` §3, `K ≤ 30000`, 29999 levels.

| quantity | value |
|---|---|
| `T(K) = ρ(K)+1`, all levels with `ρ` defined | 22864 / 29994 = 0.7622 |
| `T(K) = ρ(K)+1`, RESETTING levels (16394) | **16394 / 16394 = 1.0000** — Theorem 3.1 |
| `T(K) = ρ(K)+1`, INHERITING levels (13600 with `ρ` defined) | 6470 / 13600 = 0.4757 |
| `ρ(K)` undefined | 5 levels (`K = 4,5,6,7,8`) |

Histogram of the gap `T(K) - ρ(K) - 1`, over the 29994 levels where `ρ` is
defined:

```
    gap    0     1     2    3    4    5    6   7   8   9  10  11  12  13
    count 22864 3728  1663  879  432  198  111  56  36  13   7   4   1   2
```

The zero bucket splits exactly as `22864 = 16394 + 6470`: the first summand is
forced by Theorem 3.1, the second is the set of INHERITING levels at which
`w_{T(K)-1}(K-1) = 1` happens to hold (Proposition 3.2). The per-level split is
in `RESET_SCHEDULE_RESULTS.csv`.

**So the honest answer to the brief's question is not "usually".** The aggregate
76 % is the average of a theorem (100 %) and a coincidence (48 %), and the
coincidence rate is close to what a single unbiased bit would give — which is
all Proposition 3.2 says it is.

---

## 3. Reset increments

For a RESETTING level, `δ(K) := T(K) - T(K-1) = τ_K + 1 - T(K-1)`, i.e. one plus
the **wait** `τ_K - T(K-1)` for the first `1` in coordinate `K-1` at or after
`T(K-1)`.

Wait histogram (16394 RESETTING levels, `K ≤ 30000`):

```
    wait    0     1     2     3    4    5    6   7   8   9  10  11  12  13  14
    count 4386  6228  2825  1458  733  375  200  84  48  32  11   2   6   5   1
```

(The increment histogram is this table shifted by one: `δ = 1` at 4386 levels,
`δ = 2` at 6228, …, `δ = 15` at 1.)

Increment histogram (`δ = wait + 1`), mean `2.451873`, maximum `15`.

### THEOREM 3.4 (increment bound)

> If coordinate `K-1` is not eventually zero, then `δ(K) ≤ P(K-1)`, hence
> `T(K) - T(K-1) ≤ P(K-1)` for every level.

*Proof.* Coordinate `K-1` is `P(K-1)`-periodic from `T(K-1)` (Theorem 1.1 at
level `K-1`). If it is not eventually zero, its cycle word contains a `1`, so
some `t ∈ [T(K-1), T(K-1)+P(K-1))` has `w_t(K-1) = 1`; hence
`τ_K ≤ T(K-1)+P(K-1)-1` and `δ(K) ≤ P(K-1)`. Non-COLLAPSING levels have
`δ = 0` by B2/B3. ∎

*Verified:* 0 violations over `K ≤ 30000`; the largest observed increment is
`15` against the bound `P = 16`, so the bound is **tight to within one**.

The hypothesis "coordinate `K-1` is not eventually zero" fails for exactly
`K-1 ∈ {2, 7, 28, 399}` in the computed range — a **BOUNDED OBSERVATION**
(Phase 2D). It is **not** claimed that no further eventually-zero coordinate
exists; Theorem 3.4 is stated conditionally for that reason.

---

## 4. Spacing of the reset events

Gaps between consecutive RESETTING levels, `K ≤ 30000`:

```
    gap     1     2     3    4    5   6   7  8  9  10  12
    count 5928  8786  830  420  325  62  26  7  3   3   3
```

maximum gap **12**, mean gap **1.8299**.

### What this does and does not say

* **BOUNDED OBSERVATION.** No run of more than 12 consecutive INHERITING levels
  occurs for `K ≤ 30000`.
* **NOT a theorem.** Nothing here bounds the gap for all `K`. A long run of
  INHERITING levels is not excluded by any argument in this package; it would
  simply mean a long stretch on which the transient stops growing, and
  `T(K)/K` would dip.
* The gap histogram is emphatically **not** fitted to any distribution, and no
  asymptotic constant is extracted from it. Per the brief, the mean gap
  `1.8299` is reported as a number measured at `K = 30000`, and nothing is
  extrapolated from it.

---

## 5. What the reset schedule would have to do

`T(K)/K = d(K)·m(K)` (Corollary B5). At `K = 30000`, `d = 0.5465` and
`m = 2.4519`. To make `T(K)/K > 1` provably one needs a **joint** lower bound on
the density of RESETTING levels and the mean wait — and the two are coupled:
a level is RESETTING iff its defect bit is `1`, and the wait is determined by
where the next `1` sits in the previous coordinate's cycle word. Neither factor
has a proved lower bound here.

Two specific facts that a proof would have to supply, neither of which Phase 2E
obtains:

1. a positive lower bound on the density of levels with defect bit `D(K) = 1`;
2. a lower bound on the mean wait `τ_K - T(K-1)` that survives conditioning on
   `D(K) = 1`.

Both are registered as open in `TRANSIENT_LEMMA_REGISTRY.md` (F1, F2).

---

## 6. The complete reset schedule (Phase 2E brief §3)

§§1–5 use only `ρ(K)` and `τ(K)`. The brief asks for the **complete** sequence
of reset times of each level, i.e. every `t` with `w_t(K-1) = 1`, since at each
such `t` the fibre map `u ↦ a_t XOR (1 OR u)` is constant and the level forgets
its past.

**Definition.** `Reset(K) = { t : w_t(K-1) = 1 }`, and
`σ(K) = min Reset(K)`, `τ(K) = min (Reset(K) ∩ [T(K-1), ∞))`,
`ρ(K) = max (Reset(K) ∩ [0, T(K)))`.

### THEOREM 6.1 (the schedule is eventually periodic, and how)

> `Reset(K) ∩ [T(K-1), ∞)` is the union of `n_K` arithmetic progressions of
> common difference `P(K-1)`, where `n_K` is the number of `1`s in the base
> cycle word. It is non-empty iff level `K` is COLLAPSING.

*Proof.* Coordinate `K-1` is `P(K-1)`-periodic from `T(K-1)` (Theorem 1.1 at
level `K-1`); the positions of its `1`s within one period are the `n_K` residues.
∎

So the *whole* reset schedule of a level, from `T(K-1)` onwards, is one 16-bit
cycle word plus a phase — and the entire content of the transient theory is in
the finitely many resets **before** `T(K-1)`, which the tower does not
summarise.

### THEOREM 6.2 (what the first reset controls)

> `σ(K)` is exactly the fibre dependence horizon `R_dep(K)` of
> `TRANSIENT_DEFINITIONS.md` §4: after `σ(K)`, coordinate `K` no longer depends
> on its own initial value.

Proved there (Theorem 4.1), verified two independent ways.

### Measured, first 2000 levels

| | count |
|---|---|
| `σ(K) < τ(K)` | 1988 |
| `σ(K) = τ(K)` | 8 |
| `τ(K) = ρ(K)` (exactly the RESETTING levels) | 1078 |
| `ρ(K) < τ(K)` (the INHERITING levels) | 914 |
| `σ(K) = ρ(K)` | 6 |

**`σ`, `τ` and `ρ` are three distinct times.** The near-universal strict
inequality `σ < τ` says the level has already forgotten its own initial
condition long before the base settles — which is precisely why the transient
that remains is *inherited from below*, and why Theorem A decomposes `T(K)`
into coordinate preperiods rather than into anything local to level `K`.

### The formula relating T, ρ, T(K-1), P(K-1) — as far as it goes

Collecting Theorems 3.1, 3.4 and 6.1:

```
    RESETTING :   T(K) = ρ(K) + 1 = τ(K) + 1,      T(K-1) < T(K) ≤ T(K-1) + P(K-1)
    INHERITING:   T(K) = T(K-1),                   ρ(K) unconstrained by T
```

and there is **no formula** for which of the two holds that does not read the
defect bit `D(K)` — see `FRONTIER_DYNAMICS.md` §5, where that is proved rather
than merely observed.
