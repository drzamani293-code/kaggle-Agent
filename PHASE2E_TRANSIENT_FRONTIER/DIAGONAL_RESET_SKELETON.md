# Diagonal Reset Ancestry — Phase 2E §6

Where the diagonal cell `(t, t)` sits relative to the reset events that build
the transient frontier.

---

## 1. The reset skeleton

**Definition.** The *reset skeleton* is the set of points in the (time, level)
plane

```
    S  =  { ( τ_K , K )  :  level K is RESETTING }.
```

By Theorem B1 the level-`K` prefix settles at `τ_K + 1`, so `S` is exactly the
set of space-time points at which new transient is created. `|S| = 16394` for
`K ≤ 30000`; the levels not in `S` inherit their preperiod unchanged.

**Definition.** The *backward cone* of the diagonal cell `(t, t)` is, at
time `s ≤ t` (Phase 2D **Theorem DD1**, proved there):

```
    cone(t, s)  =  [ max(0, 2s - t) ,  min(t, 2s) ] .
```

---

## 2. The diagonal reads every level before that level resets

### THEOREM 6.1

> Let `K ≥ 18`. Then `T(K) > K`, so the diagonal cell `(K, K)` is evaluated
> **strictly before** level `K` settles, and in particular strictly before
> `τ_K` if level `K` is RESETTING.

*Status.* The inequality `T(K) > K` is a **BOUNDED OBSERVATION**, verified for
`18 ≤ K ≤ 30000`; it is Phase 2B's target G5 in its weakest form and is **not**
proved. `T(K) ≤ K` holds exactly for `K ∈ {0,…,17}`, and those 18 levels are
genuine counterexamples to the unqualified statement.

*Given* `T(K) > K`, the conclusion is immediate: `τ_K = T(K) - 1 ≥ K` on
RESETTING levels, so the diagonal at time `K` is at or before the reset.

**What this does not give.** It does not follow that the diagonal value is
"unconstrained", "random", or "aperiodic". It follows only that the diagonal is
never a read of already-settled cycle data — exactly the observation Phase 2D
made from the other direction, and exactly why the cycle-side theory of Phases
2C/2D does not reach the centre column.

---

## 3. Which reset events the diagonal's cone contains

A reset event `(τ_K, K)` lies in the cone of `(t,t)` iff `τ_K ≤ t` and

```
    max(0, 2 τ_K - t)  ≤  K  ≤  min(t, 2 τ_K).
```

Since `τ_K = T(K) - 1` and `T(K) > K` on the computed range, the upper
constraint `K ≤ 2τ_K` is automatic; the binding constraint is
`K ≥ 2 τ_K - t`, i.e. `t ≥ 2T(K) - K - 2`.

Measured (`frontier_lab.ancestry_skeleton`):

| `t` | reset events inside the cone | highest level reached | ratio `K_max/t` |
|---|---|---|---|
| 200 | 64 | 112 | 0.5600 |
| 600 | 221 | 403 | 0.6717 |
| 1500 | 518 | 971 | 0.6473 |
| 3000 | 1002 | 1856 | 0.6187 |

So the diagonal's ancestry contains the reset events of the lower part of the
level range — the highest level whose reset event is in the cone sits at
`0.56 t` to `0.67 t` across these four samples — and none above.
**BOUNDED OBSERVATION**, `t ≤ 3000`; the ratio is not monotone across the four
samples and **no limit is claimed**.

---

## 4. Where the cone stops touching the periodic region

At time `s`, the periodic frontier is at `K_per(s)`; the cone occupies
`[max(0,2s-t), min(t,2s)]`. The cone is *entirely transient* once
`2s - t > K_per(s)`.

| `t` | first `s` with the cone entirely transient | as a fraction of `t` |
|---|---|---|
| 200 | 157 | 0.7850 |
| 600 | 499 | 0.8317 |
| 1500 | 1236 | 0.8240 |
| 3000 | 2423 | 0.8077 |

This reproduces Phase 2D's `s ≳ 0.81 t` from the frontier side and with the
exact `K_per` table rather than the linear approximation. **BOUNDED
OBSERVATION**, `t ≤ 3000`, four samples, not monotone, no constant fitted.

Cell counts for `t = 3000`: the cone has `4 503 001` cells, of which
`1 416 350` (31.45 %) lie in the periodic region — so **68.5 % of the diagonal's
ancestry has never settled**, and the settled third is confined to the earliest
part of the cone. The same ratio at `t = 200, 600, 1500` is
`0.2945, 0.3208, 0.3244`.

---

## 5. What the skeleton would have to do to matter

The reset skeleton records *when levels settle*. The diagonal reads
*unsettled* cells. For the skeleton to constrain the diagonal one needs a
statement of the form

> the value `w_t(t)` is determined (or constrained) by the reset events in its
> cone

and no such statement is available. §7 measures the closest computable
approximation — how much the periodicity rewrite shrinks the diagonal's proof
DAG — and finds a constant-factor saving only.

**This is the same wall as Phase 1 §10.3, reached along a fourth route.** An
argument built from the settled part of the diagram is blind to the transient
that carries the seed, and the diagonal lives in the transient.
