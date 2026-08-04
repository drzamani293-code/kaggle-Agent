# Edge-Aligned Coordinates

The strongest structural result Phase 2B produced. It converts Phase 1's
*empirical* observation about the regular left region (§7.3 of
`PHASE1_AUDIT.md`) into a theorem, and explains exactly why the centre column
escapes it.

---

## 1. The change of coordinates

The single-cell orbit has a deterministic left support edge: `x_t(j) = 0` for
`j < -t`, and `x_t(-t) = 1` for every `t` (Phase 1 Facts F1, F2). Define

```
    w_t(k) = x_t(-t + k),      0 <= k <= 2t,      w_t(k) = 0 for k < 0 or k > 2t.
```

So `k` counts sites rightwards **from the left light-cone edge**, which is
`k = 0` at every time.

## 2. The update becomes one-sided — **THEOREM EA1**

> ```
>     w_{t+1}(k) = w_t(k-2)  XOR  ( w_t(k-1) OR w_t(k) ).
> ```

*Proof.* Substitute `j = -(t+1) + k` into the rule:
```
    w_{t+1}(k) = x_{t+1}(-(t+1)+k)
               = x_t(-(t+1)+k-1) XOR ( x_t(-(t+1)+k) OR x_t(-(t+1)+k+1) )
               = x_t(-t + (k-2)) XOR ( x_t(-t + (k-1)) OR x_t(-t + k) )
               = w_t(k-2) XOR ( w_t(k-1) OR w_t(k) ).                        ∎
```

*Index check.* At `k = 0`: `w_{t+1}(0) = w_t(-2) XOR (w_t(-1) OR w_t(0))
= 0 XOR (0 OR w_t(0)) = w_t(0)`, reproducing the frozen left edge `x_t(-t) = 1`.
At `k = 2t+2` (the new right edge): the arguments are `w_t(2t), w_t(2t+1),
w_t(2t+2) = w_t(2t), 0, 0`, giving `w_{t+1}(2t+2) = w_t(2t) = x_t(t) = 1`,
reproducing the frozen right edge. Both edges come out right.

*Verification.* `verify_edge_update(400, 300)` checks the identity cell by cell
against the real orbit: **50,397 cells, 0 mismatches.**

**The essential point: the new rule is ONE-SIDED.** `w_{t+1}(k)` depends only
on `w_t(k')` for `k' ∈ {k-2, k-1, k}` — never on `k+1`. In the original
coordinates rule 30 reads one cell to the right; after edge alignment that
dependence is exactly cancelled by the edge's leftward motion.

## 3. Prefixes evolve autonomously — **THEOREM EA2**

> For every `K >= 0` define `W_t^K = ( w_t(0), …, w_t(K) ) ∈ {0,1}^{K+1}`.
> Then there is a map `F_K : {0,1}^{K+1} → {0,1}^{K+1}` with
> `W_{t+1}^K = F_K(W_t^K)` for all `t >= K/2`.
> Consequently `(W_t^K)_t` is **eventually periodic**, with preperiod and period
> together at most `2^{K+1}`.

*Proof.* By EA1 each `w_{t+1}(k)` with `k <= K` is a function of
`w_t(k-2), w_t(k-1), w_t(k)`, all of index `<= K`. So the prefix determines its
own successor. An orbit of a map on a finite set is eventually periodic
(pigeonhole). ∎

This is a *theorem*, not an observation: the eventual periodicity of the
regular left region is forced by one-sidedness plus finiteness. No property of
rule 30 beyond EA1 is used.

## 4. The centre column is the diagonal — **THEOREM EA3**

> ```
>     x_t(0) = w_t(t).
> ```
> Hence the `p`-periodicity condition `x_{t+p}(0) = x_t(0)` becomes the
> **diagonal condition**
> ```
>     w_{t+p}(t+p) = w_t(t),
> ```
> and the defect field's wall becomes `d_t(0) = w_{t+p}(t+p) XOR w_t(t) = 0`.

*Proof.* `w_t(t) = x_t(-t + t) = x_t(0)`. ∎

*Verification.* Checked for `t = 1..800`: **exact**.

**Index warning.** The condition is *not* a relation between two entries of one
row. It compares entry `k = t` of row `t` with entry `k = t+p` of row `t+p`:
both the row index and the offset advance. Writing it as "`w_t(t+p)` versus
`w_t(t)`" would be wrong — those are two entries of the *same* row, which is a
different (and false) statement. The correct pairing is
`(t, t) ↔ (t+p, t+p)`.

## 5. Measured prefix periods

`prefix_period(K, 2600)` — first repeat of `W_t^K`, giving preperiod `T(K)` and
period `P(K)`:

| `K` | `T(K)` | `P(K)` | `T(K)/K` |
|---|---|---|---|
| 1 | 1 | 1 | 1.000 |
| 2 | 2 | 1 | 1.000 |
| 4 | 2 | 2 | 0.500 |
| 8 | 4 | 4 | 0.500 |
| 16 | 16 | 4 | 1.000 |
| 32 | 36 | 8 | 1.125 |
| 64 | 91 | 8 | 1.422 |
| 128 | 171 | 8 | 1.336 |
| 256 | 319 | 8 | 1.246 |
| 384 | 487 | 8 | 1.268 |
| **448** | 549 | **16** | 1.225 |
| 512 | 654 | 16 | 1.277 |
| 768 | 978 | 16 | 1.273 |
| 1024 | 1299 | 16 | 1.269 |
| 1280 | 1666 | 16 | 1.302 |

**COMPUTATIONAL OBSERVATION.** The periods observed are `1, 2, 4, 8, 16` — a
doubling hierarchy. `P(K) = 8` for `K` up to about 400 and `P(K) = 16` from
about 448 to at least 1280. The preperiod grows linearly, `T(K) ≈ 1.27 K` in
the measured range.

### 5.1 Independent cross-check against Phase 1

Phase 1 measured, in the original coordinates, that the diagonal shift
`(t,i) → (t+16, i-16)` holds out to a band of width `≈ 0.74 t` from the left
edge, while `τ = 8` held only in a band of *constant* width **400**. In
edge-aligned coordinates that shift is exactly `w_{t+16}(k) = w_t(k)`. So the
two statements must agree, and they do:

| Phase 1 (original coords) | Phase 2B (edge-aligned) | agreement |
|---|---|---|
| `τ = 8` band has constant width ≈ 400 | `P(K) = 8` exactly for `K ≲ 400` | ✔ |
| `τ = 16` band width `≈ 0.74 t` | `T(K) ≈ 1.27 K`, i.e. `K ≈ 0.79 T` | ✔ (see below) |
| slope drifts downward with `t` | ratio drifts with `K` | ✔ |

At `K = 1280`, `T = 1666`, so `K/T = 0.768`; Phase 1's fitted band width at
`t = 1666` is `≈ 0.765 · 1666`. The two independent measurements agree to
within 0.5%. This is a genuine cross-coordinate confirmation: the Phase 1
number was measured by scanning for matching runs in the original lattice, the
Phase 2B number by hashing prefixes in the transformed lattice.

## 6. Why the centre column escapes the periodic structure

Combining EA2 and EA3:

* the prefix `W_t^K` becomes periodic once `t >= T(K) ≈ 1.27 K`;
* the centre column at time `t` reads `w_t(t)` — it needs `K = t`;
* but `T(t) ≈ 1.27 t > t`.

> **COMPUTATIONAL OBSERVATION (central).** The diagonal that carries the centre
> column runs **strictly inside the transient** of the prefix dynamics, at every
> time. It never enters the eventually-periodic regime, because the preperiod
> grows faster than the diagonal advances (`1.27 > 1`).

This is, as far as Phase 2B can tell, the structural reason the centre column
resists the tools that settle the left region. The regular region *is*
eventually periodic — provably (EA2) — but the centre column samples the part
of each row that has not yet settled.

**Label discipline.** EA1, EA2, EA3 are **THEOREMS**. The constant `1.27`, the
period hierarchy `1,2,4,8,16`, and the claim `T(K)/K > 1` are **COMPUTATIONAL
OBSERVATIONS** over `K <= 1280`, `t <= 2600`. In particular:

* it is **not proved** that `T(K)/K > 1` for all `K`. If `T(K)/K` ever dropped
  below 1, the diagonal would enter the periodic regime and the centre column
  would be eventually periodic — i.e. **Rule 30 Prize Problem 1 would have the
  opposite answer to the expected one.** So this ratio is not a curiosity: the
  conjecture `liminf T(K)/K > 1` is *equivalent in spirit* to the problem, and
  is registered as **G5** in `GLOBAL_LEMMA_REGISTRY.md`;
* it is **not proved** that the period stays bounded by 16, nor that it doubles
  at a definite rate. The observed doubling points (`K ≈ 400` for 8→16) are
  measurements, and only two doublings were observed.

## 7. What this does *not* establish

* It does **not** show the centre column is aperiodic. EA2 gives eventual
  periodicity of every *fixed-width prefix*; the centre column is not a fixed
  prefix entry, and no argument here transfers.
* The finite-state map `F_K` has state space `2^{K+1}`, so EA2's bound on the
  period is astronomically weak; the measured periods (8, 16) are far smaller
  than the bound and that gap is unexplained.
* Nothing here uses the defect field. The connection between §6 and the
  zero-wall picture — whether a wall would force the diagonal into the periodic
  regime — is open and is registered as **G5**.
