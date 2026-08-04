# Diagonal Bridge Attempt

The centre column is the diagonal `x_t(0) = w_t(t)` (Theorem EA3). The prefix
tower is completely understood (Theorems P1, P2, S1–S4). This document asks
whether the two can be connected — and reports that the one bridge we found is
**blocked by a measurement**, in the direction opposite to what Phase 2B
suggested.

---

## 1. The exact transfer identity — **THEOREM D1**

> For every `K` and every `t` with
> ```
>     T(K)  <=  t  <=  K - P(K),
> ```
> one has
> ```
>     x_{t + P(K)}(0)  =  x_t( P(K) ).
> ```

*Proof.* Prefix periodicity gives `w_{t+P}(k) = w_t(k)` for all `t >= T(K)` and
all `k <= K`, where `P = P(K)`. Choose `k = t + P`; this is legitimate exactly
when `t + P <= K`. Then

```
    x_{t+P}(0) = w_{t+P}(t+P)          [EA3 at time t+P]
               = w_t(t+P)              [prefix periodicity, k = t+P <= K]
               = x_t(-t + t + P)       [definition of w]
               = x_t(P).                                                    ∎
```

**Index check.** The left side reads the diagonal at time `t+P`; the right side
reads row `t` at site `P`. Both row index and offset advance on the left, which
is why the identity is *not* a statement about one row. (Phase 2B warned
against the mis-statement "`w_t(t+p)` versus `w_t(t)`"; D1 is the correct
pairing.)

**D1 is not vacuous.** Its range is non-empty for exactly eight values,
`K ∈ {4, 5, 6, 7, 8, 9, 10, 12}`, giving **16 `(K,t)` pairs**, and it was
verified against the directly simulated orbit at all 16: **exact**.

**But it is vacuous everywhere else.** For every `K >= 13` up to 30000,
`T(K) > K - P(K)`, so no admissible `t` exists. The identity has content only
in the tiny-`K` corner.

## 2. The bridge — **THEOREM D2 (conditional)**

> Suppose there is a `K0` and a constant `P` such that
> ```
>     P(K) = P     and     T(K) <= K - P        for every K >= K0.       (HYP-D)
> ```
> Then the centre column of rule 30 is **not** eventually periodic.

*Proof.* Fix `t >= K0`. Apply D1 with `K = t + P`: the hypothesis `T(K) <= K-P`
reads `T(t+P) <= t`, and `t <= K - P = t` holds, so D1 applies and gives
`x_{t+P}(0) = x_t(P)`. As `t >= K0` was arbitrary,

```
    x_{t+P}(0) = x_t(P)        for every t >= K0.                          (*)
```

Now suppose the centre column were eventually periodic with period `q`. By (*),
`x_t(P) = x_{t+P}(0)`, and `t ↦ x_{t+P}(0)` is eventually `q`-periodic because
`t ↦ x_t(0)` is. Hence column `P` is eventually periodic. Since `P >= 1`,
columns `0` and `P` are two **distinct** eventually periodic columns,
contradicting Phase 1 **Theorem W2′** (at most one column of the single-cell
diagram is eventually periodic). ∎

D2 is a genuine reduction: it converts a purely *combinatorial* statement about
the prefix transients into the aperiodicity of the centre column, using only
already-proved machinery (D1, EA3, W2′).

## 3. The bridge is blocked — and this corrects Phase 2B

**(HYP-D) is false on every value measured.** `P(K) = 16` is constant for
`400 <= K <= 30000`, so the first half holds; but the second half requires
`T(K) <= K - 16`, whereas

```
    T(K)  ≈  1.34 K        for large K            (exact values in the CSV).
```

`T(K) > K` for every `18 <= K <= 30000`, hence `T(K) > K - P(K)` throughout, and
(HYP-D) fails at every `K` in range.

> **This is the substantive finding of Phase 2C, and it points the other way
> from Phase 2B.**
>
> Phase 2B (`EDGE_ALIGNED_DYNAMICS.md` §6, registry item G5) framed
> `T(K)/K > 1` as the interesting structural fact and suggested that
> `T(K) <= K` "would be the shape of a positive answer to the periodicity
> question". **That framing was wrong, and Phase 2C corrects it.** The direction
> is the opposite: `T(K) <= K - P(K)` eventually is exactly the hypothesis that
> would *prove aperiodicity* (D2). The measured `T(K) > K` therefore **blocks
> the only bridge we have**, rather than supporting the expected answer.
>
> Restated plainly: **`T(K) > K` is not evidence for aperiodicity.** It is not
> evidence for periodicity either — it is simply the condition under which this
> particular route yields nothing.

## 4. Translating an assumed centre period

The brief asks to translate `w_{t+p}(t+p) = w_t(t)` into relations among the
nested prefix systems. Doing so precisely:

Assume the centre column is eventually `p`-periodic from `T*`. Then for
`t >= T*`,

```
    w_{t+p}(t+p) = w_t(t).                                                 (C)
```

Two observations, both negative:

**(a) (C) relates different prefix systems.** The left side is coordinate `t+p`
of the `(t+p)`-prefix at time `t+p`; the right side is coordinate `t` of the
`t`-prefix at time `t`. There is no fixed `K` for which (C) is a statement
*inside* the `K`-prefix system. The projection tower (P1) only ever relates
`K` to `K-1` at the **same** time, so (C) is transverse to the whole tower.
This is the structural reason no contradiction drops out.

**(b) The one place the tower does speak is D1, and (HYP-D) is how it would
speak.** Combining (C) with D1 (where D1 applies) gives
`x_t(P) = x_{t+P}(0) = x_t(0)` — column `P` agreeing with column 0 pointwise.
That is the contradiction exploited in D2. It needs D1's range to be non-empty
for large `K`, i.e. exactly (HYP-D).

**Searched and not found:** a variant of D2 whose hypothesis is compatible with
`T(K) > K`. We tried:

* using `k = t + jP` for `j >= 2` — needs `t + jP <= K`, a *stronger*
  requirement, so strictly worse;
* using two different `K` with different periods — blocked because `P(K)` is
  constant (16) across the whole measured range, so there is no second period
  to play against the first;
* running D1 backwards (from a hypothetical centre period to a statement about
  `T(K)`) — this yields only the contrapositive of D2, i.e. "if the centre
  column is eventually periodic then `T(K) > K - P(K)` for all large `K`",
  which the measurements already satisfy. **No information.**

That last item deserves emphasis: the contrapositive of D2 says periodicity
*implies* `T(K) > K - P(K)` eventually. The data satisfy that implication's
conclusion. So the measurement is **consistent with periodicity**, and is
equally consistent with aperiodicity. It discriminates nothing.

## 5. Status

| item | label |
|---|---|
| D1 transfer identity | **THEOREM**, verified at all 16 `(K,t)` pairs where it has content |
| D1 is vacuous for `K >= 13` | **BOUNDED OBSERVATION** (`K <= 30000`) |
| D2 bridge | **THEOREM (conditional)** — a genuine reduction |
| (HYP-D) | **FALSE on `18 <= K <= 30000`** — the route is blocked |
| "`T(K) > K` supports aperiodicity" | **RETRACTED** (Phase 2B G5 framing) |
| a D2 variant compatible with `T(K) > K` | **searched, not found** |

**No contradiction with the skew-product structure was obtained, and none is
claimed.** Phase 2C ends with the diagonal still transverse to the tower.

## 6. What would revive the route

Any of the following would make D2 usable, and each is a concrete target:

1. A proof that `P(K) -> infinity` fast enough that `K - P(K)` overtakes
   `T(K)` — but `P` grows *slower* than `T`, so this looks hopeless as stated
   (and would in any case need `P(K) > K - T(K) ≈ -0.34K`, i.e. nothing:
   the inequality `T(K) <= K - P(K)` gets *harder* as `P` grows). **Dead.**
2. A proof that `T(K) <= K - c` for some constant `c` — contradicted by
   measurement up to 30000, so it would have to fail for small `K` and hold
   asymptotically, which the observed upward drift of `T(K)/K` makes
   implausible. **Dead unless the drift reverses.**
3. A different identity of D1 type reading the diagonal off a *slower-moving*
   line — e.g. relating `x_t(0)` to `w_t(k)` for `k` growing more slowly than
   `t`. This is the only direction not yet excluded, and it is open.

Item 3 is the recommended next target if this route is pursued at all. Phase 2C
does not pursue it.
