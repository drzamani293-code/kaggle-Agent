# The Exact Recurrence for T(K) — Phase 2E §2

Phase 2D proved the **inequality** `T(K) ≤ max(T(K-1), τ_K + 1)` (result R3).
This section turns it into an **equality with a necessary and sufficient
condition**, and classifies every level accordingly.

---

## 1. The defect bit

Fix `K ≥ 2` and write

```
    p    = P(K),        T_-  = T(K-1),
    a_t  = w_t(K-2),    c_t  = w_t(K-1),    u_t = w_t(K),
    u_{t+1} = a_t XOR ( c_t OR u_t ).
```

**Definition.** The *defect bit* of level `K` is

```
    D(K)  :=  w_{T_-}(K)  XOR  w_{T_-+p}(K)      ( = u_{T_-} XOR u_{T_-+p} ).
```

It is one bit, read off the orbit at two explicit times.

---

## 2. The defect propagation lemma

### LEMMA 2.1

> Put `D_s := u_s XOR u_{s+p}` for `s ≥ T_-`. Then for every `s ≥ T_-`:
> * if `c_s = 0` then `D_{s+1} = D_s`;
> * if `c_s = 1` then `D_{s+1} = 0`.

*Proof.* By Theorem 1.1 applied at level `K-1`, the coordinates `K-2` and `K-1`
are `p`-periodic from `T_-` (their minimal periods divide `P(K-1) | p`, and
their coordinate preperiods are `≤ T(K-1) = T_-`). So `a_{s+p} = a_s` and
`c_{s+p} = c_s` for `s ≥ T_-`.

If `c_s = 0`: `u_{s+1} = a_s XOR u_s` and `u_{s+p+1} = a_s XOR u_{s+p}`, so the
XOR is preserved.
If `c_s = 1`: `u_{s+1} = a_s XOR 1` and `u_{s+p+1} = a_s XOR 1`, so both sides
are the same constant and `D_{s+1} = 0`. ∎

This is the exact sense in which a collapse **erases**: it erases the *lag-`p`
discrepancy*, not just the fibre value. Phase 2D's R1 is the special case that
the value itself becomes `1 XOR a_τ`.

### COROLLARY 2.2

> `D_s = D(K)` for `T_- ≤ s ≤ τ_K`, and `D_s = 0` for `s ≥ τ_K + 1`
> (with `τ_K = +∞`, i.e. the second clause vacuous, when the level is not
> COLLAPSING).

*Proof.* By definition of `τ_K` we have `c_s = 0` for `T_- ≤ s < τ_K`, so the
first clause of Lemma 2.1 applies throughout that range; then `c_{τ_K} = 1`
gives `D_{τ_K+1} = 0`, and zero is preserved by both clauses. ∎

---

## 3. THEOREM B — the exact recurrence

### THEOREM B1 (levels with `P(K) = P(K-1)`)

> Let `K` satisfy `P(K) = P(K-1) = p`. Then
>
> ```
>     D(K) = 0   ⟹   T(K) = T(K-1)                     (INHERITING)
>     D(K) = 1   ⟹   T(K) = τ_K + 1  and  τ_K is defined  (RESETTING)
> ```
>
> and these are the only two possibilities. In particular
> `T(K) > T(K-1)` **iff** `D(K) = 1`.

*Proof.* By Theorem 1.5, `T(K) = max( T_-, R(K) )` where `R(K) = r_p(K)` and
`R(K)` is the least `t` such that `D_s = 0` for all `s ≥ t`.

*Case `D(K) = 0`.* By Corollary 2.2, `D_s = 0` for every `s ≥ T_-`
(it equals `D(K) = 0` up to `τ_K` and is `0` after). Hence `R(K) ≤ T_-` and
`T(K) = max(T_-, R(K)) = T_-`.

*Case `D(K) = 1`.* Then `D_{T_-} = 1 ≠ 0`, so `R(K) > T_-`; by Corollary 2.2 the
sequence `D_s` can only become `0` after a collapse, so `τ_K` must be defined
(otherwise `D_s = 1` for all `s ≥ T_-`, contradicting eventual `p`-periodicity
of coordinate `K`, which holds by EA2 + `P(K) = p`). Corollary 2.2 gives
`D_s = 1` for `T_- ≤ s ≤ τ_K` and `D_s = 0` for `s ≥ τ_K + 1`, so exactly
`R(K) = τ_K + 1`. Since `τ_K ≥ T_-` we get
`T(K) = max(T_-, τ_K + 1) = τ_K + 1`. ∎

### THEOREM B2 (levels with `P(K) = 2 P(K-1)`, i.e. DOUBLING)

> If `P(K) = 2 P(K-1)` then `T(K) = T(K-1)`.

*Proof.* A period-doubling level has `c_s = 0` for every `s ≥ T_-`: by the
Phase 2C trichotomy, the one-period fibre map is a negation, which requires the
`c`-word on the base cycle to be identically zero, and the base cycle repeats
forever from `T_-`. Apply Lemma 2.1 at the lag `p = P(K) = 2P(K-1)` — the
lemma's hypotheses hold at this lag too, since `P(K-1) | p` — and only the first
clause ever fires, so `D_s` is constant on `[T_-, ∞)`. Coordinate `K` is
eventually `p`-periodic (EA2), forcing that constant to be `0`. Hence
`R(K) ≤ T_-`, and Theorem 1.5 gives
`T(K) = max( T^{[p]}(K-1), R(K) ) ≤ T(K-1)` (using Theorem 1.4 for
`T^{[p]}(K-1) ≤ T(K-1)`). With Theorem 1.3 (`T` non-decreasing) this forces
`T(K) = T(K-1)`. ∎

### THEOREM B3 (NEUTRAL levels)

> If level `K` is NEUTRAL then `T(K) = T(K-1)`.

*Proof.* NEUTRAL also has `c_s = 0` for all `s ≥ T_-`, and `P(K) = P(K-1) = p`.
The argument of B2 applies verbatim at the lag `p`. ∎

NEUTRAL has never been observed (Phase 2D: exactly four opportunities, all
resolved as DOUBLING). B3 is proved anyway so that the classification below is
exhaustive **without** relying on that observation.

### COROLLARY B4 (the dichotomy)

> Every level `K ≥ 2` is exactly one of
> * **INHERITING**: `T(K) = T(K-1)`, or
> * **RESETTING**: `T(K) = τ_K + 1 > T(K-1)`;
>
> and **RESETTING ⟹ COLLAPSING**. Equivalently: a level can add transient only
> by collapsing, and when it does, the new preperiod is exactly one step after
> the first collapse.

*Proof.* B2 and B3 make every non-COLLAPSING level INHERITING; B1 splits the
COLLAPSING levels by `D(K)`. ∎

---

## 4. Verification

`run_phase2e.py` §2, `K ≤ 30000` (29999 levels):

| check | result |
|---|---|
| `T_predicted = T(K)` from Theorem B | **29999 / 29999, 0 mismatches** |
| `D(K) = 1 ⟺ RESETTING` | **exact, 0 violations** |
| RESETTING ⟹ COLLAPSING | **exact, 0 violations** |
| RESETTING ⟹ `T(K) = τ_K + 1` | **exact, 0 violations** |
| all four DOUBLING levels (`K = 3, 8, 29, 400`) | `T(K) = T(K-1)` in all four |
| NEUTRAL levels found | **none** (as in Phase 2D) |

Class counts, `2 ≤ K ≤ 30000`:

```
    RESETTING   16394      (54.65 %)
    INHERITING  13605      (45.35 %)
```

The full level-by-level table is `TRANSIENT_CLASSIFICATION.csv`
(columns `K, P_prev, P_K, fibre_kind, T_prev, T_K, tau_K, D_bit, class,
T_predicted, matches, R_K, increment`).

---

## 5. The exact decomposition of the transient

### COROLLARY B5 (telescoping)

> For any `K_0 < K`,
> ```
>     T(K)  =  T(K_0)  +  Σ_{K' RESETTING, K_0 < K' ≤ K}  ( τ_{K'} + 1 - T(K'-1) ).
> ```

*Proof.* Telescope `T(K) - T(K_0)` and apply Corollary B4 term by term. ∎

Writing `δ(K') := τ_{K'} + 1 - T(K'-1) ≥ 1` for the **reset increment**, and
letting `d(K)` be the fraction of RESETTING levels up to `K` and `m(K)` their
mean increment,

```
    T(K) / K  =  d(K) · m(K)  +  O(1/K).
```

Measured at `K = 30000` (**BOUNDED OBSERVATION**, this `K` only):

```
    d = 0.546485 ,   m = 2.451873 ,   d·m = 1.339911 ,   T(K)/K = 1.339900
```

The two agree to `1.1 × 10^-5`, the `T(K_0)/K` remainder.

**This is the content of Phase 2E §2 for the open problem.** Phase 2B's
target G5 (`liminf T(K)/K > 1`) is now *exactly equivalent* to

> `liminf_K  d(K) · m(K)  >  1`,

i.e. to a statement about **how often levels collapse late**. That is a
reformulation, **not** a reduction: no bound on either factor is proved, and
the two factors are not independent. See `TRANSIENT_LEMMA_REGISTRY.md` F1.

---

## 6. Controls

* Theorems B1–B5 are proved for **all `K`**. The 29999-level table checks the
  proofs; it does not establish them, and no all-`K` claim rests on it.
* The **only** bounded ingredients used anywhere below are: "every level in
  `[401, 30000]` is COLLAPSING", "no NEUTRAL level occurs", and "coordinates
  `2, 7, 28, 399` are the only eventually-zero ones". All three are labelled
  BOUNDED OBSERVATION wherever used, and B2/B3 are proved so that the
  classification does not depend on them.
* No search here is described as UNSAT: "no NEUTRAL level in `K ≤ 30000`" means
  exactly that and nothing more.
