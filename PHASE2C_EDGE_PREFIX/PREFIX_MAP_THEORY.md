# Prefix Map Theory

The autonomous prefix dynamics of rule 30 in edge-aligned coordinates.
Phase 2C only; no final proof is attempted.

Base facts imported from Phase 2B (`EDGE_ALIGNED_DYNAMICS.md`, all proved
there and re-verified here): with `w_t(k) = x_t(-t+k)`,

```
    EA1   w_{t+1}(k) = w_t(k-2) XOR ( w_t(k-1) OR w_t(k) ),   w_t(m)=0 for m<0
    EA3   x_t(0) = w_t(t)                                     (centre = diagonal)
```

`verify_against_rule30(200, 300)`: 50,400 cells, **0 mismatches**.

---

## 1. The map `F_K` in coordinates

**Definition.** `F_K : {0,1}^{K+1} -> {0,1}^{K+1}`,

```
    (F_K z)_k  =  z_{k-2}  XOR  ( z_{k-1} OR z_k ),        z_m := 0 for m < 0.
```

Written out, the first three coordinates are special only because of the
convention `z_m = 0` for `m < 0`:

| `k` | `(F_K z)_k` | remark |
|---|---|---|
| 0 | `0 XOR (0 OR z_0) = z_0` | **coordinate 0 is a fixed point** |
| 1 | `0 XOR (z_0 OR z_1) = z_0 OR z_1` | monotone, never returns to 0 once `z_0=1` |
| `k >= 2` | `z_{k-2} XOR (z_{k-1} OR z_k)` | the generic form |

Since `w_0(0) = 1`, coordinate 0 is identically 1 and coordinate 1 is
identically 1 from `t = 1` on. So `T(0) = 0, P(0) = 1` and `T(1) = 1, P(1) = 1`
— the base of every induction below.

**Bit-parallel form.** Packing `w(k)` into bit `k` of an integer `W`,

```
    F(W) = ( (W << 2) XOR ( (W << 1) OR W ) )  &  mask_K .
```

*Verification.* `verify_F_forms(10)`: the list form and the bit-parallel form
agree on 4,094 assignments (exhaustive for `K <= 10`). Truncation by `mask_K`
is exact because `(F z)_k` reads no index above `k`.

## 2. Projection — **THEOREM P1**

> Let `pi_K : {0,1}^{K+2} -> {0,1}^{K+1}` drop the last coordinate. Then
> ```
>     pi_K ( F_{K+1}(z) )  =  F_K ( pi_K(z) )       for all z.
> ```

*Proof.* For `0 <= k <= K`, `(F_{K+1} z)_k = z_{k-2} XOR (z_{k-1} OR z_k)`,
which involves only indices `k-2, k-1, k`, all `<= K`. Those coordinates are
unchanged by `pi_K`, so the value equals `(F_K (pi_K z))_k`. ∎

*Verification.* `verify_projection(9)`: exhaustive over 4,092 assignments,
**commutes everywhere**.

**Corollary P1a.** The prefix systems form an inverse (projective) tower

```
    ... --> {0,1}^{K+2} --pi--> {0,1}^{K+1} --pi--> {0,1}^K --> ...
```

with `F` commuting with every arrow. Consequently:

* any eventual period `Q` of the `K`-prefix orbit is an eventual period of the
  `(K-1)`-prefix orbit, so **`P(K-1)` divides `P(K)`**;
* any preperiod `T` of the `K`-prefix works for the `(K-1)`-prefix, so
  **`T(K-1) <= T(K)`** (this is statement E3, and it is a theorem).

## 3. How coordinate `K+1` is driven — **THEOREM P2**

> ```
>     (F_{K+1} z)_{K+1}  =  z_{K-1}  XOR  ( z_K  OR  z_{K+1} ).
> ```
> The new coordinate depends on **exactly two coordinates of the old prefix**
> (`K-1` and `K`) and on **its own value**. It is a one-bit fibre over the
> `K`-prefix base.

*Proof.* Specialise the definition at `k = K+1`. ∎

*Verification.* `verify_new_coordinate_drive(9)`: exhaustive over 4,088
assignments, **exact**.

**This is the entire structure Phase 2C exploits.** Writing
`a_t = w_t(K-1)`, `c_t = w_t(K)`, `u_t = w_t(K+1)`:

```
    u_{t+1} = a_t XOR ( c_t OR u_t ),
```

a one-bit recurrence *driven* by the base orbit `(a_t, c_t)`, which is already
known once the `K`-prefix has been solved. `SKEW_PRODUCT_ANALYSIS.md` works out
the consequences.

## 4. The orbit that matters

The initial state is `W_0^K = (1, 0, 0, ..., 0)`. The support constraint
`w_t(k) = 0` for `k > 2t` is **not an extra condition**: it is produced by the
dynamics from this initial state, and the truncated bit-parallel iteration
reproduces the true edge-aligned rows exactly (verified above).

Therefore

```
    T(K), P(K)  =  the tail and cycle length of the F_K-orbit of (1,0,...,0),
```

and because `F_K` is a **deterministic** map on a finite set, a single
coincidence `W_T^K = W_{T+Q}^K` implies periodicity forever. That is what makes
the computations in `PREFIX_PERIOD_RESULTS.md` **exact** rather than
estimates: the first repeat gives the true `rho` and `lambda`.

## 5. What P1 and P2 do not give

* They say nothing about the *sizes* of `T(K)` and `P(K)` beyond the recursive
  bounds of `SKEW_PRODUCT_ANALYSIS.md`.
* The tower is an inverse limit of finite systems, but the centre column reads
  the **diagonal** `w_t(t)` (EA3), which is not a coordinate of any fixed
  prefix. Nothing in this section transfers to it; see
  `DIAGONAL_BRIDGE_ATTEMPT.md`.
