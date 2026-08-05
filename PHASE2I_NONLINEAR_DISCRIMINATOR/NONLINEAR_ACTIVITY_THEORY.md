# The nonlinear activity field of Rule 30

**Phase 2I, section 1.** Every Boolean identity below is verified exhaustively
over its finite domain, and every orbit identity is checked cell by cell
against direct simulation. Problem 1 is not solved and nothing here claims
progress on it.

---

## 0. Setup

Over GF(2), `a OR b = a + b + ab`. Substituting into rule 30:

```
    x_{t+1}(j) = x_t(j-1) XOR ( x_t(j) OR x_t(j+1) )
               = x_t(j-1) + x_t(j) + x_t(j+1) + x_t(j) x_t(j+1)
               = (L x_t)(j) + n_t(j)                                   (R30)
```

with

```
    (L x)(j) := x(j-1) + x(j) + x(j+1)          the RULE 150 operator
    n_t(j)   := x_t(j) x_t(j+1)                 the NONLINEAR ACTIVITY FIELD
```

*Exhaustive verification.* The Boolean form and the GF(2) form agree on all
8 triples `(l,c,r)`, and `rule30 = rule150 + n` on all 8 triples
(`run_phase2i_tests.py`, group A).

> ### Theorem 1.1 (the forcing degree separates the three rules).
> Write each rule as `x_{t+1} = L x_t + F(x_t)` and take the algebraic normal
> form of `F` in the three local variables `(l, c, r)`:
>
> | rule | `F(l,c,r)` | ANF monomials | **degree** |
> |---|---|---|---|
> | **30** | `c·r` | `{cr}` | **2** |
> | **90** | `c` | `{c}` | **1** |
> | **150** | `0` | `{}` | **-** |
>
> Rule 30's forcing is **quadratic**; rules 90 and 150 have forcing of degree
> at most 1, i.e. they are affine and `L`-conjugate to nothing new.

*Proof.* Compute the Möbius transform of `f(l,c,r) + l + c + r` for each rule;
done exhaustively over the 8 inputs in `discriminator_lab.P1_forcing_degree`. ∎

**This is the property Phase 2H's lesson demands.** Every candidate in this
phase must use degree-2 forcing somewhere, or it is refuted in advance by
rule 90.

---

## 1. Exact identities for `n`

> ### Lemma 1.2 (basic identities). For all `t, j`:
> 1. `n_t(j) = x_t(j) x_t(j+1)`, so `n_t(j) ≤ x_t(j)` and `n_t(j) ≤ x_t(j+1)`.
> 2. `x_t(j) = 0  ⟹  n_t(j) = n_t(j-1) = 0`.
> 3. `x_t(j) OR x_t(j+1) = x_t(j) + x_t(j+1) + n_t(j)`.
> 4. `n_t(j) = x_t(j) · x_t(j+1)` is symmetric under `j ↦ -j-1` composed with
>    the spatial mirror, **not** under `j ↦ -j`. This asymmetry is the whole
>    content of section 6.

All four are immediate from the definition; (3) is the GF(2) expansion of OR.

> ### Theorem 1.3 (the centre-adjacent field is a function of the centre column).
> For rule 30 and **every** configuration,
> ```
>       n_t(-1) = c_t · ( 1 + c_{t+1} ) = c_t + c_t c_{t+1} ,
>       where c_t := x_t(0).
> ```

*Proof.* If `x_t(0) = 0` then `n_t(-1) = x_t(-1)·0 = 0`, and the right side is
`0` too. If `x_t(0) = 1` then by (R30)
`c_{t+1} = x_t(-1) + 1 + x_t(1) + 1·x_t(1) = x_t(-1) + 1`, so
`x_t(-1) = 1 + c_{t+1}`, and `n_t(-1) = 1·x_t(-1) = 1 + c_{t+1}`, which is the
right side at `c_t = 1`. ∎

**The cancellation is the point.** When `c_t = 1` the term `x_t(1)` appears
twice — once linearly and once inside `n_t(0) = c_t x_t(1) = x_t(1)` — and the
two copies cancel over GF(2). This is exactly the OR-blindness of Phase 2H,
now visible as an algebraic cancellation rather than a case split.

> ### Corollary 1.4 (the successor law).
> ```
>       c_t = 1  ⟹  c_{t+1} = 1 + x_t(-1)      (the right neighbour drops out)
>       c_t = 0  ⟹  c_{t+1} = x_t(-1) + x_t(1)
> ```

> ### Corollary 1.5.
> `n_t(0) = c_t · x_t(1)`. Unlike `n_t(-1)`, this is **not** a function of the
> centre column: `x_t(1)` survives on the rows where `c_t = 1`. The asymmetry
> between Theorem 1.3 and Corollary 1.5 is the reason the centre column
> determines its left neighbour but not its right one.

**Rule 90 verdict** (mandatory control): Theorem 1.3 is **false** for rule 90.
Checked on the single-cell rule-90 orbit, `t ≤ 800`: the identity fails.
For rule 150 it fails at 67 of the first 200 rows. The identity is therefore
genuinely rule-30 content — but it holds for **every** rule-30 seed, so it is
RULE-SPECIFIC and by itself cannot pass filter F3.

---

## 2. Temporal lag defects of `n`

For a lag `p` define the defect fields

```
    d_t(j) := x_{t+p}(j) + x_t(j) ,      e_t(j) := n_{t+p}(j) + n_t(j) .
```

> ### Theorem 1.6 (the defect evolution is affine in `d`, with quadratic source).
> ```
>       d_{t+1} = L d_t + e_t ,
>       e_t(j)  = x_t(j) d_t(j+1) + x_t(j+1) d_t(j) + d_t(j) d_t(j+1) .
> ```

*Proof.* The first is (R30) applied at `t+p` and `t` and added. For the second,
write `x_{t+p}(j) = x_t(j) + d_t(j)` and expand
`n_{t+p}(j) = (x_t(j)+d_t(j))(x_t(j+1)+d_t(j+1))`; the `x_t(j)x_t(j+1)` term
cancels against `n_t(j)`. ∎

> ### Corollary 1.7 (defects cannot be created from nothing).
> `d_t ≡ 0 ⟹ e_t ≡ 0 ⟹ d_{t+1} ≡ 0`. So the set of lags at which the whole
> configuration is `p`-periodic is closed forward in time — the usual
> statement, recorded because the quadratic term does not break it.

> ### Corollary 1.8 (the source is quadratic and *local to the defect support*).
> `e_t(j) = 0` whenever `d_t(j) = d_t(j+1) = 0`. Hence
> `supp e_t ⊆ supp d_t ∪ (supp d_t − 1)`. The nonlinearity never widens the
> defect support faster than the linear part does.

**Rule 90/150 verdict.** For a linear rule, `e_t ≡ 0` identically and the
defect obeys the *same* linear equation as the field. That is precisely why
linear rules are tractable and rule 30 is not — and it is a degree-2 fact.

---

## 3. Difference from the Rule 150 orbit

Let `y_t` be the **rule 150** orbit from the same seed, and `Δ_t := x_t + y_t`.

> ### Theorem 1.9. `Δ_0 = 0` and `Δ_{t+1} = L Δ_t + n_t`. Hence
> ```
>       Δ_t(j) = sum_{s<t} sum_d T(t-1-s, d) n_s(j-d) ,
> ```
> where `T` is the mod-2 Green function of `L` (section 2). **The entire
> difference between rule 30 and its linear part is the accumulated nonlinear
> activity, propagated by the rule-150 kernel.**

*Proof.* Subtract the two evolutions; apply Duhamel (Theorem 2.2). ∎

> ### Corollary 1.10 (centre form). Since `T(m,·)` is symmetric,
> ```
>       c_t + y_t(0) = sum_{s<t} sum_j T(t-1-s, j) n_s(j) .
> ```

Section 2 shows `y_t(0)` is a *constant* for a mirror-symmetric seed, which
turns Corollary 1.10 into the exact parity statement `c_t = x_0(0) + |N(t)| mod 2`.

---

## 4. Propagation of `n`-events toward the centre

> ### Theorem 1.11 (light cone for events). An event `n_s(j) = 1` can affect
> `c_t` only if `|j| ≤ t - 1 - s`, i.e. only if
> ```
>       t ≥ s + |j| + 1 .
> ```
> Moreover it affects `c_t` **iff** `T(t-1-s, j) = 1`, which is a pure
> binary-digit condition on `(t-1-s, j)` and does not depend on the orbit.

*Proof.* `T(m,d) = 0` for `|d| > m` (degree bound on `P_m`), and the
coefficient of `n_s(j)` in `c_t` is exactly `T(t-1-s, j)` by Corollary 1.10. ∎

So the influence of an event is governed by **two independent things**: a
geometric cone (`|j| ≤ t-1-s`) and an arithmetic filter (`T(t-1-s,j) = 1`).
Section 4 (`MOD2_KERNEL_GEOMETRY.md`) is entirely about the second.

---

## 5. Relations involving `n_t(-1)`, `n_t(0)` and the centre

Collecting Theorem 1.3 and Corollaries 1.4, 1.5:

| quantity | exact expression | determined by `c` alone? |
|---|---|---|
| `n_t(-1)` | `c_t (1 + c_{t+1})` | **yes** |
| `n_t(0)` | `c_t · x_t(1)` | no |
| `x_t(-1)` on `{c_t = 1}` | `1 + c_{t+1}` | **yes** |
| `x_t(-1) + x_t(1)` on `{c_t = 0}` | `c_{t+1}` | **yes** |
| `x_t(1)` on `{c_t = 1}` | — | **no** (one free bit per row) |

> ### Corollary 1.12 (restated Phase 2H H1, in GF(2) form).
> If the centre column is eventually `p`-periodic then `n_·(-1)` is eventually
> `p`-periodic, automatically, being a function of `c`. The only undetermined
> data is `x_t(1)` on the rows where `c_t = 1` — one bit per such row.

**This is the exact residue.** Phase 2H reached the same residue from the
backward step; the GF(2) route reaches it as a cancellation. Neither route
constrains that bit, and rule 90 shows no argument of this shape can.

---

## 6. Statement inventory

| id | statement | status | domain | rule 90 verdict |
|---|---|---|---|---|
| **N-1.1** | forcing degrees 2 / 1 / 0 for rules 30 / 90 / 150 | **THEOREM**, exhaustive | rule level | separates ✔ |
| **N-1.2** | basic `n` identities | **THEOREM** | any | — |
| **N-1.3** | `n_t(-1) = c_t(1+c_{t+1})` | **THEOREM**, verified 800 rows × 3 seeds | any rule-30 config | **fails for rule 90** ✔ |
| **N-1.4** | successor law (right neighbour cancels when `c_t=1`) | **COROLLARY** | any | fails for rule 90 |
| **N-1.6** | `d_{t+1} = L d_t + e_t`, `e` quadratic in `d` | **THEOREM** | any | `e ≡ 0` for rule 90 |
| **N-1.8** | `supp e_t ⊆ supp d_t ∪ (supp d_t − 1)` | **THEOREM** | any | vacuous for rule 90 |
| **N-1.9** | `Δ_t` = accumulated activity through the 150-kernel | **THEOREM** | any | `Δ ≡ 0` |
| **N-1.11** | event influence iff `T(t-1-s,j)=1` | **THEOREM** | any | — |
| **N-1.12** | the residue is one bit per `c_t = 1` row | **COROLLARY** | FIN | — |

Every entry above holds for **all** rule-30 seeds. That makes them
RULE-SPECIFIC, which is necessary but — by the Phase 2H lesson — not
sufficient. Section 5 is where seed specificity has to come from.
