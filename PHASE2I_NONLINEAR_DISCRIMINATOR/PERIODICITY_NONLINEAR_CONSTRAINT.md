# The parity constraint `Q_p(t)` implied by a periodic centre

**Phase 2I, section 3.**

The hypothesis, stated once and never assumed true:

> ### Hypothesis EP(T, p). `c_{t+p} = c_t` for every `t ≥ T`.

Problem 1 asks whether EP holds for the single-cell seed. Everything below is
conditional on EP and proves nothing about whether EP holds.

---

## 1. The canonical decomposition

From Corollary 2.3, `c_t = S(t) + Σ_{s<t} Σ_j T(t-1-s, j) n_s(j)`. Adding the
same at `t+p`:

> ### Definition 3.1. `Q_p(t) := c_{t+p} + c_t` (GF(2)). EP(T,p) is exactly
> `Q_p(t) = 0` for all `t ≥ T`.

> ### Theorem 3.2 (canonical three-part form). For every `p ≥ 1`, `t ≥ 0`,
> ```
>     Q_p(t)  =  SEED_p(t)  +  OLD_p(t)  +  SLAB_p(t)
> ```
> where
> ```
>     SEED_p(t) = Σ_d [ T(t+p, d) + T(t, d) ] · x_0(-d)
>     OLD_p(t)  = Σ_{s<t} Σ_j K_p(t-1-s, j) · n_s(j) ,   K_p(m,·) := T(m+p,·)+T(m,·)
>     SLAB_p(t) = Σ_{s=t}^{t+p-1} Σ_j T(t+p-1-s, j) · n_s(j)
> ```
> `SLAB_p(t)` involves **only the `p` rows of the spacetime slab `[t, t+p)`**.

*Proof.* Split the `s`-sum of `c_{t+p}` at `s = t` and add termwise. ∎

*Verification.* `Q = SEED + OLD + SLAB` agreed with the directly computed
`c_{t+p} + c_t` in **15 of 15** tested `(p,t)` pairs, `p ∈ {1,2,3,4,8}`.

> ### Corollary 3.3 (the seed term vanishes for symmetric seeds).
> If `x_0` is mirror-symmetric then `S(·)` is constant (Theorem 2.6), so
> `SEED_p(t) = 0` for every `p, t`. In particular for the single-cell seed
> ```
>     Q_p(t) = OLD_p(t) + SLAB_p(t)
> ```
> — the constraint is a statement about the **nonlinear event field alone**.

Verified: `SEED_p(t) = 0` in all 15 tested pairs.

---

## 2. The self-similar event family

> ### Theorem 3.4 (difference-kernel convolution). If `bits(m) ∩ bits(p) = ∅`
> then
> ```
>     K_p(m, ·)  =  ( T(p, ·) + δ_0 ) * T(m, ·)                     (convolution over GF(2))
> ```
> and consequently, since `T(p,·)` is symmetric,
> ```
>     Σ_j K_p(m, j) n_s(j)  =  Σ_j T(m, j) · (D_p n_s)(j) ,
>     (D_p u)(j) := Σ_e T(p, e) u(j+e)  +  u(j) .
> ```

*Proof.* Disjoint bits give `P_{m+p} = P_m · P_p` (Frobenius), so
`T(m+p,·) = T(p,·) * T(m,·)`; add `T(m,·) = δ_0 * T(m,·)`. The second display
is the self-adjointness of convolution by a symmetric kernel. ∎

> ### Corollary 3.5 (the `p = 2^a` case — an explicit self-similar family).
> If bit `a` of `m` is `0`,
> ```
>     K_{2^a}(m, j) = T(m, j - 2^a) + T(m, j + 2^a) ,
>     (D_{2^a} u)(j) = u(j - 2^a) + u(j + 2^a) .
> ```
> So at those times the old part of the constraint is **the same kernel
> `T(m,·)` applied to the `2^a`-shifted difference of the event field.**

*Verification.* The split identity `T(m+2^a,d) = T(m,d+2^a)+T(m,d)+T(m,d-2^a)`
holds in **365 466 of 365 466** checks with bit `a` of `m` zero, and the
convolution theorem holds in **419 of 419** disjoint-bit cases across
`p ∈ {1,2,3,4,5,6,8}`. When bit `a` of `m` is `1` the shifted form **fails at
every tested `m`** — the hypothesis is not decorative.

> ### The isolated quantity.
> For a mirror-symmetric seed and `p = 2^a`,
> ```
>   Q_{2^a}(t) =  Σ_{s<t,  bit a of (t-1-s) = 0}  Σ_j T(t-1-s, j) (D_{2^a} n_s)(j)
>               + Σ_{s<t,  bit a of (t-1-s) = 1}  Σ_j K_{2^a}(t-1-s, j) n_s(j)
>               + Σ_{s=t}^{t+2^a-1} Σ_j T(t+2^a-1-s, j) n_s(j) ,
> ```
> and EP(T, `2^a`) forces this to vanish for every `t ≥ T`.

---

## 3. What section 3 asked for, and what was achieved

The brief: *"Seek the smallest support, canonical parity sum, or self-similar
event family."* Three targets; the scoreboard is mixed and is reported as such.

| target | outcome |
|---|---|
| **canonical parity sum** | **ACHIEVED.** Theorem 3.2, verified 15/15, with the seed term proved to vanish for symmetric seeds. |
| **self-similar event family** | **ACHIEVED.** Theorem 3.4 / Corollary 3.5: the same kernel on a `2^a`-shifted difference of `n`, on an explicitly characterised set of times. |
| **smallest support** | **NOT ACHIEVED — refuted.** See below. |

> ### Failed conjecture 2I-X-01 (support reduction).
> *"The difference kernel `K_p(m,·)` has smaller support than `T(m,·)`, so
> `Q_p` is a sparser constraint than `c_t` itself."* **False.** Measured over
> all `m < 400` with `bits(m) ∩ bits(p) = ∅`:
>
> | `p` | mean `\|supp K_p(m,·)\|` | mean `\|supp T(m,·)\|` | ratio | `K` smaller in |
> |---|---:|---:|---:|---:|
> | 1 | 53.04 | 42.92 | **1.236** | 67/200 |
> | 2 | 65.60 | 53.04 | **1.237** | 66/200 |
> | 4 | 60.72 | 49.20 | **1.234** | 68/200 |
> | 8 | 62.80 | 50.60 | **1.241** | 64/200 |
>
> The difference kernel is on average about 24 % **larger**. Taking a temporal
> difference costs support rather than saving it, and the conjecture is
> retained rather than deleted. The only genuine support saving in this section
> is `SLAB_p(t)`, which touches exactly `p` rows.

---

## 4. Two consequences that are proved, and one that is not

> ### Proposition 3.6 (strip transfer — this is NLB2, proved for one strip).
> Under EP(T,p), the nonlinear field at column `-1` is eventually `p`-periodic.

*Proof.* `n_t(-1) = c_t(1 + c_{t+1})` (Theorem 1.3) is a function of `c` alone. ∎

> ### Proposition 3.7 (partial second strip).
> Under EP(T,p), `x_t(-2)` — and hence `n_t(-2)` — is determined and eventually
> `p`-periodic on the set `{ t ≥ T : c_t = 1 and c_{t+1} = 1 }`, which is
> itself `p`-periodic.

*Proof.* By Corollary 1.4, `c_s = 1 ⟹ x_s(-1) = 1 + c_{s+1}`. Apply the
backward step at `j = -1`:
`x_t(-2) = x_{t+1}(-1) + x_t(-1) + x_t(0) + n_t(-1)`, whose terms are all
determined when `t` and `t+1` both lie in `{c = 1}`. ∎

> ### What is **not** proved.
> `NLB2` in full — that the event field becomes `p`-periodic in **two distinct
> fixed strips** — is open. Proposition 3.6 gives one strip for free (because
> that strip is a function of the centre), and Proposition 3.7 gives a second
> strip only on a proper subset of rows. The missing bit is exactly the residue
> of `NONLINEAR_ACTIVITY_THEORY.md` §5: `x_t(1)` on the rows where `c_t = 1`.

**And it cannot be closed by this machinery.** Propositions 3.6 and 3.7 use
only left-permutivity and the local rule; rule 90 satisfies both and has a
periodic centre column. The nonlinearity enters Theorem 1.3 but not in a way
that constrains the residue.

---

## 5. Honest status of `Q_p`

`Q_p(t) = 0` is **equivalent** to `c_{t+p} = c_t`. Theorem 3.2 rewrites it;
Theorem 3.4 rewrites it again in a self-similar form on a characterised set of
times. **No rewriting makes it a weaker statement**, and by the brief's rule any
candidate equivalent to Problem 1 is rejected as a bridge. `Q_p` earns its
place only as the object in which sections 4, 8 and 9 look for structure — not
as progress.

## 6. Statement inventory

| id | statement | status |
|---|---|---|
| **Q-3.2** | `Q_p = SEED + OLD + SLAB` | **THEOREM**, verified 15/15 |
| **Q-3.3** | `SEED_p ≡ 0` for mirror-symmetric seeds | **THEOREM**, proved |
| **Q-3.4** | difference-kernel convolution, disjoint bits | **THEOREM**, 419/419 |
| **Q-3.5** | `p = 2^a` shifted-difference form | **COROLLARY**, 365 466/365 466 |
| **P-3.6** | strip `-1` is `p`-periodic under EP | **THEOREM** |
| **P-3.7** | strip `-2` on `{c_t = c_{t+1} = 1}` | **THEOREM** |
| **2I-X-01** | `K_p` has smaller support than `T` | **REFUTED**, preserved |
| **NLB2** | two full strips | **OPEN** |
