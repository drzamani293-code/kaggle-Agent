# Mod-2 kernel geometry: exact binary-digit identities

**Phase 2I, section 4.** Only exact digit identities appear here. **No
asymptotic or randomness statement is made or used**, per the brief and the
standing rules.

The single tool is the Frobenius factorisation over GF(2):

```
    (1 + z + z^2)^{2^i} = 1 + z^{2^i} + z^{2^{i+1}}
        ⟹   P_m(z) = (1+z+z^2)^m = Π_{i ∈ bits(m)} ( 1 + z^{2^i} + z^{2^{i+1}} )   (FROB)
        ⟹   bits(a) ∩ bits(b) = ∅  ⟹  P_{a+b} = P_a · P_b                          (DISJ)
```

Recall `T(m,d) := [z^{d+m}] P_m(z)`, symmetric in `d`, supported on `|d| ≤ m`.

---

## 1. `m = 2^a` — the three-atom kernel

> ### Theorem 4.1. `T(2^a, ·)` is supported on exactly `{ -2^a, 0, +2^a }`,
> with value 1 at each.

*Proof.* `P_{2^a} = 1 + z^{2^a} + z^{2^{a+1}}` by (FROB); shift by `-2^a`. ∎

Verified for `1 ≤ a ≤ 11`.

**Reading.** At a time-gap `m = 2^a` the Green function is a *perfect
three-slit*: an event at `(s, j)` reaches the centre `2^a + 1` steps later iff
`j ∈ {-2^a, 0, 2^a}`. All other events are annihilated by parity.

## 2. `m = 2^a - 1` — the Stern kernel, with a correction

By (FROB), `bits(2^a - 1) = {0, …, a-1}`, so `T(2^a-1, d)` counts
representations of `e = d + 2^a - 1` as `Σ_{i<a} e_i 2^i` with `e_i ∈ {0,1,2}`,
mod 2 — the **hyperbinary** count, which for `e < 2^a` equals Stern's diatomic
`s(e+1)`.

> ### Theorem 4.2. For `d ≤ 0` (equivalently `e = d + 2^a - 1 < 2^a`),
> ```
>     T(2^a - 1, d) = 1   ⟺   3 ∤ (d + 2^a) ,
> ```
> and the values for `d > 0` follow by the symmetry `T(m,-d) = T(m,d)`.

*Proof.* For `e < 2^a` no digit `i ≥ a` can be used, so the restricted count is
the unrestricted hyperbinary count `s(e+1)`; and `s(n)` is even iff `3 | n`. ∎

> ### Correction 2I-C-02.
> The first version of this statement asserted the criterion **for all `d`**.
> That is **false**: it fails for every `d > 0` at every `a` from 1 to 11,
> because `e = d + 2^a - 1` then exceeds `2^a` and the identification with
> Stern's sequence breaks. The over-broad claim is retracted; Theorem 4.2 is
> the corrected form, and the refutation is preserved rather than deleted.

## 3. `m = 2^a + r` — three disjoint scaled copies

> ### Theorem 4.3. For `0 ≤ r < 2^a`,
> ```
>     T(2^a + r, d) = T(r, d + 2^a) + T(r, d) + T(r, d - 2^a) ,
> ```
> i.e. **three translated copies of `T(r,·)`**, centred at `-2^a, 0, +2^a`. The
> copies are pairwise disjoint **iff** `2r < 2^a`, and then
> `|supp T(2^a+r, ·)| = 3 · |supp T(r,·)|` exactly.

*Proof.* `bits(2^a)` and `bits(r)` are disjoint, so (DISJ) gives
`P_{2^a+r} = (1 + z^{2^a} + z^{2^{a+1}}) P_r`. Disjointness of the copies is the
support bound `supp T(r,·) ⊆ [-r, r]`. ∎

Verified: **254 of 254** cases (`1 ≤ a ≤ 7`, all `0 ≤ r < 2^a`), including the
support-count accounting in the disjoint regime.

## 4. `m = q·2^a + r` — the general self-similar decomposition

> ### Theorem 4.4. For `0 ≤ r < 2^a` and any `q ≥ 0`,
> ```
>     T(q·2^a + r, d) = Σ_e T(q, e) · T(r, d - e·2^a) .
> ```
> The kernel is the `2^a`-**scaled copy of `T(q,·)`** with every atom replaced by
> a translated copy of `T(r,·)`. The copies are disjoint iff `2r < 2^a`, and
> then `|supp T(q·2^a+r,·)| = |supp T(q,·)| · |supp T(r,·)|`.

*Proof.* `P_{q·2^a}(z) = P_q(z^{2^a})` by (FROB) applied `a` times, and
`bits(q·2^a) ∩ bits(r) = ∅`, so (DISJ) applies. ∎

Verified: **1240 of 1240** cases (`q < 20`, `1 ≤ a ≤ 5`, all `r < 2^a`), with
support multiplicativity confirmed in every disjoint case.

**This is the decomposition into disjoint scaled copies the brief asks for.**
Writing `t - 1 - s` in base 2 and splitting at any digit position gives an
exact recursive description of which events reach the centre.

## 5. The difference kernel

> ### Theorem 4.5. If `bits(m) ∩ bits(p) = ∅` then
> ```
>     T(m+p, ·) + T(m, ·) = ( T(p,·) + δ_0 ) * T(m,·) .
> ```
> For `p = 2^a` (so: bit `a` of `m` is zero) this is
> `K_{2^a}(m, d) = T(m, d-2^a) + T(m, d+2^a)`.

Verified: **419 of 419** disjoint-bit cases; and the underlying split identity
`T(m+2^a,d) = T(m,d+2^a) + T(m,d) + T(m,d-2^a)` in **365 466 of 365 466**
checks. When bit `a` of `m` is `1`, the shifted form **fails at every tested
`m`** — the disjointness hypothesis is essential, not cosmetic.

## 6. Support counting

Let `A(m) := |supp T(m,·)|`, the number of odd coefficients of `P_m`.

> ### Theorem 4.6 (exact recurrence).
> ```
>     A(2m)   = A(m) ,
>     A(2m+1) = 3 A(m) - 2 C(m) ,      C(m) := | S_m ∩ (S_m + 1) | ,
> ```
> where `S_m := supp P_m` (as a set of exponents) and `C(m)` counts its adjacent
> pairs.

*Proof.* `P_{2m}(z) = P_m(z^2)` gives the even case. For the odd case,
`P_{2m+1}(z) = P_m(z^2)(1+z+z^2)`: the odd-exponent terms `z^{2e+1}`, `e ∈ S_m`,
are pairwise distinct and contribute `A(m)`; the even part is
`Σ_{e∈S_m} (z^{2e} + z^{2e+2})`, whose support is the symmetric difference
`S_m Δ (S_m+1)`, of size `2A(m) - 2C(m)`. ∎

Verified for all `m < 400`; `A(2m) = A(m)` verified for all `m ≤ 3000`.

**Consequences, exact.** `A(2^a) = 3` for every `a ≥ 1` (iterate the even case
from `A(1) = 3`). And

| `a` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `A(2^a - 1)` | 3 | 5 | 11 | 21 | 43 | 85 | 171 | 341 | 683 | 1365 | 2731 |
| `(2^{a+2} + (-1)^{a+1})/3` | 3 | 5 | 11 | 21 | 43 | 85 | 171 | 341 | 683 | 1365 | 2731 |

> **Status of the closed form.** The two rows agree for `1 ≤ a ≤ 11`. This is
> **not** a proved identity: closing Theorem 4.6's recurrence at `m = 2^a - 1`
> needs a companion recurrence for `C(2^a-1)`, which this phase did **not**
> establish. It is recorded as an exact match over a bounded range, and it is
> not used as a premise anywhere.

## 7. What this geometry does not do

1. It describes **which events can reach the centre**, never **which events are
   active**. Activity is orbit data; the kernel is arithmetic. Section 8 keeps
   the two strictly apart.
2. It is **seed-independent and rule-150 data**. The kernel is the same for any
   forcing term, so nothing here can discriminate seeds, and by itself nothing
   here can discriminate rule 30 from rule 90 either — only the *forcing* does
   that (Theorem 1.1).
3. **Powers of two are not evidence.** The `2^a` structure above is an identity
   about `P_m`, not an observation about rule 30's orbit, and per the standing
   rules no theorem is inferred from powers-of-two data.

## 8. Statement inventory

| id | statement | status |
|---|---|---|
| **K-4.1** | `T(2^a,·)` is a three-slit | **THEOREM**, proved, verified `a ≤ 11` |
| **K-4.2** | Stern criterion for `m = 2^a-1`, `d ≤ 0` | **THEOREM**, proved, verified `a ≤ 11` |
| **2I-C-02** | the same criterion for all `d` | **RETRACTED**, refuted at every `a ≤ 11` |
| **K-4.3** | three disjoint copies for `m = 2^a + r` | **THEOREM**, 254/254 |
| **K-4.4** | scaled self-similar decomposition | **THEOREM**, 1240/1240 |
| **K-4.5** | difference kernel under disjoint bits | **THEOREM**, 419/419 |
| **K-4.6** | `A(2m)=A(m)`, `A(2m+1)=3A(m)-2C(m)` | **THEOREM**, proved, `m < 400` |
| **BO-4.7** | `A(2^a-1) = (2^{a+2}+(-1)^{a+1})/3` | **BOUNDED MATCH**, `a ≤ 11`, unproved, unused |
