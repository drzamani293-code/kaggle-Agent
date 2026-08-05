# The nonlinear event skeleton `N(t)`

**Phase 2I, section 8.** The brief's warning is the section's governing rule:
**growth alone is not a proof of aperiodicity.** Nothing below is offered as
evidence of aperiodicity.

---

## 1. Two strictly separated objects

> ### Definition 8.1 (kernel support — arithmetic, orbit-independent).
> ```
>     Ker(t) := { (s, j) : 0 ≤ s < t ,  T(t-1-s, j) = 1 }
> ```
> the set of spacetime positions whose nonlinear event *could* affect `c_t`.
> This is a pure binary-digit object; it does not look at the orbit.

> ### Definition 8.2 (nonlinear skeleton — orbit-dependent).
> ```
>     N(t) := { (s, j) ∈ Ker(t) :  n_s(j) = 1 }
> ```
> the events that are actually **active**.

By Corollary 2.3, `c_t = S(t) + |N(t)| mod 2`, and for a mirror-symmetric seed
`S(t) = x_0(0)`, so for the single cell

```
        c_t  =  1 + |N(t)|  (mod 2).                                    (★)
```

Verified at every `t ≤ 40`; and `c_t = S(t) + |N(t)| mod 2` verified for
`D_two_adjacent` and `E_111` as well, where `S` is non-constant.

## 2. The kernel support, before evaluating the orbit

`|Ker(t)| = Σ_{m<t} A(m)` with `A(m) = |supp T(m,·)|`. Write `B(t) := |Ker(t)|`.

| `t` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `\|Ker(t)\|` | 0 | 1 | 4 | 7 | 12 | 15 | 24 | 29 | 40 | 43 | 52 | 61 | 76 |

> ### Theorem 8.3 (exact recurrence for the kernel size).
> ```
>     B(2t) = 4 B(t) - 2 Σ_{m<t} C(m) ,      C(m) = |S_m ∩ (S_m + 1)| ,
> ```
> with `S_m = supp P_m`, and `B` determined from `A` by Theorem 4.6.

*Proof.* Split `Σ_{m<2t} A(m)` into even and odd `m` and apply Theorem 4.6:
`Σ A(2m) = B(t)` and `Σ A(2m+1) = 3B(t) - 2Σ C(m)`. ∎

Verified for all `t < 300`.

At powers of two: `B(2^a) = 1, 4, 12, 40, 128, 416, 1344, 4352, 14080, 45568`
for `a = 0 … 9`. **No growth law is claimed** — the recurrence above is the
exact statement, and the brief forbids inferring a theorem from
powers-of-two data.

## 3. Which potential events are active in the real orbit

Single-cell seed, `t ≤ 19`:

| `t` | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `\|Ker(t)\|` | 4 | 7 | 12 | 15 | 24 | 29 | 40 | 43 | 52 | 61 | 76 | 81 | 96 | 107 | 128 | 131 | 140 | 149 |
| `\|N(t)\|` | 1 | 2 | 2 | 4 | 5 | 7 | 4 | 10 | 9 | 17 | 9 | 16 | 17 | 20 | 12 | 21 | 27 | 30 |
| `\|N(t)\| mod 2` | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 |
| `c_t` | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 1 |

The last two rows are complementary at every `t`, which is `(★)`.

## 4. Parity cancellations

`|N(t)|` is far smaller than `|Ker(t)|` — at `t = 19`, 30 active out of 149
possible. Two mechanisms are exactly identified, and neither is statistical:

> ### Lemma 8.4 (activity is sparse for a structural reason).
> `n_s(j) = 1` requires **two adjacent 1s** in row `s`. By Lemma 1.2(2), any
> `0` at column `j` kills both `n_s(j)` and `n_s(j-1)`. So `|N(t)|` is bounded
> by the number of adjacent-1 pairs inside `Ker(t)`, not by `|Ker(t)|`.

> ### Lemma 8.5 (kernel cancellation is arithmetic, not dynamical).
> `Ker(t)` is thinned from the full cone `{(s,j) : |j| ≤ t-1-s}` — of size
> `Σ_{m<t}(2m+1) = t^2` — down to `B(t)` by the mod-2 condition `T(m,j) = 1`
> alone. At `t = 16`, `t^2 = 256` versus `B(16) = 128`; at `t = 32`,
> `1024` versus `416`. **This thinning has nothing to do with the orbit** and is
> completely described by section 4's digit identities.

## 5. Recursive structure at powers of two

> ### Theorem 8.6 (exact self-similarity of the kernel). Write the time gap as
> `m = q·2^a + r` with `0 ≤ r < 2^a`. By Theorem 4.4, the row of `Ker` at gap
> `m` is the `2^a`-scaled copy of the row at gap `q`, each atom replaced by a
> translated copy of the row at gap `r`; the copies are disjoint when `2r < 2^a`,
> and then the row sizes multiply exactly.

This is a statement about `Ker`, i.e. about **which events could matter**. It is
exact and it is proved. It says nothing about `N(t)`.

> ### What was sought and not found.
> The brief asks whether the **active-event count or parity** has exact
> recurrences. It does **not**, as far as this phase could determine:
>
> * `|N(t)| mod 2` obeys an exact recurrence if and only if `c_t` does, by `(★)`
>   — so asking for one is asking for Problem 1, and is rejected under the
>   brief's "no candidate equivalent to Problem 1" rule.
> * `|N(t)|` as an **integer** was checked against the self-similar splitting of
>   Theorem 8.6 at `t = 2^a`: the integer counts `4, 2, 5, 12, …` at
>   `t = 2, 4, 8, 16` show no relation of the scaled form, because `N` is the
>   intersection of the (self-similar) `Ker` with the (non-self-similar)
>   activity set. **No recurrence is claimed, and none was found.**

## 6. What the skeleton does not establish

1. **`|Ker(t)|` grows; that is not evidence of anything.** It is a statement
   about the rule-150 Green function, identical for rule 30, rule 90 and rule
   150 forcing, and rule 90's centre column is eventually periodic.
2. **`(★)` is a restatement.** `|N(t)| mod 2` aperiodic ⟺ `c_t` aperiodic. Any
   "result" about the parity of the skeleton is a result about Problem 1 in
   different notation, and is rejected as a bridge candidate.
3. **The powers-of-two structure is arithmetic.** It belongs to `Ker`, not to
   `N`, and per the standing rules no theorem is inferred from it.

## 7. Statement inventory

| id | statement | status |
|---|---|---|
| **S-8.1/8.2** | `Ker(t)` and `N(t)`, strictly separated | **DEFINITIONS** |
| **S-8.3** | `B(2t) = 4B(t) - 2ΣC(m)` | **THEOREM**, proved; verified `t < 300` |
| **S-8.4** | activity needs adjacent 1s | **LEMMA**, proved |
| **S-8.5** | kernel thinning is arithmetic (`t²` → `B(t)`) | **LEMMA**, proved |
| **S-8.6** | exact self-similarity of `Ker` | **THEOREM** (= K-4.4) |
| **(★)** | `c_t = S(t) + \|N(t)\| mod 2` | **THEOREM**, verified `t ≤ 40`, 3 seeds |
| **target** | exact recurrence for the active count or parity | **NOT FOUND**; parity version rejected as equivalent to Problem 1 |
