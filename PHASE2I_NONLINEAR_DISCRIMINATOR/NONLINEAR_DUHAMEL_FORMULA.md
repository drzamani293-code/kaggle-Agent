# The Duhamel / forced-linear representation

**Phase 2I, section 2.** Everything is over GF(2). Every formula is checked
against direct rule-30 simulation for **all** cells of small light cones and
for three different seeds.

---

## 1. The Green function of Rule 150

Encode a row as the Laurent polynomial `X(z) = Σ_j x(j) z^j`. The rule-150
operator `(L x)(j) = x(j-1) + x(j) + x(j+1)` is multiplication by
`z + 1 + z^{-1} = z^{-1}(1 + z + z^2)`, so

```
      L^m  <->  z^{-m} P_m(z) ,        P_m(z) := (1 + z + z^2)^m ,
      (L^m x)(j) = Σ_d T(m, d) x(j-d) ,   T(m,d) := [z^{d+m}] P_m(z) .   (GF)
```

> ### Lemma 2.1 (basic properties of `T`).
> 1. `T(m,d) = 0` for `|d| > m`; `T(m, ±m) = 1`.
> 2. `T(m,-d) = T(m,d)` — `P_m` is palindromic.
> 3. **(Frobenius)** `P_m = Π_{i ∈ bits(m)} (1 + z^{2^i} + z^{2^{i+1}})`, so
>    `T(m,d)` is the number of representations `d + m = Σ_{i∈bits(m)} e_i 2^i`
>    with `e_i ∈ {0,1,2}`, reduced mod 2.
> 4. **(explicit binomial form)** since `1+z+z^2 = (1+z^3)/(1+z)` over GF(2),
>    ```
>        T(m,d) = Σ_{3i ≤ e}  C(m, i) · C(m + e - 3i - 1, e - 3i)   (mod 2),
>        e := d + m .
>    ```

*Verification.* (3) is checked against direct polynomial powering for
`m ≤ 40`, and against an independent hyperbinary-representation counter for
`m ≤ 24` over the full support. (4) is checked for `1 ≤ m < 60` over every
exponent — **3599 checks, 0 failures**.

---

## 2. The representation

> ### Theorem 2.2 (Duhamel for Rule 30). For every configuration and every
> `t ≥ 0`, `j ∈ ℤ`,
> ```
>     x_t(j)  =  Σ_d T(t, d) x_0(j-d)   +   Σ_{s=0}^{t-1} Σ_d T(t-1-s, d) n_s(j-d)
>                \______ linear evolution ______/   \____ forced part ____/
> ```
> with `n_s(j) = x_s(j) x_s(j+1)`.

*Proof.* From `x_{t+1} = L x_t + n_t` (R30), induction gives
`x_t = L^t x_0 + Σ_{s<t} L^{t-1-s} n_s`; expand each `L^m` by (GF). ∎

> ### Corollary 2.3 (centre form). Using `T(m,-j) = T(m,j)`,
> ```
>     c_t = x_t(0) = Σ_d T(t,d) x_0(-d)  +  Σ_{s<t} Σ_j T(t-1-s, j) n_s(j) .
> ```
> Write `S(t) := Σ_d T(t,d) x_0(-d)` — the **linear shadow** — and
> `N(t) := { (s,j) : s < t, T(t-1-s, j) = 1, n_s(j) = 1 }`. Then
> ```
>     c_t  =  S(t)  +  |N(t)| mod 2 .                                  (★)
> ```

*Verification.* Theorem 2.2 checked cell by cell on the full light cone:

| seed | `t ≤` | cells | mismatches |
|---|---:|---:|---:|
| `{0}` single cell | 40 | 1763 | **0** |
| `{0,1}` two adjacent | 32 | 1155 | **0** |
| `{0,1,2}` = 111 | 32 | 1155 | **0** |

and (★) verified at every `t ≤ 40` for all three seeds.

---

## 3. Specialisation to the centre: the constant shadow

> ### Theorem 2.4 (the central trinomial coefficient is always odd).
> `T(t, 0) = 1` for every `t ≥ 0`.

*Proof.* By Lemma 2.1(3), `T(t,0)` counts the representations
`t = Σ_{i ∈ bits(t)} e_i 2^i` with `e_i ∈ {0,1,2}`, mod 2. One representation
is `e_i = 1` for all `i`. Suppose another exists; subtracting,
`Σ_i (e_i - 1) 2^{a_i} = 0` with `e_i - 1 ∈ {-1,0,1}` and `a_1 < … < a_k` the
distinct set bits. Since `Σ_{i<k} 2^{a_i} ≤ 2^{a_k} - 1 < 2^{a_k}`, the top
coefficient must vanish; induct downward. So the representation is unique and
the count is `1`. ∎

*Verification.* `T(t,0) = 1` for every `t ≤ 5000`.

> ### Corollary 2.5 (single-cell seed). For `x_0 = δ_0`, `S(t) = T(t,0) = 1`,
> so
> ```
>     c_t  =  1 + |N(t)| mod 2         for every t ≥ 0.
> ```
> **The centre column of Rule 30 is the parity of the number of active
> nonlinear events in its kernel, with the linear part contributing a constant.**

> ### Theorem 2.6 (which seeds have a constant shadow).
> For a finite seed, `S(t)` is constant in `t` **iff** `x_0` is mirror-symmetric
> about column 0 (`x_0(-j) = x_0(j)` for all `j`), and then `S(t) = x_0(0)`.

*Proof.* (⇐) `L` is a symmetric operator, so it preserves mirror symmetry; if
`y_t` is symmetric then `y_{t+1}(0) = y_t(-1) + y_t(0) + y_t(1) = y_t(0)`
because the outer terms are equal and cancel. (⇒) Induct on `t`. `S(0) = x_0(0)`.
Suppose `x_0(±r)` agree for all `r < t`. By Lemma 2.1(1), `T(t,±t) = 1`, and
by the inductive hypothesis all terms with `|d| < t` cancel in pairs, so
`S(t) = x_0(0) + x_0(t) + x_0(-t)`. Constancy forces `x_0(t) = x_0(-t)`. ∎

*Verification, exhaustive.* Over all nonzero seeds with support in `[-W, W]`:

| `W` | seeds | constant shadow | predicted `2^{W+1} - 1` | all symmetric |
|---|---:|---:|---:|---|
| 3 | 127 | 15 | 15 | ✔ |
| 4 | 511 | 31 | 31 | ✔ |
| 5 | 2047 | 63 | 63 | ✔ |
| 6 | 8191 | 127 | 127 | ✔ |

> **Correction 2I-C-01.** An earlier reading of the seed suite suggested that a
> constant shadow characterised the **single-cell** seed. The exhaustive scan
> above refutes that: it characterises **mirror-symmetric** seeds, of which the
> single cell is the smallest. The narrower claim is retracted; the corrected
> statement is Theorem 2.6.

---

## 4. Algebraic normal form, and a second derivation

The ANF of `c_t` in the seed variables, computed two independent ways — by
symbolic evolution of the rule, and by symbolic evaluation of Theorem 2.2:

| `t` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| monomials | 1 | 4 | 10 | 30 | 122 | 346 | 1360 | 4852 |
| degree | 1 | 2 | 3 | 5 | 7 | 9 | 11 | 13 |
| the two derivations agree | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |

Degree `2t-1` for `2 ≤ t ≤ 7`. **No growth law is claimed** — eight data
points, and the standing rules forbid fitting one. The agreement column is the
point: Theorem 2.2 and the rule produce the *same* Boolean function, monomial
for monomial, not merely the same values on the orbit.

---

## 5. What this representation does and does not buy

**Does.** It isolates the nonlinearity into a single additive term with an
explicitly computable mod-2 kernel, reduces the single-cell centre column to
the parity `(★)` with a *constant* linear part, and gives an exact
digit-theoretic criterion (`T(t-1-s,j) = 1`) for which events matter.

**Does not.** It is a change of variables, not new information. Equation `(★)`
says `c_t` is aperiodic iff `|N(t)| mod 2` is aperiodic — the two statements
are **equivalent**, so `(★)` on its own is a restatement of Problem 1 and is
rejected as a bridge candidate under the brief's rule. Its value is that
`|N(t)|` factors through the kernel geometry of section 4 and the event field
of section 8, which are separately analysable; whether that separation buys a
proof is the subject of section 11's decision gate.

---

## 6. Statement inventory

| id | statement | status |
|---|---|---|
| **D-2.1** | properties of `T`, incl. the explicit binomial form | **THEOREM**, 3599 identity checks |
| **D-2.2** | the Duhamel representation | **THEOREM**, 4073 cells × 3 seeds, 0 mismatches |
| **D-2.3** | centre form `c_t = S(t) + \|N(t)\| mod 2` | **COROLLARY**, verified |
| **D-2.4** | `T(t,0) = 1` for all `t` | **THEOREM**, proved; verified to `t = 5000` |
| **D-2.5** | `c_t = 1 + \|N(t)\| mod 2` for the single cell | **COROLLARY** |
| **D-2.6** | constant shadow ⟺ mirror-symmetric seed | **THEOREM**, proved; exhaustive to `W = 6` |
| **2I-C-01** | "constant shadow ⟺ single cell" | **RETRACTED**, preserved |
