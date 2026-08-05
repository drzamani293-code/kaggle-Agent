# Symbolic and SAT controls

**Phase 2I, section 10.** Symbolic computation and SAT are used **only** for
finite falsification and identity checking. **No finite SAT result is reported
as an infinite theorem**, and a search that finds nothing is reported as
"no witness found in the searched range" — never as UNSAT.

---

## 1. Representation

GF(2) polynomials in the seed variables `x_j := x_0(j)` are stored in algebraic
normal form: a set of monomials, each a set of variables. Addition is symmetric
difference; multiplication is pairwise union; `a OR b = a + b + ab`.

**On "minimisation".** Over GF(2) the ANF of a Boolean function is **unique**.
There is nothing to minimise: the monomial count and degree reported below are
canonical invariants of the function, not artefacts of a representation choice.
That is the honest reading of the brief's "minimize them".

## 2. ANF of `c_t`, and two independent derivations

Two derivations are compared:

* **direct** — symbolic evolution of the rule-30 local map on a symbolic row;
* **Duhamel** — symbolic evaluation of Theorem 2.2, i.e. the linear shadow plus
  `Σ_{s<t} Σ_j T(t-1-s,j) · x_s(j) x_s(j+1)` with the `x_s` themselves symbolic.

| `t` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| monomials | 1 | 4 | 10 | 30 | 122 | 346 | 1360 | 4852 |
| degree | 1 | 2 | 3 | 5 | 7 | 9 | 11 | 13 |
| constant term | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **derivations agree** | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |

The agreement is monomial-for-monomial as Boolean functions, not merely
value-for-value on one orbit. This is the strongest identity check in the
phase: it confirms Theorem 2.2 as an algebraic identity, independently of any
seed.

Degree is `2t - 1` for `2 ≤ t ≤ 7`. **No growth law is claimed** — eight data
points, and the standing rules forbid fitting one.

## 3. ANF of `Q_p(t)`

| `p` | `t` | monomials | degree |
|---|---|---:|---:|
| 1 | 2 | 32 | 5 |
| 1 | 4 | 342 | 9 |
| 1 | 6 | 4968 | 13 |
| 2 | 2 | 128 | 7 |
| 2 | 4 | 1398 | 11 |
| 2 | 6 | 23320 | 15 |

`Q_p(t)` is never the zero polynomial in the range computed — i.e. no
`(p, t)` in this range makes `c_{t+p} = c_t` an identity in the seed. That is a
statement about *all* seeds simultaneously and is exactly what an ANF buys;
it is still a **bounded** statement, over `p ≤ 2`, `t ≤ 6`.

**Rule comparison at `t = 4`** (the discriminator, in ANF form):

| rule | 30 | 90 | 150 |
|---|---|---|---|
| ANF degree of `c_4` | **7** | 1 | 1 |

Rules 90 and 150 have **affine** centre columns at every `t` — degree 1 — which
is the ANF-level restatement of Theorem 1.1. Any candidate property that does
not raise the degree above 1 cannot separate rule 30 from rule 90.

## 4. Finite search across seeds, two methods

Question posed: *is there a nonzero seed with support in `[-W, W]` whose centre
column satisfies `c_{t+p} = c_t` for every `t ∈ [3, 9]`?*

Two independent methods: exhaustive enumeration of all `2^{2W+1} - 1` seeds,
and a SAT encoding solved by MiniSat (pysat) in which each violating assignment
is forbidden by a clause.

| `W` | `p` | brute force | SAT | agree |
|---|---|---|---|---|
| 5 | 1 | witnesses found (2) | witness found | ✔ |
| 5 | 2 | **no witness found in the searched range** | no witness found in the searched range | ✔ |
| 5 | 3 | witnesses found (2) | witness found | ✔ |
| 4 | 1 | no witness found in the searched range | no witness found in the searched range | ✔ |
| 4 | 2 | no witness found in the searched range | no witness found in the searched range | ✔ |
| 4 | 3 | no witness found in the searched range | no witness found in the searched range | ✔ |

Witnesses at `W = 5`, `p = 1`: supports `{-5,-2,1,4}` and `{-5,-2,1,4,5}`.
At `p = 3`: `{-5,-3,-1,0,2,3,4}` and `{-5,-3,-1,0,2,3,5}`.

> ### What a witness is, and is not.
> A witness is a seed whose centre column happens to agree at lag `p` across the
> **seven rows** `t = 3 … 9`. It is a **finite coincidence**. It is not eventual
> periodicity, it does not become eventual periodicity at any larger window, and
> none of these seeds is claimed to have a periodic centre column. The searches
> are falsification tools for the *bounded* statement only.

> ### Failed check 2I-X-04 (recorded, not repaired silently).
> The first run of the driver reported "witnesses found" from brute force at
> `p = 1` while the SAT branch reported "no witness found" — an apparent
> contradiction between two methods that must agree. **Cause: a window
> mismatch**, `W = 5` in the brute-force call and a hard-coded `W = 4` in the
> SAT call. It was not a mathematical discrepancy. The driver now passes the
> same `W` to both and asserts their agreement; the table above is the corrected
> run, and the failure is preserved here rather than deleted.

## 5. What the symbolic layer establishes

**Establishes.** That Theorem 2.2 is an algebraic identity in the seed
variables, not merely an orbit coincidence (§2). That the ANF degree separates
rule 30 from rules 90 and 150 at every computed `t` (§3). That two independent
search methods agree, once given the same question (§4).

**Does not establish.** Anything about infinite time. Every table here has an
explicit finite range. `Q_p` being nonzero as a polynomial for `p ≤ 2`, `t ≤ 6`
says nothing about `p = 10^6`, and no extrapolation is made.

## 6. Statement inventory

| id | statement | status |
|---|---|---|
| **Y-10.1** | direct and Duhamel ANFs coincide, `t ≤ 7` | **IDENTITY CHECK**, 8/8 |
| **Y-10.2** | ANF degree of `c_t` is `2t-1`, `2 ≤ t ≤ 7` | **BOUNDED OBSERVATION**; no law claimed |
| **Y-10.3** | `c_t` is affine for rules 90 and 150 | **THEOREM** (linearity), verified |
| **Y-10.4** | `Q_p(t) ≢ 0` as a polynomial, `p ≤ 2`, `t ≤ 6` | **BOUNDED OBSERVATION** |
| **Y-10.5** | brute force and SAT agree on all 6 aligned questions | **CROSS-CHECK** |
| **2I-X-04** | the window-mismatch discrepancy | **PRESERVED** |
