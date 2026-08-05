# Definitions and Notation

One consistent notation for the whole consolidated corpus. Every earlier phase
is translated into this notation; where a phase used a different convention it
is named here and the conversion is given.

---

## 1. The cellular automaton

**Lattice and time.** `j ∈ Z` (space), `t ∈ N = {0,1,2,…}` (time). Time runs
downwards in every printed diagram.

**Local rule.** Rule 30 in Wolfram's numbering is the map

```
    f : {0,1}^3 → {0,1},        f(l, c, r)  =  l  XOR  ( c  OR  r ).
```

Indexed by `4l + 2c + r`, the output table is `(0,1,1,1,1,0,0,0)` reading
`(l,c,r) = (0,0,0), (0,0,1), …, (1,1,1)` — i.e. Wolfram's `30 = 00011110₂`
read from neighbourhood `111` down to `000`.

**Global evolution.**

```
    x_{t+1}(j)  =  f( x_t(j-1), x_t(j), x_t(j+1) )
                =  x_t(j-1)  XOR  ( x_t(j)  OR  x_t(j+1) ).
```

**Orientation, stated once.** The **left** argument `l = x_t(j-1)` is the one
the XOR acts on. Consequently rule 30 is **left-permutive**: for fixed
`(c, r)`, `l ↦ f(l,c,r)` is a bijection. It is **not** right-permutive:
`f(l,c,0) = f(l,c,1)` whenever `c = 1`. Every asymmetry in this corpus traces
back to this one sentence.

**Mirror.** The left–right mirror of rule 30 is rule 86. A mirrored engine is
used throughout as an independent implementation.

**Seeds.** Three domains appear and must never be conflated:

| symbol | meaning |
|---|---|
| **ANY** | every bi-infinite configuration `x_0 ∈ {0,1}^Z` |
| **FIN** | every finitely supported `x_0` (all but finitely many cells `0`) |
| **SEED** | the single-cell seed `x_0(j) = [j = 0]` only |

`SEED ⊂ FIN ⊂ ANY`. A theorem proved for ANY is strictly stronger. **Phase 1
§10.3's seed-blindness obstruction** says: a statement true for all of FIN
cannot settle Problem 1, because Problem 1 is about SEED and FIN contains
configurations with eventually periodic centre columns.

**Centre column.** `c_t := x_t(0)`. **Problem 1** asks whether `(c_t)_{t≥0}` is
eventually periodic.

**Eventually periodic.** A sequence `(u_t)` is *eventually `p`-periodic from
`T`* if `u_{t+p} = u_t` for all `t ≥ T`. Its **preperiod** is the least such
`T` for the least such `p`, and its **period** is that least `p`. For an orbit
of a deterministic map these are the usual `rho` and `lambda`.

---

## 2. The temporal defect field

For a lag `p ≥ 1`:

```
    d_t(j)  :=  x_{t+p}(j)  XOR  x_t(j).
```

`p` is suppressed from the notation; every statement about `d` fixes one `p`.

**(H)** — the periodicity hypothesis — is: there exist `p ≥ 1`, `T* ≥ 0` with
`c_{t+p} = c_t` for all `t ≥ T*`. Equivalently `d_t(0) = 0` for all `t ≥ T*`,
the **zero wall** at column 0. Phase 2B writes this **(H-WALL)**; they are the
same hypothesis and this document uses **(H)**.

---

## 3. Edge-aligned coordinates

```
    w_t(k)  :=  x_t(-t + k),        k ∈ Z,   t ∈ N.
```

So `k` counts rightwards from the left edge of the light cone. Conventions:

* `w_t(k) = 0` for `k < 0` (a *definition*, consistent with `x_t(j) = 0` for
  `j < -t` under SEED and FIN);
* `w_0 = (1, 0, 0, …)` under SEED;
* `w_t(k) = 0` for `k > 2t` (the **support bound**, proved as Theorem T7.3).

**The one-sided recurrence** (Theorem T7):

```
    w_{t+1}(k)  =  w_t(k-2)  XOR  ( w_t(k-1)  OR  w_t(k) ).
```

**Prefix.** `W_t^K := ( w_t(0), …, w_t(K) ) ∈ {0,1}^{K+1}`.
**Prefix map.** `F_K : {0,1}^{K+1} → {0,1}^{K+1}`, `(F_K z)_k = z_{k-2} XOR
(z_{k-1} OR z_k)` with `z_{-1} = z_{-2} = 0`. Then `W_{t+1}^K = F_K(W_t^K)`.
**Projection.** `π_K : {0,1}^{K+2} → {0,1}^{K+1}` drops the last coordinate.

**Prefix preperiod and period.**

```
    P(K) := the exact (minimal) period of the orbit ( W_t^K )_{t≥0}
    T(K) := the exact preperiod: least t with W_t^K = W_{t+P(K)}^K
```

**Coordinate preperiod.** For a lag `q` with `P(k) | q`,

```
    r_q(k) := least t such that w_s(k) = w_{s+q}(k) for every s ≥ t.
```

`T(K) = max_{k ≤ K} r_{P(K)}(k)` (Theorem T21.1). Note `r_{P(K)}(K) ≤ T(K)`,
usually with equality but **not always** — the two are different functions.

---

## 4. Fibre, reset times, classes

At level `K ≥ 2`, with

```
    a_t := w_t(K-2),      c_t := w_t(K-1),      u_t := w_t(K),
```

the top coordinate obeys the **skew-product fibre recursion**

```
    u_{t+1}  =  a_t  XOR  ( c_t  OR  u_t ) ,
```

a one-bit fibre over the `(K-1)`-prefix base. **`c_t = 1` is a reset**: it
makes the fibre map constant, and the fibre forgets its own value.

**The three reset times** (all distinct in general — Theorem T21.4):

```
    σ(K) := min{ t ≥ 0      : w_t(K-1) = 1 }     first reset ever
    τ(K) := min{ t ≥ T(K-1) : w_t(K-1) = 1 }     first reset after the base settles
    ρ(K) := max{ t < T(K)   : w_t(K-1) = 1 }     last reset before the level settles
```

Each may be undefined; the documents say where.

**Fibre classes**, determined by the base cycle word `(a_t, c_t)_{t ∈ [T(K-1),
T(K-1)+P(K-1))}`:

| class | condition | one-period fibre map |
|---|---|---|
| **COLLAPSING** | some `c_t = 1` on the cycle | constant |
| **NEUTRAL** | all `c_t = 0` and `XOR_t a_t = 0` | identity |
| **DOUBLING** | all `c_t = 0` and `XOR_t a_t = 1` | negation |

**Defect bit.** `D(K) := w_{T(K-1)}(K) XOR w_{T(K-1)+P(K)}(K)`.

**Transient age and frontier.**

```
    A(t, K) := max( 0, T(K) - t ),        A(t) := A(t, t)
    K_per(t) := max{ K : T(K) ≤ t }       (-1 if none)
```

**Moving strip.**

```
    q_t(r)  :=  w_t(t + r)  =  x_t(r),        -R ≤ r ≤ R.
```

The second equality is Theorem T21.6; it is why the strip is not a new object.

---

## 5. Conversion table between coordinate systems

| object | original `x` | edge-aligned `w` | strip `q` |
|---|---|---|---|
| a cell | `x_t(j)` | `w_t(t + j)` | `q_t(j)` |
| index change | — | `k = t + j`, `j = k - t` | `r = j` |
| centre column | `x_t(0)` | `w_t(t)` — the **diagonal** | `q_t(0)` |
| column `r` | `x_t(r)` | `w_t(t+r)` — a line of slope 1 | `q_t(r)` |
| left light-cone edge | `x_t(-t)` | `w_t(0)` | — |
| right light-cone edge | `x_t(t)` | `w_t(2t)` | — |
| local rule | `x_t(j-1) XOR (x_t(j) OR x_t(j+1))` | `w_t(k-2) XOR (w_t(k-1) OR w_t(k))` | `q_t(r-1) XOR (q_t(r) OR q_t(r+1))` |
| autonomous sets | right half-lines `j ≥ j₀`? **no** | left prefixes `k ≤ K`? **yes** | any strip? **no** |
| support | `x_t(j) = 0` for `|j| > t` | `w_t(k) = 0` for `k < 0` or `k > 2t` | `q_t(r) = 0` for `|r| > t` |

The single content of the change of variables is the last-but-one row: in `w`
coordinates the rule is **one-sided to the left**, so every prefix is closed.
In `x` and `q` coordinates no finite window is closed, because rule 30 is not
right-permutive.

---

## 6. Indexing conventions, resolved explicitly

1. **Time indices in the local rule.** `x_{t+1}` is computed from `x_t`. Every
   defect and reset statement uses the same convention; the identity
   `d_{t+1}(j) = …` therefore reads inputs at time `t`.
2. **`k-2` not `k+2`.** In edge-aligned coordinates the XOR argument is at
   `k-2` because `j-1 = (k-2) - (t)` after the frame shifts by one. A sign
   error here would exchange left-permutivity for right-permutivity; the
   convention is pinned by Theorem T7 and re-verified cell-by-cell in the test
   suite.
3. **Bit packing.** Where a row is stored as an integer, **bit `k` is
   `w_t(k)`** (edge-aligned) and **bit `j + C` is `x_t(j - C)`** for an offset
   `C` (original). Strip states pack **bit `j` as `q(j - R)`**.
4. **Cycle-word phase.** A cycle word of coordinate `k` is always read *from
   phase `T(k)`* unless stated otherwise, i.e. it is the word
   `w_{T(k)}(k), …, w_{T(k)+P(k)-1}(k)`. Cycle words read from a different
   phase are rotations of it and are never compared across phases without
   saying so.
5. **Minimal vs non-minimal periods.** `P(K)` is always **minimal**. Where a
   statement holds at a non-minimal lag `q` (a multiple of `P(K)`), the lag is
   written explicitly as `r_q(·)`. Preperiods at larger lags can be **strictly
   smaller** (Theorem T21.2), so the distinction is load-bearing.
6. **Preperiod synchronisation.** Two eventually periodic sequences with data
   `(T₁,p₁)`, `(T₂,p₂)` are simultaneously `(max(T₁,T₂), lcm(p₁,p₂))`-periodic.
   `max` for preperiods, `lcm` for periods — never `max` for both.

---

## 7. Claim-ID scheme used in the ledger

```
    T-nn    formally proved theorem                     (ledger category A)
    CT-nn   formally proved conditional theorem         (B)
    CC-nn   finite computational certificate            (C)
    BO-nn   bounded observation                         (D)
    CJ-nn   conjecture                                  (E)
    RF-nn   refuted claim                               (F)
    VQ-nn   vacuous or equivalent-to-target claim       (G)
    BR-nn   open bridge                                 (H)
```

IDs are stable: once assigned they are never reused or renumbered, even if a
claim is later downgraded. A downgraded claim keeps its old ID and gains a
status note.
