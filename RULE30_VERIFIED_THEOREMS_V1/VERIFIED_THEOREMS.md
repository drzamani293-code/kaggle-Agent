# Verified Theorems

Only statements with a complete proof appear here. Every proof is written from
the definitions in `DEFINITIONS_AND_NOTATION.md`, with **no reference to code**.
Computations appear only in `COMPUTATIONAL_CERTIFICATES.md`; where a theorem
depends on one, it is stated as **conditional** and the dependency is named.

Each entry gives: the fully quantified statement, its **domain** (ANY / FIN /
SEED), its dependencies, the proof, and a paragraph **"why this does not solve
Problem 1"**.

Numbering follows the Phase 2F brief's T1–T21. Original phase labels are given
so the corpus can be traced back.

---

## T1 — Left-permutivity and exact left reconstruction

> **(a)** For every `(c, r) ∈ {0,1}²`, the map `l ↦ f(l,c,r) = l XOR (c OR r)`
> is a bijection of `{0,1}`.
> **(b)** For every configuration `x_0` (domain **ANY**), every `t ∈ N` and
> every `j ∈ Z`,
> ```
>     x_t(j-1)  =  x_{t+1}(j)  XOR  ( x_t(j)  OR  x_t(j+1) ).
> ```
> **(c)** `f` is **not** right-permutive: `f(l,1,0) = f(l,1,1)` for both `l`.

*Domain:* ANY. *Depends on:* nothing.

*Proof.* (a) `l ↦ l XOR b` is an involution for each fixed `b = c OR r`.
(b) Apply XOR by `(x_t(j) OR x_t(j+1))` to both sides of the defining
recurrence `x_{t+1}(j) = x_t(j-1) XOR (x_t(j) OR x_t(j+1))` and use
`b XOR b = 0`. (c) With `c = 1`, `c OR r = 1` for both `r`, so the output does
not depend on `r`. ∎

*Consequence used later.* Given two adjacent columns `col_j`, `col_{j+1}` in
full (all times), (b) determines `col_{j-1}` at every time, hence — by
induction — the entire left half-plane. This is the **left reconstruction**
property.

*Original label:* Phase 1 Lemma 1 / Facts F3, F4.

**Why this does not solve Problem 1.** T1 is a statement about the local rule
alone. It holds for every configuration, and in particular for configurations
whose centre column *is* eventually periodic. It supplies the machinery for
T2 and T3 and no more.

---

## T2 — No two adjacent columns are both eventually periodic

> Let `x_0` be finitely supported (domain **FIN**). Then there is no `j ∈ Z`
> such that both `t ↦ x_t(j)` and `t ↦ x_t(j+1)` are eventually periodic.

*Domain:* FIN (hence SEED). *Depends on:* T1, T2.1, T2.2, T2.3, T2.4.

### T2.1 (light cone). For `x_0` finitely supported with support in `[-m, m]`, `x_t(j) = 0` whenever `|j| > m + t`.

*Proof.* Induction on `t`: if all three inputs at `(t, j-1), (t,j), (t,j+1)`
are `0` then `f(0,0,0) = 0`. ∎

### T2.2 (frozen left edge). Under **SEED**, `x_t(-t) = 1` for every `t`, and more generally under FIN with leftmost support point `-m`, `x_t(-m-t) = 1` for every `t`.

*Proof.* Induction: `x_{t+1}(-m-t-1) = x_t(-m-t-2) XOR (x_t(-m-t-1) OR
x_t(-m-t))`. By T2.1 the first two arguments are `0`, and the third is `1` by
induction, so the value is `0 XOR (0 OR 1) = 1`. ∎

### T2.3 (normalisation). If `(u_t)` is eventually `(T₁,p₁)`-periodic and `(v_t)` eventually `(T₂,p₂)`-periodic, both are eventually `(max(T₁,T₂), lcm(p₁,p₂))`-periodic.

*Proof.* Immediate. ∎

### T2.4 (leftward propagation, uniform in `T`). If `col_j` and `col_{j+1}` are both `(T,p)`-periodic — the *same* `T` and `p` — then `col_{j-k}` is `(T,p)`-periodic for every `k ≥ 0`, **with the same `T` and `p`**.

*Proof.* Induction on `k`. For `k = 1`, T1(b) at time `t ≥ T` gives
`x_t(j-1) = x_{t+1}(j) XOR (x_t(j) OR x_t(j+1))`. Every term on the right is
`p`-periodic in `t` for `t ≥ T` (the first uses `t+1 ≥ T` too), so the left is
`p`-periodic for `t ≥ T`. The inductive step applies the same identity to the
pair `(col_{j-k}, col_{j-k+1})`, which by hypothesis and the previous step are
both `(T,p)`-periodic. **The preperiod does not grow**, because T1(b) reads
only times `t` and `t+1`. ∎

*Proof of T2.* Suppose `col_j` and `col_{j+1}` are both eventually periodic.
By T2.3 they are both `(T,p)`-periodic for some common `(T,p)`. By T2.4 every
column `col_{j-k}`, `k ≥ 0`, is `(T,p)`-periodic with the same `T`.

Fix `k` large enough that `j - k < -m - T`. By T2.1, `x_T(j-k) = 0`, since
`|j-k| > m + T`. By `(T,p)`-periodicity, `x_t(j-k) = 0` for **all** `t ≥ T`
with `t ≡ T (mod p)`. But by T2.2 the leftmost cell of the light cone at time
`t` is `1`: taking `t = m + k - j` (which is `≥ T` for `k` large and satisfies
`-m-t = j-k`) gives `x_t(j-k) = 1`. Choosing `k` so that in addition
`m + k - j ≡ T (mod p)` — possible since `k` ranges over an infinite arithmetic
progression — yields `0 = 1`, a contradiction. ∎

*Original label:* Phase 1 Theorem W2.

**Why this does not solve Problem 1.** T2 forbids *two adjacent* periodic
columns. Problem 1 asks about *one* column. Nothing in T2 rules out a single
eventually periodic column, and T3 shows that is the exact limit of this line
of argument. Moreover T2 holds verbatim for **every** finitely supported seed,
so by the seed-blindness obstruction it cannot distinguish the single-cell
seed and cannot settle Problem 1.

---

## T3 — At most one column of the diagram is eventually periodic

> Let `x_0` be finitely supported (domain **FIN**). Then the set
> `{ j ∈ Z : t ↦ x_t(j) is eventually periodic }` has at most one element.

*Domain:* FIN (hence SEED). *Depends on:* T2, T3.1, T3.2.

### T3.1 (boundary forcing). For `i < j`, the interior values `x_t(i+1..j-1)` for `t ≥ t₀` are determined by `x_{t₀}(i..j)` together with the two boundary columns `col_i` and `col_j` restricted to `[t₀, ∞)`.

*Proof.* The rule has radius 1, so each interior cell at time `t+1` reads only
cells in `[i, j]` at time `t`; those at the two ends are supplied by the
boundary columns. Induct on `t`. ∎

### T3.2 (interior inherits eventual periodicity). If `col_i` and `col_j` (`i < j`) are both eventually periodic, then so is every `col_h` with `i < h < j`.

*Proof.* By T2.3 take a common `(T,p)`. Consider the state
`S_t := x_t(i..j) ∈ {0,1}^{j-i+1}` for `t ≥ T`. By T3.1, `S_{t+1}` is
determined by `S_t` and by the boundary values at time `t`, which are
`p`-periodic. So the map `(S_t, t mod p) ↦ (S_{t+1}, t+1 mod p)` is a
deterministic map on the finite set `{0,1}^{j-i+1} × Z/p`, and the orbit of
`S_T` is eventually periodic in `t` with period dividing
`p · 2^{j-i+1}`. Every interior column is a coordinate of `S_t`, hence
eventually periodic. ∎

*Proof of T3.* Suppose `col_i` and `col_j` are both eventually periodic with
`i < j`. If `j = i+1` this contradicts T2. Otherwise T3.2 makes `col_{i+1}`
eventually periodic, and then `col_i, col_{i+1}` are two adjacent eventually
periodic columns — contradicting T2 again. ∎

*Original label:* Phase 1 Theorem W2′.

**Why this does not solve Problem 1.** T3 says the centre column may be the
**unique** eventually periodic column — which is exactly the scenario Problem 1
asks us to exclude. T3 is the sharpest statement obtainable without using the
seed, and it stops one column short. Every later phase's route ends by needing
a *second* periodic column; that gap is the **Bridge** (`BR-01`).

---

## T4 — Factor-complexity lower bound (CONDITIONAL)

> **Lemma (factor counting).** If `(u_t)` is eventually periodic with preperiod
> `T` and period `p`, then for every `n ≥ 1` the number of distinct factors
> (contiguous subwords) of `(u_t)` of length `n` is at most `T + p`.
>
> **Corollary (conditional).** *Given* the certified count `CC-02` — that the
> first `10⁶` bits of the centre column of the single-cell orbit contain
> **998140** distinct factors of length 28 — if the centre column is eventually
> periodic with preperiod `T` and period `p`, then
> ```
>     T + p  ≥  998140.
> ```

*Domain of the lemma:* ANY sequence. *Domain of the corollary:* SEED.
*Depends on:* the lemma, plus the finite certificate `CC-02`.

*Proof of the lemma.* Let `u` be eventually `p`-periodic from `T`. Any factor
of length `n` starting at position `i ≥ T` equals the factor starting at
`i - p` if `i - p ≥ T`; so every such factor equals one starting in
`[T, T+p)`. Together with the factors starting in `[0, T)` there are at most
`T + p` distinct positions to consider, hence at most `T + p` distinct factors.
∎

**This is a conditional theorem, not an unconditional one.** The number 998140
is a *measured* factor count of a finite prefix, produced by a program. The
mathematics contributed by this entry is the lemma; the numeral is a
certificate. Anyone re-deriving the bound must re-run or re-verify `CC-02`.
Recorded as ledger category **B** (conditional theorem) with a **C**
(certificate) dependency, never as category **A**.

**Why this does not solve Problem 1.** The bound grows only linearly with the
length of the computed prefix and can never exceed it: with `N` bits one gets
at best `T + p ≲ N`. No finite computation of this kind can rule out
periodicity, only push the threshold. This is the exact sense in which "finite
computation does not prove non-periodicity".

---

## T5 — The exact temporal-defect equation

> Fix `p ≥ 1` and set `d_t(j) = x_{t+p}(j) XOR x_t(j)`. Then for every
> configuration (domain **ANY**), every `t` and every `j`, over GF(2):
> ```
>     d_{t+1}(j) = d_t(j-1) + d_t(j) + d_t(j+1)
>                + x_t(j)·d_t(j+1) + x_t(j+1)·d_t(j) + d_t(j)·d_t(j+1).
> ```
> Equivalently, writing `Δ(u, v, e, g) := (u XOR e) OR (v XOR g)  XOR  (u OR v)`
> for `u = x_t(j)`, `v = x_t(j+1)`, `e = d_t(j)`, `g = d_t(j+1)`:
> ```
>     d_{t+1}(j) = d_t(j-1) XOR Δ( x_t(j), x_t(j+1), d_t(j), d_t(j+1) ).
> ```

*Domain:* ANY. *Depends on:* nothing.

*Proof.* Write `x_{t+p}(i) = x_t(i) XOR d_t(i)`. Then

```
    d_{t+1}(j) = x_{t+p+1}(j) XOR x_{t+1}(j)
               = [ x_{t+p}(j-1) XOR ( x_{t+p}(j) OR x_{t+p}(j+1) ) ]
                 XOR [ x_t(j-1) XOR ( x_t(j) OR x_t(j+1) ) ]
               = d_t(j-1) XOR [ (u XOR e) OR (v XOR g) ] XOR [ u OR v ],
```

with `u,v,e,g` as above. That is the second form. The ANF form follows by
expanding `OR` as `a OR b = a + b + ab` over GF(2) and simplifying; the
expansion has 16 cases in `(u,v,e,g)` and both forms agree on all of them, so
the identity is a Boolean identity in four variables and holds identically. ∎

*Two corollaries used later, both immediate from the case table:*

* **T5.1 (masking).** `Δ(1, v, 0, g) = 0` for all `v, g`: a black centre cell
  with no defect masks any defect on its right.
* **T5.2 (support edges).** Under **SEED**, `d_t(±(t+p)) = 1` for every `t` and
  every `p ≥ 1`. *Proof:* by T2.1, `x_t(t+p) = 0` while `x_{t+p}(t+p) = 1` by
  T2.2 mirrored; similarly on the left. Hence the defect field is **never
  empty**, at any time, for any lag.

**Why this does not solve Problem 1.** T5 is an identity, true for every
configuration and every lag. It is the correct algebra in which to phrase the
wall hypothesis, and it constrains nothing by itself. T5.2 does distinguish the
real orbit from a seed-free relaxation (in which `d ≡ 0` is invariant), but it
does so at the *support edges*, which move away from column 0 at speed 1.

---

## T6 — Zero-wall identities, in their strongest actually proved forms

All four are **conditional theorems**: they assume the wall

> **(H-local)** `d_s(0) = 0` for `s` in a stated finite range.

They are *not* assumed to hold globally, and each states its own range. Domain:
**ANY** configuration (the proofs use only the local rule).

### T6.1 (= W1, the wall equation). If `d_t(0) = d_{t+1}(0) = 0`, then

```
    d_t(-1)  =  d_t(1) · ( 1 XOR x_t(0) ) .
```

*Proof.* Apply T5 at `j = 0`: `d_{t+1}(0) = d_t(-1) XOR Δ(x_t(0), x_t(1),
d_t(0), d_t(1))`. With `d_t(0) = 0` the case table gives
`Δ(u, v, 0, g) = g · (1 XOR u)`. Setting `d_{t+1}(0) = 0` and solving for
`d_t(-1)` gives the claim. ∎

### T6.2 (= W2, half-plane decoupling). If `d_s(0) = 0` for all `s ∈ [t₀, t₁]`, then for `t ∈ [t₀, t₁-1]` the values `d_{t+1}(j)` for `j ≥ 1` are determined by `x_t(·)` and by `d_t(·)` restricted to `j ≥ 1` — no value from `j ≤ -1` enters.

*Proof.* T5 at `j ≥ 1` reads `d_t(j-1), d_t(j), d_t(j+1)` with `j - 1 ≥ 0`, and
`d_t(0) = 0` is known. ∎

### T6.3 (= W3, the zero triangle). If `d_s(0) = 0` for all `s ∈ [t₀, t₁+1]` and `x_s(0) = 1` for all `s ∈ [t₀, t₁]`, then

```
    d_s(-m) = 0    for every m ∈ [1, t₁-t₀+1]  and every s ∈ [t₀, t₁-(m-1)].
```

*Proof.* `m = 1` is T6.1 with `x_s(0) = 1`. For the inductive step, apply T5 at
`j = -m+1`, whose inputs at time `s` are `d_s(-m), d_s(-m+1), d_s(-m+2)`; the
last two vanish on the stated range by the inductive hypothesis, and
`d_{s+1}(-m+1) = 0` likewise, so the identity forces `d_s(-m) = 0`. Each step
costs one time unit at each end, giving the triangle. ∎

### T6.4 (= W6, runs of zeros). If `d_s(0) = 0` for `s ∈ [t₀, t₁+1]` and `x_s(0) = 0` for `s ∈ [t₀, t₁]`, then `d_s(-1) = d_s(1)` for `s ∈ [t₀, t₁]`, and `d_s(-2) = d_{s+1}(1) XOR d_s(1)` for `s ∈ [t₀, t₁-1]`.

*Proof.* The first is T6.1 with `x_s(0) = 0`. The second is T5 at `j = -1`
using the first. ∎

*Original labels:* Phase 2B W1, W2, W3, W6. **W4, W5, W7 are corollaries or
summaries of these and are not restated as independent theorems.**

**Why these do not solve Problem 1.** Every statement is conditional on a wall
that is conjecturally never realised for long. Under the expected answer to
Problem 1 the hypotheses are satisfied only on short ranges (the longest real
wall found over 83 lags × 30000 steps has length **18**), so the theorems are
close to vacuous on the real orbit; and under (H) they are consequences of an
assumption believed false. They constrain the *left* half-plane in terms of the
*right*, which is the correct direction, but they never close the loop back to
column 0.

---

## T7 — The edge-aligned one-sided recurrence

> For every configuration (domain **ANY**), with `w_t(k) := x_t(-t+k)`:
> ```
>     w_{t+1}(k)  =  w_t(k-2)  XOR  ( w_t(k-1)  OR  w_t(k) )       for all t, k.
> ```

*Domain:* ANY. *Depends on:* nothing.

*Proof.* `w_{t+1}(k) = x_{t+1}(-(t+1)+k) = x_{t+1}(-t + (k-1))`. The rule gives
`x_{t+1}(j) = x_t(j-1) XOR (x_t(j) OR x_t(j+1))` at `j = -t + (k-1)`, whose
three arguments are `x_t(-t + k - 2) = w_t(k-2)`, `x_t(-t+k-1) = w_t(k-1)` and
`x_t(-t+k) = w_t(k)`. ∎

### T7.1 (boundary). Under **FIN** with support in `[-m, m]`, `w_t(k) = 0` for `k < 0` when `m = 0`; in general `w_t(k) = 0` for `k < -m + 0`. Under **SEED**, `w_t(k) = 0` for `k < 0`.

*Proof.* T2.1. ∎

### T7.2 (seed). Under **SEED**, `w_0 = (1,0,0,…)`.

### T7.3 (support bound). Under **SEED**, `w_t(k) = 0` for `k > 2t`.

*Proof.* `w_t(k) = x_t(k - t)`, and by T2.1 this vanishes when `k - t > t`. ∎

**Why this does not solve Problem 1.** T7 is a change of variables. It moves
the asymmetry of rule 30 into a convenient place — and by T21.6 the centre
column becomes a *diagonal*, which is precisely the object none of the prefix
theory reaches.

---

## T8 — Every fixed prefix is eventually periodic

> Under **SEED** (indeed under **ANY** configuration with `w_t(k) = 0` for
> `k<0`), for every `K ≥ 0` the sequence `(W_t^K)_{t ≥ 0}` satisfies
> `W_{t+1}^K = F_K(W_t^K)` for **every** `t ≥ 0`, and is therefore eventually
> periodic with preperiod `T(K)` and period `P(K)` satisfying
> `T(K) + P(K) ≤ 2^{K+1}`.

*Domain:* ANY with a left boundary; in particular SEED. *Depends on:* T7.

*Proof.* By T7 the value `w_{t+1}(k)` depends only on `w_t(k')` for
`k' ∈ {k-2, k-1, k}`, all `≤ k ≤ K`. Hence `W_{t+1}^K` is a function of
`W_t^K` alone; call it `F_K`. The state space `{0,1}^{K+1}` is finite, so the
orbit of a deterministic map on it is eventually periodic, with tail plus cycle
at most `2^{K+1}`. ∎

**Correction recorded.** Phase 2B stated this with the qualifier "for all
`t ≥ K/2`". **That qualifier is unnecessary and is removed**: the argument
above uses no lower bound on `t`. See `CORRECTIONS_AND_RETRACTIONS.md` C-08.

*Original label:* Phase 2B Theorem EA2.

**Why this does not solve Problem 1.** T8 gives eventual periodicity of every
*fixed* prefix. The centre column is not read off a fixed prefix: by T21.6 it
is `w_t(t)`, whose level grows with `t`. The whole difficulty of Phases 2C–2E
is that the diagonal escapes every fixed `K`.

---

## T9 — The projection relation

> For every `K ≥ 0` and every `z ∈ {0,1}^{K+2}`,
> ```
>     π_K ( F_{K+1}(z) )  =  F_K ( π_K(z) ).
> ```

*Domain:* algebraic identity, no configuration needed. *Depends on:* the
definition of `F_K`.

*Proof.* For `k ≤ K`, `(F_{K+1} z)_k = z_{k-2} XOR (z_{k-1} OR z_k)`, which
involves only coordinates `≤ K` of `z`, i.e. only `π_K(z)`; and that expression
is exactly `(F_K π_K(z))_k`. Dropping coordinate `K+1` on the left is exactly
the restriction to `k ≤ K`. ∎

### T9.1 (monotone preperiod, = T12). `T(K-1) ≤ T(K)` and `P(K-1) | P(K)`.

*Proof.* By T9, `π_{K-1}` maps the `K`-orbit onto the `(K-1)`-orbit and
intertwines the dynamics. If the `K`-orbit is periodic from `T(K)` with period
`P(K)`, its image is periodic from `T(K)` with period dividing `P(K)`; hence
the *minimal* preperiod of the image satisfies `T(K-1) ≤ T(K)` and the minimal
period `P(K-1)` divides `P(K)`. ∎

*Original label:* Phase 2C Theorem P1, S3.

**Why this does not solve Problem 1.** T9 organises the prefixes into a tower.
The tower is entirely a statement about fixed levels.

---

## T10 — The skew-product trichotomy

> Let `K ≥ 2`, let the base `(K-1)`-prefix be periodic from `T_- = T(K-1)` with
> period `P_- = P(K-1)`, and read the base cycle word
> `(a_j, c_j)_{j=0}^{P_- - 1}` with `a_j = w_{T_-+j}(K-2)`, `c_j = w_{T_-+j}(K-1)`.
> Let `G := g_{P_- - 1} ∘ … ∘ g_0` where `g_j(u) = a_j XOR (c_j OR u)`.
> Then exactly one of:
> ```
>   COLLAPSING :  some c_j = 1                  =>  G is constant
>   NEUTRAL    :  all c_j = 0 and XOR_j a_j = 0 =>  G = identity
>   DOUBLING   :  all c_j = 0 and XOR_j a_j = 1 =>  G = negation
> ```

*Domain:* any base orbit; applied under SEED. *Depends on:* T7, T8.

*Proof.* Each `g_j` is either constant (`c_j = 1`, value `a_j XOR 1`) or the
affine bijection `u ↦ a_j XOR u` (`c_j = 0`). A composition of affine maps on
`{0,1}` is constant as soon as one factor is constant, giving the first case.
If no factor is constant, the composition is `u ↦ (XOR_j a_j) XOR u`, which is
the identity or the negation according to the parity. ∎

### T10.1 (= T11, period relation). `P(K) ∈ { P(K-1), 2·P(K-1) }`, and `P(K) = 2 P(K-1)` **iff** the level is DOUBLING. Consequently `P(K) = 2^{m(K)}` with `m` non-decreasing and `m(K) - m(K-1) ∈ {0,1}`.

*Proof.* On the base cycle the fibre value after one base period is `G(u)`; the
level-`K` orbit is periodic with period `P(K-1)` iff `G` has a fixed point that
the orbit reaches — which happens in the COLLAPSING and NEUTRAL cases — and
with period `2P(K-1)` in the DOUBLING case, since `G` is the negation and
`G² = id`. Minimality and `P(K-1) | P(K)` (T9.1) leave no other option. ∎

*Original label:* Phase 2C Theorems S1, and the fibre classification.

**Why this does not solve Problem 1.** The trichotomy describes what happens
**after** the base has settled. The diagonal never reaches that regime (T21.7).

---

## T11 — The period relation

> For every `K ≥ 1`, `P(K) ∈ { P(K-1), 2·P(K-1) }`, with `P(K) = 2P(K-1)`
> **iff** level `K` is DOUBLING. Consequently `P(K) = 2^{m(K)}` with `m`
> non-decreasing and `m(K) - m(K-1) ∈ {0,1}`.

*Domain:* SEED (any orbit with the tower defined). *Depends on:* T9.1, T10.

*Proof.* This is T10.1, proved there. ∎

*Original label:* Phase 2C Theorem S1.

**Why this does not solve Problem 1.** The period hierarchy is a statement
about the cycle region. Measured, only four doublings occur for `K ≤ 30000`, so
`P(K) = 16` throughout the bulk of the range — the tower's periods are tiny and
carry almost no information, while the preperiods grow linearly.

---

## T12 — Monotonicity of the preperiod

> For every `K ≥ 1`, `T(K-1) ≤ T(K)`, and `P(K-1)` divides `P(K)`.

*Domain:* SEED (any orbit with the tower defined). *Depends on:* T9.

*Proof.* This is T9.1, proved there via the projection `π_{K-1}`. ∎

*Original label:* Phase 2C Theorem S3 / E3.

**Why this does not solve Problem 1.** Monotonicity says the frontier never
retreats. It gives no lower bound on how fast it advances, and a lower bound is
what the transient-outruns-the-diagonal statement would need.

---

## T13 — The strongest valid upper bound for `T(K)`

Three bounds, in increasing strength. All under **SEED**; all depend on T8–T10.

### T13.1 (Phase 2C S2). `T(K) ≤ T(K-1) + P(K-1)`.

*Proof.* For `s ≥ T_-` the fibre value `u_{s+P_-}` equals `G(u_s)`, a function
of the phase only (T10) — in the COLLAPSING case it is a constant, in the other
two cases `G` is a bijection with `G² = id`. In all three cases `u_{s+2P_-}`
depends on `s` only through its phase, so coordinate `K` is `P(K)`-periodic
from `T_- + P_-` at the latest. Combining with T9.1 gives the bound. ∎

### T13.2 (Phase 2D R3, sharpened). If level `K` is COLLAPSING then `T(K) ≤ τ(K) + 1`.

*Proof.* See T16.3 below. Note `τ(K) ≥ T(K-1)` by definition, so the `max`
written in Phase 2D is redundant; see correction C-09.

### T13.3 (Phase 2E, exact — this is T18). `T(K)` is given exactly by the recurrence of T18. Since T18 is an equality, it is the strongest possible statement, and T13.1/T13.2 are corollaries.

**Why this does not solve Problem 1.** All three bound `T(K)` from **above**.
What a proof of the transient-outruns-the-diagonal statement needs is a lower
bound, and the best proved lower bound remains `T(K) + P(K) > ⌊K/2⌋` (T21.3) —
a factor of about `2.7` below the measured values.

---

## T14 — Non-COLLAPSING iff the predecessor coordinate is eventually zero

> Level `K` is **not** COLLAPSING **iff** coordinate `K-1` is zero throughout
> its own eventual cycle (i.e. `w_t(K-1) = 0` for all `t ≥ T(K-1)`).

*Domain:* SEED (and any orbit for which the prefix tower is defined).
*Depends on:* T10, T8.

*Proof.* COLLAPSING is by definition "`c_j = 1` for some `j` on the base
cycle", and `c_j = w_{T_-+j}(K-1)`. Its negation is "`w_t(K-1) = 0` for every
`t` on the cycle", which by periodicity is "for every `t ≥ T(K-1)`". ∎

*Original label:* Phase 2D Theorem N1.

**Why this does not solve Problem 1.** T14 is a reformulation that makes the
rarity of NEUTRAL intelligible: NEUTRAL requires an eventually-zero
predecessor, and those are rare. It replaces one unexplained scarcity by
another and touches nothing on the diagonal.

---

## T15 — The zero-cycle identity

> If coordinate `m` is zero throughout its cycle, then for every `t` on that
> cycle (i.e. `t ≥ T(m)`),
> ```
>     w_t(m-2)  =  w_t(m-1).
> ```

*Domain:* SEED (or any orbit). *Depends on:* T7.

*Proof.* Apply T7 at coordinate `m` and time `t ≥ T(m)`:
`0 = w_{t+1}(m) = w_t(m-2) XOR ( w_t(m-1) OR w_t(m) ) = w_t(m-2) XOR
( w_t(m-1) OR 0 ) = w_t(m-2) XOR w_t(m-1)`. ∎

*Original label:* Phase 2D Theorem N2.

**Indexing warning.** The hypothesis is about **coordinate `m`**, not about
level `m`. The eventually-zero coordinates found in the computed range are
`{2, 7, 28, 399}`; the corresponding non-COLLAPSING **levels** are
`{3, 8, 29, 400}`. Applying T15 with the level indices is a genuine error and
was made and caught during this audit — see `PROOF_AUDIT_REPORT.md` A-04.

**Why this does not solve Problem 1.** T15 transfers a parity question one
coordinate down without resolving it. No identity found forces the parity that
decides NEUTRAL versus DOUBLING.

---

## T16 — Exact collapse and reset theorems

Throughout: level `K` COLLAPSING, `T_- = T(K-1)`, `P_- = P(K-1)`, fibre as in
§4 of the notation. Domain **SEED**; depends on T7, T8, T10.

### T16.1 (= R1, erasure). Let `τ = τ(K)`. Then `u_{τ+1} = 1 XOR a_τ`, independently of `u_τ`. Consequently two orbits differing only in the fibre value at some time `≤ τ` **merge** at time `τ+1`.

*Proof.* `c_τ = 1` by definition of `τ`, so `u_{τ+1} = a_τ XOR (1 OR u_τ) =
a_τ XOR 1`. ∎

### T16.2 (= R2, the one-period reset constant). Let `σ*` be the last phase in `[T_-, T_-+P_-)` with `c = 1`. Then for every `s ≥ T_-` with phase `T_-`,

```
    u_{T_- + P_-}  =  Φ_K  :=  ( 1 XOR a_{σ*} )  XOR  XOR_{t = σ*+1}^{T_-+P_- - 1} a_t ,
```

independently of `u_{T_-}`.

*Proof.* By T16.1 applied at `σ*`, `u_{σ*+1} = 1 XOR a_{σ*}`, with no
dependence on the past. For `t > σ*` within the period, `c_t = 0`, so
`u_{t+1} = a_t XOR u_t`; unrolling to `T_- + P_-` gives the stated XOR. ∎

### T16.3 (= R3, periodicity onset). `T(K) ≤ τ(K) + 1`.

*Proof.* For `s ≥ τ+1` let `ς(s) := max{ t < s : c_t = 1 }`, which exists and
is `≥ τ`. As in T16.2, `u_s = (1 XOR a_{ς(s)}) XOR XOR_{t=ς(s)+1}^{s-1} a_t`,
a function of the base orbit only. Since `ς(s) ≥ τ ≥ T_-` and the base is
`P(K)`-periodic from `T_-` (using `P(K-1) | P(K)`), we get `ς(s+P(K)) = ς(s) +
P(K)` and hence `u_{s+P(K)} = u_s` for every `s ≥ τ+1`. With T9.1 this gives
`T(K) ≤ max(T_-, τ+1) = τ+1`. ∎

*Original labels:* Phase 2D R1, R2, R3.

**Why these do not solve Problem 1.** They describe the fibre once the base
below it has settled. The diagonal sits at level `t` at time `t`, and `T(t) > t`
on the whole computed range, so the diagonal is read before any of this applies.

---

## T17 — Unique periodic solution along a collapse chain

> Suppose level `K` is COLLAPSING. Then the cycle word of coordinate `K`,
> ```
>     C_K := ( w_{T_-}(K), …, w_{T_-+P(K)-1}(K) ),
> ```
> is the **unique** `P(K)`-periodic solution of the fibre recursion driven by
> the cycle words `C_{K-2}, C_{K-1}`.

*Domain:* SEED (or any orbit). *Depends on:* T10, T16.

*Proof.* A `P(K)`-periodic solution `(u_t)` must satisfy `u_{T_-} =
G^{P(K)/P_-}(u_{T_-})`. In the COLLAPSING case `G` is constant (T10), so the
composite is constant and there is exactly one such `u_{T_-}`, and it
determines the whole word. Existence is the real orbit. ∎

### T17.1 (bounded memory). Consequently the map `(C_{K-2}, C_{K-1}) ↦ C_K` is a deterministic transducer on `2·P(K)` bits; along a maximal chain of consecutive COLLAPSING levels with constant `P`, the cycle words are generated by a fixed finite-state machine of `2^{2P}` states.

### T17.2 (pigeonhole — with a vacuous hypothesis). A chain of more than `2^{2P}` consecutive COLLAPSING levels forces the sequence `(C_K)` to be eventually periodic in `K`.

**T17.2 has never been applicable.** At `P = 16` it needs `2^{32} ≈ 4.3 × 10⁹`
consecutive levels; the longest chain computed is `29 600`. It is recorded as a
theorem with a hypothesis that no computation has met.

*Original labels:* Phase 2D CH1, CH2, CH3.

**Why these do not solve Problem 1.** They describe the cycle region only.
Phase 2D's own conclusion stands: the entire cycle region of the tower is
generated by a 32-bit transducer, and it constrains the diagonal not at all.

---

## T18 — The exact recurrence for `T(K)`

> Let `K ≥ 2`, `T_- = T(K-1)`, `P_- = P(K-1)`, and let
> `D(K) = w_{T_-}(K) XOR w_{T_-+P(K)}(K)`. Then:
>
> **(a)** If `P(K) = P(K-1)` and level `K` is COLLAPSING:
> ```
>     D(K) = 0   =>   T(K) = T(K-1)
>     D(K) = 1   =>   T(K) = τ(K) + 1     (and τ(K) is defined)
> ```
> **(b)** If level `K` is DOUBLING: `T(K) = T(K-1)`.
> **(c)** If level `K` is NEUTRAL: `T(K) = T(K-1)`.
>
> In particular `T(K) > T(K-1)` **iff** `D(K) = 1`, and that forces the level
> to be COLLAPSING.

*Domain:* SEED (or any orbit with the tower defined). *Depends on:* T7–T10,
T16, and T18.1.

### T18.1 (defect propagation at a level). With `p := P(K)` and `D_s := u_s XOR u_{s+p}` for `s ≥ T_-`: if `c_s = 0` then `D_{s+1} = D_s`; if `c_s = 1` then `D_{s+1} = 0`.

*Proof.* Coordinates `K-2` and `K-1` are `p`-periodic from `T_-` (their minimal
periods divide `P(K-1) | p`, and their coordinate preperiods are `≤ T(K-1)`),
so `a_{s+p} = a_s` and `c_{s+p} = c_s`. If `c_s = 0` then `u_{s+1} = a_s XOR
u_s` and `u_{s+p+1} = a_s XOR u_{s+p}`, preserving the XOR. If `c_s = 1` both
become the same constant `a_s XOR 1`, so `D_{s+1} = 0`. ∎

### T18.2 (corollary). `D_s = D(K)` for `T_- ≤ s ≤ τ(K)`, and `D_s = 0` for `s ≥ τ(K)+1`.

*Proof of T18.* Write `R(K) := r_{P(K)}(K)`, the least `t` with `D_s = 0` for
all `s ≥ t`. By T21.1, `T(K) = max( max_{k<K} r_{P(K)}(k), R(K) )`, and the
first term is `≤ T(K-1)` by T21.2.

*(a), `D(K) = 0`.* By T18.2, `D_s = 0` for all `s ≥ T_-`, so `R(K) ≤ T_-` and
`T(K) = T_-` (equality, using T9.1).

*(a), `D(K) = 1`.* Then `R(K) > T_-`. Coordinate `K` is eventually
`P(K)`-periodic (T8), so `D_s` must vanish eventually; by T18.1 it can only do
so at a step with `c_s = 1`, so `τ(K)` exists. T18.2 then gives exactly
`R(K) = τ(K)+1`, and `T(K) = max(T_-, τ(K)+1) = τ(K)+1` since `τ(K) ≥ T_-`.

*(b), (c).* Both have `c_s = 0` for every `s ≥ T_-` (T10), so by T18.1, `D_s`
is constant on `[T_-, ∞)` at the lag `P(K)`. Eventual `P(K)`-periodicity forces
that constant to be `0`, so `R(K) ≤ T_-` and `T(K) ≤ T(K-1)`; with T9.1,
equality. ∎

### T18.3 (telescoping). For `K₀ < K`,

```
    T(K)  =  T(K₀)  +  Σ_{K' : D(K')=1, K₀ < K' ≤ K}  ( τ(K') + 1 - T(K'-1) ).
```

*Original label:* Phase 2E Theorem B.

**Why this does not solve Problem 1.** T18 is an exact description of the
frontier `T(K)`. It says nothing about the *values* `w_t(t)` the diagonal
reads, and by T21.5 the recurrence cannot be driven from cycle data alone.

---

## T19 — `T(K) = ρ(K) + 1` on resetting levels

> If `T(K) > T(K-1)` then `ρ(K)` is defined and `T(K) = ρ(K) + 1 = τ(K) + 1`.
> More generally, for every `K` with `ρ(K)` defined:
> `T(K) = ρ(K)+1` **iff** `w_{T(K)-1}(K-1) = 1`.

*Domain:* SEED. *Depends on:* T18.

*Proof.* By T18, `T(K) = τ(K)+1` and `w_{τ(K)}(K-1) = 1` with `τ(K) < T(K)`.
Hence `τ(K)` lies in the set defining `ρ(K)`, so `ρ(K) ≥ τ(K)`; and
`ρ(K) ≤ T(K)-1 = τ(K)`. The general statement is the definition of `ρ` as a
maximum. ∎

*Original label:* Phase 2E Theorem 3.1 / Proposition 3.2.

**Why this does not solve Problem 1.** It is an identity about *where* the
frontier lands, not about what the diagonal reads. On the inheriting levels the
same equality holds only about 47.6 % of the time and is a coincidence with no
mechanism (Proposition 3.2).

---

## T20 — The XOR-spine lower bound on the dependency skeleton

> Define the **reset-erased dependency skeleton** `Sk(t)` of the diagonal cell
> `(t,t)`: the set of cells reachable from `(t,t)` by backward edges
> `(s,k) → (s-1,k-2)`, `(s,k) → (s-1,k-1)` (present iff `w_{s-1}(k) = 0`), and
> `(s,k) → (s-1,k)` (present iff `w_{s-1}(k-1) = 0`), stopping at cells with
> `k < 0`, `k > 2s`, or `s = 0`.
> Then for every `t ≥ 0`,
> ```
>     |Sk(t)|  ≥  ⌊t/2⌋ + 1 .
> ```

*Domain:* SEED (the erasure conditions read real values); the *spine* argument
holds for ANY configuration. *Depends on:* T7, T7.3.

*Proof.* The edge `(s,k) → (s-1,k-2)` is never erased, because `w_s(k) =
w_{s-1}(k-2) XOR (…)` is XOR-permutive in `w_{s-1}(k-2)`: flipping that input
always flips the output, whatever the other two inputs are (T1(a) transported
by T7). Follow it repeatedly from `(t,t)`: the cells are `(t-j, t-2j)` for
`j = 0, 1, 2, …`. Each satisfies `k = t-2j ≤ 2(t-j) = 2s`, so none is folded by
the support bound T7.3, and `k ≥ 0` for `j ≤ ⌊t/2⌋`. These `⌊t/2⌋+1` cells are
distinct and all lie in `Sk(t)`. ∎

*Original label:* Phase 2E Theorem E1.

**Why this does not solve Problem 1.** Unbounded derivation size is **not**
evidence of aperiodicity: an eventually periodic sequence can have derivations
of unbounded size (the constant sequence `1,1,1,…` has arbitrarily large
cones). T20 rules out one hope — that reset erasure would collapse the
diagonal's ancestry to something finite — and nothing more.

---

## T21 — Further theorems that survive the audit

### T21.1 (coordinate decomposition of `T`). For every `K`, `T(K) = max_{0 ≤ k ≤ K} r_{P(K)}(k)`.

*Proof.* `W_s^K = W_{s+P(K)}^K` for all `s ≥ t` iff each coordinate agrees at
lag `P(K)` from `t` on, i.e. iff `t ≥ r_{P(K)}(k)` for every `k ≤ K`; the least
such `t` is the maximum. That least `t` is `T(K)` because the orbit of a
deterministic map has the rho shape and fails at `T(K)-1` by minimality. ∎
*Domain:* SEED / any orbit. *Original label:* Phase 2E Theorem A.

### T21.2 (lag monotonicity). If `q | q'` then `r_{q'}(k) ≤ r_q(k)`. In particular `r_{P(K)}(k) ≤ r_{P(k)}(k) ≤ T(k)` for `k ≤ K`.

*Proof.* `q`-periodicity from `t` implies `q'`-periodicity from `t`. ∎
**This is why preperiods measured at a longer lag can be strictly smaller, and
why T18(b) does not contradict T9.1.**

### T21.3 (support lower bound). If `w_t(K) = 1` for at least one `t`, then `T(K) + P(K) > ⌊K/2⌋`.

*Proof.* By T7.3, `w_t(K) = 0` for `t < K/2`. Let `m` be least with
`w_m(K) = 1`; then `m ≥ ⌈K/2⌉`. If `T(K) + P(K) ≤ m` then `t = m - P(K) ≥ T(K)`
and periodicity would give `w_m(K) = w_{m-P(K)}(K) = 0`, a contradiction. ∎
*Original label:* Phase 2C Theorem S4.

### T21.4 (the three reset times are distinct). `σ(K) ≤ τ(K)` always. `τ(K) = ρ(K)` exactly on the levels with `D(K) = 1`. Neither inequality is an equality in general.

*Proof.* The first is immediate from the definitions (the set for `τ` is a
subset of the set for `σ`). The second is T19. Non-degeneracy is exhibited by
explicit levels; see `COMPUTATIONAL_CERTIFICATES.md` CC-09. ∎

### T21.5 (the frontier recurrence is not finite-state in cycle data). For a COLLAPSING level, `D(K) = w_{T(K-1)}(K) XOR Φ_K`, where `Φ_K` (T16.2) is a function of the base cycle word alone. Consequently flipping the single value `w_{T(K-1)}(K)` flips `D(K)` while leaving every cycle word at every level `< K` unchanged, so **`D(K)` is not a function of the cycle words below `K`, at any depth**.

*Proof.* T16.2 says `u_{T_-+P_-} = Φ_K` independently of `u_{T_-}`; and
`P(K) = P(K-1)` for a COLLAPSING level by T10.1. Hence
`D(K) = u_{T_-} XOR u_{T_-+P(K)} = w_{T(K-1)}(K) XOR Φ_K`. The second claim is
immediate: `Φ_K` and all lower cycle words are unchanged by the flip. ∎
*Original label:* Phase 2E Theorems C2, C3.

**Scope warning, required by the brief.** T21.5 rules out exactly one class of
descriptions: those whose state is a **fixed-width window of cycle words of
coordinates below `K`, read from phase `T(K-1)`**. It does **not** prove that
no finite-state description of any kind exists — a machine reading other data
(for instance transient values, or a differently-phased window) is untouched.
Earlier wording that said "no finite-state description exists" is corrected in
`CORRECTIONS_AND_RETRACTIONS.md` C-12.

### T21.6 (the strip is the original diagram). For all `t` and all `r` with `t + r ≥ 0`, `q_t(r) := w_t(t+r) = x_t(r)`. In particular `x_t(0) = w_t(t)`: the centre column is the diagonal.

*Proof.* Substitute `k = t+r` in `w_t(k) = x_t(-t+k)`. ∎
*Original labels:* Phase 2B EA3; Phase 2E Theorem 5.1.

### T21.7 (no strip is autonomous). For every `R ≥ 0`, `q_{t+1}(r)` for `r` in `[-R, R]` is not a function of `q_t` restricted to `[-R, R]`: the value at `r = R` reads `q_t(R+1)`, and by T1(c) that dependence is genuine whenever `q_t(R) = 0`. Backwards, the dependency cone of `q_t(-R..R)` at time `t-n` is exactly `[-R-n, R+n]`, one column wider on each side per step.

*Proof.* T7/T21.6 give the update `q_{t+1}(r) = q_t(r-1) XOR (q_t(r) OR
q_t(r+1))`. Genuineness on the right: `l XOR (m OR n)` depends on `n` exactly
when `m = 0`. On the left it is always genuine by T1(a). ∎

### T21.8 (the strip automaton excludes nothing, all radii). Consider the nondeterministic automaton on `{0,1}^{2R+1}` whose transitions are `q ↦ q'` for some choice of the two boundary bits. Then for every `R`, every state has **in-degree exactly 4**, the transition relation is surjective, and the maximal recurrent set (states on a bi-infinite admissible path) is the **entire** state space.

*Proof.* Fix a target `q'`, a value for `q(R)` and a value for the right
boundary bit `q(R+1)`. Read the constraint `q'(r) = q(r-1) XOR (q(r) OR
q(r+1))` downwards from `r = R`: at each step `q(r-1)` is uniquely determined
by already-known quantities, ending with the left boundary bit `q(-R-1)`. So
each of the 4 choices yields exactly one predecessor, and distinct choices
yield distinct predecessors (they differ in `q(R)` or in the boundary bit used).
In a finite digraph with every in-degree `≥ 1`, every vertex is reachable from
a cycle, so the maximal recurrent set is everything. ∎
*Original label:* Phase 2E Theorem D2.

### T21.9 (strip lookahead). `q_t(-R..R)` determines `q_{t+n}(0)` for every `n ≤ R`, and for no larger `n` in general.

*Proof.* The backward cone of `(t+n, 0)` at time `t` is `[-n, n] ⊆ [-R, R]`
(T21.7), giving the positive half. The negative half is exhibited by explicit
pairs of times; see CC-10. ∎

### T21.10 (the diagonal's reset schedule is the centre column). The self-edge of the diagonal cell `(t,t)` — the edge `(t,t) → (t-1,t)` — is erased exactly when `w_{t-1}(t-1) = 1`, i.e. exactly when `x_{t-1}(0) = 1`.

*Proof.* By the erasure rule in T20, that edge is erased iff `w_{t-1}(k-1) = 1`
with `k = t`, i.e. iff `w_{t-1}(t-1) = 1`, which is `x_{t-1}(0)` by T21.6. ∎
*Original label:* Phase 2E Theorem G1.

**Consequence, and it is negative.** "Eventual periodicity of the reset
schedule along the diagonal" is *the same statement* as (H). Any lemma of that
shape is circular. See `VQ-01`.

### T21.11 (Phase 2A translation invariance). Satisfiability of the Model A patch constraints for a given look-back depth `a` and lag `p` does not depend on the base time `t`.

*Proof.* Model A constrains a finite space-time patch by the local rule plus
hypotheses that are stated relative to `t`; translating every index by a
constant is a bijection between the solution sets at `t` and at `t'`. ∎
*Original label:* Phase 2A Lemma T. **This is what makes the Phase 2A `t`
search complete rather than bounded**, and it is the only reason the Model A
sweep is a theorem rather than a scan.

### T21.12 (Phase 2A soundness of Model A). If a Model A patch exists for `(a,p)` then no argument using only the local rule and a window of look-back depth `a` — and no property of the initial configuration — can prove MB1-loc(`a`) at lag `p`.

*Proof.* A Model A patch is a rule-30-consistent assignment on the patch that
satisfies the hypotheses and violates the conclusion. Any purported proof using
only those hypotheses and the local rule within the window would apply to the
patch and derive a contradiction, so no such proof is sound. ∎

**Converse fails.** A Model A patch is **not** a counterexample to MB1-loc: its
base row is free and need not occur in the real orbit. This asymmetry is the
whole reason Phase 2A's 832 SAT results are a statement about *proof
strategies*, not about the conjecture.

**Why this does not solve Problem 1** (covering T21.1–T21.12). Six of these are
statements about the prefix tower, which lives in the cycle region; three are
about the strip, which by T21.6 is the original diagram re-indexed and by
T21.8 admits a local relaxation that excludes nothing; T21.10 turns an
apparently new hypothesis into (H) itself; and T21.11/12 are about the
*soundness of a proof search*, not about the conjecture it searched. None of
them constrains the values the diagonal reads.

---

## T22 — Coordinate 7 is permanently zero

> Under **SEED**, `w_t(7) = 0` for every `t ≥ 0`.

*Domain:* SEED. *Depends on:* T7, T8.

*Proof.* The 8-prefix orbit reaches a 2-cycle: the states
`W = 11001000` and `W' = 11011110` (bit `k` written left to right for
`k = 0..7`) satisfy `F_7(W) = W'` and `F_7(W') = W`, and both have bit `7`
equal to `0`. The orbit from `W_0^7 = 10000000` reaches `W` at `t = 2`, and the
transient values `W_0^7, W_1^7` also have bit 7 equal to `0` because
`w_t(7) = 0` for `t < 4` by the support bound T7.3. Hence `w_t(7) = 0` for
every `t`. ∎

*Original label:* Phase 2C, restated in Phase 2D.

**Uniqueness is NOT claimed.** That `7` is the *only* permanently zero
coordinate is a **bounded observation** (`BO-04`), checked for `K ≤ 3000`, not
a theorem. See correction C-10.

**Why this does not solve Problem 1.** A single permanently zero coordinate at
level 7 says nothing about level `t` at time `t`.

---

## T23 — The backward cone of a diagonal cell

> Under **ANY** configuration, the set of cells at time `s ≤ t` on which
> `w_t(t)` can depend is exactly the interval
> ```
>     [ max(0, 2s - t) ,  min(t, 2s) ] .
> ```

*Domain:* ANY (with the left boundary convention). *Depends on:* T7, T7.3.

*Proof.* One step back from `(u,k)` reaches `k-2, k-1, k`, so going back `n`
steps from `(t,t)` reaches `[t - 2n, t]`. With `n = t - s` the reachable
interval is `[2s - t, t]`, intersected with the support constraints `k ≥ 0`
(T7.1) and `k ≤ 2s` (T7.3). ∎

*Original label:* Phase 2D Theorem DD1.

**Why this does not solve Problem 1.** The cone has `Θ(t²)` cells and no
sub-structure that a finite argument can exploit; T20 and the DAG measurements
say the strongest available simplifications shrink it by a constant factor
only.

---

## Ledger ID mapping

The stable ledger IDs in `MASTER_CLAIM_LEDGER.md` are `T-01 … T-23`,
corresponding to `T1 … T23` above. Sub-results (`T2.4`, `T21.5`, …) are cited
by their sub-label and inherit the parent's ledger ID.

---

## Summary of domains

| domain | theorems |
|---|---|
| **ANY** configuration | T1, T5, T7, T9, T21.7, T21.8 |
| **FIN** (finite support) | T2, T2.1–T2.4, T3, T3.1, T3.2 |
| **SEED** only | T4 (corollary), T7.1–T7.3, T8, T10, T13, T14, T15, T16, T17, T18, T19, T20, T21.1–T21.6, T21.9, T21.10 |

Everything in the FIN row is subject to the **seed-blindness obstruction**: it
cannot settle Problem 1, because FIN contains configurations whose centre
column is eventually periodic. Everything in the SEED row is about the prefix
tower or the transient, and none of it reaches the diagonal's *values*.
