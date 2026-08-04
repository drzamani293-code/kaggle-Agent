# Formal Proof — At Most One Eventually Periodic Column (Theorem W2′)

Fully quantified reconstruction, self-contained. Nothing is imported from the
literature. Companion to `WIDTH2_PROOF_RECONSTRUCTION.md` (which gives the same
argument in expository form) and `PROOF_DEPENDENCY_GRAPH.md`.

Claim proved here:

> **Theorem W2′.** Let `a` be the rule-30 orbit of the single-cell seed. Then
> the set `E = { i ∈ Z : col_i is eventually periodic }` satisfies `|E| ≤ 1`.

---

## 0. Notation and standing conventions

* `Σ = {0,1}`. `⊕` is addition mod 2 (XOR); `∨` is boolean OR.
* `N = {0, 1, 2, ...}` — **time is one-sided**. There is no `t < 0` anywhere in
  this document.
* `Z` — the lattice. **Space is two-sided**, and this is used (Theorem W2 needs
  columns arbitrarily far to the left). See §7.3 for why this is harmless.
* For `x, y ∈ Z` with `x ≤ y`, `[x, y] = { z ∈ Z : x ≤ z ≤ y }`.
* `lcm(p, q)` is the least common multiple of positive integers `p, q`.

**Definition 0.1 (local rule).** `f : Σ³ → Σ`, `f(l, c, r) = l ⊕ (c ∨ r)`.

*(That this is Wolfram's rule 30 — i.e. `f(l,c,r) = (30 >> (4l+2c+r)) & 1` — is
checked by exhaustion over all eight triples in `rule30_lab.verify_local_rule`,
test `table_equals_xor_or_formula`. It is a finite verification of a definition,
not an assumption.)*

**Definition 0.2 (the orbit).** `a : N × Z → Σ` is defined by primitive
recursion on the first argument:

```
a(0, i)   = 1 if i = 0, else 0                                  (∀ i ∈ Z)
a(t+1, i) = f( a(t, i-1), a(t, i), a(t, i+1) )                  (∀ t ∈ N, ∀ i ∈ Z)
```

**This is a definition, not an existence claim.** `a(t, ·)` is determined by
`a(t-1, ·)` cellwise, so `a` exists and is unique by ordinary recursion on `t`.
No fixed-point theorem, no compactness, no choice.

**Definition 0.3 (column).** For `i ∈ Z`, `col_i : N → Σ` is `col_i(t) = a(t, i)`.

**Definition 0.4 (eventual periodicity).** For `x : N → Σ`, and `T ∈ N`,
`p ∈ Z_{≥1}`, say *`x` is `(T, p)`-periodic* iff

```
∀ t ∈ N :  t ≥ T  ⟹  x(t + p) = x(t).
```

Say `x` is *eventually periodic*, written `EP(x)`, iff

```
∃ T ∈ N  ∃ p ∈ Z_{≥1} :  x is (T, p)-periodic.
```

**Definition 0.5 (strip / interior state).** For `i, j ∈ Z` with `i < j`, put
`m = j - i - 1 ∈ N`. The *strip* is the index set `[i, j]`; its *interior* is
`[i+1, j-1]` (empty iff `m = 0`). The *interior state at time `t`* is

```
v(t) = ( a(t, i+1), a(t, i+2), ..., a(t, j-1) ) ∈ Σ^m
```

and the *boundary pair at time `t`* is `u(t) = ( a(t, i), a(t, j) ) ∈ Σ²`.

---

## 1. Elementary facts

**Fact F1 (light cone).** `∀ t ∈ N ∀ i ∈ Z : |i| > t ⟹ a(t, i) = 0.`

*Proof.* Induction on `t`. For `t = 0`: if `|i| > 0` then `i ≠ 0` so
`a(0,i) = 0` by Definition 0.2. Inductive step: assume the statement at `t`,
and let `i` satisfy `|i| > t+1`. Then each of `i-1, i, i+1` has absolute value
`≥ |i| - 1 > t`, so by the inductive hypothesis
`a(t,i-1) = a(t,i) = a(t,i+1) = 0`. Hence
`a(t+1,i) = f(0,0,0) = 0 ⊕ (0 ∨ 0) = 0`. ∎

**Fact F2 (the left light-cone edge is frozen at 1).**
`∀ t ∈ N : a(t, -t) = 1.`

*Proof.* Induction on `t`. For `t = 0`: `a(0,0) = 1`. Inductive step: assume
`a(t,-t) = 1`. The site `-(t+1)` has left neighbour `-(t+2)` and centre
`-(t+1)`, both of absolute value `> t`, so `a(t,-(t+2)) = a(t,-(t+1)) = 0` by
F1. Its right neighbour is `-t`. Hence

```
a(t+1, -(t+1)) = f( 0, 0, a(t,-t) ) = 0 ⊕ ( 0 ∨ a(t,-t) ) = a(t,-t) = 1.   ∎
```

**Fact F3 (left-permutivity).**
`∀ c, r ∈ Σ : f(0,c,r) ≠ f(1,c,r)`, because `f(l,c,r) = l ⊕ (c ∨ r)` and
`l ↦ l ⊕ b` is a bijection of `Σ` for each `b ∈ Σ`.

*Remark.* F3 is a statement about the **local rule in its first argument only**.
It is *not* a statement that the global map is invertible; see §7.2.

---

## 2. Lemma 1 — local inversion in the left argument

**Lemma 1.** `∀ t ∈ N ∀ i ∈ Z :`

```
a(t, i-1) = a(t+1, i) ⊕ ( a(t,i) ∨ a(t,i+1) ).
```

*Proof.* Fix `t, i`. By Definition 0.2,
`a(t+1,i) = a(t,i-1) ⊕ ( a(t,i) ∨ a(t,i+1) )`. Add
`( a(t,i) ∨ a(t,i+1) )` to both sides mod 2 and use `z ⊕ b ⊕ b = z`. ∎

*Direction.* Lemma 1 expresses a value at time `t` and site `i-1` in terms of
values at times `t` and `t+1` and sites `i, i+1`. It moves **one site to the
left** and uses data one step **forward** in time. There is no backward-in-time
inference.

---

## 3. Lemma 2 — normalising two different preperiods and periods

**Lemma 2.** Let `x, y : N → Σ`. Suppose `x` is `(T₁, p₁)`-periodic and `y` is
`(T₂, p₂)`-periodic. Put `T = max(T₁, T₂)` and `p = lcm(p₁, p₂)`. Then `x` and
`y` are both `(T, p)`-periodic.

*Proof.* It suffices to prove: if `x` is `(T₁,p₁)`-periodic and `p₁ | p` and
`T ≥ T₁`, then `x` is `(T,p)`-periodic. Write `p = k·p₁` with `k ∈ Z_{≥1}`.
We show by induction on `s ∈ {0,1,...,k}` that
`∀ t ≥ T₁ : x(t + s·p₁) = x(t)`.

* `s = 0`: trivial.
* `s → s+1`: for `t ≥ T₁`, `x(t + (s+1)p₁) = x( (t + s·p₁) + p₁ )`. Since
  `t + s·p₁ ≥ t ≥ T₁`, the defining property of `(T₁,p₁)`-periodicity applies
  at the index `t + s·p₁`, giving `x( (t+s·p₁) + p₁ ) = x(t + s·p₁)`, which by
  the inductive hypothesis equals `x(t)`.

Taking `s = k` gives `x(t+p) = x(t)` for all `t ≥ T₁`, hence in particular for
all `t ≥ T`. The same argument applies to `y` with `p₂ | p`. ∎

**Why the common multiple is needed.** The leftward recursion (Lemma 3) mixes
values of two columns at the *same* time index. Two columns with periods `p₁`
and `p₂` need a shift under which *both* are invariant; the set of such shifts
is exactly the multiples of `lcm(p₁,p₂)`. Likewise the preperiod must be taken
as the maximum, so that both periodicity statements are available at every
index used.

---

## 4. Theorem W2 — no two *adjacent* eventually periodic columns

**Lemma 3 (leftward propagation with uniform constants).** Let `i ∈ Z` and
suppose `col_i` and `col_{i+1}` are both `(T, p)`-periodic. Then

```
∀ k ∈ N :  col_{i-k}  is (T, p)-periodic.
```

Note `T` and `p` are **independent of `k`**.

*Proof.* Induction on `k`, with the strengthened hypothesis

> `P(k)`: both `col_{i-k}` and `col_{i-k+1}` are `(T, p)`-periodic.

`P(0)` is the assumption. Assume `P(k)` and write `m = i - k`. Let `t ≥ T` be
arbitrary. Lemma 1 at `(t, m)` and at `(t+p, m)` gives

```
col_{m-1}(t)     = col_m(t+1)     ⊕ ( col_m(t)     ∨ col_{m+1}(t)     )
col_{m-1}(t+p)   = col_m(t+1+p)   ⊕ ( col_m(t+p)   ∨ col_{m+1}(t+p)   )
```

By `P(k)`, `col_m` and `col_{m+1}` are `(T,p)`-periodic. The three indices used
on the right are `t+1`, `t` and `t`; all are `≥ T` because `t ≥ T`. Hence
`col_m(t+1+p) = col_m(t+1)`, `col_m(t+p) = col_m(t)` and
`col_{m+1}(t+p) = col_{m+1}(t)`. The two right-hand sides are therefore equal,
so `col_{m-1}(t+p) = col_{m-1}(t)`. As `t ≥ T` was arbitrary, `col_{m-1}` is
`(T,p)`-periodic, and with `P(k)` this gives `P(k+1)`. ∎

**The uniformity is essential.** If the preperiod grew by `1` per column, the
bound `T ≤ n` in Step 3 of Theorem W2 below would fail: the growing preperiod
would outrun the light cone. The proof of Lemma 3 shows it does not grow at
all, because the only time indices consulted are `t` and `t+1`, both `≥ T`.

**Theorem W2.** `¬ ∃ i ∈ Z : EP(col_i) ∧ EP(col_{i+1})`.

*Proof.* Suppose, for contradiction, that such an `i` exists. By Definition 0.4
choose `(T₁,p₁)` for `col_i` and `(T₂,p₂)` for `col_{i+1}`; by Lemma 2 both are
`(T,p)`-periodic with `T = max(T₁,T₂)`, `p = lcm(p₁,p₂)`. By Lemma 3, for every
`k ∈ N` the column `col_{i-k}` is `(T,p)`-periodic.

Define the integer

```
j = min( i , -(T+p) )        and       n = -j.
```

Then `j ≤ i`, so `j = i - k` for some `k ∈ N` and `col_j` is `(T,p)`-periodic.
Also `j ≤ -(T+p) < 0`, so `n = -j ≥ T + p ≥ 1`.

*Step 1.* `∀ t ∈ N : t < n ⟹ col_j(t) = 0.` Indeed `|j| = n > t`, so F1
applies.

*Step 2.* `∀ t ∈ N : col_j(t) = 0.` Let `t ≥ T`. Since `p ≥ 1`, there is
`k' ∈ N` with `T ≤ t - k'p < T + p`. (Take `k' = ⌊(t-T)/p⌋`.) Because
`T + p ≤ n`, we have `t - k'p < n`, so `col_j(t - k'p) = 0` by Step 1. Applying
`(T,p)`-periodicity `k'` times upward from index `t - k'p ≥ T` gives
`col_j(t) = col_j(t - k'p) = 0`. For `t < T`: `T ≤ T + p ≤ n`, so `t < n` and
Step 1 gives `col_j(t) = 0` directly.

*Step 3.* By F2, `a(n, -n) = 1`, i.e. `col_j(n) = 1`, contradicting Step 2. ∎

---

## 5. Boundary forcing: the strip is a finite-state machine

**Lemma 4 (boundary forcing / strip determinism).** Let `i < j` in `Z`, and let
`m = j - i - 1`. There exists a function `G : Σ^m × Σ² → Σ^m`, depending only
on `f` and `m`, such that

```
∀ t ∈ N :  v(t+1) = G( v(t), u(t) ),
```

with `v`, `u` as in Definition 0.5.

*Proof.* Let `l ∈ [i+1, j-1]`, so `v(t)` contains `a(t,l)` as its
`(l-i)`-th coordinate. By Definition 0.2,
`a(t+1,l) = f( a(t,l-1), a(t,l), a(t,l+1) )`. The three sites `l-1, l, l+1` all
lie in `[i, j]`, since `i+1 ≤ l ≤ j-1`. Every site of `[i,j]` is either an
interior site (its value is a coordinate of `v(t)`) or one of the two boundary
sites `i, j` (its value is a coordinate of `u(t)`). Hence `a(t+1,l)` is a
function of `(v(t), u(t))` alone. Define `G` coordinatewise by these
functions. ∎

**Why boundary forcing determines the interior.** The local rule has radius 1.
A site strictly inside the strip therefore reads only sites within distance 1,
which cannot escape `[i, j]`. So the two boundary columns supply *exactly* the
information that crosses into the strip from outside; given them, the interior
evolves autonomously. Nothing outside `[i, j]` is consulted at any time. This
is a statement about one forward step, applied for each `t` separately — it
involves no limit and no global argument.

*Mechanical check.* `rule30_lab.verify_lemma_strip_determinism` drives the
interior columns `-5..5` for 3000 steps from `u` alone and reproduces the true
diagram exactly.

**Lemma 5 (interior columns inherit eventual periodicity).** Let `i < j` in
`Z`. If `EP(col_i)` and `EP(col_j)`, then `EP(col_l)` for every
`l ∈ [i+1, j-1]`.

*Proof.* If `m = j-i-1 = 0` the statement is vacuous, so assume `m ≥ 1`.

By Definition 0.4 and Lemma 2, choose `T ∈ N` and `p ∈ Z_{≥1}` such that both
`col_i` and `col_j` are `(T, p)`-periodic. Then

```
(∗)      ∀ t ≥ T :  u(t + p) = u(t),
```

since `u(t) = (col_i(t), col_j(t))` coordinatewise.

Define `H : Σ^m → Σ^m` as the composition of one period of driving steps:

```
H(w) = G( G( ... G( G(w, u(T)), u(T+1) ) ..., u(T+p-2) ), u(T+p-1) ),
```

i.e. `H = G(·, u(T+p-1)) ∘ ... ∘ G(·, u(T+1)) ∘ G(·, u(T))`. This is a
well-defined function `Σ^m → Σ^m`: it is a finite composition of `p` functions.

*Claim 5.1.* `∀ k ∈ N : v(T + k·p) = H^k( v(T) )`.

*Proof of claim.* Induction on `k`. For `k = 0` both sides are `v(T)`. Assume
it for `k`. For `0 ≤ s < p`, we show `v(T + k·p + s + 1) = G( v(T+k·p+s), u(T+s) )`:
by Lemma 4, `v(T+k·p+s+1) = G( v(T+k·p+s), u(T+k·p+s) )`, and
`u(T+k·p+s) = u(T+s)` by `(∗)` applied `k` times (each application is legal
because `T + s' ≥ T` for every intermediate index `s'`). Composing these `p`
steps for `s = 0, 1, ..., p-1` gives
`v(T + (k+1)p) = H( v(T+k·p) ) = H( H^k(v(T)) ) = H^{k+1}(v(T))`. ∎

*Claim 5.2 (pigeonhole, not compactness).* There exist `k₀ ∈ N` and
`L ∈ Z_{≥1}` with `k₀ + L ≤ 2^m` and `H^{k₀+L}( v(T) ) = H^{k₀}( v(T) )`.

*Proof of claim.* The `2^m + 1` elements `H^0(v(T)), H^1(v(T)), ..., H^{2^m}(v(T))`
all lie in the set `Σ^m`, which has exactly `2^m` elements. By the pigeonhole
principle two of them are equal, say `H^{k₀}(v(T)) = H^{k₀+L}(v(T))` with
`0 ≤ k₀ < k₀ + L ≤ 2^m`. ∎

Now set

```
T' = T + k₀·p        and        P = L·p   (so P ≥ 1).
```

By Claims 5.1 and 5.2, `v(T' + P) = H^{k₀+L}(v(T)) = H^{k₀}(v(T)) = v(T')`.

*Claim 5.3.* `∀ t ≥ T' : v(t + P) = v(t)`.

*Proof of claim.* Induction on `t ≥ T'`. Base `t = T'`: just shown. Step:
assume `v(t+P) = v(t)` for some `t ≥ T'`. Then by Lemma 4 twice,

```
v(t+1+P) = G( v(t+P), u(t+P) ) = G( v(t), u(t) ) = v(t+1),
```

where `u(t+P) = u(t)` holds because `t ≥ T' ≥ T` and `P = L·p` is a multiple of
`p`, so `(∗)` applies `L` times. ∎

Finally, for `l ∈ [i+1, j-1]`, `col_l(t)` is a fixed coordinate of `v(t)`, so
Claim 5.3 gives `col_l(t+P) = col_l(t)` for all `t ≥ T'`, i.e. `col_l` is
`(T', P)`-periodic, hence `EP(col_l)`. ∎

*Quantitative remark.* The period obtained is `P = L·p ≤ 2^{j-i-1}·p`, i.e.
exponential in the gap. This is irrelevant for W2′ (any finite period suffices)
but blocks any attempt to make the argument quantitative in the gap.

---

## 6. Theorem W2′

**Theorem W2′.** `|{ i ∈ Z : EP(col_i) }| ≤ 1`. Equivalently:

```
¬ ∃ i, j ∈ Z :  i < j  ∧  EP(col_i)  ∧  EP(col_j).
```

*Proof.* Suppose `i < j` with `EP(col_i)` and `EP(col_j)`.

* **Case `j = i + 1`.** Immediate contradiction with Theorem W2.
* **Case `j ≥ i + 2`.** Then `m = j-i-1 ≥ 1` and `j - 1 ∈ [i+1, j-1]`, so
  Lemma 5 gives `EP(col_{j-1})`. Now `col_{j-1}` and `col_{(j-1)+1} = col_j`
  are both eventually periodic, contradicting Theorem W2.

In both cases a contradiction. Hence no two distinct columns are both
eventually periodic, i.e. the set `E` has at most one element. ∎

**Corollary W2′a.** If the centre column `col_0` is eventually periodic, then
`col_i` is **not** eventually periodic for every `i ≠ 0`.

**Corollary W2′b (relation to Prize Problem 1).** Theorem W2′ does *not* decide
Problem 1: the centre column is a priori permitted to be the unique element of
`E`. The missing input is exactly the statement "if `EP(col_0)` then
`EP(col_j)` for some `j ≠ 0`" (registry item C2).

---

## 7. What is *not* used

The brief asks for explicit confirmation on three points.

### 7.1 No compactness, no König's lemma, no choice

The only step that produces an infinite conclusion from finite data is
Claim 5.2, and it is the **pigeonhole principle on a finite set**
(`2^m + 1` elements in a set of size `2^m`). No subsequence is extracted, no
limit of configurations is taken, `Σ^Z` is never topologised, and Tychonoff /
König / compactness of the shift space play no role. Every other step is either
(i) induction on `t ∈ N`, (ii) induction on `k ∈ N`, or (iii) a finite algebraic
identity in `Σ`. The axiom of choice is not used: all objects are explicitly
defined (`a` by recursion, `G` and `H` by explicit composition, `k₀` and `L` as
the least witnesses of the pigeonhole collision, `j` as an explicit `min`).

### 7.2 No reversibility of the global map

Rule 30's global map is **not** injective, and we never assume it is. The only
inversion performed is in **Lemma 1**, which inverts the map

```
l ↦ f(l, c, r)      for FIXED (c, r)
```

— a bijection of the two-element set `Σ` (Fact F3). This is left-permutivity of
the *local* rule, an entirely different statement from invertibility of the
*global* map `F`. Concretely, Lemma 1 recovers a value at the **same time `t`**
and one site to the left; it does not recover `a(t-1, ·)` from `a(t, ·)`. No
step of this document computes any value at a time strictly before the earliest
time already known.

### 7.3 No bi-infinite time; and the role of bi-infinite space

* **Time.** Every time index in this document lies in `N`. Definition 0.2
  defines `a` only for `t ≥ 0`; Definition 0.4 quantifies over `t ≥ T ≥ 0`;
  every induction is upward in `t`. Nothing is assumed about a bi-infinite-time
  orbit, and the theorem is *not* a statement about the natural extension of
  the shift.
* **Space.** `i` ranges over `Z`, and Theorem W2 genuinely uses arbitrarily
  negative `i` (the choice `j = min(i, -(T+p))`). This is not an idealisation
  that could hide an infinite argument: by Fact F1, at each fixed time `t` all
  but the finitely many sites in `[-t, t]` are `0`, so the whole proof could be
  rewritten over the finite windows `[-t-1, t+1]` at each time step. The
  bi-infinite lattice is a notational convenience, and no property of `Z` beyond
  "unbounded below" is used.

### 7.4 What *is* used about the initial condition

Exactly two facts: **F1** (light cone) and **F2** (frozen left edge). Both hold
for every initial condition of finite support. Consequently Theorem W2′ holds
verbatim for *every* finitely supported seed, and therefore cannot distinguish
the single-cell seed from any other — which is precisely why it cannot settle
Prize Problem 1 (see `WIDTH2_PROOF_RECONSTRUCTION.md` §10.3).

---

## 8. Mechanical checks of the computational content

A proof by contradiction about infinite sequences has no finite computational
content, so **none of these tests verifies Theorem W2 or W2′.** They verify the
identities and determinism claims the proof manipulates, which is where an
implementation or orientation error would hide.

| Lemma | Test in `rule30_lab.py` | Coverage | Result |
|---|---|---|---|
| Definition 0.1 = rule 30 | `verify_local_rule` | all 8 neighbourhoods | PASS |
| Fact F1 (implicitly) | `verify_boundary_independence` | `t ≤ 2000`, two array radii | PASS |
| Fact F2 | `verify_structural_invariants` | `t = 0..3000` | PASS |
| Fact F3 / F4 | `verify_local_rule` | all 8 neighbourhoods | PASS |
| Lemma 1 | `verify_lemma_left_reconstruction` | 202,000 `(t,i)` pairs | PASS |
| Lemma 3 (mechanism) | `verify_lemma_leftward_determinism` | columns `-1..-200` rebuilt from columns `0,1` | PASS |
| Lemma 4 | `verify_lemma_strip_determinism` | interior `-5..5` driven 3000 steps by boundary `-6, 6` | PASS |
| Lemma 2, Claims 5.1–5.3, Theorems W2, W2′ | — | **no computational content; proof only** | — |
