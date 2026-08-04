# Width-2 Non-Eventual-Periodicity for Rule 30 — Reconstruction from First Principles

**Status of this document.** Everything in Sections 1–8 is proved here from the
definition of rule 30 alone. **No theorem is imported from the literature**
(see Section 2 for the explicit accounting). Sections 9–12 are analysis of the
gap between what is proved and Rule 30 Prize Problem 1; they contain *no*
claimed proof of Problem 1.

**What is proved.** Theorem W2 (Section 7): no two adjacent columns of the
rule-30 single-cell space-time diagram are both eventually periodic.
Theorem W2′ (Section 8), a strengthening: **at most one column of the entire
diagram is eventually periodic.**

**What is not proved, and is not close to being proved.** That the *center*
column specifically is not eventually periodic. Section 11 isolates the
smallest lemma that would bridge the gap; Section 12 reports a computational
result showing that this lemma has **no local/finitary form**, which rules out
an entire class of proof strategies.

---

## 1. Definitions

Every definition used anywhere in this document is listed here. Nothing else is
assumed.

**D1 (Lattice and configurations).** The lattice is `Z` (the integers). A
*configuration* is a map `a : Z -> {0,1}`. We write `a(i)` for the value at
site `i`.

**D2 (Elementary CA, Wolfram numbering).** For a *rule number*
`n` in `{0, ..., 255}`, the *local rule* is the map
`f_n : {0,1}^3 -> {0,1}` defined by

```
f_n(l, c, r) = (n >> (4*l + 2*c + r)) & 1.
```

The *global map* `F_n` sends `a` to `a'` with
`a'(i) = f_n(a(i-1), a(i), a(i+1))` for every `i`.

**D3 (Rule 30).** `n = 30`. Its table, indexed by `4l + 2c + r`, is

| `lcr` | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|
| `f_30`|  0  |  1  |  1  |  1  |  1  |  0  |  0  |  0  |

so `f_30` has binary code `00011110 = 30`. Throughout we use the equivalent
closed form (Fact F0)

```
f_30(l, c, r) = l XOR (c OR r).
```

**D4 (Orientation).** Site index `i` increases to the *right*; "left" means
decreasing `i`. Time `t` increases *downward*. This convention is fixed once
and used everywhere; the reconstruction below is not symmetric under reflection
(see Section 10), so orientation errors are fatal and are guarded against in
`rule30_lab.py` by the mirror check against rule 86.

**D5 (Single-cell seed).** `a_0(0) = 1` and `a_0(i) = 0` for `i != 0`.

**D6 (Space-time diagram).** `a_t = F_30^t(a_0)` for `t = 0, 1, 2, ...`.
We write `a_t(i)` for the value at site `i` at time `t`.

**D7 (Column).** For `i` in `Z`, the *column at `i`* is the infinite binary
sequence `col_i = (a_t(i))_{t >= 0}`.

**D8 (Center column).** `c = col_0`, i.e. `c(t) = a_t(0)`, with `c(0) = 1`.

**D9 (Eventually periodic).** A sequence `x = (x(t))_{t>=0}` is *eventually
periodic* if there exist an integer `T >= 0` (a *preperiod*) and an integer
`p >= 1` (a *period*) with `x(t + p) = x(t)` for every `t >= T`. It is
*purely periodic* if this holds with `T = 0`.

**D10 (Width-`w` window sequence).** For `i` in `Z` and `w >= 1`, the
*width-`w` window at `i`* is the sequence
`W_i^w(t) = (a_t(i), a_t(i+1), ..., a_t(i+w-1))` with values in the finite
alphabet `{0,1}^w`. Definition D9 applies verbatim to sequences over any
alphabet.

**D11 (Permutivity).** A local rule `f` is *left-permutive* if for every fixed
`(c, r)` the map `l -> f(l, c, r)` is a bijection of `{0,1}`, and
*right-permutive* if for every fixed `(l, c)` the map `r -> f(l, c, r)` is a
bijection of `{0,1}`.

**D12 (Light cone).** The *light cone* of the seed is the set of space-time
points `(t, i)` with `|i| <= t`.

---

## 2. What is imported from the literature: nothing

The user's brief asks to identify every theorem imported from the literature.
The accounting is:

| Ingredient | Source | Status here |
|---|---|---|
| Definition of rule 30 (D2, D3) | Wolfram's elementary-CA numbering | Definition, not a theorem. Verified against the closed form `l XOR (c OR r)` exhaustively over all 8 neighbourhoods (`verify_local_rule`). |
| Left-permutivity of rule 30 | folklore | **Proved here** (F3), by exhaustion over 8 cases. |
| Light cone / speed-1 growth | folklore | **Proved here** (F1), by induction. |
| Frozen left edge `a_t(-t) = 1` | folklore | **Proved here** (F2), by induction. |
| "Two adjacent columns cannot both be eventually periodic" | stated in Wolfram's *Announcing the Rule 30 Prizes* (2019) | **Proved here from scratch** (Theorem W2). We did not consult a written proof; the argument below was reconstructed and every step is verified mechanically (Section 7.1). |
| "At most one column is eventually periodic" | Wolfram's prize announcement remarks that the argument also works for non-adjacent columns | **Proved here** (Theorem W2′). Our proof is self-contained; we have not seen the argument alluded to and do not claim ours is the same one. |
| "…and if one doesn't know every cell in both columns" (Wolfram) | Wolfram's prize announcement | **NOT reconstructed. Unverified.** We neither use nor endorse this claim. Reconstructing it is a Phase-2 task (Registry item C7). |

Nothing else from the literature is used. In particular this document does
**not** rely on any result about rule 30's entropy, its randomness properties,
its algebraic reformulations, or any published numerical data.

---

## 3. Elementary facts

**Fact F0 (closed form).** `f_30(l, c, r) = l XOR (c OR r)`.

*Proof.* Exhaustion over the 8 neighbourhoods, comparing against the table in
D3. Machine-checked in `verify_local_rule` (check
`table_equals_xor_or_formula`). ∎

**Fact F1 (light cone).** `a_t(i) = 0` whenever `|i| > t`.

*Proof.* Induction on `t`. True at `t = 0` by D5. Assume it at time `t` and
let `|i| > t + 1`. Then `|i - 1|, |i|, |i + 1| >= |i| - 1 > t`, so all three
arguments of `f_30` vanish, and `f_30(0,0,0) = 0`. ∎

**Fact F2 (the left edge is frozen at 1).** `a_t(-t) = 1` for every `t >= 0`.

*Proof.* Induction on `t`. At `t = 0`, `a_0(0) = 1`. Assume `a_t(-t) = 1`.
By F1, `a_t(-(t+2)) = a_t(-(t+1)) = 0`. Hence

```
a_{t+1}(-(t+1)) = f_30( a_t(-(t+2)), a_t(-(t+1)), a_t(-t) )
               = f_30(0, 0, 1) = 0 XOR (0 OR 1) = 1.
```
∎

**Fact F3 (rule 30 is left-permutive).** For every `(c, r)`,
`f_30(0, c, r) != f_30(1, c, r)`.

*Proof.* `f_30(l, c, r) = l XOR (c OR r)`, and `x -> x XOR b` is a bijection
of `{0,1}` for either value of `b`. ∎

**Fact F4 (rule 30 is NOT right-permutive).** For every `l`,
`f_30(l, 1, 0) = f_30(l, 1, 1) = l XOR 1`.

*Proof.* `c = 1` forces `c OR r = 1` regardless of `r`. ∎

F3 and F4 are the entire reason the argument below has the shape it has, and
the entire reason it stops where it stops.

---

## 4. Lemma 1 — left reconstruction

**Lemma 1.** For every `t >= 0` and every `i` in `Z`,

```
a_t(i-1) = a_{t+1}(i) XOR ( a_t(i) OR a_t(i+1) ).
```

*Proof.* By D2 and F0, `a_{t+1}(i) = a_t(i-1) XOR (a_t(i) OR a_t(i+1))`.
XOR both sides with `(a_t(i) OR a_t(i+1))` and use `x XOR y XOR y = x`. ∎

This is the *only* place left-permutivity is used, and Lemma 1 is exactly
left-permutivity made explicit: F3 says the map `l -> f_30(l, c, r)` is
invertible, and Lemma 1 writes down the inverse.

*Mechanical check.* `verify_lemma_left_reconstruction` evaluates both sides at
202,000 space-time points (`t < 2000`, `|i| <= 50`): all agree.

---

## 5. Lemma 2 — two adjacent columns determine the whole left half-plane

**Lemma 2.** Fix `i`. The two sequences `col_i` and `col_{i+1}` determine
`col_j` for every `j <= i`.

*Proof.* Induction leftwards. By Lemma 1 with the pair `(i, i+1)`,

```
col_{i-1}(t) = col_i(t+1) XOR ( col_i(t) OR col_{i+1}(t) )        (*)
```

for every `t >= 0`, so `col_{i-1}` is determined. Now the pair
`(col_{i-1}, col_i)` is known, and the same step applies to it, giving
`col_{i-2}`; and so on. ∎

Two remarks that matter later.

* The recursion **consumes a pair and produces a pair**: `(col_i, col_{i+1})`
  yields `(col_{i-1}, col_i)`. This is why the natural unit is *two* adjacent
  columns, not one.
* Equation (*) needs `col_i` at time `t+1`. Over the infinite sequence nothing
  is lost; in a finite simulation each leftward step costs one time step of
  horizon, which is why the mechanical check below reconstructs 200 columns
  from a 3000-step diagram.

*Mechanical check.* `verify_lemma_leftward_determinism` reconstructs columns
`-1, -2, ..., -200` from columns `0` and `1` alone and compares against the
simulated diagram: exact match everywhere.

---

## 6. Lemma 3 — eventual periodicity propagates leftwards with *uniform* constants

**Lemma 3.** Suppose `col_i` and `col_{i+1}` are both eventually periodic.
Then there are `T >= 0` and `p >= 1` such that for **every** `j <= i`,
`col_j(t + p) = col_j(t)` for all `t >= T`. The constants `T` and `p` do not
depend on `j`.

*Proof.* Let `(T_1, p_1)` and `(T_2, p_2)` be preperiod/period pairs for
`col_i` and `col_{i+1}`. Put `T = max(T_1, T_2)` and `p = lcm(p_1, p_2)`; both
columns are `p`-periodic from time `T` on.

We show by induction on `k >= 0` that `col_{i-k}` and `col_{i-k+1}` are both
`p`-periodic from time `T` on. The base case `k = 0` is the hypothesis.
For the step, assume it for `k`, write `m = i - k`, and take any `t >= T`.
Since `t >= T` and `t + 1 >= T`, all three terms on the right of

```
col_{m-1}(t) = col_m(t+1) XOR ( col_m(t) OR col_{m+1}(t) )
```

are unchanged when `t` is replaced by `t + p`. Hence
`col_{m-1}(t + p) = col_{m-1}(t)` for all `t >= T`. So `col_{m-1}` and
`col_m` are both `p`-periodic from `T` on, which is the statement for
`k + 1`. ∎

**The uniformity is the crux.** If the preperiod were allowed to grow by even
`1` per column, the argument of Section 7 would collapse: the contradiction is
obtained by going far enough left that the light cone has not yet arrived by
time `T`, and a growing `T` would outrun the light cone. Lemma 3 shows it does
not grow at all.

---

## 7. Theorem W2 — no two adjacent eventually periodic columns

**Theorem W2.** For rule 30 started from the single-cell seed (D5), there is no
`i` in `Z` such that `col_i` and `col_{i+1}` are both eventually periodic.

*Proof.* Suppose such an `i` exists. Let `T` and `p` be the uniform constants
from Lemma 3, so every column `col_j` with `j <= i` is `p`-periodic from time
`T` on.

Choose an integer `j` with

```
j <= i        and        j <= -(T + p),
```

and set `n = -j`, so `n >= T + p`.

*Step 1: `col_j` vanishes below time `n`.* By F1, `a_t(j) = 0` whenever
`|j| > t`, i.e. `col_j(t) = 0` for every `t < n`.

*Step 2: `col_j` vanishes identically.* The interval `[T, T + p)` is contained
in `[0, n)` because `T + p <= n`. So `col_j(t) = 0` for all `t` in
`[T, T + p)` — a full period's worth. Since `col_j` is `p`-periodic from `T`
on, `col_j(t) = 0` for every `t >= T`. Together with Step 1 (which covers
`t < T`, as `T <= n`), `col_j(t) = 0` for every `t >= 0`.

*Step 3: contradiction.* By F2, `a_n(-n) = 1`, that is `col_j(n) = 1`.
This contradicts Step 2. ∎

**Corollary W2a.** The width-2 window sequence `W_0^2(t) = (a_t(0), a_t(1))` is
not eventually periodic. The same holds for `W_i^2` for every `i`.

*Proof.* A sequence with values in a product alphabet is eventually periodic
if and only if each coordinate sequence is (take the max of the preperiods and
the lcm of the periods in one direction; project in the other). So eventual
periodicity of `W_i^2` would make `col_i` and `col_{i+1}` both eventually
periodic, contradicting W2. ∎

### 7.1 What the mechanical checks do and do not certify

`rule30_lab.py` verifies the *computational content* of every step: Lemma 1 as
an identity, Lemma 2 by explicit reconstruction, F1/F2 as invariants over
thousands of rows. It does **not** verify Theorem W2 — a proof by contradiction
about infinite sequences has no finite computational content, and no amount of
simulation could supply one. The checks guard against implementation and
orientation errors in the objects the proof talks about; the proof itself is
the text above.

---

## 8. Theorem W2′ — at most one eventually periodic column in the whole diagram

Theorem W2 forbids *adjacent* pairs. It extends to arbitrary pairs, and the
extension is elementary.

**Lemma 4 (strip determinism).** Let `i < j` and put `m = j - i - 1 >= 0`.
Define the *interior state* `v_t = (a_t(i+1), ..., a_t(j-1))` in `{0,1}^m`.
There is a map `G : {0,1}^m x {0,1}^2 -> {0,1}^m`, depending only on `f_30`,
with

```
v_{t+1} = G( v_t , (a_t(i), a_t(j)) ).
```

*Proof.* For `i+1 <= l <= j-1`, the update `a_{t+1}(l) = f_30(a_t(l-1), a_t(l),
a_t(l+1))` uses only sites in `[i, j]`, i.e. only the entries of `v_t` together
with the two boundary values `a_t(i)` and `a_t(j)`. ∎

*Mechanical check.* `verify_lemma_strip_determinism` drives the interior
columns `-5..5` for 3000 steps from the boundary columns `-6` and `6` plus the
initial row, and reproduces the true diagram exactly.

**Lemma 5 (interior columns inherit eventual periodicity).** If `col_i` and
`col_j` are both eventually periodic (`i < j`), then `col_l` is eventually
periodic for every `l` with `i < l < j`.

*Proof.* Let `T`, `p` be a common preperiod and period for `col_i` and
`col_j`, and let `u_t = (a_t(i), a_t(j))`, so `u_{t+p} = u_t` for `t >= T`.
Define `H : {0,1}^m -> {0,1}^m` as the composition of one full period of
driving steps,

```
H(v) = G( ... G( G(v, u_T), u_{T+1} ) ..., u_{T+p-1} ).
```

By Lemma 4 and the `p`-periodicity of `u` after `T`, `v_{T + kp} = H^k(v_T)`
for every `k >= 0`. The set `{0,1}^m` is finite, so the orbit
`v_T, H(v_T), H^2(v_T), ...` is eventually periodic: there exist `k_0 >= 0`
and `1 <= L <= 2^m` with `H^{k_0 + L}(v_T) = H^{k_0}(v_T)`.

Put `T' = T + k_0 p` and `P = L p`. Then `v_{T'+P} = v_{T'}`, and the driving
sequence satisfies `u_{t+P} = u_t` for `t >= T` since `p | P`. An immediate
induction on `t` using Lemma 4 gives `v_{t+P} = v_t` for every `t >= T'`.
Hence every interior column is eventually periodic with period dividing `P`. ∎

**Theorem W2′.** For rule 30 from the single-cell seed, **at most one column of
the space-time diagram is eventually periodic.**

*Proof.* Suppose `col_i` and `col_j` are both eventually periodic with `i < j`.
If `j = i + 1` this contradicts Theorem W2. If `j > i + 1`, Lemma 5 makes
`col_{j-1}` eventually periodic; then `col_{j-1}` and `col_j` are adjacent and
both eventually periodic, again contradicting W2. ∎

**Remark (the name "width 2" is slightly misleading).** The natural statement
of what is known is not about width-2 windows but about the *cardinality of the
set of eventually periodic columns*: it is at most 1. Prize Problem 1 asks
whether that set contains column 0. Note also that Lemma 5 gives a period bound
`P <= 2^{j-i-1} * p` that is exponential in the gap; this is harmless for W2′
(any finite period suffices) but is a real obstruction to any quantitative
version.

---

## 9. Exactly why the theorem gives width 2

Three independent things must be true for the argument to run, and each of them
requires *two* adjacent columns.

1. **Invertibility needs the left neighbour to be the permutive coordinate.**
   Lemma 1 solves the local rule for `a_t(i-1)`. Its right-hand side involves
   `a_{t+1}(i)`, `a_t(i)` and `a_t(i+1)` — three values living in *two* columns
   (`i` and `i+1`). One column supplies `a_{t+1}(i)` and `a_t(i)`; nothing in
   column `i` supplies `a_t(i+1)`.

2. **The recursion is a map on pairs.** Lemma 2 is an iteration of the map
   `(col_i, col_{i+1}) -> (col_{i-1}, col_i)`. A pair is the smallest object
   this map acts on. There is no map `col_i -> col_{i-1}`; see Section 10.

3. **The contradiction is at spatial infinity.** Steps 1–3 of Theorem W2 need
   the conclusion at *arbitrarily negative* `j`, which needs Lemma 2 to run
   arbitrarily far left, which needs the pair structure at every stage. It also
   needs the preperiod not to degrade (Lemma 3), which again holds because the
   pair is regenerated exactly at each step.

A compact way to say all of this: rule 30's information flow is **strictly
one-directional**. The pair `(col_i, col_{i+1})` is precisely the "state" of
that flow, and eventual periodicity of a state propagates. A single column is
not a state.

---

## 10. Exactly why this does not prove width 1

### 10.1 The direct obstruction: no rightward reconstruction

To apply Theorem W2 to the center column one would need a second eventually
periodic column adjacent to it. The natural attempt is to recover `col_1` from
`col_0`. This is impossible as a matter of local information: by F4, when
`a_t(0) = 1` the update

```
a_{t+1}(0) = a_t(-1) XOR ( a_t(0) OR a_t(1) ) = a_t(-1) XOR 1
```

does not depend on `a_t(1)` at all. The value of the right neighbour is
**invisible** in the center column's own evolution at every time step where the
center cell is black — asymptotically about half of them. There is no formula,
local or otherwise, of the shape `col_1 = (something)(col_0)`.

### 10.2 The counting obstruction: no soft argument can work

Consider the dynamics **without** the seed constraint, i.e. arbitrary
bi-infinite orbits of rule 30. Lemma 2 says a pair `(col_i, col_{i+1})`
determines the left half-plane; conversely, *any* pair of sequences whatsoever
arises from some orbit's two adjacent columns (run Lemma 2's recursion to
define everything to the left, which is consistent by construction). Therefore
the set of orbits whose column `i` equals a *prescribed* eventually periodic
sequence is in bijection with `{0,1}^N` — uncountably many, of which
uncountably many have non-periodic `col_{i+1}`.

Consequence: **no dimension count, entropy count, or other "soft" argument can
upgrade width 1 to width 2.** The upgrade must use the specific initial
condition D5. Any proposed proof that never uses the seed is wrong.

### 10.3 The seed-blindness obstruction: the theorem is too cheap

Inspect what Theorem W2 actually used about the seed: only F1 (light cone) and
F2 (frozen left edge). Both hold for *every* initial condition with finite
support, and F2 holds for every configuration whose leftmost `1` exists. So:

> Theorem W2 holds verbatim for every finitely supported initial condition.

But the center column's behaviour is emphatically *not* the same for every
finitely supported seed — for many seeds the center column *is* eventually
periodic. (Trivially: the all-zero seed. Less trivially, the seed determines
which column is "the center".) Therefore W2 cannot possibly distinguish the
single-cell seed from other seeds, while Problem 1 is a statement about one
specific seed. **A proof of Problem 1 must use a property of the single-cell
seed that F1 and F2 do not capture.** This is the sharpest available statement
of why the gap is not a technicality.

### 10.4 What is *not* an obstruction

For the record, these are *not* reasons the argument fails, and should not be
cited as such:

* It is not a failure of the light cone on the right. The right edge is also
  frozen (`a_t(t) = 1` for all `t`, by the mirror of F2's proof — verified in
  `verify_structural_invariants`). Nothing about the right boundary is
  irregular.
* It is not an issue of the center column being "special" in the sense of
  Lemma 5: by W2′ the obstruction is uniform across all columns.
* It is not a preperiod-growth problem; Lemma 3 rules that out.

---

## 11. The smallest missing lemma

By Theorem W2′, Prize Problem 1 follows from:

> **(Bridge)** If `col_0` is eventually periodic, then *some other column* is
> eventually periodic.

The cheapest instance takes the other column to be `col_{-1}`, and it can be
made completely explicit, because the identity of Lemma 1 turns it into a
statement about `col_1` on a set of density about `1/2`.

**Proposition 6 (exact reduction).** Suppose `col_0` is eventually periodic
with preperiod `T` and period `p`. Put

```
Z = { t >= T : a_t(0) = 0 }        (the zero set of the center column).
```

Then for every `t >= T`:

```
a_{t+p}(-1) XOR a_t(-1)  =  [ t in Z ] * ( a_t(1) XOR a_{t+p}(1) ).
```

Consequently `col_{-1}` is `p`-periodic from time `T` on **if and only if**
`a_{t+p}(1) = a_t(1)` for every `t` in `Z`.

*Proof.* By Lemma 1 at `i = 0`, for `t >= T`,

```
a_t(-1)     = a_{t+1}(0)   XOR ( a_t(0)   OR a_t(1)   ),
a_{t+p}(-1) = a_{t+1+p}(0) XOR ( a_{t+p}(0) OR a_{t+p}(1) ).
```

XOR the two lines. Since `t + 1 >= T`, periodicity of `col_0` kills the first
terms: `a_{t+1}(0) XOR a_{t+1+p}(0) = 0`. Also `a_{t+p}(0) = a_t(0)`. So

```
a_t(-1) XOR a_{t+p}(-1) = ( a_t(0) OR a_t(1) ) XOR ( a_t(0) OR a_{t+p}(1) ).
```

If `a_t(0) = 1` both parentheses equal `1` and the right side is `0`. If
`a_t(0) = 0` the right side is `a_t(1) XOR a_{t+p}(1)`. That is the displayed
identity, and the equivalence is immediate. ∎

This yields the nominated missing lemma.

> ### MB1 — the smallest missing lemma
>
> **Statement.** Let `col_0` be eventually periodic with preperiod `T` and
> period `p`, and let `Z = { t >= T : a_t(0) = 0 }`. Then there exists `T'`
> such that
> ```
> a_{t+p}(1) = a_t(1)   for every t in Z with t >= T'.
> ```
>
> **Consequence.** MB1 + Proposition 6 make `col_{-1}` eventually periodic;
> then `col_{-1}` and `col_0` are adjacent and both eventually periodic,
> contradicting Theorem W2. Hence **MB1 implies that the center column of rule
> 30 is not eventually periodic** — Prize Problem 1, in the expected direction.

### 11.1 Why MB1 is genuinely smaller than the obvious candidates

| Candidate | Statement | Relation |
|---|---|---|
| MB0 | `col_0` eventually periodic ⟹ `col_1` eventually periodic | MB0 ⟹ MB1 (strictly stronger: MB0 demands agreement at *all* large `t`, MB1 only on `Z`, of density ≈ 1/2) |
| MB1 | as above: agreement of `col_1` at lag `p`, on the zero set of `col_0` only | **nominated** |
| MB1′ | `col_0` eventually periodic ⟹ `col_{-1}` eventually periodic | equivalent to MB1 up to the choice of period (Prop. 6 gives MB1 ⟺ "`col_{-1}` is `p`-periodic"; MB1′ allows any period, so MB1 ⟹ MB1′) |
| Bridge | `col_0` eventually periodic ⟹ some other column is | MB1′ ⟹ Bridge; Bridge is what W2′ needs |

So the implication chain is `MB0 ⟹ MB1 ⟹ MB1′ ⟹ Bridge ⟹ Problem 1`, and MB1
is the weakest of these that is still fully explicit. MB1 asks for agreement
at **one specific lag** (`p`, the period of `col_0`), on **half the times**,
and only **eventually**.

### 11.2 A logical warning about MB1

MB1 is a conditional whose hypothesis is conjecturally false. If Problem 1 has
the expected answer, MB1 is *vacuously true*. Therefore:

* MB1 **cannot be confirmed or refuted by computation**, ever. Any experiment
  claiming to "test MB1" is testing something else.
* MB1 is nonetheless a legitimate proof target: a proof of MB1 that does not
  presuppose Problem 1 would settle Problem 1.
* What *can* be tested is a **finitary strengthening** of MB1, and that is what
  Section 12 does. Falsifying a finitary strengthening does not falsify MB1 —
  but it does rule out proof strategies.

---

## 12. Computational result: MB1 has no local form

Define, for an integer `a >= 0`, the finitary statement

> **MB1-loc(`a`).** For all `t > a` and all `p >= 1`: if
> `a_s(0) = a_{s+p}(0)` for every `s` in `[t-a, t]`, and `a_t(0) = 0`, then
> `a_t(1) = a_{t+p}(1)`.

MB1-loc(`a`) is universally quantified over a computable predicate, so a single
counterexample refutes it — this is the one direction in which finite
computation is logically sufficient. MB1-loc(`a`) for any fixed `a` would
immediately give MB1 (apply it at every `t` in `Z` beyond `T + a`).

**Result (from `run_bridge_probe.py` and `find_mb1loc_witnesses.py`, `N = 200001`
center bits, lags `p <= 4096` resp. `p <= 20000`).**

* **MB1-loc(`a`) is FALSE for `a = 0, 4, 8, 12, 16, 20, 24, 28`**, with explicit
  witnesses `(t, p)` re-verified against a freshly simulated diagram. See
  PHASE1_AUDIT.md §8 for the witness table. For `a >= 32` the 10^6-bit sample
  contains only 3 eligible positions; that case is **untested, not verified**.
* The conditional agreement probability
  `P[ a_t(1) = a_{t+p}(1) | col_0 agrees at lag p on [t-a, t], a_t(0) = 0 ]`
  rises steeply from `0.5000` at `a = 0` and then flattens in the `0.75`–`0.78`
  band:

  | look-back `a` | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | `P` | .50002 | .62287 | .65468 | .69387 | .70268 | .72570 | .74018 | .74906 | .75559 | .75762 | .75544 |

  (lags `p <= 4096`; the independent `p <= 20000` run gives `.70245`, `.74012`,
  `.75545`, `.75975`, `.76949`, `.77551` at `a = 4, 8, 12, 16, 20, 24`.)

* **The trend is still increasing at the largest look-backs, and the sample
  sizes there are small.** The 10^6-bit run gives `.7031, .7401, .7545, .7627,
  .7829, .8082, .9583` at `a = 4, 8, 12, 16, 20, 24, 28` on samples of
  `3.1e8, 1.9e7, 1.2e6, 7.6e4, 4.7e3, 292, 24`. The data are consistent with
  convergence to a limit strictly below `1`, and also with a drift towards `1`.
  **Neither can be excluded from this sample.** Do not cite "saturates below 1"
  as established.
* The corresponding **forward** statistic — conditioning on `col_0` agreeing on
  `[t, t+b]` instead — is flat at `0.5000` for every `b` tested. The effect is
  purely retrospective.

**Interpretation, stated carefully.**

1. The forward/backward asymmetry has an exact explanation and is *not*
   evidence of anything deep: `a_t(1)` is a function of the cells at time
   `t-k` in positions `[1-k, 1+k]`, a set that contains `a_{t-1}(0),
   a_{t-2}(0), ...` but no cell at any time `> t`. Conditioning on the past of
   `col_0` matches part of `a_t(1)`'s own dependency cone; conditioning on its
   future matches nothing.

2. The substantive finding is the *refutation of MB1-loc at every look-back
   depth we could test* (`a <= 28`). Within that range, **no finite amount of
   observed local periodicity of the center column forces its right neighbour
   to repeat**, so a proof of MB1 by a local-window / compactness argument at
   any of those depths is dead: the implication it would need is false.

   Whether this persists for all `a` is **open**, and the trend is not in the
   comfortable direction: the agreement probability keeps climbing with `a`
   (0.76 at `a = 16`, 0.81 at `a = 24`, 0.96 at `a = 28` on 24 samples). If it
   tends to `1`, MB1-loc(`a`) could hold for some large `a` — which would be a
   *route to a proof of Problem 1*, not an obstruction. Deciding this is the
   single highest-value experiment identified in Phase 1 (Registry C5).

3. This does **not** refute MB1, and must not be reported as evidence against
   it (see §11.2). It refutes MB1-loc, which is a different statement.

**Open sub-question.** Does the saturation value equal `3/4` exactly? The naive
heuristic "the center cell is `1` half the time, and otherwise the neighbour is
a fair coin" predicts a per-step survival of `3/4`; the measured limit
(`0.7554 ± 0.008` at `a = 16`) is close but the sequence is still drifting
upward at the largest look-backs. Registered as C6 in CONJECTURE_REGISTRY.md.

---

## 13. Summary of the gap

| Level | Statement | Status |
|---|---|---|
| Width 2 | no two adjacent columns both eventually periodic | **proved** (Thm W2) |
| Any pair | at most one eventually periodic column in the diagram | **proved** (Thm W2′) |
| Bridge | `col_0` eventually periodic ⟹ some other column is | **open** |
| MB1 | `col_1` agrees at lag `p` on the zero set of `col_0` | **open**; smallest explicit sufficient lemma |
| MB1-loc(`a`) | the finite-window version of MB1 | **false**, refuted computationally for all `a` tested |
| Problem 1 | the center column is not eventually periodic | **open** |

The single sentence version: *rule 30's information flows only leftwards, a
pair of adjacent columns is the smallest object that flow acts on, and the
center column alone is half a state.*
