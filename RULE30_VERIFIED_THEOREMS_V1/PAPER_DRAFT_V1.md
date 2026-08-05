# Nested Prefix Dynamics and Transient Structure in the Single-Seed Rule 30 Space-Time Diagram

> ## SUPERSEDED BY `PHASE2G_LITERATURE_COMPARISON/REVISED_PAPER_OUTLINE.md`
>
> Phase 2G's comparison shows that this draft **claims too much**. On the
> available (secondary, unverified) evidence:
>
> * §§3–4's period structure — left-justified coordinates, eventual periodicity
>   of each diagonal, periods a power of two, and the **doubling criterion** —
>   is due to **E. S. Rowland, "Local Nested Structure in Rule 30", Complex
>   Systems 16 (2006) 239–258**, not to us. Theorem 4.1/Corollary 4.2 and
>   Theorems 5.1/5.2 must be demoted to recalled results with a citation.
> * §8's `T(K)/K = 1.3399` is a re-measurement of the known rule-30 order/chaos
>   boundary speed `≈ 0.252` (`1/(1-0.252) = 1.337`), not an observation of ours.
> * §5's column results (T-02/T-03) sit on territory occupied by **Jen (1990)**
>   and **Kopra (2022, Thm 3.5)**; their status is **uncertain**, and no priority
>   may be implied.
> * The title's "Nested" is Rowland's word for his mechanism. The revised title
>   is **"Preperiods in the left-diagonal tower of Rule 30"**.
> * Rowland's `2^n` nested-restart mechanism and its conditional time-reversal
>   have **no analogue here** — a gap in this work, now stated explicitly.
>
> **This draft must not be submitted or circulated in its current form.**
> The revised outline supersedes it. The text below is kept unaltered for the
> record.


**Draft v1.** This paper does **not** solve Wolfram's Rule 30 Prize Problem 1
and makes no partial claim on it. Section 9 states precisely why the results
here do not resolve centre-column aperiodicity.

Every statement is labelled **Theorem** (complete proof given), **Proposition**
/ **Lemma** (likewise), **Certificate** (the output of a stated finite
computation), **Observation** (measured on a stated finite range), or
**Conjecture** (precise, unproved). No claim of novelty is made anywhere; see
§10 and `NOVELTY_STATUS.md`.

---

## 1. Introduction

Rule 30 is the elementary cellular automaton

```
    x_{t+1}(j) = f( x_t(j-1), x_t(j), x_t(j+1) ),
    f(l, c, r) = l XOR ( c OR r ),
```

on `{0,1}^Z`. Started from the single-cell seed `x_0(j) = [j=0]` it produces a
space-time diagram whose centre column `c_t := x_t(0)` has resisted analysis
since 1983. Wolfram's Prize Problem 1 asks whether `(c_t)` is eventually
periodic; the expected answer is no, and the problem is open.

This paper collects what we can prove about the structure surrounding that
column. The results fall into two groups that, as we explain in §9, do not
meet.

**Group I — column structure, seed-blind.** We prove that no two adjacent
columns are both eventually periodic (Theorem 2.4), and strengthen this to: at
most **one** column of the entire diagram is eventually periodic (Theorem 2.6).
Both hold for every finitely supported seed. Theorem 2.6 is sharp in the only
way that matters: it permits the centre column to be that one column, which is
exactly the scenario Problem 1 asks us to exclude.

**Group II — prefix structure, seed-aware.** In light-cone coordinates the rule
becomes one-sided, so every prefix evolves autonomously and is eventually
periodic. This produces a tower of skew products with one-bit fibres
(§3), a period-doubling hierarchy (§4), an exact collapse/reset calculus (§5),
and an **exact recurrence for the preperiod** `T(K)` driven by a single bit
(Theorem 4.6). We also give a defect calculus in the original coordinates (§6)
and a dependency-skeleton analysis of the diagonal (§7).

The obstruction, made precise in §9, is that Group I cannot see the seed while
Group II cannot see the diagonal.

## 2. Rule 30 and coordinate systems

### 2.1 Permutivity

**Lemma 2.1.** *For every `(c,r)`, `l ↦ f(l,c,r)` is a bijection; consequently
for every configuration, every `t` and every `j`,*
`x_t(j-1) = x_{t+1}(j) XOR (x_t(j) OR x_t(j+1))`. *`f` is not right-permutive.*

Two adjacent columns therefore determine the entire left half-plane.

### 2.2 Two changes of coordinates

**Edge-aligned:** `w_t(k) := x_t(-t+k)`. **Co-moving strip:**
`q_t(r) := w_t(t+r)`.

**Theorem 2.2 (one-sided form).**
`w_{t+1}(k) = w_t(k-2) XOR ( w_t(k-1) OR w_t(k) )`.

**Proposition 2.3.** `q_t(r) = x_t(r)`: the co-moving strip is the original
column set, and `x_t(0) = w_t(t)` — *the centre column is the diagonal of the
edge-aligned diagram.*

The content of the change of variables is asymmetry: in `w` coordinates every
left prefix is closed under the dynamics, while in `x` (or `q`) coordinates no
finite window is, because `f` is not right-permutive.

### 2.3 Column periodicity

**Theorem 2.4.** *For every finitely supported seed there is no `j` such that
both `col_j` and `col_{j+1}` are eventually periodic.*

The proof uses only the light cone, the frozen left edge `x_t(-m-t) = 1`, and
the fact — essential — that leftward propagation of eventual periodicity
preserves the **preperiod**, not merely the period.

**Lemma 2.5 (interior inheritance).** *If `col_i` and `col_j` are eventually
periodic then so is every column strictly between them.*

**Theorem 2.6.** *For every finitely supported seed, at most one column of the
space-time diagram is eventually periodic.*

## 3. The prefix skew-product tower

**Theorem 3.1.** *For every `K ≥ 0` there is a map `F_K` on `{0,1}^{K+1}` with
`W_{t+1}^K = F_K(W_t^K)` for every `t ≥ 0`, where
`W_t^K = (w_t(0),…,w_t(K))`. Hence `(W_t^K)` is eventually periodic; write
`T(K)` for its exact preperiod and `P(K)` for its exact minimal period.*

**Theorem 3.2 (projection).** `π_K ∘ F_{K+1} = F_K ∘ π_K`, where `π_K` drops
the last coordinate. Hence `T(K-1) ≤ T(K)` and `P(K-1) | P(K)`.

So the levels form a tower of factors, and level `K` is a **one-bit fibre** over
level `K-1`:

```
    u_{t+1} = a_t XOR ( c_t OR u_t ),
    a_t = w_t(K-2),  c_t = w_t(K-1),  u_t = w_t(K).
```

`c_t = 1` makes the fibre map constant — we call such a step a **reset**.

## 4. Period and preperiod theorems

**Theorem 4.1 (trichotomy).** *Composing the fibre maps over one base period
gives a map that is **constant** (some `c = 1` on the cycle: COLLAPSING), the
**identity** (`c ≡ 0` and `XOR a = 0`: NEUTRAL), or the **negation** (`c ≡ 0`
and `XOR a = 1`: DOUBLING).*

**Corollary 4.2.** `P(K) ∈ { P(K-1), 2P(K-1) }`, with doubling exactly in the
DOUBLING case. Hence `P(K)` is a power of two.

**Theorem 4.3 (coordinate decomposition).**
`T(K) = max_{k ≤ K} r_{P(K)}(k)`, where `r_q(k)` is the least time from which
coordinate `k` agrees at lag `q`.

*Caution.* `r_q(k)` decreases as `q` grows through multiples, so the lag must be
`P(K)` and not each coordinate's own period.

**Theorem 4.4 (collapse calculus).** *For a COLLAPSING level with first reset
`τ(K) := min{t ≥ T(K-1) : c_t = 1}`:*
(i) `u_{τ+1} = 1 XOR a_τ`, independently of `u_τ`;
(ii) the fibre value one base period after `T(K-1)` is a constant `Φ_K`
determined by the base cycle word;
(iii) `T(K) ≤ τ(K) + 1`.

**Lemma 4.5 (defect propagation).** *Put `D_s := u_s XOR u_{s+P(K)}`. Then
`D_{s+1} = D_s` if `c_s = 0`, and `D_{s+1} = 0` if `c_s = 1`.*

**Theorem 4.6 (exact recurrence).** *Let
`D(K) := w_{T(K-1)}(K) XOR w_{T(K-1)+P(K)}(K)`. Then*

```
    D(K) = 0                        ⟹  T(K) = T(K-1)
    D(K) = 1                        ⟹  T(K) = τ(K) + 1   (and τ(K) exists)
    K is DOUBLING or NEUTRAL        ⟹  T(K) = T(K-1)
```

*In particular `T(K) > T(K-1)` iff `D(K) = 1`, which forces the level to be
COLLAPSING.*

**Corollary 4.7.** `T(K) = ρ(K) + 1` at every level with `D(K) = 1`, where
`ρ(K) = max{ t < T(K) : c_t = 1 }`.

**Corollary 4.8 (telescoping).**
`T(K) = T(K₀) + Σ_{D(K')=1} ( τ(K') + 1 - T(K'-1) )`, so `T(K)/K` factors as
(density of levels with `D = 1`) × (mean increment) up to `O(1/K)`.

**Theorem 4.9 (the recurrence is not driven by cycle data).**
*`D(K) = w_{T(K-1)}(K) XOR Φ_K`. Since `Φ_K` depends only on the base cycle
word, flipping the single transient bit `w_{T(K-1)}(K)` flips `D(K)` and leaves
every cycle word below `K` unchanged. Hence no fixed-width window of cycle
words below `K`, read from phase `T(K-1)`, determines `D(K)`.*

## 5. Collapse and reset structure

**Theorem 5.1.** *Level `K` is not COLLAPSING iff coordinate `K-1` is zero
throughout its own cycle.*

**Theorem 5.2.** *If coordinate `m` is zero on its cycle then
`w_t(m-2) = w_t(m-1)` for `t` on that cycle.*

**Theorem 5.3.** *For a COLLAPSING level the cycle word `C_K` is the unique
`P(K)`-periodic solution of the fibre recursion driven by `C_{K-2}, C_{K-1}`;
hence along a chain of consecutive COLLAPSING levels with constant period the
cycle words are generated by a transducer on `2P(K)` bits.*

**Observation 5.4** (`K ≤ 30000`). The coordinates zero on their own cycle are
exactly `{2, 7, 28, 399}`; the period-doubling levels are exactly
`{3, 8, 29, 400}`; every level in `[401, 30000]` is COLLAPSING; no NEUTRAL
level occurs. **This is a bounded observation, and the evidence for the last
clause is four events, not thirty thousand**: by Theorem 5.1 there were exactly
four opportunities for a non-COLLAPSING level, and each resolved as DOUBLING
because a parity came out odd.

**Theorem 5.5.** `w_t(7) = 0` for every `t` (exhibited by an explicit 2-cycle).
*Uniqueness of 7 is not claimed.*

## 6. Temporal-defect identities

For a lag `p`, put `d_t(j) := x_{t+p}(j) XOR x_t(j)`.

**Theorem 6.1.** *Over GF(2), for every configuration,*
```
    d_{t+1}(j) = d_t(j-1) + d_t(j) + d_t(j+1)
               + x_t(j)d_t(j+1) + x_t(j+1)d_t(j) + d_t(j)d_t(j+1).
```

**Corollary 6.2.** A black cell with no defect masks any defect to its right.

**Corollary 6.3.** For the single-cell seed, `d_t(±(t+p)) = 1` for every `t` and
every `p`: *the defect field is never empty.* (It is, however, never near
column 0 for that reason: the support edges recede at speed 1.)

Eventual `p`-periodicity of the centre column is exactly a **zero wall**
`d_t(0) = 0` for `t ≥ T`. Under a wall on a finite range one gets, e.g.,
`d_t(-1) = d_t(1)·(1 XOR x_t(0))`, and a run of `k` ones in the centre column
forces a zero triangle of height `k` reaching `k` columns to the left. These
are conditional theorems; on the real orbit the longest wall observed over 83
lags × 30000 steps has length **18**.

## 7. Transient dependency skeleton

Delete from the backward cone of `w_t(t)` every edge whose dependence a reset
erases: the edge to `(s-1, k-2)` is never erasable (permutivity), while the
edges to `(s-1,k-1)` and `(s-1,k)` vanish when `w_{s-1}(k) = 1` and
`w_{s-1}(k-1) = 1` respectively. Call what remains the **skeleton** `Sk(t)`.

**Theorem 7.1 (XOR spine).** *The cells `(t-j, t-2j)`, `0 ≤ j ≤ ⌊t/2⌋`, all lie
in `Sk(t)`. Hence `|Sk(t)| ≥ ⌊t/2⌋ + 1`.*

**Theorem 7.2.** *The self-edge of the diagonal cell `(t,t)` is erased exactly
when `x_{t-1}(0) = 1`: the diagonal's reset schedule **is** the centre column,
shifted by one.*

Theorem 7.2 is the reason one apparently promising route is empty — see §9.

## 8. Computational observations

Every number here is measured on a stated finite range, computed by at least
two independent implementations, and **no asymptotic constant is fitted**.

* Exact `T(K), P(K)` for `K ≤ 30000` by three independent algorithms.
  `T(K)/K = 1.3399` at `K = 30000`; the ratio is **not monotone** (`1.3400` at
  `K = 100`, `1.2525` at `K = 400`). No limit is claimed to exist.
* `T(K) > K` for `18 ≤ K ≤ 30000`; **false for `K ≤ 17`** (18 counterexamples).
* Theorem 4.6 verified at 29 999 / 29 999 levels, 0 mismatches; the five-way
  case classification cross-derived two independent ways with 0 disagreements.
* Density of levels with `D(K) = 1`: `0.546485`; mean increment `2.451873`;
  product `1.339911` against `T(K)/K = 1.339900` — Corollary 4.8 in numbers.
* **Certificate.** The first `10⁶` centre bits contain 998140 distinct length-28
  factors (three algorithms). With the factor-counting lemma: *if* the centre
  column is eventually periodic then `T + p ≥ 998140`. The method's ceiling is
  the length computed; it can never prove aperiodicity.
* The reset-erased skeleton has `≈ 0.39 t²` cells (`t ≤ 1200`); with the
  periodicity rewrite as well, `≈ 0.24 t²` — a reduction of about `2.1×`,
  still quadratic. No recursive family of derivations was found at `2^n`,
  `2^n − 1`, the doubling levels, or `T(K)`.

## 9. Why these results do not resolve centre-column aperiodicity

Three independent reasons, each of which alone is fatal to the obvious hopes.

**9.1 The column results are seed-blind.** Theorems 2.4 and 2.6 hold for
*every* finitely supported seed. Among such seeds there are ones whose centre
column is eventually periodic. So no consequence of these theorems can decide
Problem 1; a proof must use the single-cell initial condition in an essential
way.

**9.2 The prefix results describe the cycle region; the centre column is in the
transient.** By Proposition 2.3 the centre column is the diagonal `w_t(t)`, at
level `K = t` at time `t`. On the whole computed range `T(K) > K` for
`K ≥ 18`, so the diagonal is read strictly before its level settles.
Theorems 4.1–4.9 and 5.1–5.5 all apply *after* a level settles. Quantitatively,
about 69 % of the diagonal's backward cone has never settled, and the cone
becomes entirely transient from about `0.8 t` onwards.

**9.3 Growing complexity is not evidence.** Theorem 7.1 gives unbounded
derivations, and the measured skeleton is quadratic; the strongest sound
simplifications available (reset erasure plus periodicity rewriting) reduce it
only by a constant factor. **None of this bears on periodicity.** An eventually
periodic sequence may have derivations of unbounded size. We record the
measurements and draw no inference from them.

Finally, two statements that look like progress and are not:

* "(H) forces the diagonal's reset schedule to be eventually periodic" is
  **vacuous**: by Theorem 7.2 that schedule *is* the centre column.
* "Equal diagonal tails force equal prefix states, contradicting minimality of
  `T(K)`" is **Problem 1 restated**: a repeated prefix state at `t < t'` forces
  `T(K) ≤ t`, so the configuration it asks for is impossible outright, and the
  statement reduces to `¬(H)`. (Its natural intermediate step, "equal value ⟹
  equal derivation", is false: at `t = 300` and `t = 329` the centre bits agree
  while the derivations have 28 553 and 33 843 nodes.)

## 10. Open bridge problems

> **Bridge (open).** If the centre column of the single-seed rule-30 diagram is
> eventually periodic, then some other column is eventually periodic.

With Theorem 2.6 this settles Problem 1 immediately. Five formulations of it
appear in our work — a local lemma about column `−1`, a zero-wall statement, a
statement about reset schedules, and two others — and all are the same
statement. We did not move it.

Three subsidiary open problems, in decreasing order of how much we would learn:

1. **Does a positive density of levels have `D(K) = 1`?** By Corollary 4.8 this
   is one factor of `liminf T(K)/K`; it is the smallest statement in our
   framework whose proof would be new information. Measured: `0.546485` at
   `K = 30000`, longest observed run of `D = 0` is 12.
2. **Is `liminf_K T(K)/K > 1`?** Equivalently, does the transient outrun the
   diagonal? A single `K > 17` with `T(K) ≤ K` refutes it.
3. **Is every level beyond 400 COLLAPSING?** Equivalently, is `399` the last
   coordinate that is eventually zero? Our evidence is four events.

We note explicitly that a proof of (1), (2) or (3) would **not** settle
Problem 1: none of them constrains the values the diagonal reads. They would
sharpen the description of the transient, which is where we believe the
difficulty lives.

---

## Notes on status and novelty

* Novelty is **unresolved**. We could not reach any primary source (every
  attempt returned HTTP 403), so we neither claim any result is new nor assert
  that any is known. `NOVELTY_STATUS.md` lists our priors, labelled as priors,
  together with search terms and the specific results most in need of an
  expert's judgement — chiefly Theorem 2.6 and the prefix tower of §§3–4.
* The rule-numbering convention (`f(l,c,r) = l XOR (c OR r)`) is verified only
  against a 14-bit prefix from a secondary source. This is the corpus's largest
  unverified assumption and is stated as such.
* All computations are deterministic; artifacts, hashes, solver versions and
  independent implementations are catalogued in
  `COMPUTATIONAL_CERTIFICATES.md`.
