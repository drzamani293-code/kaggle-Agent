# Co-moving Transient Strip — Phase 2E §5

The brief asks for the dynamics of the strip `q_t(r) = w_t(t + r)` for
`r = 1 … 12`, moving with the diagonal.

**The main result of this section is negative and should be stated first: the
co-moving strip is not a new object. It is literally the original columns of the
rule-30 diagram, and it is not a finite-state system.** Both halves are proved
below, and the second is proved by explicit finite counterexamples at every
radius from 1 to 12.

---

## 1. The strip is the original diagram

### THEOREM 5.1 (strip identity)

> For all `t ≥ 0` and all `r` with `t + r ≥ 0`,   `q_t(r) := w_t(t+r) = x_t(r)`.

*Proof.* Phase 2B Theorem EA1 defines `w_t(k) = x_t(-t+k)`. Put `k = t + r`. ∎

### COROLLARY 5.2 (the strip's update rule is rule 30 in original orientation)

> `q_{t+1}(r) = q_t(r-1) XOR ( q_t(r) OR q_t(r+1) )`.

*Proof.* Substitute `k = t+1+r` into the edge-aligned rule
`w_{t+1}(k) = w_t(k-2) XOR (w_t(k-1) OR w_t(k))`; the three arguments become
`w_t(t+r-1), w_t(t+r), w_t(t+r+1)`, i.e. `q_t(r-1), q_t(r), q_t(r+1)`. ∎

### COROLLARY 5.3

> `q_t(0) = x_t(0)` is the centre column, and `t ↦ q_t(r)` is exactly column `r`
> of the original diagram.

*Verified:* `frontier_lab.strip_identity`, cell-by-cell against an independent
original-coordinate simulation (`_orbit_rows`, the bit-parallel engine
`row = ((row<<1) ^ (row | (row>>1))) & mask`): **5081 cells, 0 mismatches**,
`t ≤ 300`, `|r| ≤ 8`.

**Consequence for the research programme.** Any statement about the co-moving
strip is a statement about the original columns `0 … R` of rule 30. In
particular a proof that the strip is eventually periodic *is* a positive answer
to Problem 1, and a proof that it is not *is* the negative answer. The strip is
a change of viewpoint with zero information gain. This is worth recording
because the edge-aligned picture makes the strip look like a fresh finite-state
object, and it is not.

---

## 2. The strip is NOT autonomous

### THEOREM 5.4

> For every `R ≥ 0`, the map `q_t(0..R) ↦ q_{t+1}(0..R)` is **not** well defined:
> the successor of the width-`(R+1)` strip word is not a function of that word.

*Proof.* By Corollary 5.2, `q_{t+1}(R)` reads `q_t(R+1)`, which lies outside the
strip. Rule 30 is left-permutive but **not right-permutive** (Phase 1 Fact F4),
so this dependence cannot be eliminated: `q_{t+1}(R)` genuinely varies with
`q_t(R+1)` whenever `q_t(R-1) XOR (q_t(R) OR ·)` distinguishes the two values,
which happens whenever `q_t(R) = 0`. ∎

This is the exact mirror of the reason the *edge-aligned prefix* IS autonomous:
information in rule 30 flows leftwards in original coordinates, so a left-hand
prefix of the edge-aligned row is closed and a left-hand strip of the original
row is not.

### Explicit finite counterexamples

`frontier_lab.strip_not_autonomous` searches the real orbit for two times with
the same strip word and different successors. One is found at **every** radius
`R = 1 … 12`; these are finite, checkable refutations, not statistics.

States are integers with **bit `r` equal to `q_t(r)`**, `r = 0 … R`.

| `R` | width | witness `(t₁, t₂)` | common state | `q_{t₁+1}` | `q_{t₂+1}` |
|---|---|---|---|---|---|
| 1 | 2 | (1, 3) | 3 | 0 | 1 |
| 2 | 3 | (1, 5) | 3 | 4 | 0 |
| 3 | 4 | (0, 4) | 1 | 3 | 11 |
| 4 | 5 | (6, 14) | 8 | 28 | 29 |
| 5 | 6 | (6, 14) | 8 | 60 | 61 |
| 6 | 7 | (6, 14) | 72 | 124 | 125 |
| 7 | 8 | (5, 13) | 59 | 72 | 200 |
| 8 | 9 | (32, 44) | 70 | 235 | 234 |
| 9 | 10 | (32, 44) | 70 | 747 | 234 |
| 10 | 11 | (31, 43) | 60 | 1094 | 70 |
| 11 | 12 | (8, 94) | 263 | 905 | 2952 |
| 12 | 13 | (7, 93) | 252 | 263 | 4359 |

Sanity check on the smallest case, done by hand: at `R = 1`, `t₁ = 1` and
`t₂ = 3` both have `q(0) = q(1) = 1`, yet `q_2(0..1) = (0,0)` and
`q_4(0..1) = (1,0)`. The two futures differ because `q_1(2) ≠ q_3(2)` — a
column outside the strip.

---

## 3. Strip word census

`t ≤ 20000`:

| `R` | width | distinct words seen | `2^(R+1)` | all seen | first repeat |
|---|---|---|---|---|---|
| 1 | 2 | 4 | 4 | yes | (1, 3) |
| 2 | 3 | 8 | 8 | yes | (0, 4) |
| 3 | 4 | 16 | 16 | yes | (0, 4) |
| 4 | 5 | 32 | 32 | yes | (5, 13) |
| 5 | 6 | 64 | 64 | yes | (5, 13) |
| 6 | 7 | 128 | 128 | yes | (5, 13) |
| 7 | 8 | 256 | 256 | yes | (5, 13) |
| 8 | 9 | 512 | 512 | yes | (30, 42) |
| 9 | 10 | 1024 | 1024 | yes | (30, 42) |
| 10 | 11 | 2048 | 2048 | yes | (30, 42) |
| 11 | 12 | 4062 | 4096 | **no** (34 missing) | (6, 92) |
| 12 | 13 | 7514 | 8192 | **no** (678 missing) | (7, 93) |

### How to read this table, and how not to

* **The repeats prove nothing.** A repeated strip word does not imply
  periodicity, because the strip is not autonomous (Theorem 5.4): the same
  word recurs with different successors, by construction of the witnesses in
  §2. Every "first repeat" entry above is therefore a **statistic, not a
  cycle**.
* **The missing words at `R = 11, 12` prove nothing either.** They are words not
  observed within `t ≤ 20000`; whether they are unreachable is a different
  question that no computation here settles. Recorded as *not observed in the
  sampled range* — the same convention Phase 2B used for the wall automaton.
* Per the brief: **a finite-state pattern observed in a bounded run is not a
  theorem.** Nothing in this table is promoted to one.
* **No randomness test was run** and no statistical-regularity claim is used as
  evidence anywhere (standing constraint from Phase 2B).

---

## 4. Cone growth

### THEOREM 5.5

> For `0 ≤ s ≤ t`, the values `q_t(0..R)` are determined by
> `q_s(r)` for `r ∈ [-(t-s), R + (t-s)]`, and by no smaller interval in general:
> the backward cone opens by exactly one column on **each** side per step.

*Proof.* Corollary 5.2 makes each cell depend on `r-1, r, r+1` one step back;
induction gives the interval. Minimality on the right: the local rule
`l XOR (m OR n)` depends on its right argument `n` exactly when `m = 0`, so
whenever `q_t(R) = 0` the value `q_{t+1}(R)` genuinely varies with `q_t(R+1)`.
Minimality on the left is immediate from left-permutivity (Phase 1 Fact F4):
the rule is `l XOR (…)`, so flipping `l` always flips the output. ∎

*Verified:* `frontier_lab.strip_dependency_growth` — cone `[-50, 54]` after 50
steps from `R = 4`, one column per side per step; the sensitivity table
confirms that the right argument matters exactly at the local patterns
`(l, m) ∈ {(0,0), (1,0)}`.

**This is the obstruction in one line.** The strip of any fixed width is
determined by an initial segment whose length grows with `t`; the edge-aligned
prefix of any fixed width is not. That asymmetry is why Phases 2C–2E work on
the prefix side, and it is also why the prefix side never reaches the diagonal:
the diagonal sits at level `K = t`, which is not fixed.

---

## 6. The full-width strip `-R ≤ r ≤ R` (Phase 2E brief §5)

§§1–5 worked with the one-sided strip `r = 0 … R`. The brief's strip is
two-sided, `q_t(r) = w_t(t+r)` for `-R ≤ r ≤ R`, width `2R+1`. Everything above
carries over; this section adds what the two-sided version makes visible.

### 6.1 The exact update law

Packing bit `j` of a state as `q(j - R)`:

```
    q'  =  ( (q << 1) | left )  XOR  ( q | (q >> 1) | (right << 2R) )     (mod 2^{2R+1})
```

with `left = q_t(-R-1)` and `right = q_t(R+1)` the two boundary bits. Written
bit by bit this is `q'(r) = q(r-1) XOR (q(r) OR q(r+1))` (Corollary 5.2).

*Verified:* the bit-parallel and bit-by-bit forms agree exhaustively for
`R = 1, 2, 3`; the law reproduces the real orbit exactly with the true boundary
bits (`R = 6`, 2000 steps, 0 mismatches).

### 6.2 Minimal additional variables

### THEOREM D1

> To run the strip `[-R, R]` forward for `n` steps one needs, and needs no more
> than, the initial values on `[-R-n, R+n]`: exactly `n` extra coordinates on
> each side. **No finite extension is autonomous for all time.**

*Proof.* Sufficiency by induction from Corollary 5.2. Necessity on the left is
left-permutivity (flipping `q(r-1)` always flips `q'(r)`); on the right, the
local rule `l XOR (m OR n)` depends on `n` whenever `m = 0`, so the dependence
is genuine (Theorem 5.5). ∎

### 6.3 The strip automaton, and the answer for **every** radius

Guessing the two boundary bits makes the strip a nondeterministic automaton on
`2^{2R+1}` states, with four transitions from each state.

### THEOREM D2

> For every `R`, the transition relation is **surjective with in-degree exactly
> four**: given any target state `q'` and any choice of the top bit `q(R)` and
> the right boundary bit, the predecessor is uniquely determined. Consequently
> every state lies on a bi-infinite admissible path, and
> **the maximal recurrent core is the entire state space `{0,1}^{2R+1}`.**

*Proof (constructive).* Read the constraint `q'(r) = q(r-1) XOR (q(r) OR q(r+1))`
from `r = R` downwards. With `q(R)` and `q(R+1) = right` chosen, `q(R-1)` is
forced; then `q(R-2)`, and so on down to `q(-R-1) = left`. Each step is a
forced XOR, so the preimage exists and is unique for each of the four choices.
A finite digraph in which every node has in-degree ≥ 1 has every node reachable
from a cycle, so the recurrent core is everything. ∎

*Verified exhaustively* for `R = 1 … 8` (up to 131072 states): in-degree exactly
4 at every state, 0 preimage failures.

**This is a stronger negative result than Phase 2B's wall automaton.** There the
maximal invariant set was merely non-empty at every radius computed; here it is
*everything*, at every radius, **proved**. The local relaxation of the co-moving
strip excludes nothing whatsoever, so no bounded impossibility can ever come
from it.

### 6.4 The real orbit inside that space

`t ≤ 50000`:

| `R` | width | states total | observed | sampling-limited? |
|---|---|---|---|---|
| 2 | 5 | 32 | 32 | no |
| 4 | 9 | 512 | 512 | no |
| 6 | 13 | 8192 | 8172 | no |
| 8 | 17 | 131072 | 41251 | **yes** (only 50001 samples) |

The orbit is **not** confined to a small recurrent subset — there is no small
recurrent subset to be confined to (Theorem D2). At `R = 8` the observed count
is bounded by the length of the run, not by the dynamics; that row measures the
experiment, not rule 30.

### 6.5 How far a strip determines the centre bit

### THEOREM D4

> `q_t(-R..R)` determines `q_{t+n}(0)` for every `n ≤ R`, because the backward
> cone of `(t+n, 0)` at time `t` is exactly `[-n, n]`. For `n = R+1` it does
> not.

The positive half is the cone argument. The negative half is exhibited
explicitly: two times with the **same** strip state and different `q_{t+R+1}(0)`.

| `R` | theory: determined up to `n =` | first `n` with an explicit counterexample |
|---|---|---|
| 1 | 1 | **2** |
| 2 | 2 | **3** |
| 3 | 3 | **4** |
| 4 | 4 | **5** |
| 6 | 6 | **7** |

The bound is exactly attained at every radius tested: the strip sees exactly
`R` steps into the future of the centre column and not one more.

### 6.6 Does the centre column determine itself?

Testing whether the last `m` centre bits determine the next one — i.e. whether
the centre column is generated by a finite automaton reading only its own past:

| `m` | 1 | 2 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|---|
| determined? | **no** | **no** | **no** | **no** | **no** | **no** | **no** |

Each "no" is an explicit pair of times with the same length-`m` history and a
different next bit — a finite refutation, e.g. at `m = 20` the times
`t = 227` and `t = 1994`.

**Control, stated because this is exactly where over-reading happens.** These
refutations say the centre column is not `m`-step Markov for `m ≤ 20`. They say
**nothing** about aperiodicity: an eventually periodic sequence with a long
preperiod also fails such tests at small `m`. Per the brief, a finite-state
pattern observed (or refuted) in a bounded run is not a theorem about eventual
periodicity in either direction.
