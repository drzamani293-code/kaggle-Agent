# Consequences of the Periodicity Hypothesis — Phase 2E §8

Standing hypothesis, as in Phase 2B:

> **(H)** the centre column is eventually `p`-periodic: there are `p ≥ 1` and
> `T* ≥ 0` with `x_{t+p}(0) = x_t(0)` for all `t ≥ T*`.

**No contradiction is derived in this section, and none is claimed.** Every
statement below is either proved outright from (H) together with results already
established in Phases 1–2E, or explicitly labelled as depending on an unproved
lemma. Per the brief: a contradiction is not asserted unless every dependency is
formally proved, and none of the candidate routes here meets that bar.

Note also that (H) is conjecturally **false**, so all conditional statements are
vacuous if Problem 1 has the expected answer, and none of them can be tested
computationally. That is a property of the logical form, inherited from Phase 1
registry item C3 and Phase 2B.

---

## 1. What (H) says in transient-frontier language

### PROPOSITION 8.1 (translation)

> (H) holds with `(p, T*)` **iff** `w_{t+p}(t+p) = w_t(t)` for all `t ≥ T*`.

*Proof.* Theorem EA3, `x_t(0) = w_t(t)`. ∎

Note the shape: the index moves on **both** coordinates. This is not a
statement about a fixed level, and it is not the lag-`p` periodicity of any
coordinate. That mismatch is the entire difficulty, and it is why the exact
level-by-level theory of §§1–4 does not attach to (H).

### PROPOSITION 8.2 (the diagonal is read out of the transient)

> Assume (H) and assume `T(K) > K` for all `K ≥ 18` (**BOUNDED OBSERVATION**,
> verified `K ≤ 30000`, not proved). Then for every `t ≥ max(T*, 18)` the value
> `w_t(t)` is a transient value of the level-`t` prefix dynamics: `t < T(t)`.
> So under (H) an eventually periodic sequence is being read entirely out of
> non-settled cells.

*Proof.* Immediate. ∎

**This is not a contradiction and must not be presented as one.** A transient
region can emit an eventually periodic sequence — for instance the constant
sequence. Proposition 8.2 rules nothing out; it locates the problem.

---

## 2. What follows rigorously

### THEOREM 8.3 (Phase 2C Theorem D1 is vacuous under (H))

> Phase 2C's D1 states `x_{t+P(K)}(0) = x_t(P(K))` for `T(K) ≤ t ≤ K - P(K)`.
> That range is non-empty only when `T(K) ≤ K - P(K)`.

Measured over `K ≤ 30000`: the range is non-empty for **0 levels** at every lag
`p ≥ 7`, and for at most 14 tiny levels at `p ∈ {1,2,3}` (all with `K ≤ 16`).
So D1 carries no information about the real orbit at any period of interest.
Recorded as a **negative result**, and it is the reason Phase 2E did not pursue
D1 further.

### THEOREM 8.4 (at most one line can be periodic)

> Assume (H). Then for every `r ≠ 0` the sequence `t ↦ w_t(t + r)` is **not**
> eventually periodic.

*Proof.* By Theorem 5.1, `w_t(t+r) = x_t(r)` is column `r` of the original
diagram. Phase 1 Theorem W2′ states that at most one column of the rule-30
diagram from a single-cell seed is eventually periodic. Under (H) column `0` is
that one column, so no other column can be. ∎

This is a genuine, fully proved conditional consequence, and it is the sharpest
one Phase 2E obtains. It also converts into a **falsification target**: a proof
that *some* column `r ≠ 0` is eventually periodic would, with W2′, refute (H)
and settle Problem 1. Registered as **F5**.

### COROLLARY 8.5 (the width-2 relation, in strip form)

> Assume (H) with period `p`. For `t ≥ T*`, Corollary 5.2 at `r = 0` gives
> ```
>     q_{t+1}(0)  XOR  q_t(-1)  =  q_t(0) OR q_t(1) .
> ```
> On the set `Z := { t ≥ T* : q_t(0) = 0 }` this forces
> `q_t(1) = q_{t+1}(0) XOR q_t(-1)`, so on `Z` column `+1` is determined by
> column `−1` together with eventually periodic data.

*Proof.* Substitute; when `q_t(0) = 0` the right side is `q_t(1)`. ∎

`Z` is itself eventually periodic under (H) (it is defined by the periodic
column `0`), and is co-infinite unless column `0` is eventually all ones — which
Phase 1's factor-counting bound excludes for any small `(T*, p)`.

**This is Phase 1's Proposition 6 again**, in strip coordinates. Corollary 8.5
is therefore not new mathematics; it is a demonstration that the transient
frontier picture lands back on the same relation. That is worth recording
precisely because it shows §§1–7 did *not* find a new handle on (H).

### PROPOSITION 8.6 (quantitative refutation of small budgets)

> For each line `r ∈ [-4, 4]` and each lag `q ≤ 32`, the lag-`q` comparison on
> that line still fails at the index tabulated below, so any eventual
> `q`-periodicity of that line must begin later than that index.

| `r` | length | min over `q ≤ 32` of (preperiod + `q`) | at `q` = |
|---|---|---|---|
| −4 | 20001 | 19997 | 30 |
| −3 | 20001 | 19997 | 14 |
| −2 | 20001 | 19997 | 3 |
| −1 | 20001 | 19997 | 10 |
| **0** | 20001 | **19996** | 29 |
| +1 | 20001 | 19996 | 29 |
| +2 | 20001 | 19992 | 29 |
| +3 | 20001 | 19996 | 7 |
| +4 | 20001 | 19994 | 13 |

This is a **refutation of bounded (preperiod, period) budgets only**. It
establishes nothing about aperiodicity, and it is far weaker than Phase 1's
factor-counting bound for the centre column (`T* + p ≥ 998140` for
`p ≤ 20000`), which remains the strongest finite statement in the project.
It is included because the brief asks for the hypothesis to be pushed on every
line, not only `r = 0`.

---

## 3. Routes that do NOT close, and why

### Route A — "the diagonal is transient, therefore aperiodic"

**Invalid.** Proposition 8.2 says the diagonal reads transient cells; that is
not an obstruction to periodicity of the read-out sequence. Recorded here
because it is the most tempting misreading of the whole Phase 2B–2E line of
work, and because Phase 2B's registry entry G5 carries the same warning.

### Route B — "resets are aperiodic in `K`, therefore the diagonal is aperiodic in `t`"

**Not closed: two unproved dependencies.** It would require
(i) a proof that the RESETTING/INHERITING word is not eventually periodic in
`K` (open, see F3), and
(ii) a proof that the diagonal value at time `t` is determined by the reset
events in its cone (**false as stated** — §7 shows the cone's un-rewritable
part is quadratic and no finite summary of the resets determines `w_t(t)`).
Since (ii) fails, Route B does not merely lack a proof; the bridge itself is
missing.

### Route C — "the frontier outruns the diagonal, therefore the diagonal never
enters a cycle"

**Circular.** "Never enters a cycle" here means only "never reads a settled
cell", which is Proposition 8.2 again, not a statement about the value sequence.

### Route D — via Corollary 8.5 to the Bridge

**Open, and identical to the Phase 1 Bridge (registry C2) / Phase 2B G1.**
Corollary 8.5 gives column `+1` in terms of column `−1` on an eventually
periodic set. Making both eventually periodic would contradict Phase 1 Theorem
W2. Phase 2E adds nothing to this beyond restating it in strip coordinates.

---

## 4. Controls

* Every conditional statement lists its hypotheses explicitly, including the
  bounded ones (`T(K) > K` for `18 ≤ K ≤ 30000`).
* **No contradiction with (H) is claimed anywhere in this package.**
* Nothing here is inferred from growing complexity, from DAG size, from strip
  word counts, or from any statistical regularity. No randomness tests were run.
* Failed routes A–D are kept in full rather than deleted, per the standing rule
  on preserving negative results.
