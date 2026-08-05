# Proof Audit Report

Every theorem in `VERIFIED_THEOREMS.md` was re-derived from
`DEFINITIONS_AND_NOTATION.md` without consulting the original phase's prose,
then compared with the original. This report records what the re-derivation
found: hidden assumptions, boundary cases, index and orientation checks, and
the places where a claim had to be **downgraded, restricted or corrected**.

Computational checks were used **only** as falsification tools, never as
evidence that a proof is correct. Where a check is quoted, its role is
"an error would have shown up here", not "therefore the proof is right".

---

## 0. Audit outcome summary

| outcome | count | items |
|---|---|---|
| reconstructed unchanged | 14 | T1, T2, T3, T5, T7, T9, T10, T14, T16, T17, T19, T20, T21.1–T21.4 |
| reconstructed with a **strengthening** | 1 | T8 (spurious hypothesis removed) |
| reconstructed with a **restriction** | 2 | T21.5 (scope of "no finite-state"), T6 (ranges made explicit) |
| reconstructed with a **redundancy removed** | 1 | T13.2 (`max` was redundant) |
| **downgraded** from theorem to conditional | 1 | T4 |
| **downgraded** from theorem to vacuous | 1 | F3 → `VQ-01` |
| **downgraded** from lemma to restatement | 1 | F5 → `VQ-02` |
| audit **caught a live indexing error** | 1 | A-04 (T15) |

No theorem failed to reconstruct. One (T17.2) reconstructs correctly but has a
hypothesis that no computation has ever satisfied, and is labelled accordingly.

---

## A. Findings, one per issue

### A-01 — T8's hypothesis "`t ≥ K/2`" is spurious. **STRENGTHENING.**

*Original (Phase 2B EA2):* "there is a map `F_K` with `W_{t+1}^K = F_K(W_t^K)`
for all `t ≥ K/2`."

*Reconstruction:* the autonomy argument reads only the one-sided recurrence
T7, which holds at every `t ≥ 0`; the coordinates it reads (`k-2, k-1, k`) are
all `≤ K`. No lower bound on `t` is used anywhere.

*Falsification check:* `F_K` applied to `W_t^K` reproduces `W_{t+1}^K` for
`K = 40`, every `t ∈ [0, 78]` — including `t < K/2 = 20`. No mismatch.

*Resolution:* the qualifier is removed. Recorded as correction **C-08**. The
original statement was not wrong, merely weaker than necessary; but a weaker
statement invites the reader to think something happens at `t ≈ K/2`, and
nothing does.

### A-02 — T4 is conditional, not absolute. **DOWNGRADE.**

*Original (Phase 1 D1):* "If the centre column is eventually periodic then
`T + p ≥ 998140`," presented alongside proved lemmas.

*Reconstruction:* the *lemma* (an eventually periodic sequence has at most
`T+p` factors of each length) is unconditional and elementary. The *bound* also
requires the numeral 998140, which is the output of a program counting
length-28 factors of a stored 10⁶-bit file. That is a finite certificate.

*Resolution:* T4 is stated as **category B (conditional theorem)** with an
explicit **category C** dependency (`CC-02`). It is never listed as an
unconditional theorem. Phase 1's own ledger already labelled it `COMP + C10`,
so this is a presentational tightening rather than a substantive correction —
but the brief asks for the distinction to be enforced, and it is.

### A-03 — T13.2's `max` is redundant. **REDUNDANCY REMOVED.**

*Original (Phase 2D R3):* `T(K) ≤ max( T(K-1), τ(K) + 1 )`.

*Reconstruction:* `τ(K) := min{ t ≥ T(K-1) : … }`, so `τ(K) ≥ T(K-1)` by
definition, hence `τ(K) + 1 > T(K-1)` always and the `max` is the second
argument. The bound is simply `T(K) ≤ τ(K) + 1`.

*Why it matters:* stated with the `max`, the bound looks like it might be tight
via the first argument, which invites a wrong reading of when equality holds.
Phase 2E's exact recurrence (T18) settles equality by a different mechanism (the
defect bit), and the redundancy obscured that. Recorded as **C-09**.

### A-04 — T15 applied with level indices instead of coordinate indices. **LIVE ERROR CAUGHT DURING THIS AUDIT.**

While re-deriving T15 (Phase 2D N2) the auditor tested
`w_t(m-2) = w_t(m-1)` at `m ∈ {3, 8, 29, 400}` — the **non-COLLAPSING levels** —
and it **failed at all four**. The theorem's hypothesis is about
**coordinate `m` being zero on its cycle**, and the eventually-zero
*coordinates* are `{2, 7, 28, 399}`, one less than the levels. Re-tested at
`m ∈ {2, 7, 28, 399}` the identity **holds at all four**, with cycle parities
`XOR_t w_t(m-1) = XOR_t w_t(m-2) = 1` in every case.

*Resolution:* the original Phase 2D statement is **correct**; the auditor's
first reading was wrong. An explicit indexing warning is now attached to T15,
and the off-by-one between "eventually-zero coordinate `k`" and
"non-COLLAPSING level `k+1`" is called out in `DEFINITIONS_AND_NOTATION.md` §6.
This is the audit working as intended and is recorded rather than quietly
fixed.

### A-05 — "No finite-state description exists" was too broad. **RESTRICTION.**

*Original (Phase 2E Corollary C3, as worded in the Phase 2E README):* "No
finite-state description of the frontier increments exists."

*Reconstruction:* what is proved is T21.5: `D(K) = w_{T(K-1)}(K) XOR Φ_K`,
where `Φ_K` is a function of the base cycle word. Flipping `w_{T(K-1)}(K)`
flips `D(K)` and changes no cycle word below `K`. That refutes exactly the class
of descriptions whose state is a **fixed-width window of cycle words of
coordinates below `K`, read from phase `T(K-1)`**.

It does **not** refute a machine reading anything else — transient values, a
differently-phased window, or data from above `K`. Recorded as **C-12**; the
scope sentence is now attached to T21.5 itself.

### A-06 — F3 is vacuous, not a lemma. **DOWNGRADE to `VQ-01`.**

Phase 2E's registry item F3 ("eventual periodicity of `c_t` forces eventual
periodicity of the reset schedule along the diagonal") reconstructs as: the
diagonal's reset schedule **is** the centre column shifted by one (T21.10).
So F3 reads "(H) ⟹ (H)". True, and carries no information. Filed in ledger
category **G**.

### A-07 — F5 is Problem 1 restated. **DOWNGRADE to `VQ-02`.**

F5's conclusion asks for `W_t^K = W_{t'}^K` with `t < T(K) ≤ t'`. For a
deterministic map, a repeated prefix state at times `t < t'` forces periodicity
from `t`, hence `T(K) ≤ t` — so the requested configuration is **impossible
outright**, with no reference to the diagonal. F5 therefore reads
"(H) ⟹ FALSE", i.e. F5 *is* `¬(H)`.

*Falsification check:* searching for a repeat with `t < T(k)` at `k ∈ {5,20,100}`
over the full computed range finds none, as the argument requires.

Filed in ledger category **G**. Its natural intermediate step ("equal value ⟹
equal derivation") is independently **refuted** by the pair `t = 300`,
`t = 329`; that refutation is `RF-05`.

### A-08 — T17.2's hypothesis has never been met. **LABELLING.**

The pigeonhole theorem is correct: a chain of more than `2^{2P}` consecutive
COLLAPSING levels forces `K`-periodicity of the cycle words. At `P = 16` that
is `4.3 × 10⁹` levels against a longest computed chain of `29 600`. It stays in
the theorem list because it *is* a theorem, with a bold note that it has never
applied.

### A-09 — T3.2's period bound is exponential and blocks quantification.

The interior-inheritance lemma gives period at most `p · 2^{j-i+1}`. That is
fine for the qualitative statement T3 but rules out any quantitative version
("if two columns at distance `d` are periodic then the interior period is at
most …" is useless for large `d`). Recorded so that no future phase tries to
extract a bound from it.

### A-10 — T2.4's uniformity of `T` is the load-bearing step.

If the preperiod were allowed to grow by one per column, the contradiction in
T2 would fail: one could not fix a single `T` and push `k` to infinity. The
proof of T2.4 keeps `T` fixed because T1(b) reads only times `t` and `t+1`.
This is the step most likely to be mis-transcribed, and it is flagged in both
the theorem and here.

---

## B. Systematic checks applied to every theorem

### B-1 Hidden assumptions

| theorem | assumption that had to be made explicit |
|---|---|
| T2, T3 | **finite support** — used twice (light cone T2.1, frozen edge T2.2). Neither holds for general configurations. |
| T2.2 | the leftmost support cell is `1` **by definition of support**; for FIN with support `[-m,m]` this means `x_0(-m) = 1`. |
| T6.* | a wall on a **stated finite range**, never globally. Phase 2B's `(H-WALL)` is global; the theorems only need local ranges and are stated that way here. |
| T10, T13, T16–T19 | the base `(K-1)`-prefix has **already settled** (`t ≥ T(K-1)`). Everything about the fibre is a cycle-region statement. |
| T18.1 | `P(K-1) | P(K)` — needed so that both `a` and `c` are `P(K)`-periodic from `T(K-1)`. Supplied by T9.1. |
| T21.3 | coordinate `K` takes the value `1` **at least once**. Fails for the eventually-zero coordinates `{2,7,28,399}`, where the bound is vacuous. |
| T20 | the erasure conditions read **real orbit values**, so the skeleton is SEED-specific; only the *spine* sub-argument is configuration-independent. |

### B-2 Boundary cases

* `K = 0, 1`: the fibre recursion needs `K ≥ 2` (it reads `w(K-2)`). T10 and
  everything downstream is stated for `K ≥ 2`. `T(0) = 0`, `P(0) = 1`.
* `t = 0`: T8 checked at `t = 0` (A-01). `w_0 = (1,0,0,…)` is the seed.
* `k < 0` and `k > 2s`: both are constant-`0` leaves; the DAG and skeleton
  constructions fold them, and the folding is justified by T7.1 and T7.3.
* `ρ(K)` undefined: happens exactly when coordinate `K-1` is identically `0`
  before `T(K)`; in the computed range at `K ∈ {4,5,6,7,8}` only. Statements
  about `ρ` carry the caveat.
* `τ(K)` undefined: exactly on non-COLLAPSING levels (T14).
* Period-doubling levels: T18(b) gives `T(K) = T(K-1)`, checked at all four
  computed doubling points. Note this is **not** in tension with monotonicity
  (T9.1) — see B-5.

### B-3 Time-index shifts

Each theorem was re-derived with the convention "the rule computes time `t+1`
from time `t`" written out explicitly.

* T5: inputs at `t`, output at `t+1`. A shift here would put `x_{t+1}` in the
  coefficients and break the identity; the 16-case table is the check.
* T6.1: needs the wall at **both** `t` and `t+1`. Stated.
* T16.1: erasure occurs at `τ+1`, not at `τ`. The value `1 XOR a_τ` uses `a` at
  time `τ`. An off-by-one here would move `T(K)` by one and break T19.
* T21.10: the diagonal's self-edge at `(t,t)` is erased by `w_{t-1}(t-1)`, i.e.
  by `x_{t-1}(0)` — the centre column at time `t-1`, **not** `t`. The shift is
  what makes T21.10 a shift-by-one identity rather than an equality.

### B-4 Left/right orientation

The single orientation fact is that `f(l,c,r) = l XOR (c OR r)` is permutive in
`l` and not in `r`. Consequences, all checked for consistent direction:

* left reconstruction (T1(b)) recovers `col_{j-1}` from `col_j, col_{j+1}` —
  **leftwards**;
* T2.4 propagates periodicity **leftwards**;
* in edge-aligned coordinates the closed sets are **prefixes** `k ≤ K` (T8);
* in the original/strip coordinates the closed sets would have to be **right**
  half-lines, and are not (T21.7);
* the mirror rule 86 reverses all of the above, and is used as an independent
  implementation precisely to catch a global orientation slip.

The `k-2` in T7 (rather than `k+2`) is forced by the frame moving one step left
per time step; it was re-derived from `w_{t+1}(k) = x_{t+1}(-(t+1)+k)` rather
than copied.

### B-5 Minimal versus non-minimal periods

This is where the corpus is most exposed, and two places need care.

* **`r_q(k)` depends on `q`** (T21.2): at a *longer* lag the coordinate can
  appear to settle *earlier*. Therefore `T(K) = max_k r_{P(K)}(k)` must use the
  lag `P(K)`, not each coordinate's own `P(k)`. Using the wrong lag would give a
  wrong `T`.
* **T18(b) and monotonicity.** At a doubling level the bound derived from the
  decomposition is `T(K) ≤ T(K-1)`, and monotonicity gives `T(K) ≥ T(K-1)`;
  together, equality. Without T21.2 one might think the two are contradictory.
  They are not: the first is computed at lag `2P(K-1)`, the second is a
  statement about the minimal preperiods.
* `P(K)` is always minimal in this corpus. Where a proof needs a multiple, the
  multiple is written out.

### B-6 Preperiod synchronisation by max / lcm

T2.3 is used in T2 and T3. The correct normalisation is **`max` for
preperiods, `lcm` for periods**. Two failure modes were checked for and are
absent: (i) taking `lcm` of preperiods; (ii) assuming the common preperiod can
be taken to be either of the two originals.

In the prefix tower the analogue is T9.1 (`T` non-decreasing, `P(K-1) | P(K)`),
which is proved by the projection rather than by normalisation.

### B-7 Finite-support versus single-cell

| result | needs |
|---|---|
| T1, T5, T7, T9, T21.7, T21.8 | nothing — ANY configuration |
| T2, T3 | FIN (finite support). **Not** SEED-specific. |
| T4, T8, T10, T13–T20, T21.1–T21.6, T21.9, T21.10 | SEED, or at least a specific orbit — they quantify over the actual values of the single-cell orbit |

The FIN row is the seed-blindness obstruction in table form: those theorems
hold for seeds whose centre column *is* eventually periodic (e.g. any
configuration eventually reaching a spatially periodic state has plenty of
eventually periodic columns — though never two adjacent, by T2), so they cannot
decide Problem 1. Any future route must live in the SEED row.

---

## C. Supplementary falsification checks run during the audit

These are listed for completeness. **None of them is offered as evidence that a
proof is correct**; each is a test that would have exposed a specific error.

| # | check | would have caught | result |
|---|---|---|---|
| 1 | `F_K(W_t^K) = W_{t+1}^K` for `K=40`, `t ∈ [0,78]` | a spurious `t ≥ K/2` hypothesis, an index slip in `F_K` | 0 mismatches |
| 2 | `w_t(k) = 0` for `k > 2t`, `t ≤ 120`, `k ≤ 200` | a wrong support bound in the DAG folding | 0 violations |
| 3 | `T(K) ≤ T(K-1) + P(K-1)`, `K ≤ 3000` | a broken T13.1 | 0 violations |
| 4 | `T(K) + P(K) > ⌊K/2⌋`, `K ≤ 3000` | a broken T21.3 | 0 violations |
| 5 | T15 at coordinates `{2,7,28,399}` | the index error of A-04 | holds at all four |
| 6 | `q_t(r) = x_t(r)` for `|r| ≤ 6`, `t ≤ 300`, including negative `r` | a sign error in the strip identity | 0 mismatches |
| 7 | `T(K) + P(K) ≤ 2^{K+1}`, `K ≤ 20` | a wrong state-space bound in T8 | holds |
| 8 | no prefix-state repeat with `t < T(k)`, `k ∈ {5,20,100}` | a wrong argument in A-07 | none found |

---

## D. What the audit did not do

* It did **not** re-verify the certificates' numerals by re-running the long
  computations; those are catalogued separately with hashes, and the test suite
  checks the hashes, not the mathematics.
* It did **not** attempt to reconstruct any literature claim. Phase 2B's
  literature reconstruction failed (every primary source returned HTTP 403) and
  nothing in this corpus depends on it.
* It did **not** verify Wolfram's rule-numbering convention against a primary
  source. If that convention were misremembered, the entire corpus would
  concern a different rule. The only external anchor is a 14-bit prefix match
  to a secondary source (Phase 1 B6), and it is weak. This is the largest
  single unaudited assumption in the project and is stated as such in
  `EXPERT_REVIEW_REQUEST.md`.
