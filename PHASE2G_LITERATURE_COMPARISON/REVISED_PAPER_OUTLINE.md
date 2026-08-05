# Revised Paper Outline

Replaces the structure of `RULE30_VERIFIED_THEOREMS_V1/PAPER_DRAFT_V1.md` in
the light of the (unverified, secondary) literature comparison.

**Three things change.** The period structure moves from "our results" to
"recalled from Rowland 2006". The preperiod material becomes the paper's
centre. And a new section is added on what the comparison shows we are
*missing*.

**Standing constraint unchanged:** the paper must not claim to solve Problem 1,
and must not claim novelty for anything in §1 below.

---

## Revised title

> **Preperiods in the left-diagonal tower of Rule 30**

The old title ("Nested Prefix Dynamics and Transient Structure…") claimed
"nested prefix dynamics" as a contribution. That is Rowland's territory —
"nested" is his word and his mechanism. The new title names what may actually
be ours: **preperiods**.

## Revised structure

### 1. Introduction
State Problem 1. State immediately that the *period* structure of the left
diagonals is due to **Rowland (2006)** and is recalled, not proved, here. State
that the paper's subject is the **preperiods**, which we could not find treated
anywhere. State that Problem 1 is not solved and no partial claim is made.

### 2. Preliminaries, with attribution
* Rule 30; left-permutivity; the light cone. *Standard.*
* Left-justified / edge-aligned coordinates `w_t(k) = x_t(-t+k)` and the
  one-sided recurrence. **Attribute the presentation to Rowland**; the explicit
  recurrence is at most a convenience.
* **Recalled from Rowland 2006** (with citation, no proof):
  every left diagonal is eventually periodic with period a power of two; and
  Proposition 2 — a doubling occurs exactly when a diagonal becomes eventually
  white and the diagonal to its left has odd parity per block.
* Our T-10/T-14 appear here **only** as a restatement in fibre language, flagged
  as such.

### 3. The prefix tower as a skew product
`F_K`, the projection relation, the one-bit fibre. Present as **bookkeeping**
for §4, not as a result. The trichotomy is the fibre-map form of Rowland's
criterion plus the (never-observed) neutral case.

### 4. Preperiods — **the core of the paper**
* The reset times `σ`, `τ`, `ρ`, and the proof that they are distinct.
* The defect bit `D(K)`; the propagation lemma.
* **Theorem (main).** The exact recurrence: `D(K)=0 ⟹ T(K)=T(K-1)`;
  `D(K)=1 ⟹ T(K)=τ(K)+1`; doubling and neutral levels inherit.
* **Corollary.** `T(K) = ρ(K)+1` at every resetting level.
* **Corollary.** `D(K) = w_{T(K-1)}(K) XOR Φ_K` — the recurrence consumes one
  new transient bit per level and is not a function of the cycle data at any
  depth.
* **Corollary.** The telescoping factorisation of `T(K)/K`.

Each result must carry an explicit sentence: *we could not find this in the
literature, and the paper most likely to contain it (Rowland 2006) was not
available to us.*

### 5. Columns, and what is known
* T-02, T-03 with proofs — but stated **after** a paragraph recording that
  **Jen (1990)** proves aperiodicity of columns for a class of injective rules
  and that **Kopra (2022, Thm 3.5)** generalises it to rapidly left expansive
  CA including rule 30. State plainly that we could not read either, that T-03
  may be known, may be an immediate corollary, or may be a strengthening, and
  that we do not claim it.
* Keep the seed-blindness observation: both ours and (apparently) Jen's hold
  for all finite seeds and therefore cannot decide Problem 1.

### 6. Computational certificates
Unchanged in content. Two additions:
* the 48-term reproduction of Rowland's diagonal-period sequence, as an
  external check on conventions;
* the boundary-slope comparison (ours `0.25368` vs the known `≈ 0.252`), stated
  as **agreement with a known constant**, not as a measurement of ours.

Drop any suggestion that `T(K)/K ≈ 1.34` is a discovery.

### 7. Transient dependency skeleton
Shortened. The XOR spine is three lines and probably folklore; keep the bound,
drop the framing that made it look substantial. Keep the honest negative: the
strongest simplifications available reduce the diagonal's derivation by a
constant factor only.

### 8. Why these results do not resolve aperiodicity
Unchanged, plus one addition: Jen's result, as far as we can tell, stops at the
same place — statements true for every finite seed cannot single out the
single-cell seed. That is evidence the obstruction is structural rather than an
artefact of our approach.

### 9. What we could not do
**New section, and it should be prominent.**
* No primary source was readable from our environment; every attribution in
  this paper is second-hand and flagged.
* Rowland's `2^n` nested-restart mechanism and its conditional
  time-reversibility have **no analogue in our work**; the right-hand side of
  the diagram is opaque to us.
* Kopra's expansivity framework and distribution-mod-1 methods are likewise
  outside our apparatus.

### 10. Open problems
The Bridge, unchanged. Plus the three subsidiary questions (density of
`D(K)=1`; `liminf T(K)/K > 1`, now identified as the known boundary-speed
statement; whether every level past 400 collapses). And a new one: **does
Rowland's `2^n` restart mechanism constrain the preperiods?** — the obvious
question at the junction of his work and ours, which neither of us has asked.

---

## Sections that must be cut or demoted

| in the current draft | action |
|---|---|
| §4 Corollary 4.2 (`P(K) ∈ {P,2P}`) presented as ours | **demote** to a recalled result, cite Rowland |
| §5 Theorems 5.1/5.2 (non-COLLAPSING ⟺ white stripe) as ours | **demote**, cite Rowland Proposition 2 |
| §8 "`T(K)/K = 1.3399`" as an observation of ours | **reframe** as agreement with the known boundary constant |
| §2 the "content of the change of variables" paragraph | **keep**, but note the presentation is Rowland's |
| §9.1 seed-blindness | **keep and strengthen** with the Jen parallel |
| the title's "Nested Prefix Dynamics" | **cut** — "nested" is Rowland's |

## Submission readiness

**Not ready, and should not be submitted.** The paper turns on a novelty claim
for §4, and that claim rests on not having read the one paper most likely to
contain it. Required before any submission:

1. Read Rowland 2006 in full; settle whether preperiods appear.
2. Read Kopra Theorem 3.5; settle the quantifier in the column result.
3. Read Jen 1990's proposition; settle T-03's status.
4. Re-run this outline against what is found.
