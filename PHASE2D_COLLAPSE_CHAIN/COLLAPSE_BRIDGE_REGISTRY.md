# Collapse Bridge Registry

Candidate statements connecting the collapse structure to the diagonal.
Labels as in earlier phases. **(H-PER)** denotes the hypothesis that the centre
column is eventually periodic — conjecturally false, hence every conditional on
it is vacuous if Problem 1 has the expected answer.

---

## C1 — arbitrarily long collapse chains force a bounded-memory description of diagonal bits

**Exact statement.** There are constants `M, K0` such that for every `t >= K0`
the bit `w_t(t)` is computable from at most `M` bits of state carried along the
chain containing level `t`.

**Evidence.** Partial and negative.
* **For:** within a COLLAPSING chain the *cycle-determined* region of the tower
  is a function of exactly `2P = 32` bits (**THEOREM CH2/DD3**), independent of
  how many cells it contains. The longest measured chain is 2600 levels.
* **Against:** the diagonal is not in that region. `DIAGONAL_DEPENDENCY.md` §3
  measures the backward cone of `(t,t)` as **entirely transient for the last
  ≈19% of its history** (`s >~ 0.81 t`), and the transient part is not covered
  by CH2.

**Known obstacles.** The transient carries the seed's information, and Phase 2C
showed the transient outruns the diagonal (`T(K) ≈ 1.34K > K`). A bounded-memory
description would contradict that unless the transient portion is itself
compressible, for which there is no evidence.

**Relaxed-model counterexamples.** In the seed-free tower, cycle words can be
chosen arbitrarily at the bottom of a chain, so `2P` bits genuinely parametrise
the cycle-determined region — CH2 is tight, and gives nothing beyond it.

**Would combine with.** Nothing yet; if `M` existed, the diagonal would be an
automatic sequence, which would settle Problem 1 either way.

**Status: NOT PROVED. Evidence currently points against.**

---

## C2 — eventual absence of DOUBLING and NEUTRAL implies eventual periodicity or contradiction for the diagonal

**Exact statement.** If there is `K0` with every level `K >= K0` COLLAPSING,
then either the diagonal `(w_t(t))_t` is eventually periodic, or a
contradiction follows.

**Evidence.** The hypothesis is *measured to hold* on `[401, 3000]` and, by
Phase 2C, `P(K) = 16` throughout `[400, 30000]` — so no doubling point occurs
in that whole range, i.e. **every level in `[401, 30000]` is COLLAPSING**. The
hypothesis of C2 is therefore satisfied on the entire computed range.

**And yet no conclusion follows.** This is the sharpest negative result of
Phase 2D: 29,600 consecutive COLLAPSING levels produce a `32`-bit transducer for
the cycle words (CH2) and **no constraint whatsoever on the diagonal**, because
the diagonal never enters the cycle region (Phase 2C `T(K) > K`;
`DIAGONAL_DEPENDENCY.md` §3).

**Known obstacles.** As stated, C2's conclusion does not follow from its
hypothesis by any argument we could construct; the disjunction "or a
contradiction follows" is doing all the work and is unsubstantiated.

**Relaxed-model counterexamples.** In the seed-free tower one can have all
levels COLLAPSING with an arbitrary transient, so the hypothesis alone certainly
does not force diagonal periodicity.

**Status: the hypothesis is essentially verified on the computed range and the
conclusion still does not follow. C2 as stated is not a viable target.**

---

## C3 — every sufficiently long collapse chain transfers periodicity from one fixed column to another

**Exact statement.** There is `L` such that if levels `[A, A+L]` are all
COLLAPSING and column `j` (in the original coordinates) is eventually periodic,
then some column `j' != j` is eventually periodic.

**Why it would matter.** C3 + Phase 1 **Theorem W2′** (at most one eventually
periodic column) would give Problem 1 immediately. C3 is the Phase 1 "Bridge"
(registry C2 there) restated in collapse language.

**Evidence.** None direct. The collapse structure transfers information *within
the cycle region* of the edge-aligned tower; a fixed original-coordinate column
`j` corresponds to the moving diagonal `k = t + j`, which is exactly what the
tower does not reach.

**Known obstacles.** Same as C1/C2: cycle-region statements do not touch moving
diagonals.

**Relaxed-model counterexamples.** Not applicable — C3's hypothesis is about the
real orbit; but note that the seed-free tower admits arbitrarily long
COLLAPSING chains with no periodic column at all, so C3 is false without the
seed.

**Status: OPEN, and unmoved by Phase 2D.** It remains the same Bridge that
Phases 1, 2B and 2C all failed to cross.

---

## C4 — the real orbit cannot sustain a permanently bounded prefix period together with an eventually periodic diagonal

**Exact statement.** It is not the case that both (i) `P(K)` is bounded (only
finitely many doubling points) and (ii) the centre column is eventually
periodic.

**Evidence, and the honest reading.**
* (i) is *consistent with* everything measured: `P(K) = 16` for all
  `400 <= K <= 30000`, and whether a fifth doubling point exists is unknown
  (`DOUBLING_POINT_ANALYSIS.md` §3).
* Under (i), Theorem **CH3** applies in principle: a chain longer than `2^{2P}`
  forces the cycle words to be periodic in `K`. At `P = 16` that needs
  `4.3 x 10^9` levels — far beyond anything computed, but a genuine consequence
  if `P` is truly bounded and the chain is infinite.
* Under (i) **and** an infinite COLLAPSING chain, the periodic region of the
  tower would be doubly periodic (period `P` in `t`, period `<= 2^{2P}` in `K`).
  That is a very rigid structure — but the diagonal still lies outside it.

**Known obstacles.** Even full double periodicity of the cycle region does not
constrain the transient, and Phase 2C's `T(K) > K` keeps the diagonal in the
transient at every measured `K`.

**Relaxed-model counterexamples.** A seed-free tower can have `P` bounded, all
levels COLLAPSING, doubly periodic cycle region, and an arbitrary transient — so
no contradiction arises from (i) alone.

**Would combine with.** Phase 1 W2′, if one could show the doubly periodic
region forces a second periodic column.

**Status: OPEN. The most structurally promising of the four, because it is the
only one whose hypothesis (bounded `P`) is a concrete, falsifiable property of
the real orbit** — a fifth doubling point would refute (i) and remove the
branch entirely. **Falsification test:** extend the exact `P(K)` table past
`K = 30000` looking for `P = 32`. Cheap; not done here (the brief asks not to
extend ranges merely for larger tables).

---

## Cross-cutting conclusion of Phase 2D

The collapse structure is now completely understood at the level of the
**cycle**: exact reset formulas (R1–R3), a `2P`-bit transducer (CH1–CH2), a
pigeonhole theorem (CH3), and a full account of why NEUTRAL is scarce (N1, N2).

**None of it reaches the diagonal.** Every bridge candidate fails at the same
place: the collapse machinery describes the periodic region, and the diagonal
lives strictly in the transient (`T(K) ≈ 1.34K > K`, and the backward cone is
entirely transient for the last ≈19% of its history).

That is the honest summary: Phase 2D solved the structure it set out to solve
and thereby **sharpened, rather than crossed, the same gap** identified in
Phase 1 §10.3 — an argument that never touches the seed-carrying transient
cannot settle Problem 1.
