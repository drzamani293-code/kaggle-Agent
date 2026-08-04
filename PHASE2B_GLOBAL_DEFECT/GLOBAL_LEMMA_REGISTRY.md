# Global Lemma Registry — Phase 2B

Proof targets for the global (zero-wall) attack on Rule 30 Prize Problem 1.
Labels follow the Phase 1 convention: **THEOREM** (proved here), **CONJECTURE**
(precise, unproved), **COMPUTATIONAL OBSERVATION** (measured, bounded),
**STRATEGY ASSESSMENT** (a judgement about a route, not a mathematical claim).

Standing hypothesis throughout:

> **(H-WALL)** the centre column is eventually `p`-periodic after time `T`,
> equivalently `d_t(0) = 0` for all `t >= T`, where
> `d_t(j) = x_{t+p}(j) XOR x_t(j)`.

(H-WALL) is conjecturally false, so every conditional statement below is
**vacuous if Problem 1 has the expected answer** and cannot be tested
computationally. That is a property of the logical form, not a defect of the
experiments — the same point as Phase 1 registry item C3.

---

## G1 — an infinite zero wall forces a second periodic column

**Formal statement.** Assume (H-WALL). Then there exists `j ≠ 0` and integers
`T'`, `q >= 1` with `x_{t+q}(j) = x_t(j)` for all `t >= T'`.

**Why it matters.** G1 + Phase 1 Theorem W2′ ("at most one column of the
diagram is eventually periodic") immediately gives Problem 1. G1 is the
defect-language form of Phase 1's Bridge (registry C2).

**Current evidence.** None direct — G1 is conditional on a hypothesis believed
false. What Phase 2B adds is *structure* under (H-WALL): by **W2** the defect
field on `j >= 1` closes on itself and by **W5** the entire left half-plane is
determined by `(d_t(1))` together with the centre column. So under (H-WALL) the
whole defect field is a function of one column's defect sequence. G1 asks that
this sequence be eventually periodic.

**Known counterexamples in relaxed models.** Yes, and they matter. The
zero-wall automaton (`ZERO_WALL_AUTOMATON.md` §2) has a non-empty maximal
invariant set at every radius `r <= 5` — bi-infinite walls exist in the
seed-free local relaxation, and nothing forces a second column to be periodic
there. So **G1 is false without the seed**, and any proof must use the
single-cell initial condition.

**Weakest useful version.** `d_t(1)` is eventually `p`-periodic under (H-WALL)
— i.e. exactly Phase 1's MB1 restated. By W1, that already forces `d_t(-1)`
eventually periodic, giving column −1 and column 0 both periodic and
contradicting Phase 1 Theorem W2.

**Falsification test.** None exists (vacuity). The finitary shadow — MB1-loc —
was refuted for `a <= 28` in Phase 2A; that refutes a *proof strategy*, not G1.

**Dependencies.** Phase 1 W2, W2′ (proved). Literature: **unresolved** —
`LITERATURE_GAP_MAP.md` could not retrieve any source.

**Status: OPEN. Highest priority. This is the Phase 1 Bridge in new clothes,
and Phase 2B did not move it.**

---

## G2 — the support-edge defects obstruct the wall

**Formal statement (as originally proposed).** The deterministic defects at the
support edges, `d_t(±(t+p)) = 1` for all `t` (**THEOREM**, D7), are incompatible
with (H-WALL).

**Verdict: STRATEGY ASSESSMENT — REFUTED AS A ROUTE. Recorded, not deleted.**

**Why.** The edge defects sit at `±(t+p)` and move **outward** at speed 1, away
from column 0. A wall constrains a region the edges are receding from. Measured
support: `DEFECT_GEOMETRY_RESULTS.md` §1 confirms the edges exactly, and §2
finds defect density `≈ 0.500` and column-0 crossing rate `≈ 0.5` at every one
of 83 lags, with no interaction visible between the edge structure and the
centre.

**What survives of the idea.** D7 does establish that the defect field is
**never empty** for any lag at any time — a fact with no analogue in the local
relaxation, where `d ≡ 0` is a legitimate fixed point. So D7 *does* distinguish
the real orbit from the relaxation. It simply does not do so **at column 0**.
A repaired version would have to connect the edge to the centre; §G3 is the
natural attempt and is also open.

**Falsification test.** Already performed: the geometry sweep. G2 predicted some
interaction between edge defects and the wall region; none was found.

---

## G3 — every defect component eventually meets every width-2 strip

**Formal statement.** For every `p >= 1` and every `j0 ∈ Z`, and for every
non-zero connected component `C` of the space-time defect set, there exists a
time at which `C` intersects `{j0, j0+1}`.

**Current evidence.** COMPUTATIONAL OBSERVATION only: across 83 lags the
fraction of rows in which the defect component covering column 0 exists is
`0.425`–`0.561`, centred on `0.5`
(`DEFECT_GEOMETRY_RESULTS.md` §3). Defects reach the centre constantly in the
real orbit — but that is a statement about the real orbit, in which there is no
wall.

**Known counterexamples in relaxed models.** Yes. In the seed-free relaxation,
`d ≡ 0` is invariant and a defect component confined to a half-line is easy to
construct (the automaton's invariant set contains states with defects only on
one side). So **G3 is false without the seed.**

**Weakest useful version.** Under (H-WALL), some defect component crosses
column 0 at arbitrarily large times — which directly contradicts (H-WALL) if
"crosses" is read as `d_t(0) = 1`. Stated that way the weakest version is
*equivalent* to the negation of (H-WALL), i.e. to Problem 1, so it is not
weaker at all. This circularity is the trap in G3 and is the reason it is
ranked below G1.

**Falsification test.** For the real orbit: measure, for each lag, the longest
interval on which the component covering some fixed strip is absent — this is
exactly the wall-run statistic (longest 18 over 83 lags × 30000 steps). Cannot
falsify the conditional form.

**Status: OPEN, but likely circular as stated. Needs reformulation before it is
worth effort.**

---

## G4 — a wall forces a closed finite-state cycle incompatible with the seed

**Formal statement.** Under (H-WALL) the local wall data around column 0
eventually enters a cycle of the radius-`r` wall automaton; and no such cycle
is compatible with the single-cell initial condition.

**Verdict on the first half: FALSE AS STATED.** The wall automaton is
nondeterministic (four bits are guessed each step,
`ZERO_WALL_AUTOMATON.md` §1), so an infinite wall gives an infinite *path*, not
an eventual cycle of a deterministic system. A path in a finite graph need not
be eventually periodic. The first half must be weakened to "eventually stays
inside the maximal invariant set", which is true by construction and carries no
information.

**Verdict on the second half: NOT ESTABLISHED, and the route is structurally
blocked.** The maximal invariant set is non-empty for every radius `r <= 5`
(sizes 16, 128, 992, 7616, 59136), so no bounded impossibility was obtained;
and the growth `|Inv(r)| ≈ C · 7.7^r` (CONJECTURE) suggests it will remain
non-empty. More decisively, a fixed-radius automaton around column 0 **cannot
see the seed**, and Phase 1 §10.3 proves seed-blind arguments cannot settle
Problem 1. The automaton is seed-blind by construction.

**What survives.** A quantitative measurement: the wall condition removes a
constant fraction (`≈ 0.52`) of the local state space per unit radius. And the
comparison in `ZERO_WALL_AUTOMATON.md` §3 — at `r = 4`, only 1 763 of 7 616
invariant states were observed in the real orbit — is a genuine gap, though
"not observed" is not "unreachable".

**Falsification test.** Compute the invariant set at larger `r` with a symbolic
method (BDD or bounded model checking). Emptiness at any `r` would be a
theorem; the extrapolation says not to expect it.

**Status: PARTIALLY REFUTED (first half false as stated, second half blocked).
Retained as a failed route.**

---

## G5 — the transient outruns the diagonal *(new in Phase 2B)*

**Formal statement.** In edge-aligned coordinates `w_t(k) = x_t(-t+k)`, let
`T(K)` be the preperiod of the autonomous prefix dynamics on
`(w_t(0..K))` (well defined by **Theorem EA2**). Then

```
    liminf_{K → ∞}  T(K) / K  >  1.
```

**Why it matters — this is the sharpest new statement Phase 2B produces.**
By **EA3** the centre column is the diagonal, `x_t(0) = w_t(t)`. The prefix of
length `K` becomes periodic only from time `T(K)`. If `T(K)/K > 1` then at
every time `t` the diagonal entry `w_t(t)` lies strictly inside the *transient*
of the prefix dynamics — the centre column never reads the periodic regime.
Conversely, **if `T(K) <= K` for all large `K`, the centre column would be
reading an eventually periodic structure**, which is the shape of a positive
answer to the periodicity question. G5 is therefore not a side remark: it is
close to a restatement of the problem in coordinates where one half of it
(EA2, eventual periodicity of every fixed prefix) is a *theorem*.

**Current evidence.** COMPUTATIONAL OBSERVATION, `K <= 1280`, `t <= 2600`:
`T(K)/K` measured at `1.422, 1.336, 1.246, 1.268, 1.225, 1.277, 1.273, 1.269,
1.302` for `K = 64 … 1280`; mean `1.291` for `K >= 64`. Independently
cross-checked against Phase 1's measurement in the *original* coordinates
(band width `≈ 0.74 t`, i.e. `K/T ≈ 0.77`), agreeing to within 0.5 %
(`EDGE_ALIGNED_DYNAMICS.md` §5.1).

**Known counterexamples in relaxed models.** None — G5 is a statement about the
single-cell orbit specifically, and it has no seed-free analogue. **This is
what distinguishes it from G1–G4**, all of which are false or vacuous without
the seed. G5 is *intrinsically* seed-dependent, which by Phase 1 §10.3 is a
necessary feature of any route that can work.

**Weakest useful version.** `T(K) > K` for all sufficiently large `K` (rather
than a `liminf` bound with a margin).

**Falsification test — genuine and cheap.** Extend the measurement of `T(K)`
to `K = 10^4`–`10^5` (cost `O(t_max · K)`, well within reach). **A single `K`
with `T(K) <= K` would refute G5** and would be a major structural fact. This
is the only registry item with a real, non-vacuous, finitely checkable
falsification test.

**Dependencies.** EA1, EA2, EA3 (**THEOREMS**, proved and verified here). No
literature dependency.

**Status: OPEN. New. Recommended as the highest-value target after G1, and the
only one with a live falsification test.**

**Caveat.** Proving G5 is not obviously easier than Problem 1, and no route
from G5 to a contradiction with (H-WALL) has been constructed. G5 is a
*reformulation with one half proved*, not a reduction. Do not present it as
progress on the problem itself.

---

## Priority ordering

| Rank | Item | Why | Status |
|---|---|---|---|
| 1 | **G1** | equivalent to the Phase 1 Bridge; would close the problem with W2′ | OPEN, untouched by Phase 2B |
| 2 | **G5** | new; seed-dependent (necessary); half of it is a theorem; has a live falsification test | OPEN |
| 3 | G3 | needs reformulation — currently circular | OPEN, low value as stated |
| 4 | G2 | edges recede from the centre | **refuted as a route** |
| 5 | G4 | first half false as stated; seed-blind by construction | **partially refuted** |

## Cross-cutting facts that any future attempt must respect

1. **Seed-blindness is fatal** (Phase 1 §10.3). G2, G3, G4 and the whole
   automaton apparatus are seed-blind; G5 is not.
2. **Conditionals on (H-WALL) cannot be tested** — only their finitary shadows,
   and Phase 2A showed those can be refuted without touching the conditional.
3. **The defect field is never empty** (D7) — the one exact global fact Phase 2B
   established about the real orbit that has no seed-free analogue. It is
   currently unconnected to column 0; connecting it is what a repaired G2 would
   have to do.
