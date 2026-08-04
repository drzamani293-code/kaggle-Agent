# Phase 2A — Method

Scope: **build and validate SAT/SMT formulations of the finite statement
MB1-loc(a).** No final proof is attempted, and none is claimed. Phase 2A ends
with the package; Phase 2B is not started.

Formal definitions and all soundness proofs are in
`SAT_MODEL_SPECIFICATION.md`. Verdicts are in `PHASE2A_RESULTS.md`.

---

## 1. The one methodological fact that shapes everything

MB1-loc(`a`) quantifies over `t` and `p` in the **actual** rule-30 orbit of the
single-cell seed. That orbit is *deterministic*: once `t` and `p` are fixed,
every cell in the relevant region is forced by the seed. So:

> **MODEL B — the faithful encoding of the real orbit — has zero free
> variables.** A SAT solver applied to it decides the instance by unit
> propagation alone. It is a *certificate checker*, not a search engine.

Searching the real orbit for witnesses is not a satisfiability problem at all;
it is enumeration, which is what Phase 1 already did. A SAT solver can only
become a *search* engine here by relaxing something — and the natural
relaxation is to drop the seed, giving MODEL A, whose free base row is the
only source of genuine search.

This is not a limitation of the encoding but of the problem, and it determines
what each model can and cannot deliver:

| | free variables | genuine search? | what a SAT answer means | what an UNSAT answer means |
|---|---|---|---|---|
| **MODEL A** | base row | **yes** | a patch of the *seed-free relaxation*; **not** a counterexample | **THEOREM**: MB1-loc holds for that `(a,p)` and *every* `t` |
| **MODEL B** | none | no | **CERTIFICATE**: this triple is a real counterexample | this triple is not a counterexample (complete for the triple) |
| **MODEL B′** | none | no | same as B, relative to two verified orbit rows | same as B, relative to those rows |

Anyone reading a Phase 2A result must first ask which model produced it. Every
certificate file states its model on line 1.

## 2. Model A: the only real search, and what it is a search for

Model A asks: *does there exist any rule-30 space-time patch — with a
completely free base row — satisfying H1, H2 and ¬C?*

Two facts make this a complete search rather than a bounded one:

* **Lemma C (containment)** — the encoded region is the backward light cone of
  the two queried cells, so every encoded cell is determined by the base row
  and no unconstrained boundary can influence any queried cell. The model is
  exactly "some rule-30 orbit segment", with nothing left dangling.
* **Lemma T (translation invariance)** — Model A's constraints never mention
  absolute time, so satisfiability depends only on `(a, p)`. **One solve per
  `(a, p)` settles every `t` at once.** The search over `t` is therefore
  *complete*, not bounded; only `p` is bounded.

Both lemmas are proved in `SAT_MODEL_SPECIFICATION.md` §2–§3.

The payoff is asymmetric and worth stating plainly: Model A **UNSAT** would be
a theorem about the real orbit (Theorem A-sound), whereas Model A **SAT** says
nothing about the real orbit (Theorem A-incomplete). We ran the sweep hoping
for UNSAT somewhere.

## 3. Model B and the cost wall

Model B is exact (Theorem B-exact) but its cone has `Θ((t+p)²)` cells:

| witness | `t` | `p` | cells in full Model B |
|---|---|---|---|
| a=0 | 17 | 1 | 380 |
| a=8 | 3189 | 2 | 1.0 × 10⁷ |
| a=4 | 15363 | 1 | 2.4 × 10⁸ |
| a=24 | 82630 | 1324 | 7.0 × 10⁹ |
| a=12 | 94693 | 2 | 9.0 × 10⁹ |
| a=20 | 322343 | 27 | 1.0 × 10¹¹ |
| a=28 | 529506 | 2750 | 2.8 × 10¹¹ |
| a=16 | 850603 | 6 | 7.2 × 10¹¹ |

Only the `a=0` witness is tractable in the full form. It is run, and reported,
in that form — including the check that the recovered base row is literally the
single-cell seed.

## 4. Model B′: two windows instead of one tall cone

The constrained cells occupy two time windows of height `a`, at `[t-a, t]` and
`[t+p-a, t+p]`. Joining them by a single cone is what costs `(a+p)²`. Pinning
each window's base row to the true orbit row makes the two windows separately
determined, so the cost drops to `Θ(a²)`:

| witness | single-cone cells | two-window cells |
|---|---|---|
| a=24, p=1324 | 1 821 150 | 1 300 |
| a=28, p=2750 | 7 725 620 | 1 740 |

**The assumption this introduces, stated plainly:** B′ is sound *relative to
its base rows*. Those come from the Phase 1 big-integer engine, which is
cross-checked by four independent implementations, by a mirror-rule (rule 86)
regeneration with reversed shifts, and by SHA-256 against the stored sequences.
Model B assumes nothing beyond the seed; B′ assumes the simulator. Where both
are affordable, both are run and compared.

## 5. Independence: four decision procedures, three encodings

Every reported verdict is produced more than once, by machinery that shares as
little as possible:

**Encodings** (`SAT_MODEL_SPECIFICATION.md` §6)
* ENC1 "direct" — 8 width-4 clauses per transition, no auxiliary variables.
* ENC2 "Tseitin" — auxiliary `u ↔ (c ∨ r)`, then `y ↔ l ⊕ u`; 7 clauses + 1 aux.
* Z3-native — Z3 terms `x == Xor(l, Or(c, r))`; Z3 clausifies internally, so no
  CNF code is shared with ENC1/ENC2.

**Solvers** — CaDiCaL 1.5.3, Glucose 4.2, MiniSat 2.2 (via PySAT), and Z3.

**Fourth, solver-free procedure** — exhaustive enumeration of all `2^w` base
rows (`w = 2(a+p)+2`), each evolved with the Phase 1 rule-30 table. Used
wherever `w ≤ 18`. This decides the instance with no SAT solver in the loop at
all.

A verdict is reported only if all configurations agree; disagreements are
recorded in `phase2a_results.json` under `disagreements` and would be treated
as a defect, not averaged away.

## 6. Search bounds, and why they are where they are

| Sweep | Bounds | Why |
|---|---|---|
| Model A | `a ∈ {0,4,…,48}`, `p ∈ [1, 64]`, **all `t`** | `t` is free by Lemma T; `p` bounded by solver time (cone is `Θ((a+p)²)`) |
| Brute force | `2(a+p)+2 ≤ 18` bits | `2¹⁸` base rows × ~90 cells each is seconds; `2²⁰` is minutes per instance |
| Model B sweep | `a ∈ {0,4,8}`, `t ≤ 48`, `p ≤ 24` | cone is `Θ((t+p)²)`; this is an exhaustive real-orbit search in the small-`t` corner |
| Model B′ | the eight Phase 1 witnesses | `Θ(a²)` per witness; cost is dominated by simulating the base rows (`Θ(t²)`) |
| Eligibility census | `N = 200001` bits, `p ≤ 20000` | to *quantify* why the real-orbit search finds nothing at large `a` |

## 7. Witness verification protocol

For every SAT answer that is claimed to be a real counterexample:

1. Extract the base row from the solver model.
2. **Re-evolve it independently** with `rule30_lab`'s table-driven step —
   not with the SAT encoding.
3. Re-check H1, H2 and ¬C on the re-simulated grid.
4. **Independently simulate the single-cell orbit from time 0** to `t+p` with
   the Phase 1 engine and re-check H1, H2 and ¬C there
   (`real_orbit_witness_check`).
5. Save the exact space-time patch, the triple `(a,t,p)`, and a human-readable
   certificate with the patch drawn and every check printed.

Steps 4 and 5 are what separate a genuine counterexample from a Model A patch.
Model A certificates carry an explicit banner saying they are **not**
counterexamples.

## 8. UNSAT handling protocol

1. **Distinguish the two kinds of UNSAT.** A Model B UNSAT at a specific
   `(a,t,p)` is *complete for that triple* (Theorem B-exact) — it says that
   triple is not a counterexample, full stop. A negative result from a *sweep*
   is **BOUNDED UNSAT**: it is reported together with its bounds and never as
   a universal statement. A Model A UNSAT would be a **THEOREM** covering all
   `t`.
2. **Unsat cores.** Every hypothesis constraint carries a selector variable
   passed as an assumption, so cores name hypotheses rather than rule clauses.
   Cores are extracted from PySAT (`get_core`) and Z3 (`unsat_core`) and saved.
3. **Proof export.** `Cadical153(with_proof=True)` is used to request DRUP.
   *Reported limitation:* for these instances the emitted proof is **empty** —
   they are refuted by unit propagation during preprocessing, so nothing is
   logged — and **no DRAT checker (`drat-trim`) is installed in this
   environment**, so no proof is independently checked. Instead, UNSAT is
   corroborated by (i) three SAT solvers, (ii) two CNF encodings, (iii) Z3,
   (iv) exhaustive enumeration, and (v) a hand proof.
4. **Structural reason.** For every UNSAT instance found, the core is inspected
   and compared across `p` to see whether the same constraint set recurs.

## 9. Threats to validity

* **The models could be vacuously easy.** If the encoding were wrong in a way
  that made every instance satisfiable, the uniform SAT result of the Model A
  sweep would be meaningless. This is exactly why the Proposition 6 control
  instance is included: it is a *predicted UNSAT*, derived from a hand proof,
  in the same encoding. Its UNSAT verdict shows the encoding does detect
  forcing.
* **A shared misunderstanding of rule 30** would defeat all four procedures at
  once. Mitigated only by the Phase 1 chain: exhaustive check against the rule
  *number*, hand-computed rows, mirror-rule check, and the external OEIS prefix
  (itself weak — see Phase 1 `CLAIM_LEDGER.md` B6).
* **Model B′ inherits the simulator's correctness.** Stated in §4; Model B is
  run wherever affordable precisely to have one result free of that assumption.
* **`p` is bounded in the Model A sweep.** Nothing is claimed for `p > 64`.
* **Solver bugs.** Three solvers, two encodings, one SMT solver and one
  solver-free enumeration would all have to be wrong together.

## 10. What Phase 2A does not do

* It does not attempt to prove or disprove MB1-loc(`a`) for any `a`.
* It does not extend the real-orbit witness search beyond Phase 1's; §8 of the
  results explains, with counts, why that search is data-limited rather than
  method-limited.
* It does not claim MB1-loc(`a`) holds for any `a` on the basis of a failed
  search. The standing prohibition in `SAT_MODEL_SPECIFICATION.md` §9 applies
  to every line of `PHASE2A_RESULTS.md`.
