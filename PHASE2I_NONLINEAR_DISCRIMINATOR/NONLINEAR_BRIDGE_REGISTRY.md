# Nonlinear bridge registry — NLB1 … NLB5

**Phase 2I, section 9.** Each target gets: an exact quantified statement,
current status, the proof attempt, the counterexample search, the **rule 90
verdict**, the **other-rule-30-seed verdict**, whether it is genuinely weaker
than Problem 1, and the consequence if proved.

**Any candidate equivalent to Problem 1 is rejected**, per the brief. Two are.

Notation: `c_t = x_t(0)`; `n_t(j) = x_t(j)x_t(j+1)`; `T(m,d)` the mod-2
rule-150 Green function; `N(t) = {(s,j) : T(t-1-s,j) = 1, n_s(j) = 1}`;
EP(T,p) is "`c_{t+p} = c_t` for all `t ≥ T`".

---

## NLB1 — a nonzero canonical parity obstruction infinitely often

> **Statement.** For every fixed `p ≥ 1` there exist infinitely many `t` with
> `Q_p(t) = 1`, where `Q_p(t) = SEED_p(t) + OLD_p(t) + SLAB_p(t)` is the
> canonical decomposition of Theorem 3.2.

**Status: REJECTED — equivalent to Problem 1.**

*Proof attempt.* None was made past the first line. `Q_p(t) = c_{t+p} + c_t`
identically (Theorem 3.2 is an identity, not an inequality), so "`Q_p(t) = 1`
for infinitely many `t`, for every `p`" **is** "the centre column is not
eventually periodic", verbatim. Theorem 3.4's self-similar rewriting changes the
expression, not the statement.

*Counterexample search.* Not applicable.
*Rule 90 verdict.* For rule 90 with the single-cell seed the statement is
**false** (its centre column is eventually 0), which is consistent and carries
no information here.
*Other rule-30 seed.* Unknown for every seed — that is the point.
*Weaker than Problem 1?* **No. Equivalent.** Rejected under the brief's rule.
*Consequence if proved.* It would be Problem 1.

---

## NLB2 — periodicity of the event field in two fixed strips

> **Statement.** If EP(T,p) holds then there are two distinct columns `j₁ ≠ j₂`
> such that `t ↦ n_t(j₁)` and `t ↦ n_t(j₂)` are both eventually `p`-periodic.

**Status: PARTIALLY PROVED — one strip free, the second only on a subset.**

*Proof attempt.* Strip `j = -1` is **proved unconditionally**:
`n_t(-1) = c_t(1 + c_{t+1})` (Theorem 1.3) is a function of the centre column
alone, so EP(T,p) makes it eventually `p`-periodic with no extra input
(Proposition 3.6). Strip `j = -2` is **proved on the `p`-periodic row set
`{t ≥ T : c_t = 1 and c_{t+1} = 1}`** (Proposition 3.7), via the backward step
at `j = -1`, but not off it. The residue blocking the full second strip is
exactly `x_t(1)` on the rows where `c_t = 1` — one free bit per such row, which
is the same residue Phase 2H isolated as H1.

*Counterexample search.* No counterexample to the partial statements was found;
none is expected, since they are proved.

*Rule 90 verdict.* **The proved part transfers to rule 90 and is therefore
non-discriminating.** Rule 90's centre column is eventually periodic and its
`n_·(-1)` is eventually periodic too, without contradiction. So NLB2, even if
proved in full, cannot by itself settle Problem 1 — it is refuted in advance as
a bridge by the Phase 2H lesson.

*Other rule-30 seed.* The proved parts hold for every rule-30 seed.
*Weaker than Problem 1?* **Yes**, strictly — it is a consequence of EP, not an
equivalent.
*Consequence if proved in full.* Two fully periodic adjacent-ish strips would
*not* immediately give two adjacent periodic **columns**, so T-02 would not fire
directly; it would need `n_t(j) = 1` to be upgraded to column information. That
upgrade is not available: `n_t(j) = 0` is consistent with `x_t(j)` being either
value. **NLB2 is weaker than it looks.**

---

## NLB3 — a powers-of-two event pattern incompatible with any period

> **Statement.** The single-cell seed forces a mod-2 nonlinear-event pattern at
> times `2^a` that is incompatible with `c_{t+p} = c_t` for every fixed `p`.

**Status: NOT PROVED. The route is blocked, and the block is identified.**

*Proof attempt.* The `2^a` structure available is entirely in the **kernel**,
not the event field: Theorem 4.1 (`T(2^a,·)` is a three-slit), Theorem 4.3
(three disjoint copies), Theorem 4.4 (exact scaled self-similarity), Theorem 8.6
(the same for `Ker`). All are proved. **None of them touches `n`.** The active
set `N(t) = Ker(t) ∩ {n = 1}` is the intersection of a self-similar arithmetic
set with an orbit-dependent one, and section 8 found no recurrence for the
intersection. The parity version of the statement is `(★)`, hence equivalent to
Problem 1 and rejected.

*Counterexample search.* `|N(t)|` at `t = 2^a` was checked against every
scaled-copy splitting of Theorem 8.6; no relation of that form holds.

*Rule 90 verdict.* The kernel identities are properties of the rule-150 Green
function and are **identical** for rules 30, 90 and 150. So every proved
ingredient of NLB3 is shared with a rule whose centre column is eventually
periodic. **F2 fails for the whole ingredient list.**

*Other rule-30 seed.* The kernel is seed-independent, so **F3 fails too.**
*Weaker than Problem 1?* The kernel statements are much weaker (and proved);
the full NLB3 as stated is not obviously weaker.
*Consequence if proved.* It would settle Problem 1 for the single-cell seed.
Since its only proved ingredients fail both filters, the phase does **not**
recommend it.

---

## NLB4 — a seed-specific coupled-tower invariant  ★ the promoted target

> **Statement.** Let `Π3` be: *the seed is mirror-symmetric about a cell, and
> the orbit is not* — i.e. `x_0(c+d) = x_0(c-d)` for all `d`, yet
> `δ_t(r) := x_t(c-r) + x_t(c+r) ≢ 0`. Then `Π3` together with EP(T,p) is
> contradictory.

**Status: `Π3` PROVED for rule 30 with the single-cell seed and passing filters
F1–F3. The implication to a contradiction is OPEN, with a proved obstruction.**

*Proof attempt.* The exact machinery is Theorem 6.4:
`δ_{t+1} = L_{150} δ_t + ν_t` with `ν_t(j) = n_t(j) + n_t(-j)` and `δ_0 = 0`,
giving `δ_t(j) = Σ_{s<t} Σ_d T(t-1-s,d) ν_s(j-d)`. **The attempt fails at a
provable point:** `δ_t(0) = 0` identically, both trivially and through the
Duhamel form (each event is counted twice because `T(m,·)` is symmetric). The
mirror invariant is *structurally blind to the centre column*. Coupling it to
the centre via the exact relations of §5 of `NONLINEAR_ACTIVITY_THEORY.md`
yields `δ_t(1) = c_{t+1}` on `{c_t = 0}` — which is **unconditional**, using no
periodicity — and on `{c_t = 1}` leaves the familiar one-bit residue. No new
leverage was obtained.

*Counterexample search.* 247 mirror-symmetric seeds were run under rule 30;
**0** produced a symmetric orbit, so no counterexample to `Π3`'s first half was
found in the searched range (reported as "no witness found", not as UNSAT).

*Rule 90 verdict.* **`Π3` fails for rule 90 — provably.** Rules 90 and 150 are
symmetric local rules, commute with the mirror, and give `δ ≡ 0` forever from a
symmetric seed. Verified: 0 nonzero cells, against 864 for rule 30. **F2 ✔.**

*Other rule-30 seed.* **`Π3` fails for `D_two_adjacent`, `E_1001`,
`F_rand01/03/04/06`** — those seeds are not mirror-symmetric about any cell.
**F3 ✔.**

*Weaker than Problem 1?* **Yes, strictly.** `Π3` is decided by a finite
computation (exhibit one asymmetric cell); Problem 1 is not.

*Consequence if proved.* A genuine seed-specific nonlinear bridge, and the
first result in this corpus immune to the Phase 2H rule-90 objection. **But the
obstruction above must be routed around first**, and this phase did not find a
route.

---

## NLB5 — left-erased information recoverable only through a nonperiodic word

> **Statement.** The information erased by the left tower `F_K` is recoverable
> from the right tower `G_K` only through a reduced boundary word that is not
> eventually periodic.

**Status: NOT PROVED; and its second clause is equivalent to Problem 1.**

*Proof attempt.* The first clause is **proved** and is Theorem 7.3 plus Phase
2H's Theorem 2.C′: `G_K` is a bijection, so the right tower loses nothing and
in principle carries what `F_K` erased. The second clause — "only through a
**nonperiodic** word" — is the obstruction. By Phase 2H H1 and
`NONLINEAR_ACTIVITY_THEORY.md` §5, the reduced boundary word is `x_t(1)` on
`{c_t = 1}`, and "that word is not eventually `p`-periodic" is exactly what
EP(T,p) would have to contradict. Asserting it *is* asserting Problem 1.

*Counterexample search.* Not applicable to a rejected clause.
*Rule 90 verdict.* Rule 90's `G_K` and `F_K` are **both** bijections, so
nothing is erased and the first clause is vacuous rather than false. The
statement does not discriminate.
*Other rule-30 seed.* First clause holds for all seeds (rule-level).
*Weaker than Problem 1?* **No** for the second clause. **REJECTED** as stated.
*Consequence if proved.* None beyond Problem 1 itself.

---

## Summary

| target | status | F2 (rule 90) | F3 (other seed) | weaker than Problem 1 | verdict |
|---|---|---|---|---|---|
| **NLB1** | rejected | — | — | **no — equivalent** | rejected |
| **NLB2** | one strip proved, second partial | ✘ transfers to rule 90 | ✘ all seeds | yes | proved but non-discriminating |
| **NLB3** | not proved | ✘ kernel is shared | ✘ kernel is seed-free | partly | not recommended |
| **NLB4** | `Π3` proved; implication **OPEN** | ✔ | ✔ | yes | **promoted target** |
| **NLB5** | first clause proved, second rejected | ✘ vacuous for rule 90 | ✘ rule-level | no (2nd clause) | rejected as stated |

**Exactly one target survives with both filters intact: NLB4.** Its `Π3` is
proved; its bridge implication is open and carries a proved obstruction
(centre-blindness of the mirror defect). Two of the five targets — NLB1 and
NLB5's operative clause — are restatements of Problem 1 and are rejected on
that ground, not because they are hard.
