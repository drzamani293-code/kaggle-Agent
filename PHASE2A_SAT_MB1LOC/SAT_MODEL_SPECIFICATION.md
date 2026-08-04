# SAT / SMT Model Specification — MB1-loc(a)

Formal specification of the three encodings. Every claim about what a solver
verdict *means* is stated and proved here; `PHASE2A_RESULTS.md` reports only
verdicts, and refers back to this document for their interpretation.

---

## 0. Notation

* `Σ = {0,1}`, `⊕` = XOR, `∨` = OR.
* `f(l,c,r) = l ⊕ (c ∨ r)` — rule 30 (verified exhaustively against the rule
  *number* 30 in `rule30_lab.verify_local_rule`).
* `a_τ(j)` — the real rule-30 orbit of the single-cell seed:
  `a_0(0) = 1`, `a_0(j) = 0` for `j ≠ 0`, and
  `a_{τ+1}(j) = f(a_τ(j-1), a_τ(j), a_τ(j+1))`.
* `col_i(τ) = a_τ(i)`.
* Time runs upward; site index increases to the right.

## 1. The statement under test

> **MB1-loc(`a`)** — for all `t > a` and all `p ≥ 1`:
> if `col_0(s) = col_0(s+p)` for every `s ∈ [t-a, t]`   **(H1)**
> and `col_0(t) = 0`                                     **(H2)**
> then `col_1(t) = col_1(t+p)`.                          **(C)**

A **counterexample** is a triple `(a, t, p)` with H1 ∧ H2 ∧ ¬C **in the real
orbit**. MB1-loc(`a`) is universally quantified over `t` and `p`, so one
counterexample refutes it, and no finite search can confirm it.

The cells named by the constraints are:

| Constraint | cells |
|---|---|
| H1 (`a+1` equalities) | `(t-a+k, 0)` and `(t-a+k+p, 0)` for `k = 0..a` |
| H2 | `(t, 0)` |
| ¬C | `(t, 1)` and `(t+p, 1)` |

## 2. Patch geometry, and why no free boundary can reach the queried cells

**Definition (cone).** For `t_base ≤ t_top` and a top site interval
`[j_lo, j_hi]`, the *cone* is the set of cells

```
    P = { (τ, j) : t_base ≤ τ ≤ t_top ,  j_lo - (t_top - τ) ≤ j ≤ j_hi + (t_top - τ) }.
```

**Lemma C (containment).** For every `(τ, j) ∈ P` with `τ > t_base`, the three
cells `(τ-1, j-1)`, `(τ-1, j)`, `(τ-1, j+1)` lie in `P`.

*Proof.* Write `d = t_top - τ ≥ 0`, so `j_lo - d ≤ j ≤ j_hi + d`. At level
`τ-1` the admissible interval is `[j_lo - (d+1), j_hi + (d+1)]`. From
`j ≥ j_lo - d` we get `j-1 ≥ j_lo - (d+1)`; from `j ≤ j_hi + d` we get
`j+1 ≤ j_hi + (d+1)`. ∎

**Corollary C′.** Every cell of `P` above the base row is a function of the
base row alone. Hence in every model below, *no unconstrained variable outside
the encoded region can influence any queried cell* — the requirement stated in
the Phase 2A brief. In particular the encodings contain no artificial
boundary condition: cells outside `P` are never referenced.

All three models use `[j_lo, j_hi] = [0, 1]` (the two queried columns), except
the control instance of §7, which uses `[-1, 0]`.

## 3. MODEL A — unrestricted rule-30 patch

**Variables.** `x[τ,j] ∈ Σ` for every `(τ, j)` in the cone with
`t_base = t - a`, `t_top = t + p`, `[j_lo, j_hi] = [0,1]`.
The base row is **free**.

**Constraints.**
1. `x[τ,j] = f(x[τ-1,j-1], x[τ-1,j], x[τ-1,j+1])` for every cell above the base.
2. H1: `x[t-a+k, 0] = x[t-a+k+p, 0]` for `k = 0..a`.
3. H2: `x[t, 0] = 0`.
4. ¬C: `x[t, 1] ≠ x[t+p, 1]`.

**Lemma T (translation invariance).** Model A is satisfiable for `(a, p, t)` if
and only if it is satisfiable for `(a, p, t')`, for any `t, t' > a`.

*Proof.* The constraint set mentions `t` only through differences of time
indices: every constraint relates levels `t-a+k`, `t-a+k+p`, `t`, `t+p`, all of
which are `t` plus a constant depending on `a`, `p`, `k` only. The map
`x'[τ, j] := x[τ + (t - t'), j]` is a bijection between assignments for `t` and
for `t'` preserving constraint 1 (which is itself time-translation invariant)
and constraints 2–4. ∎

Consequently the code normalises `t := a`, putting the base row at time 0, and
**one solve per `(a, p)` decides Model A for every `t` simultaneously**. The
search over `t` is therefore complete, not bounded.

**Theorem A-sound.** If Model A`(a,p)` is UNSAT then for **every** `t > a` the
triple `(a, t, p)` is not a counterexample to MB1-loc(`a`).

*Proof.* Suppose `(a, t, p)` were a counterexample. Restrict the real orbit to
the cone: `x[τ,j] := a_τ(j)`. Constraint 1 holds because the orbit obeys the
rule; constraints 2–4 hold because H1, H2, ¬C hold by assumption. So Model A
has a model, contradicting UNSAT. ∎

**Theorem A-incomplete.** Model A`(a,p)` SAT does **not** imply that a
counterexample exists in the real orbit.

*Proof.* A model provides a base row, but nothing forces that row to occur as a
row of the single-cell orbit. Model A drops the seed entirely. ∎

> **Reading rule.** A-UNSAT ⇒ **THEOREM** (MB1-loc holds for that `(a,p)`, all
> `t`). A-SAT ⇒ a certificate about the *seed-free relaxation only*; it says
> nothing about MB1-loc. Model A SAT patches are **not** counterexamples and
> are labelled as such in every certificate file.

## 4. MODEL B — the actual single-cell orbit

**Variables.** `x[τ,j]` on the cone with `t_base = 0`, `t_top = t + p`,
`[j_lo, j_hi] = [0,1]`. The base row spans sites `[-(t+p), 1+(t+p)]`.

**Constraints.** Rule transitions (as above), plus

* **seed**: `x[0,0] = 1` and `x[0,j] = 0` for every other site of the base row;
* H1, H2, ¬C as in §1.

**Theorem B-exact.** Model B`(a,t,p)` is satisfiable **iff** `(a,t,p)` is a
counterexample to MB1-loc(`a`) in the real orbit.

*Proof.* By F1 (light cone, Phase 1) the true configuration at time 0 is zero
outside site 0, and the base row of the cone contains site 0, so the seed
clauses pin the base row to the true row `a_0(·)` restricted to the cone. By
Corollary C′ every other variable is then forced, and by induction on `τ` the
forced value of `x[τ,j]` equals `a_τ(j)`. Hence the transition constraints have
exactly one model, and it satisfies H1 ∧ H2 ∧ ¬C iff the real orbit does. ∎

**Remark B-nofree.** Model B has **no free variables**. A solver run on it is a
*certificate check*, not a search: unit propagation alone decides it. Model B
is therefore an independent re-implementation of "simulate and test", not a way
to find witnesses that simulation could not.

**Cost.** The cone has `Θ((t+p)²)` cells — 380 for the `a=0` witness
(`t=17, p=1`) but `7.7 × 10⁶` for the `a=28` witness (`t=529506, p=2750`).
Model B is run exactly where it is affordable; §5 gives the alternative.

> **Reading rule.** B-SAT ⇒ **COMPUTATIONAL CERTIFICATE** that this triple is a
> genuine counterexample. B-UNSAT ⇒ this **specific** triple is not a
> counterexample. That verdict is *complete for the triple* (Theorem B-exact),
> not merely bounded; only a *sweep* over `(t,p)` is bounded, and its negative
> result is labelled **BOUNDED UNSAT**.

## 5. MODEL B′ — two windows on verified orbit rows

Motivated by the cost of B. The constrained cells occupy two time windows,
`[t-a, t]` and `[t+p-a, t+p]`, of height `a` each. Once each window's base row
is pinned to the true orbit row, the windows are separately determined and need
not be joined by one tall cone.

**Variables.** Two cones, `P₁` with `t_base = t-a, t_top = t`, and `P₂` with
`t_base = t+p-a, t_top = t+p`, both with top sites `[0,1]`.

**Constraints.** Rule transitions in each cone; base row of `Pᵢ` pinned to the
real orbit row at its base time; H1 relating `(s,0)` in `P₁` to `(s+p,0)` in
`P₂`; H2 in `P₁`; ¬C relating `(t,1)` in `P₁` to `(t+p,1)` in `P₂`.

**Theorem B′-relative.** If the two base rows equal the true orbit rows at
those times, then Model B′`(a,t,p)` is satisfiable iff `(a,t,p)` is a
counterexample.

*Proof.* Identical to Theorem B-exact, applied to each cone separately, using
Corollary C′. ∎

**Assumption made explicit.** B′ is sound *relative to the base rows*, which
are produced by the Phase 1 big-integer engine. That engine is cross-checked by
four independent implementations, by a mirror-rule (rule 86) regeneration, and
by SHA-256 against stored sequences (Phase 1 package, `CLAIM_LEDGER.md`
B4/B10/B11). B, by contrast, assumes nothing beyond the seed.

**Cost.** `Θ(a²)` cells: 1740 instead of 7.7 × 10⁶ for the `a=28` witness.

## 6. Encodings

Let `T` be the number of transition constraints.

| Name | Description | Vars | Clauses |
|---|---|---|---|
| **ENC1** "direct" | for each transition, 8 clauses of width 4, one forbidding each wrong `(l,c,r,y)` | cells | `8T` |
| **ENC2** "Tseitin" | auxiliary `u ↔ (c ∨ r)` [3 clauses] then `y ↔ l ⊕ u` [4 clauses] | cells + `T` | `7T` |
| **Z3-native** | Z3 term `x[τ,j] == Xor(l, Or(c, r))`; Z3 does its own clausification | — | — |

ENC1 and ENC2 share no clause-generation code; Z3-native shares no CNF code
with either. Equality/disequality constraints are 2 clauses each.

**Selectors.** Every H1/H2/¬C constraint `φ` is added as `¬s → φ`'s clauses
guarded by a fresh selector `s`, and `s` is passed as an assumption. An UNSAT
answer then yields an **unsat core** naming exactly which hypotheses conflict
(`get_core()` in PySAT, `assert_and_track` / `unsat_core()` in Z3). Transition
and seed clauses are hard, so cores never contain "the rule" — only
hypotheses, which is what makes them readable.

## 7. The UNSAT control instance

The MB1-loc grid turned out to be uniformly SAT, so the core/proof machinery
would never be exercised by it. A control instance with a **known** UNSAT
answer is therefore included, derived from Proposition 6 of the Phase 1 package:

```
    a_τ(-1) = a_{τ+1}(0) ⊕ ( a_τ(0) ∨ a_τ(1) )
```

If `a_t(0) = 1` then `a_t(0) ∨ a_t(1) = 1` regardless of `a_t(1)` (Fact F4:
rule 30 is **not** right-permutive), so agreement of column 0 at lag `p` at
times `t` and `t+1` *forces* `a_t(-1) = a_{t+p}(-1)`.

**Control(p, forced).** Cone with `t_base = 0`, `t_top = t+p+1`, top sites
`[-1, 0]`, constraints: `col_0` agrees at lag `p` at `t` and `t+1`;
`col_0(t) = 1` (forced) or `= 0` (unforced); `col_{-1}(t) ≠ col_{-1}(t+p)`.

**Expected:** UNSAT when forced, SAT when unforced. This is a *prediction from
a hand proof*, so agreement confirms that the encoding can detect genuine
forcing — i.e. that the uniform SAT result of the MB1-loc grid is a fact about
rule 30, not an artefact of a vacuous encoding.

## 8. Fourth decision procedure: exhaustive enumeration

Model A's only free variables are its base row, of width `2(a+p)+2`. For small
`a+p` the instance can be decided by enumerating all `2^w` base rows and
evolving each with the Phase 1 rule-30 table — **no SAT solver involved**.
This is a fully independent decision procedure and is used as a cross-check
wherever `w ≤ 18`.

## 9. Label semantics used throughout Phase 2A

| Label | Meaning | Example |
|---|---|---|
| **THEOREM** | proved mathematically here or in Phase 1 | Lemma T, Lemma C, Theorems A-sound / A-incomplete / B-exact / B′-relative |
| **COMPUTATIONAL CERTIFICATE** | a specific object (patch, witness triple) exhibited and re-verified independently | the eight Model B′ witness certificates |
| **BOUNDED UNSAT** | UNSAT over a stated finite range only; **not** a universal statement | the Model B sweep finding no counterexample with `t ≤ 48, p ≤ 24, a ≥ 4` |
| **CONJECTURE** | precise, unproved | "Model A is SAT for every `(a,p)`" |

**Standing prohibition.** No section of Phase 2A asserts MB1-loc(`a`) is
universally true on the ground that no witness was found under finite bounds.
Absence of a witness under bounds is recorded as BOUNDED UNSAT together with
the bounds, and — where the bound is the real obstruction — with the count of
instances that even satisfied the hypotheses.
