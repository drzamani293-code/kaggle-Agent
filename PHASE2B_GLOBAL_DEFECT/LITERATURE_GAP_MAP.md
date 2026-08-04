# Literature Gap Map

Purpose: for each externally claimed theorem, record its exact assumptions and
conclusion, why it does or does not apply to width 1 (the centre column alone),
and what additional lemma would make it applicable.

---

## 0. ACCESS LIMITATION — read this before anything else

**No primary source could be retrieved in this environment.** Every attempt
returned HTTP 403 through the network proxy:

| target | result |
|---|---|
| `arxiv.org/abs/2202.13809`, `arxiv.org/pdf/2202.13809` | **403** |
| `export.arxiv.org/abs/2202.13809` | **403** |
| `sciencedirect.com/.../S0304397522007502` | **403** |
| `api.semanticscholar.org/graph/v1/paper/arXiv:2202.13809` | **403 (CONNECT tunnel failed)** |
| `oeis.org` (Phase 1, same proxy) | **403** |

What follows is therefore built from **search-result snippets only**, which are
themselves machine summaries. Consequently:

> **NO THEOREM IN THIS DOCUMENT HAS BEEN FORMALLY RECONSTRUCTED FROM ITS
> SOURCE.** Every statement attributed to the literature below is labelled
> **UNVERIFIED — SNIPPET ONLY**. None of it is used anywhere else in Phase 2B,
> and no Phase 2B result depends on any of it.

The task asked for formal reconstruction of these works. That part of the brief
**could not be completed** with the access available. It is recorded as an
outstanding item rather than approximated, because approximating a theorem
statement from a summary and then reasoning from it is exactly the failure mode
this project is set up to avoid.

What *can* be done without the sources — and is done in §5 — is to state
precisely what shape of theorem would be needed, so that a reader with library
access can check the real statements against the requirement.

---

## 1. Kopra — rapidly left expansive cellular automata

**Bibliographic identification (from search results, unverified):**
Johan Kopra, *A natural class of cellular automata containing fractional
multiplication automata, Rule 30, and others*, arXiv:2202.13809; and
*Rapid left expansivity, a commonality between Wolfram's Rule 30 and powers of
p/q*, Theoretical Computer Science (ScienceDirect S0304397522007502).

**Claimed content — UNVERIFIED, SNIPPET ONLY.** The snippets state that Kopra
defines a class of *rapidly left expansive* cellular automata containing rule
30 and the fractional multiplication automata, and that "previous results on
aperiodicity of columns in space-time diagrams of certain cellular automata
generalize to this new class", citing a proposition of Jen.

**Exact assumptions:** *not retrieved.* The definition of "rapidly left
expansive" is not available to us; nor are the hypotheses of the aperiodicity
proposition (in particular whether it requires a specific initial condition, a
class of initial conditions, or a shift-invariant measure).

**Exact conclusion:** *not retrieved.* Critically, **we could not determine
whether the aperiodicity conclusion concerns a single column or a window of
width ≥ 2.** That is the one fact that decides relevance, and it is exactly the
fact the snippets do not contain.

**Why it may not apply to width 1.** Our Phase 1 Theorem W2′ (proved from
scratch) shows that left-permutivity plus the light cone yields "at most one
eventually periodic column" — a width-2-flavoured statement — and Phase 1 §10
proves that no argument blind to the initial condition can do better, since W2′
holds for *every* finitely supported seed. Any theorem derived from expansivity
alone inherits that limitation: expansivity is a property of the *map*, not of
the orbit of one particular seed, so it cannot distinguish the single-cell seed
from other finitely supported seeds. Since some finitely supported seeds
*do* have eventually periodic centre columns (e.g. the all-zero seed), a
map-level hypothesis cannot by itself settle Problem 1.

**What additional lemma would make it applicable:** a statement that upgrades a
width-2 (or measure-theoretic, or generic-point) aperiodicity conclusion to the
specific orbit of `δ_0` at width 1 — i.e. precisely the Bridge / MB1 of the
Phase 1 registry. Nothing in the snippets suggests Kopra claims that.

**Status: UNVERIFIED. Not used. Reconstruction outstanding.**

---

## 2. Jen — aperiodicity in one-dimensional cellular automata

**Bibliographic identification (from search results, unverified):**
Erica Jen, *Aperiodicity in One-dimensional Cellular Automata*, Physica D **45**
(1990) 3–18.

**Claimed content — UNVERIFIED, SNIPPET ONLY.** Referred to in the Kopra
snippets as "a proposition of Jen on aperiodicity of columns in space-time
diagrams". The searches did **not** return the statement, and in particular did
not confirm the widely repeated "no two adjacent columns can both be eventually
periodic" formulation.

**Relation to what we have proved.** Phase 1 proves, from first principles and
importing nothing, both
* **Theorem W2**: no two adjacent columns of the rule-30 single-cell diagram
  are both eventually periodic; and
* **Theorem W2′**: at most one column of the whole diagram is eventually
  periodic.

If Jen's proposition is the width-2 statement, our W2 is an independent
reconstruction of it and W2′ is a strengthening. **We make no claim about
priority or equivalence**, because we have not read Jen.

**Why it does not apply to width 1:** identical to the analysis in
`WIDTH2_PROOF_RECONSTRUCTION.md` §10 — the argument's engine is
left-permutivity, which reconstructs leftwards from a *pair* of columns; a
single column is half a state.

**Status: UNVERIFIED. Not used (Phase 1 does not depend on it).**

---

## 3. Trace and ultimate-trace results

**What a trace result would say.** The *trace* of a CA at a cell is the
sequence of values that cell takes; the *trace subshift* is the closure of the
set of traces over all initial configurations. Typical theorems bound the
entropy or complexity of trace subshifts, or establish transitivity /
chain-transitivity properties (the snippets mention Kopra proving that
surjective ultimately right-expansive CA over full shifts are chain-transitive,
implying a result of Boyle).

**Why trace results are structurally the wrong tool for Problem 1 — our own
argument, not the literature's.** A trace subshift is defined by quantifying
over *all* initial configurations, or over a measure. Problem 1 concerns **one
orbit**. The centre column of the single-cell seed is a single point of the
trace subshift, and:

* a trace subshift can be complicated while containing eventually periodic
  points (it always contains some, if it contains a periodic orbit of the CA);
* conversely, knowing the whole trace subshift has positive entropy says
  nothing about which point our particular seed produces.

This is the same seed-blindness obstruction as §1, in a different dress. It is
recorded here as our reasoning, labelled **THEOREM-LEVEL ARGUMENT, ours** —
not as a claim about any specific paper.

**What additional lemma would make trace results applicable:** an equidistribution
or genericity statement placing the single-cell orbit's centre column in a
"typical" subset of the trace subshift on which the trace theorem's conclusion
holds pointwise. No such statement is known to us, and Phase 1 §7.1 shows the
centre column is statistically indistinguishable from typical at every measure
tried — which is suggestive and proves nothing.

**Status: no specific theorem reconstructed. Relevance argued, not imported.**

---

## 4. The recent algebraic Rule 30 paper

**Bibliographic identification (from search results, unverified):**
*Symmetric Nonlinear Cellular Automata as Algebraic References for Rule 30*,
arXiv:2604.00165 (v2).

**Claimed content — UNVERIFIED, SNIPPET ONLY.** The snippets describe: an
algebraic framework centred on spatial symmetry; Rule 22 (ANF
`a ⊕ b ⊕ c ⊕ abc`) as the symmetric reference; three closed-form results for
Rule 22 (support-set cardinality, a two-step recursive construction of support
sets, a continuum limit `∂ₘu = u_xx + 2u + u³`); an *empirical* power law
`m^b` with `b ≈ 1.11` for the symmetry-breaking deviation between Rule 22 and
Rule 30; and "a mechanism for the apparent randomness of Rule 30's centre
column identified through the left-permutive structure and asymmetric Boolean
sensitivity profile".

**Assessment of applicability.** On the snippet evidence:
* the closed-form results concern **Rule 22**, not Rule 30;
* the Rule 30 content is a **comparison**, with the key quantity described as
  *empirically consistent* with a power law — i.e. a measurement, not a
  theorem;
* "a mechanism for apparent randomness" is an explanatory claim, not an
  aperiodicity proof.

Left-permutivity — the ingredient named — is exactly what Phase 1 already uses,
and Phase 1 proves it cannot suffice alone (§10.3: the resulting theorems hold
for every finitely supported seed). So on the available evidence this work does
**not** supply the missing width-1 ingredient. **That assessment is provisional
and based on snippets; it is not a review of the paper.**

**Status: UNVERIFIED. Not used.**

---

## 5. "Rule 30: Solving the Chaos" (arXiv:2207.13237)

Per the standing instruction, this is **not treated as a proof**, and Phase 2B
uses nothing from it. We did not retrieve it (same 403), so we take no position
on its contents beyond that.

---

## 6. What a usable imported theorem would have to look like

Stated so that a reader with library access can test any candidate against it.
Write `X = (x_t)_{t≥0}` for the rule-30 orbit of `δ_0`.

> **Requirement R.** A theorem is usable for Problem 1 iff its hypotheses are
> satisfiable by `X` *as a single orbit* and its conclusion implies either
> (a) the centre column of `X` is not eventually periodic, or
> (b) `col_0(X)` eventually periodic ⟹ `col_j(X)` eventually periodic for some
>     `j ≠ 0` (the Bridge, Phase 1 registry C2).

Two filters that eliminate most candidates immediately:

1. **Seed-blindness filter.** If the theorem's hypotheses mention only the map
   (expansivity, permutivity, surjectivity, entropy of the CA) and not the
   initial condition, it cannot yield (a): those hypotheses hold for rule 30
   with *every* seed, including seeds whose centre column *is* eventually
   periodic. (Phase 1 `WIDTH2_PROOF_RECONSTRUCTION.md` §10.3.)
2. **Width filter.** If the conclusion is about a window of width ≥ 2, a
   product of columns, a trace subshift, or a set of initial conditions of full
   measure, it does not by itself yield (a) or (b) for one column of one orbit.

**Current status: every source examined here falls to filter 1, filter 2, or
both — on snippet evidence only.** The honest summary is that we could not
verify whether the literature already contains the bridge, and the access
failure is the reason.

---

## 7. Outstanding work (not attempted here)

1. Obtain Kopra arXiv:2202.13809 and the TCS paper; write out the definition of
   *rapidly left expansive* and the exact statement of the aperiodicity
   proposition, with its width.
2. Obtain Jen (1990) and reconstruct the proposition; compare formally with our
   Theorem W2 and W2′; establish whether W2′ is genuinely a strengthening.
3. Check every candidate against Requirement R in §6 and record the verdict.
4. Only then decide whether any registry item (G1–G5) is already known.

Until 1–3 are done, `GLOBAL_LEMMA_REGISTRY.md` marks all dependencies on the
literature as **unresolved**.
