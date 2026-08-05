# Comparison with Jen, "Aperiodicity in one-dimensional cellular automata"

**Physica D 45 (1990) 3–18.** Elsevier PII `016727899090169P`;
ADS bibcode `1990PhyD...45....3J`.

> **The paper was not retrieved**, and unlike Rowland and Kopra there is not
> even a preprint host to try — the brief's request to "retrieve the full
> primary text if possible" could not be met. ScienceDirect, ADS and OSTI were
> all unreachable (`RETRIEVAL_LOG.md`). Everything below is
> **UNVERIFIED — SECONDARY**, drawn from an abstract-level summary.

---

## 1. What is attributed to the paper

| # | attributed statement | source status |
|---|---|---|
| J-a | A certain class of one-dimensional, binary site-valued, **nearest-neighbour** automata generates **infinitely many aperiodic temporal sequences** from **arbitrary finite initial conditions** on an infinite lattice | SECONDARY (abstract-level) |
| J-b | The rules that generate aperiodic temporal sequences are characterised by **a particular form of injectivity** | SECONDARY |
| J-c | The class includes nontrivial "linear" automaton rules and certain nonlinear automata | SECONDARY |
| J-d | (via Kopra) Jen has a **proposition on aperiodicity of columns in space-time diagrams**, which Kopra's Theorem 3.5 generalises | SECONDARY, second-hand through Kopra |

"Temporal sequence" in J-a is what we call a **column**: the sequence
`t ↦ x_t(j)` at a fixed site `j`.

## 2. The three quantities that decide the comparison

Our T-02/T-03 and J-a are statements of the same *kind* but we cannot line them
up without the text. Three things must be read:

1. **The quantifier.** "Infinitely many aperiodic columns" is *weaker* than our
   T-03 ("at most one eventually periodic column", i.e. **all but at most one**
   are aperiodic). If J-a is exactly as summarised, T-03 is a **STRICT
   STRENGTHENING** for the rules in Jen's class.
2. **Membership.** Is rule 30 in Jen's class? J-b says the class is
   characterised by "a particular form of injectivity". Rule 30 is
   left-permutive, hence left-injective in the usual sense, so it plausibly
   qualifies — and Kopra's abstract, which names Rule 30 and cites Jen in the
   same breath (K-a, K-b), makes it near-certain that the connection is
   intended. **But "plausibly qualifies" is not a citation.**
3. **The seed class.** J-a says "arbitrary finite initial conditions", which is
   our **FIN** domain — the same domain as T-02/T-03. Good news for
   comparability, and it also means Jen's result is subject to the same
   seed-blindness limitation we recorded: a theorem true for all of FIN cannot
   settle Problem 1.

## 3. Correspondence table

All labels **UNCERTAIN — TEXT NOT RETRIEVED**.

| our result | Jen item | predicted label | conf. | decisive question |
|---|---|---|---|---|
| **T-02** no two adjacent EP columns | J-a / J-d | UNCERTAIN; plausibly IMMEDIATE COROLLARY of Jen's method | medium | Does Jen's proof forbid two adjacent EP columns, or only produce infinitely many aperiodic ones? |
| **T-03** at most one EP column | J-a / J-d | **STRICT STRENGTHENING** *if* J-a is as summarised; **EXACTLY KNOWN** if the proposition is sharper | **medium** | The exact statement of the proposition Kopra generalises |
| **T-01** left-permutivity | J-b ("a particular form of injectivity") | REFORMULATION — our permutivity is presumably a special case of Jen's injectivity condition | **high** | Which injectivity condition, exactly? |
| **T-2.1 / T-2.2** light cone, frozen left edge | presumably used in Jen's proof too | UNCERTAIN, likely shared | medium | Does the proof use the frozen edge of a finite-support orbit? |
| **CT-01** the factor-counting bound | — | APPARENTLY NOT PRESENT | medium | Does Jen give quantitative bounds, or only aperiodicity? |
| everything about the prefix tower (T-07 … T-21) | — | APPARENTLY NOT PRESENT | medium | Jen's object is columns; ours is diagonals |
| **BR-01** the Bridge | — | UNCERTAIN | — | Does Jen's proof produce a *second* periodic column from one, or aperiodicity directly? |

## 4. The one thing worth flagging loudly

If J-a is exactly as summarised — **infinitely many aperiodic columns for every
finite seed** — then for rule 30 it does **not** settle Problem 1, for the same
reason ours does not: it permits the centre column to be one of the
finitely-many-or-one exceptional columns. It is a strong result about the
diagram and it stops at the same place.

That coincidence of stopping-points is itself informative. It suggests the
seed-blindness obstruction we recorded in Phase 1 §10.3 is not an artefact of
our approach but a feature of this whole line of argument: **statements proved
for all finite seeds cannot distinguish the single-cell seed, and the centre
column's status is exactly what is left over.**

## 5. Honest summary

Nothing was read. Jen 1990 is the oldest and least accessible of the three
sources and the one most likely to contain the ancestor of our T-02/T-03. Until
it is read:

* **T-03 remains UNCERTAIN**, not "possibly new";
* the phrase "at most one column is eventually periodic" must not be presented
  as ours anywhere;
* and the relationship "ours strengthens Jen" must not be asserted, even though
  the summary points that way, because a summary of a 1990 abstract is not the
  proposition.
