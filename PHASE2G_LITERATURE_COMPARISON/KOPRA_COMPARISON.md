# Comparison with Kopra, "A natural class of cellular automata containing fractional multiplication automata, Rule 30, and others"

**arXiv:2202.13809 [math.DS]**, submitted 28 February 2022. Journal version:
"Rapid left expansivity, a commonality between Wolfram's Rule 30 and powers of
p/q", *Theoretical Computer Science*, December 2022 (PII `S0304397522007502`).

> **The paper was not retrieved.** `arxiv.org`, `export.arxiv.org` and the
> journal host all returned HTTP 403 at the proxy (`RETRIEVAL_LOG.md`).
> Everything below is **UNVERIFIED — SECONDARY**. **Theorem 3.5 was not read**,
> so the central question this comparison exists to answer is *open*.

---

## 1. What is attributed to the paper

| # | attributed statement | source status |
|---|---|---|
| K-a | The paper defines the class of **rapidly left expansive** cellular automata, which contains fractional multiplication automata, Wolfram's Rule 30, and many others | SECONDARY (abstract-level, consistent across results) |
| K-b | "The definition has been shaped by a proposition of **Jen** on aperiodicity of columns in space-time diagrams of certain cellular automata, which generalizes to this new class" | SECONDARY |
| K-c | "In Section 3, [the author] presents the definition of rapidly left expansive CA and proves the essential **Theorem 3.5** that has shaped this definition, a generalization of a result of Jen on aperiodicity of columns in space-time diagrams of certain CA" | SECONDARY |
| K-d | The paper also presents results originating from the theory of **distribution modulo 1** | SECONDARY |

## 2. Why this is the comparison that matters most for T-02 and T-03

Our two column theorems are:

* **T-02** — for every finitely supported seed, no two *adjacent* columns are
  both eventually periodic;
* **T-03** — for every finitely supported seed, **at most one** column of the
  whole diagram is eventually periodic.

K-b and K-c say Theorem 3.5 is a generalisation of a Jen result **on
aperiodicity of columns in space-time diagrams**, and that rule 30 is in the
class. That places Kopra's Theorem 3.5 squarely on top of our T-02/T-03
territory. Whether it *subsumes* them depends entirely on the quantifier, which
we could not read:

| if Theorem 3.5 says … | then T-02/T-03 are … |
|---|---|
| infinitely many columns are aperiodic | **STRICT STRENGTHENING** by us (at most one EP ⟹ all but one aperiodic) |
| all but finitely many columns are aperiodic | **STRICT STRENGTHENING** by us, but narrowly |
| at most one column is eventually periodic | **EXACTLY KNOWN** — T-03 is Kopra's (or Jen's) result |
| no two adjacent columns are eventually periodic | T-02 **EXACTLY KNOWN**, T-03 possibly still ours |
| something about a distinguished column only | **UNCERTAIN** |

The Jen abstract (see `JEN_COMPARISON.md`) points at the *first* row —
"infinitely many aperiodic temporal sequences" — which would leave T-03 as a
strengthening. But Kopra **generalises** Jen, and a generalisation may also
sharpen. **We cannot tell, and we do not guess.**

## 3. Correspondence table

Every label is **UNCERTAIN — TEXT NOT RETRIEVED**. The "predicted" column
records what our reading of the secondary material suggests, with confidence;
the "decisive question" column is what a reader with the paper should check.

| our result | Kopra item | predicted label | conf. | decisive question |
|---|---|---|---|---|
| **T-02** no two adjacent EP columns | Thm 3.5 | EXACTLY KNOWN or IMMEDIATE COROLLARY | medium | Does Thm 3.5, specialised to rule 30, forbid two adjacent EP columns? |
| **T-03** at most one EP column | Thm 3.5 | **UNCERTAIN** — could be anything from EXACTLY KNOWN to STRICT STRENGTHENING | — | What is the exact quantifier over columns in Thm 3.5? |
| **T-01** left-permutivity, left reconstruction | the *rapidly left expansive* definition | REFORMULATION — left expansivity is the general notion of which left-permutivity is a special case | **high** | Is left-permutivity exactly rapid left expansivity for radius-1 binary rules, or strictly stronger/weaker? |
| **T-2.4** leftward propagation preserving the **preperiod** | possibly inside the Thm 3.5 machinery | UNCERTAIN | low | Does the proof track preperiods, or only periods? Ours needs uniformity of `T`. |
| **T-3.2** interior inheritance | — | UNCERTAIN | low | Does Kopra have an interior/boundary-forcing argument? |
| **CT-01** factor-counting bound `T+p ≥ 998140` | — | APPARENTLY NOT PRESENT (it is orbit-specific and computational) | medium | Does the paper give any effective bound for rule 30's centre column? |
| **T-07 / T-08 / T-10 / T-11** the prefix tower | — | APPARENTLY NOT PRESENT | medium | Kopra's apparatus is expansivity and distribution mod 1, not a prefix tower |
| **T-18 / T-19** preperiod recurrence | — | APPARENTLY NOT PRESENT | medium | Does anything in the paper compute preperiods? |
| **BR-01** the Bridge (one EP column ⟹ a second) | — | UNCERTAIN | — | **Does Theorem 3.5 or its proof supply anything of this shape?** This is the question that would matter most to our programme |
| K-d distribution modulo 1 | — | APPARENTLY NOT PRESENT IN OURS | **high** | An apparatus we do not have at all |

## 4. The `2604.00165` lead

Search results repeatedly surfaced arXiv:2604.00165, "Symmetric Nonlinear
Cellular Automata as Algebraic References for Rule 30", alongside these
queries. Phase 2B also recorded it as unretrievable. It is **NOT RETRIEVED**
and nothing in this corpus depends on it; it is noted only so a future reader
does not have to rediscover the lead.

Likewise arXiv:2207.13237 "Rule 30: Solving the Chaos" surfaced again. The
standing instruction from the Phase 2B brief — **do not treat it as a proof** —
is unchanged, and it remains unread.

## 5. Honest summary

**This comparison did not happen.** Kopra's Theorem 3.5 is, on the available
evidence, the result in the literature most likely to bear directly on our
T-02/T-03 and conceivably on BR-01, and we could not read a word of it.

Any future claim that T-03 is new **must** wait on this paper. Until then T-03
is recorded as **UNCERTAIN**, and `REVISED_NOVELTY_LEDGER.md` treats it that way
rather than as the "POSSIBLY NEW, low confidence" that Phase 2F assigned.
