# Retrieval Log — read this before any other file in this package

## The headline, stated first

**Task 1 of the Phase 2G brief — "Read Rowland and Kopra completely, not only
abstracts or snippets" — was NOT completed. None of the three papers was
retrieved. Not one page of any of them was read.**

Every comparison in this package is therefore built on **secondary material**:
search-engine result summaries and figure captions. Every claim about what a
paper contains is labelled **UNVERIFIED — SECONDARY** and must be re-checked by
someone with the text.

This is the same failure Phase 2B recorded, and the standing rule from that
phase applies unchanged: **do not reconstruct literature from snippets as
though verified.** The brief said the sources are "now available"; they are not
available *to this environment*.

---

## 1. What was attempted, and what happened

Every outbound HTTPS request from this session goes through a policy-enforcing
egress proxy. Fetches fail at the **CONNECT** stage with **HTTP 403** — an
organization policy denial, not a network fault and not a paywall. The proxy's
own README says such denials are to be reported, not retried.

| target | method | result |
|---|---|---|
| `arxiv.org/abs/2202.13809` | WebFetch | **403 Forbidden** |
| `arxiv.org/pdf/2202.13809` | curl | **CONNECT tunnel failed, 403** |
| `export.arxiv.org/abs/2202.13809` | curl | **CONNECT tunnel failed, 403** |
| `ericrowland.github.io/papers/Local_nested_structure_in_rule_30.pdf` | WebFetch | **403 Forbidden** |
| `ericrowland.github.io/…` (same PDF) | curl | **CONNECT tunnel failed, 403** |
| `wpmedia.wolfram.com/sites/13/2018/02/16-3-4.pdf` | curl | **CONNECT tunnel failed, 403** |
| `complex-systems.com/abstracts/v16_i03_a04/` | WebFetch | **403 Forbidden** |
| `en.wikipedia.org/wiki/Rule_30` | WebFetch | **403 Forbidden** |

The last row is the diagnostic one: **even Wikipedia is blocked.** Page
retrieval is disabled wholesale in this environment, for every host. It is not
specific to publishers, paywalls, or these three papers.

Local filesystem was also searched for supplied copies
(`find / -iname "*rowland*" -o -iname "*kopra*" -o -iname "*2202.13809*"`, plus
`/mnt/attach`, `/mnt/user-data`, `/home/user`): **nothing present**.

## 2. What did work

`WebSearch` returns result titles, URLs, and a **model-generated summary of
those results**. That is a tertiary source: a summary of snippets of the
papers, not the papers. It is the only channel that returned anything.

Everything attributed to a paper in this package comes from that channel, plus
two ResearchGate **figure captions** (which are verbatim strings from the
papers' own figures, and are the strongest evidence available here — still not
the text).

## 3. Source-status labels used throughout this package

| label | meaning |
|---|---|
| **VERIFIED — OUR COMPUTATION** | computed here, reproducible, two independent implementations |
| **METADATA** | bibliographic facts (title, author, journal, volume, pages, year, DOI, arXiv ID) appearing consistently across independent search results |
| **UNVERIFIED — FIGURE CAPTION** | a verbatim caption string from a paper's own figure, surfaced by search |
| **UNVERIFIED — SECONDARY** | a search-engine summary of the paper's content |
| **NOT RETRIEVED** | no information obtained |

No statement anywhere in this package is labelled as a reading of a paper,
because no paper was read.

## 4. Bibliographic metadata obtained (**METADATA**)

* **Eric S. Rowland**, "Local Nested Structure in Rule 30", *Complex Systems*
  **16** (3) (2006) 239–258. DOI `10.25088/ComplexSystems.16.3.239`.
  Author copy at `ericrowland.github.io/papers/Local_nested_structure_in_rule_30.pdf`;
  publisher copy at `wpmedia.wolfram.com/sites/13/2018/02/16-3-4.pdf`.
* **Johan Kopra**, "A natural class of cellular automata containing fractional
  multiplication automata, Rule 30, and others", arXiv:2202.13809 [math.DS],
  submitted 28 February 2022. Journal version: "Rapid left expansivity, a
  commonality between Wolfram's Rule 30 and powers of p/q", *Theoretical
  Computer Science*, December 2022 (Elsevier PII `S0304397522007502`); open
  copy reported at `utupub.fi/bitstream/handle/10024/174540/`.
* **Erica Jen**, "Aperiodicity in one-dimensional cellular automata",
  *Physica D* **45** (1990) 3–18. Elsevier PII `016727899090169P`;
  ADS bibcode `1990PhyD...45....3J`.

Two leads not verified: **OEIS A094605** (reported as periods of rule-30
diagonals, one summary says "from the right") and **OEIS A363345** (reported as
eventual periods of the left diagonals). `oeis.org` is also unreachable, so
neither was checked.

## 5. What was salvaged, and why it is worth something

One numerical statement attributed to Rowland was specific enough to test
against our own tables: a 48-term sequence of eventual period lengths of the
left diagonals. **We reproduced all 48 terms exactly** — see
`ROWLAND_COMPARISON.md` §2 and `verify_against_literature.py`.

A 48-term agreement between an independently computed sequence and a published
one is strong evidence of two things — our rule-numbering convention and the
identification of our coordinates with Rowland's — **without being a reading of
the paper**. It also forces several novelty downgrades, which is the honest
direction and is carried out in `REVISED_NOVELTY_LEDGER.md`.

## 6. What must be done when the papers become readable

The comparison tables in this package are written as **questions with
pre-specified answer criteria**, so that a reader with the texts can fill them
in mechanically. In priority order:

1. **Rowland, Proposition 2** — read it verbatim and compare with our T-10 +
   T-14. Our reconstruction predicts they coincide exactly.
2. **Rowland**, whether preperiods/transients are treated at all — this decides
   whether T-16, T-18, T-19 survive as new.
3. **Kopra, Theorem 3.5** and the definition of *rapidly left expansive* —
   does it imply, for rule 30, anything about *how many* columns can be
   eventually periodic?
4. **Jen's proposition** as generalised by Kopra — the same question. Our T-03
   ("at most one") is formally stronger than "infinitely many are aperiodic";
   confirm that Jen's is the weaker statement and not the stronger one.
5. **Rowland's rule table and orientation conventions** — to confirm at the
   level of definitions what the 48-term match already indicates numerically.
