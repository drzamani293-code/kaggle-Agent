# Novelty Status

**Novelty is unresolved for this corpus, and this document does not resolve
it.**

Every attempt to reach a primary source from this environment failed with
**HTTP 403** through the proxy: arXiv (2202.13809 and its listing pages),
`export.arxiv.org`, the Theoretical Computer Science version, ScienceDirect,
the Semantic Scholar API, and `oeis.org`. No paper, no abstract, no OEIS
b-file was retrieved. Phase 2B recorded this in `LITERATURE_GAP_MAP.md` and
labelled every attributed statement **UNVERIFIED — SNIPPET ONLY**.

Consequently:

* **No claim of novelty is made for any result in this corpus.**
* No literature is reconstructed from search snippets as though verified.
* Assessments below are the authors' own priors about what is likely standard,
  clearly labelled as such, and are **not** evidence of anything.

---

## 1. Assessment table

Categories, as required by the brief: **KNOWN STANDARD FACT** ·
**LIKELY KNOWN BUT SOURCE NOT VERIFIED** · **REFORMULATION OF KNOWN RESULT** ·
**POSSIBLY NEW** · **NEW ONLY AS A SPECIAL-ORBIT COMPUTATIONAL CERTIFICATE** ·
**NOVELTY UNKNOWN**.

| ID | Result | Assessment | Reasoning (a prior, not a citation) |
|---|---|---|---|
| **T-01** | left-permutivity; left reconstruction | **KNOWN STANDARD FACT** | permutivity of elementary CA is textbook; rule 30's left-permutivity is stated in essentially every treatment |
| **T-02** | no two adjacent eventually periodic columns | **LIKELY KNOWN BUT SOURCE NOT VERIFIED** | the argument is short and the statement is the natural first consequence of left reconstruction; it would be surprising if it were new |
| **T-03** | at most **one** eventually periodic column | **POSSIBLY NEW**, low confidence | strictly stronger than T-02 and requires the interior-inheritance lemma. We could not check whether it appears in the literature. **This is the theorem most in need of an expert's five minutes.** |
| **T-04 / CT-01** | factor-counting lemma; `T + p ≥ 998140` | lemma **KNOWN STANDARD FACT**; the bound **NEW ONLY AS A SPECIAL-ORBIT COMPUTATIONAL CERTIFICATE** | subword complexity of eventually periodic sequences is standard combinatorics on words. The numeral is specific to this computation |
| **T-05** | exact temporal-defect equation | **REFORMULATION OF KNOWN RESULT** | it is the rule written in difference variables; any treatment of CA perturbation dynamics contains the equivalent |
| **T-06** | wall identities W1/W2/W3/W6 | **NOVELTY UNKNOWN** | conditional on a hypothesis nobody expects to hold; we have no idea whether anyone has written them down |
| **T-07** | edge-aligned one-sided recurrence | **LIKELY KNOWN BUT SOURCE NOT VERIFIED** | a light-cone change of coordinates for a left-permutive rule is a natural move; we would expect it to be folklore |
| **T-08** | every fixed prefix is eventually periodic | **LIKELY KNOWN BUT SOURCE NOT VERIFIED** | immediate from T-07 plus finiteness. Almost certainly folklore |
| **T-09 – T-12** | projection; trichotomy; `P(K) ∈ {P,2P}`; `T` monotone | **POSSIBLY NEW as a package**, individually **LIKELY KNOWN** | each step is elementary; the *tower* organisation (a skew product with one-bit fibres, giving a period-doubling hierarchy) may or may not have been set out this way |
| **T-13 – T-19** | collapse/reset theory; the **exact** recurrence for `T(K)` via `D(K)` | **POSSIBLY NEW** | this is the most substantial block. The exact dichotomy (`D(K) = 0 ⇒ inherit`, `D(K) = 1 ⇒ T(K) = τ(K)+1`) and the factorisation `T(K)/K = d(K)·m(K)` are specific enough that we would be mildly surprised to find them stated |
| **T-20** | XOR-spine lower bound on the reset-erased skeleton | **POSSIBLY NEW**, but **trivial once stated** | the proof is three lines from permutivity. Novelty here would be worth little |
| **T-21.5** | `D(K) = w_{T(K-1)}(K) XOR Φ_K`; not finite-state in cycle data | **POSSIBLY NEW** | depends on the tower being new |
| **T-21.6** | `x_t(0) = w_t(t)` — the centre column is the diagonal | **KNOWN STANDARD FACT** | it is the coordinate change, restated |
| **T-21.8** | strip automaton: recurrent core is everything, all radii | **LIKELY KNOWN BUT SOURCE NOT VERIFIED** | surjectivity of a permutive-rule window relation is the standard argument; the corollary is immediate |
| **T-21.11/12** | Phase 2A's translation invariance and Model A soundness | **REFORMULATION OF KNOWN RESULT** | these are standard SAT-encoding hygiene, specific to this formalisation |
| **T-22** | `w_t(7) = 0` for all `t` | **NEW ONLY AS A SPECIAL-ORBIT COMPUTATIONAL CERTIFICATE** | a fact about one orbit at one coordinate, provable by exhibiting a 2-cycle |
| **T-23** | backward cone of the diagonal | **KNOWN STANDARD FACT** | light-cone geometry |
| **CC-01 – CC-12** | all certificates | **NEW ONLY AS SPECIAL-ORBIT COMPUTATIONAL CERTIFICATES** | by construction. Larger centre-column computations certainly exist elsewhere (Wolfram's own are reported to reach far beyond 10⁶ bits), so even the *scale* claims nothing |
| **BR-01** | the Bridge | **NOVELTY UNKNOWN** — and it is *open*, so novelty is moot | the reduction "second periodic column ⇒ Problem 1" is immediate from T-03; whether that reduction is in the literature is exactly what we could not check |

---

## 2. Precise search terms for whoever can reach the literature

If someone with library access wants to settle this, these are the queries we
would run, in priority order:

1. `rule 30 center column eventually periodic` / `"Rule 30" periodicity conjecture`
2. `"at most one" eventually periodic column cellular automaton` — for **T-03**
3. `left-permutive cellular automaton column periodicity`
4. `rule 30 prize problems Wolfram` — for the official statements of Problems 1–3
5. `elementary cellular automaton light cone coordinates skew product`
6. `"rule 30" subword complexity` / `factor complexity rule 30 center column`
7. `permutive cellular automata injectivity columns Hedlund`
8. `cellular automaton column sequence automaticity` / `k-automatic`
9. `"rule 30" transient preperiod prefix dynamics`
10. `Kopra rule 30` (arXiv:2202.13809, "Solving the Chaos …" — **explicitly not
    to be treated as a proof of anything here**)

## 3. Authors and topics likely relevant

Named because their areas overlap, **not** because we verified any specific
paper:

* **S. Wolfram** — the original rule-30 problems and the centre-column data.
* **G. A. Hedlund** — permutivity, surjectivity, and column/factor structure of
  CA; the classical source for T-01-type facts.
* **E. Jen** — exact results on rule-30 column sequences (a 1990 paper is
  repeatedly cited in secondary sources we could not retrieve).
* **J. Kari, K. Culik** — CA dynamics and decidability.
* **J. Kopra** — the 2022 arXiv item above.
* **N. Pytheas Fogg / J. Cassaigne / J. Berstel** — combinatorics on words, for
  the factor-complexity side of T-04.
* **F. Blanchard, P. Kůrka** — CA subshifts and column factor maps, relevant to
  T-02/T-03 and to the strip automaton (T-21.8).

## 4. Which results need external expert review

In priority order — this list is repeated in `EXPERT_REVIEW_REQUEST.md`:

1. **T-03** (at most one eventually periodic column) — is this known? It is the
   only structural theorem here that is stronger than the obvious one.
2. **T-18** (the exact recurrence for `T(K)` via the defect bit) together with
   **T-10/T-11** (the prefix tower as a skew product with a period-doubling
   hierarchy) — is the tower a known object?
3. **CT-01's method** — is the factor-counting bound the standard way to get
   lower bounds on `(T, p)` for rule 30, and is a stronger method known?
4. **T-21.5** — is "the frontier recurrence needs one new transient bit per
   level" a known obstruction pattern?
5. **The rule-numbering convention** — a five-second check by anyone with a
   reference that rule 30's table is `(0,1,1,1,1,0,0,0)` under `4l+2c+r`. This
   is the corpus's single largest unverified assumption (`CC-01`).

## 5. Standing rule

Until a primary source is actually read, the corpus states:

> **Novelty unresolved.** No result here is claimed to be new, and no result
> here is claimed to be known. Where an assessment is given above it is a prior,
> not a citation.
