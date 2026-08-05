# Adapting Kopra's Theorem 3.5 — **NOT COMPLETED**

**Phase 2H, section 7.**

The brief asks: *"Reconstruct Theorem 3.5 from the primary paper. Identify the
exact step where width `w = 2` is used."*

> ## This task was not completed, and could not be.
>
> **Kopra, arXiv:2202.13809, was never read.** Every retrieval attempt in this
> phase returned HTTP **403** at the proxy's `CONNECT` stage — abstract, PDF,
> and mirror alike (`RETRIEVAL_NOTE.md` has the verbatim transcript). Theorem
> 3.5's statement is unknown to us, its proof is unknown to us, and therefore
> **no step of it can be identified, adapted, or checked.**

Nothing in this document is a reconstruction of Kopra's argument. Nothing here
may be cited as such. What follows is (§1) an explicit list of what would be
needed, and (§2) the honest substitute: the same question asked about **our
own** proofs, where the answer is known exactly.

---

## 1. What is missing, itemised

For a later phase with library access, these are the specific questions, in the
order that would make the adaptation possible:

1. **The statement.** What exactly does Theorem 3.5 assert — about which class
   of CA, which seeds (all configurations? finite seeds? a measure-one set?),
   and which columns (one? all? all but finitely many? infinitely many)?
2. **The quantifier over seeds.** Phase 2G's `KOPRA_COMPARISON.md` records that
   this is the decisive unknown. If Theorem 3.5 quantifies over *all* finite
   seeds, it is subject to the same seed-blindness obstruction as our T-02/T-03
   and cannot settle Problem 1 — the pattern already observed for Jen (1990).
3. **Where `w = 2` enters.** Secondary summaries indicate a width-2 hypothesis.
   Is it (i) two adjacent columns, as in our Theorem 2.B; (ii) a width-2 factor
   map / two-block code; or (iii) a rapid-expansivity constant that happens to
   equal 2? These are three different things and they adapt differently.
4. **Rapid left expansivity.** Is rule 30 shown to be rapidly left expansive,
   and with what constant? Our corpus has only rule-30-specific arguments.
5. **The distribution-modulo-1 apparatus.** Whether it is essential or a
   convenience.

Until at least (1) and (3) are answered, an "adaptation" would be invention
attributed to a named author. That is not done here.

## 2. The substitute: where width 2 enters **our** proofs

This is *our* question about *our* work. It is not Kopra's answer and must not
be reported as such. But it is the part of the section that could be done, and
it is exact.

Width 2 enters our corpus at **one place only**, and everything downstream
inherits it.

> ### The single origin.
> The backward step (BW) — `x_s(j-1) = x_{s+1}(j) XOR ( x_s(j) OR x_s(j+1) )` —
> reads **two same-row cells**, `x_s(j)` and `x_s(j+1)`, because they sit inside
> the `OR`. One is not enough, and three are not needed.

Why exactly two, stated as a proof rather than an observation:

* **Not one.** By T-01(c) / Lemma 1.0′, rule 30 is not right-permutive:
  `f(l,1,0) = f(l,1,1)`. So `x_s(j+1)` genuinely matters whenever `x_s(j) = 0`,
  and a single column cannot seed the sweep. Refuted concretely by
  Theorem 2.A, with the minimal counterexample `010` vs `011` at `h = 1`, and
  exhaustively for `h ≤ 12`.
* **Not three.** (BW) is an equality with exactly three inputs, one of which
  (`x_{s+1}(j)`) is on the later row. Adding a fourth same-row cell adds no
  information: the recursion already closes.

### Where the 2 propagates

| our result | the width-2 step it uses |
|---|---|
| **T-01(b)** left reconstruction | (BW) directly |
| **T-02** no two adjacent EP columns | via T2.4, which applies (BW) to the pair `(col_j, col_{j+1})` |
| **T-03** at most one EP column | via T-02, plus T3.1's radius-1 boundary forcing |
| **Theorem 1.2/1.3** the backward cone | the two-cell right seed `x_s(m), x_s(m+1)` |
| **Theorem 2.B** width-2 envelope | the columns `h, h+1` |
| **Theorem 2.C′** diagonal recursion | the two diagonals `E_k, E_{k+1}` |
| **Theorem 3.5 (ours)** reduced boundary | the same pair, thinned by OR-blindness |

### The one place our width 2 can be reduced

`PERIODIC_CENTER_TRIANGLES.md` Lemma 3.2: when `x_s(0) = 1`, the `OR` is
saturated and `x_s(1)` is irrelevant. So the *second* column is needed only at
the rows where the first is white — measured at 9880 of 20 000 rows. **The
width is still 2; the number of bits is about half.** Two further attempted
reductions failed (X-2H-04, X-2H-05) and are preserved.

For rule 90 the reduction is different in kind: `f_{90}(l,c,r) = l XOR r` has no
`OR`, so there is no blindness and no thinning — but rule 90 is *right*-
permutive too, so the sweep can be seeded from either side. Recorded as the
control.

## 3. Status

| item | status |
|---|---|
| Kopra Theorem 3.5 statement | **UNKNOWN — paper not retrieved** |
| Kopra Theorem 3.5 proof | **UNKNOWN — paper not retrieved** |
| the step where Kopra uses `w = 2` | **NOT IDENTIFIED — cannot be** |
| adaptation of Kopra's argument to the single-cell seed | **NOT ATTEMPTED** |
| where width 2 enters *our* proofs | **ANSWERED EXACTLY** (§2) |
| relationship between the two | **UNRESOLVED** |

**Section 7 of the brief is therefore reported as not completed.** It is the
only section of Phase 2H in that state.
