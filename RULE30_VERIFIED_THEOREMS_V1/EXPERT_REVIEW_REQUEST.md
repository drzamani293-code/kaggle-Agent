# Request for Expert Review

**Subject:** structural results on the single-seed rule-30 space-time diagram —
novelty check and proof scrutiny

---

## What this is

A self-contained corpus of elementary results about the rule-30 space-time
diagram from the single-cell seed, produced over six work phases. **It does not
solve Wolfram's Prize Problem 1 and makes no partial claim on it.** We are
asking for two things: (1) whether any of these results is already in the
literature, and (2) scrutiny of a small number of proofs.

We could not check the literature ourselves — every primary source we tried
returned HTTP 403 through our network. That is the main reason for this
request.

## The problem

Rule 30 is the elementary cellular automaton

```
    x_{t+1}(j) = x_t(j-1) XOR ( x_t(j) OR x_t(j+1) ),
```

started from the single-cell seed `x_0(j) = [j = 0]`. **Prize Problem 1:** is
the centre column `c_t = x_t(0)` eventually periodic? The expected answer is
no; it is open.

## What we can prove

All statements are for the single-cell seed unless a wider domain is marked.
Full proofs are in `VERIFIED_THEOREMS.md`; a claim-by-claim ledger with
categories (proved / conditional / certificate / observation / conjecture /
refuted) is in `MASTER_CLAIM_LEDGER.md`.

**Column structure** (holds for every *finitely supported* seed):

* **T-02.** No two adjacent columns are both eventually periodic.
* **T-03.** At most **one** column of the whole diagram is eventually periodic.

**A prefix tower.** In light-cone coordinates `w_t(k) = x_t(-t+k)` the rule
becomes one-sided, `w_{t+1}(k) = w_t(k-2) XOR (w_t(k-1) OR w_t(k))`, so every
prefix `W_t^K = (w_t(0),…,w_t(K))` is autonomous and eventually periodic, with
exact preperiod `T(K)` and period `P(K)`. The tower is a skew product with
one-bit fibres:

* **T-10/T-11.** The one-period fibre map is constant, the identity, or the
  negation; correspondingly `P(K) ∈ {P(K-1), 2P(K-1)}`, so `P(K)` is a power of
  two and the sequence of period doublings is the whole period structure.
* **T-18.** An **exact** recurrence for the preperiod. With
  `D(K) := w_{T(K-1)}(K) XOR w_{T(K-1)+P(K)}(K)` (one bit) and
  `τ(K) := min{t ≥ T(K-1) : w_t(K-1) = 1}`:
  ```
      D(K) = 0  ⟹  T(K) = T(K-1)
      D(K) = 1  ⟹  T(K) = τ(K) + 1        (exactly)
  ```
  with the period-doubling and neutral cases both giving `T(K) = T(K-1)`.
  Verified at all 29 999 levels `K ≤ 30000` with no exceptions.
* **T-21.5.** `D(K) = w_{T(K-1)}(K) XOR Φ_K`, where `Φ_K` depends only on the
  base cycle word. So flipping one transient bit flips `D(K)` without changing
  any cycle word below: the recurrence consumes exactly one new transient bit
  per level and is not determined by any fixed-width window of cycle data.

**A lower bound (conditional on a finite computation).** If the centre column
is eventually periodic with preperiod `T` and period `p`, then `T + p ≥ 998140`
— from the standard factor-counting lemma plus a certified count of 998140
distinct length-28 factors in the first 10⁶ bits.

## The unresolved bridge

Everything reduces to one missing implication:

> **BR-01.** If the centre column is eventually periodic, then some **other**
> column is eventually periodic.

With **T-03**, BR-01 settles Problem 1 immediately. We attempted five routes
(a local "MB1" lemma and its finitary shadow; a zero-wall analysis of the
temporal defect field; the prefix tower's transient frontier; a collapse-chain
analysis; a dependency-skeleton analysis) and **all five reduce back to BR-01
without crossing it**. Two apparent routes turned out to be degenerate: one is
"(H) ⟹ (H)", the other is Problem 1 restated. Details in
`THEOREM_DEPENDENCY_GRAPH.md` §4.

There is a structural reason we cannot get past it. T-02 and T-03 hold for
**every** finitely supported seed, so they cannot distinguish the single-cell
seed and cannot decide Problem 1. Everything that *is* seed-aware in our corpus
describes the eventually periodic region of the prefix tower, while the centre
column is the diagonal `w_t(t)` — which, on the whole computed range
(`18 ≤ K ≤ 30000`), lies strictly inside the transient, `T(K) > K`. The
seed-blind half cannot see the seed; the seed-aware half cannot see the
diagonal.

## Questions

1. **Is T-03 known?** "At most one eventually periodic column" for a
   left-permutive elementary CA from a finitely supported seed. This is the
   result we most suspect is in the literature and could not check.
2. **Is the prefix tower (T-08–T-11) a known object?** A skew product with
   one-bit fibres over nested light-cone prefixes, with a period-doubling
   hierarchy.
3. **Is the exact preperiod recurrence (T-18) new**, or is there a standard
   treatment of preperiods in such towers?
4. **Is factor counting the standard way to lower-bound `(T, p)` for rule 30?**
   Is a stronger elementary method known? Our bound scales linearly with the
   length computed and can never close.
5. **Are there results we should know about that would settle BR-01, or show it
   to be as hard as Problem 1?**

## Proofs most in need of scrutiny

1. **T-02**, specifically **T-2.4** (leftward propagation with the *same*
   preperiod). The whole proof turns on the preperiod not growing as one moves
   left; if that step is wrong, T-02 and T-03 both fail.
2. **T-03**, specifically **T-3.2** (interior inheritance). The period bound is
   exponential in the gap; we use only the qualitative statement, but the
   argument via a deterministic map on `{0,1}^{gap} × Z/p` should be checked.
3. **T-18**, specifically **T-18.1** (the defect at a level is preserved by
   non-reset steps and zeroed by reset steps) and the treatment of the
   period-doubling case, where the analysis runs at the doubled lag `2P(K-1)`
   while the monotonicity argument runs at minimal periods. We believe these
   are consistent (see `PROOF_AUDIT_REPORT.md` B-5) but it is the subtlest
   point in the corpus.
4. **T-21.5**, and in particular the *scope* of its negative half: we claim
   only that no fixed-width window of cycle words below `K`, read from phase
   `T(K-1)`, determines `D(K)`. An earlier draft overstated this as "no
   finite-state description exists"; the restriction is correction C-12.
5. **The rule-numbering convention.** We fix rule 30's table as
   `(0,1,1,1,1,0,0,0)` indexed by `4l+2c+r`, equivalently
   `f(l,c,r) = l XOR (c OR r)`. Our only external check is a 14-bit
   centre-column prefix matched against a *secondary* transcription of OEIS
   A051023 (`oeis.org` was unreachable). If this convention is wrong, the whole
   corpus concerns a different rule. **A five-second confirmation from anyone
   with a reference would remove our largest unverified assumption.**

## References we would like

Anything overlapping: column periodicity in permutive elementary CA; light-cone
/ prefix coordinates for one-sided rules; subword-complexity bounds for CA
column sequences; the status of Wolfram's rule-30 problems; and E. Jen's exact
results on rule-30 columns.

## What we are not claiming

**This work does not solve Wolfram's Rule 30 Prize Problem 1.** Specifically:

* Not that the centre column is aperiodic.
* Not that any finite computation supports aperiodicity — our bounds are lower
  bounds on `T + p` and scale linearly with the computation.
* Not that any result here is new. `NOVELTY_STATUS.md` records novelty as
  **unresolved** and gives our priors, clearly labelled as priors.
* Not that Problem 1 is solved or partially solved. The bridge above is the
  whole problem, and we did not move it.

## Reproducing

Every claim is traceable: `SOURCE_MANIFEST.md` maps each result to its
originating file and section, with original and corrected wording;
`COMPUTATIONAL_CERTIFICATES.md` gives programs, ranges, solvers, artifacts and
SHA-256 hashes; `run_verified_theorems_tests.py` re-checks the identities,
hashes and cross-references from a clean extraction. All computations are
deterministic (no randomness anywhere in the corpus).
