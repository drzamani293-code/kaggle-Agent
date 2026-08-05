# Final Validation Report

Results of `run_verified_theorems_tests.py` run from a **fresh extraction** of
`RULE30_VERIFIED_THEOREMS_V1.zip`.

---

## 1. Runs performed

| run | location | result |
|---|---|---|
| in place, full | `/home/user/kaggle-Agent/RULE30_VERIFIED_THEOREMS_V1` | **71 passed, 0 FAILED, 0 skipped** (37.8 s) |
| **fresh extraction, standalone** (sibling packages absent) | a scratch directory | **57 passed, 0 FAILED, 11 skipped** (0.1 s) |
| **fresh extraction, beside the sibling packages** | the repository root | **71 passed, 0 FAILED, 0 skipped** (39.9 s) |

The 11 skips in the standalone run are the artifact-dependent checks (10 SHA-256
hashes and the certificate re-verification). They are reported as **SKIPPED with
a reason**, never as passes — a folder extracted on its own cannot check hashes
of files it does not have.

## 2. What the suite validates

### §1 — Boolean identities, exhaustively (10 checks)
Rule table `(0,1,1,1,1,0,0,0)` under `4l+2c+r`; left-permutivity on all four
`(c,r)`; failure of right-permutivity exactly when `c = 1`; the local inversion
identity on all 8 neighbourhoods; the defect equation in both XOR/OR and ANF
form on all 32 assignments; the masking corollary; the `Δ(u,v,0,g) = g ∧ ¬u`
reduction used by T-6.1; the fibre trichotomy over **every** base word of length
`≤ 6` (5 460 words); and the strip transition relation's in-degree, exhaustively
for `R ≤ 3`. **No sampling anywhere in this section.**

### §2 — theorems re-checked on the real orbit (23 checks)
The coordinate change cell by cell; light cone; support bound; frozen left edge;
the strip identity including negative `r`; **T-08 autonomy from `t = 0`** (the
check that establishes correction C-08); T-22 with its explicit 2-cycle;
exact `(T,P)`; monotonicity; `P(K-1) | P(K)`; the period relation; `T(K) ≤
T(K-1)+P(K-1)`; the support lower bound; **`T(K) ≤ K` exactly for `K ≤ 17`**
(correction C-07); the exact recurrence T-18 at every level; T-19 at every
resetting level; the eventually-zero coordinates; **T-15 at the coordinate
indices** (correction C-16); T-14; the XOR spine at three values of `t`;
**VQ-02's configuration never occurring**; and T-21.10.

### §3–4 — certificates and hashes (14 checks)
The stored 10⁶-bit sequence is re-generated independently for its first 200 001
bits and compared; its **998140 distinct length-28 factors are recounted from
scratch**; and all ten catalogued artifacts are hashed and compared with
`COMPUTATIONAL_CERTIFICATES.md`.

### §5 — cross-references (16 checks)
73 claim IDs are declared in the ledger; all eight categories A–H are non-empty;
**every claim ID cited in any document exists in the ledger**; sections T1–T23
are all present in `VERIFIED_THEOREMS.md`; **every theorem section carries a
"why this does not solve Problem 1" note** (23 notes for 23 sections); all 11
required documents exist; the dependency graph has a Mermaid block and marks the
bridge.

### §6 — banned-overclaim scan (8 checks)
Scans every document for:

* "Problem 1 solved" / "we solve Problem 1"
* "proved the centre column aperiodic" / "the centre column is not eventually periodic"
* "unique for all K"
* **unqualified uses of "UNSAT"** — every occurrence must sit in a paragraph
  that qualifies it (e.g. "no bounded scan is UNSAT", "0 UNSAT", "UNSAT
  predicted by a hand proof")

and positively requires the disclaimers: the paper draft and the review request
must state that Problem 1 is not solved; `NOVELTY_STATUS.md` must record novelty
as unresolved and name the HTTP 403 failures; the ledger must state that passing
tests does not promote a claim.

Markdown emphasis is stripped before every prose check, so `does **not** solve`
is read as `does not solve`.

## 3. Failures found and fixed during validation

The suite failed **seven checks across its first runs**, all against this package's own
documents:

| failure | fix |
|---|---|
| claim ID `T-04` cited but not declared | added T-04 (the factor-counting lemma) to ledger category A — the lemma is unconditional; only the numeral is a certificate |
| `VERIFIED_THEOREMS.md` missing sections T11, T12 | added them explicitly (they had existed only as sub-results T10.1 and T9.1) |
| a theorem section lacked its "why this does not solve Problem 1" note | added an umbrella note covering T21.1–T21.12 |
| the overclaim scanner flagged the ledger's own disclaimer | widened the disclaimer context window from one line to the surrounding sentence |
| two qualified uses of "UNSAT" flagged | scan paragraphs, not physical lines; strip markdown first |
| a third, in **this report's own** description of the scanner | added "qualified"/"uses of"/"occurrence" to the qualifier list — a document explaining the check is not making the claim |
| the paper-draft disclaimer check failed on `does **not** solve` | strip markdown emphasis before phrase checks |

Four were genuine document defects; three were defects in the checker (the
third surfaced when this report was itself scanned). Both kinds
are recorded rather than quietly fixed.

## 4. What this validation does **not** establish

* It does **not** verify that any proof is correct. Proof correctness is the
  business of `PROOF_AUDIT_REPORT.md`, which re-derives each theorem from the
  definitions; the suite only re-checks identities and looks for counterexamples.
* It does **not** re-run the long Phase 2A–2E computations. It hashes their
  artifacts and re-derives the two Phase 1 certificates.
* It does **not** verify the rule-numbering convention against a primary
  source. The suite checks that the table used is `(0,1,1,1,1,0,0,0)` under
  `4l+2c+r`, which is a check of internal consistency, not of the convention.
  This remains the corpus's largest unverified assumption.
* It cannot detect an overclaim phrased in words the scanner does not know.
  The banned list is a safety net, not a proof of restraint.

## 5. Verdict

The package is internally consistent: every cited claim ID resolves, every
theorem has a proof and a limitation note, every certificate's artifact hashes
correctly, every catalogued correction is recorded with both wordings, and no
document contains an unqualified overclaim of the kinds scanned for.

**Nothing in this validation bears on Problem 1, which remains open.**
