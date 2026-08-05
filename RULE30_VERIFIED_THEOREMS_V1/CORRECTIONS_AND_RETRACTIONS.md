# Corrections and Retractions

Every correction made anywhere in the corpus, with the **original wording** and
the **corrected wording** side by side. Nothing is silently repaired.

Corrections are numbered `C-nn` and are stable. A correction is not a criticism
of the phase that made the claim; several were caught by that phase's own
tests, and those are noted.

---

## C-01 — the backward-transfer probability

| | |
|---|---|
| **Original (Phase 1 draft)** | "The transfer probability saturates around **0.755**, not at 1." |
| **Corrected (Phase 1 final, F10)** | The probability rises with look-back: `0.5000, 0.7031, 0.7401, 0.7545, 0.7627, 0.7829, 0.8082, 0.9583` at `a = 0,4,…,28` in the 10⁶-bit run. **Its limit as `a → ∞` is undetermined**; both `< 1` and `= 1` are consistent with the data, and they imply opposite research programmes. |
| **Found by** | extending the run from 200001 to 10⁶ bits |
| **Why it mattered** | the two possibilities imply opposite programmes: saturation below 1 would suggest MB1 is false, saturation at 1 would suggest it is true |
| **Caveat that remains** | sample sizes collapse — 4662 at `a = 20`, 292 at `a = 24`, **24** at `a = 28`. The last points are nearly meaningless individually. |

## C-02 — MB1-loc witness polarity

| | |
|---|---|
| **Original (`PHASE1_AUDIT.md` §8 draft)** | "Every MB1-loc witness has `a_t(1) = 0` and `a_{t+p}(1) = 1`." |
| **Corrected** | True for the 200001-bit witnesses; **false** for the 10⁶-bit ones at `a = 16` and `a = 20`, where the polarity is `1 → 0`. A polarity column was added to the witness table. |
| **Found by** | `run_all_tests.py` step 5, which **re-simulates** rather than trusting the stored JSON |
| **Effect on results** | none: a violation requires only `a_t(1) ≠ a_{t+p}(1)`, so all eight refutations stand |

## C-03 — a `SKIPPED` verdict counted as a dissent

| | |
|---|---|
| **Original (Phase 2A)** | the agreement set folded `SKIPPED` enumeration verdicts in with real verdicts, so a too-wide base row at `p = 13` made a test report a cross-method disagreement |
| **Corrected** | an explicit `enumeration_checked` flag; `SKIPPED` is no longer a verdict. Recorded in `PHASE2A_RESULTS.md` §5.2 |
| **Found by** | Phase 2A's own test battery |

## C-04 — what Model A rules out

| | |
|---|---|
| **Original (loose phrasing)** | "Model A rules out every possible local proof." |
| **Corrected** | Model A's 832 SAT results rule out **seed-free local arguments at look-back depth `a ≤ 48` and lag `p ≤ 64`** — and only those. Nothing is ruled out for larger `a` or `p`, and nothing is ruled out for arguments that use the initial condition. |
| **Why it matters** | the corrected form is a statement about a *grid of proof strategies*; the original sounds like a statement about all proofs |

## C-05 — `T(K) ≈ 1.29K` / `≈ 1.34K`

| | |
|---|---|
| **Original (Phase 2B §5, Phase 2C)** | "preperiod `T(K) ≈ 1.29 K`" (Phase 2B, `K ≤ 1280`); later "`T(K) ≈ 1.34 K`" (Phase 2C, `K ≤ 30000`) |
| **Corrected** | These are **values measured at the largest `K` computed**, not theorems and not limits. `T(K)/K` is **not monotone**: `1.3400` at `K = 100`, `1.2525` at `K = 400`, `1.3315` at `K = 10000`, `1.3419` at `K = 20000`, `1.3399` at `K = 30000`. **No limit is claimed to exist.** |
| **Related** | Phase 1's `BO-08` has the same shape: the left-region slope had **not converged** (0.7645 → 0.7444), and no constant was claimed there either |

## C-06 — which direction of `T(K)` vs `K` is favourable

| | |
|---|---|
| **Original (Phase 2B G5 framing)** | `T(K)/K > 1` presented as the favourable direction for proving aperiodicity |
| **Corrected (Phase 2C, `DIAGONAL_BRIDGE_ATTEMPT.md` §3)** | The opposite: it is `T(K) ≤ K - P(K)` that would give Theorem D1 content, and D1 is what would relate the diagonal to a fixed column. `T(K) > K` means D1 is **vacuous** (`VQ-05`). Phase 2B's framing was retracted. |
| **Why it matters** | the retracted framing made a measurement look like progress in the wrong direction |

## C-07 — "`T(K) > K`"

| | |
|---|---|
| **Original (Phase 2C E4)** | stated as if general |
| **Corrected** | `T(K) ≤ K` holds **exactly for `K ∈ {0,1,…,17}`** — 18 genuine counterexamples, listed and kept. `T(K) > K` is a **bounded observation** for `18 ≤ K ≤ 30000` (`BO-03`), never a theorem |

## C-08 — the spurious `t ≥ K/2` in EA2

| | |
|---|---|
| **Original (Phase 2B EA2)** | "there is a map `F_K` with `W_{t+1}^K = F_K(W_t^K)` for all **`t ≥ K/2`**" |
| **Corrected (this phase, audit A-01)** | The qualifier is **unnecessary**. Autonomy holds for **every `t ≥ 0`**: the recurrence reads only coordinates `≤ K`, at every time. Verified at `K = 40` for all `t ∈ [0,78]`, including `t < K/2`. |
| **Direction** | a **strengthening**, not a repair — but the weaker form suggested that something happens at `t ≈ K/2`, and nothing does |

## C-09 — the redundant `max` in R3

| | |
|---|---|
| **Original (Phase 2D R3)** | `T(K) ≤ max( T(K-1), τ_K + 1 )` |
| **Corrected (this phase, audit A-03)** | Since `τ(K) := min{ t ≥ T(K-1) : … } ≥ T(K-1)`, the `max` is always its second argument. The bound is simply **`T(K) ≤ τ(K) + 1`**. |
| **Why it matters** | with the `max` in place, the bound looks as if it might be tight through the first argument, obscuring the real criterion for equality — the defect bit (T-18) |

## C-10 — completeness of `{2,7,28,399}` and `{3,8,29,400}`

| | |
|---|---|
| **Original (Phase 2C/2D phrasing)** | "the eventually-zero coordinates are `{2,7,28,399}`"; "the doubling points are `{3,8,29,400}`"; "`K = 7` is the only permanently zero coordinate" |
| **Corrected** | `w_t(7) = 0` for all `t` is a **theorem** (T-22, via an explicit 2-cycle). **Everything else in that sentence is a bounded observation** for `K ≤ 30000` (`BO-04`): that these lists are *complete* is not proved, and **uniqueness of 7 is not an all-`K` theorem** |
| **Strength of the evidence, restated** | by T-14 there were exactly **four opportunities** for a non-COLLAPSING level in the whole range, and all four resolved as DOUBLING on an odd parity. That is four events, not 30 000 confirmations |

## C-11 — what collapse theory constrains

| | |
|---|---|
| **Original (implicit in Phase 2C/2D framing)** | the collapse/cycle analysis was presented as progress towards the centre column |
| **Corrected (Phase 2D §3, restated here)** | Collapse theory applies **only after each coordinate has entered its eventual cycle**. The diagonal sits at level `t` at time `t`, and `T(t) > t` on the whole computed range (`BO-03`), so **the diagonal is read strictly before any collapse result applies to its level**. Phase 2D's registry item C2 is the sharpest form: its hypothesis is verified on the entire computed range and its conclusion still does not follow |

## C-12 — "no finite-state description exists"

| | |
|---|---|
| **Original (Phase 2E, README and Corollary C3 wording)** | "No finite-state description of the frontier increments exists." |
| **Corrected (this phase, audit A-05)** | What is proved (T21.5) is: `D(K) = w_{T(K-1)}(K) XOR Φ_K` with `Φ_K` a function of the base cycle word, so flipping the single transient bit `w_{T(K-1)}(K)` flips `D(K)` and changes no cycle word below `K`. This refutes exactly the class of descriptions whose state is a **fixed-width window of cycle words of coordinates below `K`, read from phase `T(K-1)`**. It does **not** rule out a machine reading other data — transient values, a differently-phased window, or data from above `K`. |

## C-13 — "no witness found" is not UNSAT

| | |
|---|---|
| **Original (a standing temptation, never actually written as UNSAT)** | — |
| **Standing rule, restated** | A bounded orbit scan that finds no witness is reported as "**no witness found in the scanned range**". It is called **UNSAT** only when a formal finite formula was solved and the solver returned unsatisfiable — which in this corpus happens exactly twice: the Proposition 6 control (where UNSAT is *predicted* by a hand proof) and the Model B sweep outside its 122 counterexamples. Phase 2A's `BO-01` ("no new witness for `a ≥ 32`") is a **data-limited scan**, not UNSAT: at `N = 200001, p ≤ 20000` only **1** position even satisfies the hypotheses at `a = 32`, and **0** at `a ≥ 36`. |

## C-14 — F3 is vacuous

| | |
|---|---|
| **Original (Phase 2E registry, brief's F3)** | "Eventual periodicity of `c_t` forces eventual periodicity of the reset schedule along the diagonal" — offered as a target lemma |
| **Corrected** | By **T21.10** the diagonal's reset schedule **is** the centre column shifted by one. F3 therefore reads "(H) ⟹ (H)": true, and carrying no information. Filed as `VQ-01`. It is **not** a step towards anything |

## C-15 — F5 is Problem 1 restated

| | |
|---|---|
| **Original (brief's F5)** | "Two equal diagonal tails force equality of two distinct prefix states, contradicting minimality of `T(K)`" — offered as a target lemma |
| **First assessment during Phase 2E** | "REFUTED AS STATED — the implication fails at its first step" |
| **Corrected assessment (this phase, audit A-07)** | That was too strong: the implication was not shown to fail. What is true is that F5's **conclusion is impossible outright** — for a deterministic map, `W_t^K = W_{t'}^K` with `t < t'` forces `T(K) ≤ t`, so the requested configuration `t < T(K) ≤ t'` cannot occur. Hence F5 reads "(H) ⟹ FALSE", i.e. **F5 is `¬(H)`**, i.e. Problem 1 restated. Filed as `VQ-02`, not as a refutation. Its natural intermediate step ("equal value ⟹ equal derivation") **is** separately refuted — that is `RF-05` |

## C-16 — the `T15` indexing error caught during this audit

| | |
|---|---|
| **The error** | while re-deriving T15 (Phase 2D N2), the auditor tested `w_t(m-2) = w_t(m-1)` at `m ∈ {3,8,29,400}` — the non-COLLAPSING **levels** — and it failed at all four |
| **Resolution** | the theorem's hypothesis is about **coordinate `m` being zero on its cycle**, and the eventually-zero *coordinates* are `{2,7,28,399}`. Re-tested there, the identity **holds at all four**, with cycle parity 1 in every case. **Phase 2D was right; the auditor's first reading was wrong.** |
| **Kept because** | the off-by-one between "eventually-zero coordinate `k`" and "non-COLLAPSING level `k+1`" is exactly the kind of slip this corpus is most exposed to. An indexing warning is now attached to T15 and to `DEFINITIONS_AND_NOTATION.md` §6 |

## C-17 — the suffix-minimum `(T,P)` algorithm

| | |
|---|---|
| **Original (Phase 2C, first attempt)** | an exact-`(T,P)` algorithm based on suffix minima |
| **What happened** | an empty-suffix sentinel made short ranges succeed spuriously, giving `T(K) = T_max, P(K) = 1` for `K ≥ 4` |
| **Found by** | immediate disagreement with the hash-based reference implementation |
| **Corrected** | replaced by the first-repeat criterion (`P(K) = min{Q : ∃t, W_t^K = W_{t+Q}^K}`), which is exact by determinism. The failed method is **preserved** in the Phase 2C record |

## C-18 — literature was never reconstructed

| | |
|---|---|
| **Original (Phase 2B brief item 6)** | reconstruct the relevant literature |
| **Outcome** | **not completed.** Every primary source attempted returned **HTTP 403** through this environment's proxy: arXiv:2202.13809, its TCS version, Jen 1990, oeis.org, the Semantic Scholar API. `LITERATURE_GAP_MAP.md` labels every attributed statement **UNVERIFIED — SNIPPET ONLY** |
| **Standing rule** | no literature claim is reconstructed from snippets as though verified, and **nothing in the corpus depends on one** |

---

## Corrections by phase of origin

| phase | corrections originating there | caught by |
|---|---|---|
| Phase 1 | C-01, C-02, C-17(precursor) | its own tests (C-02), a longer run (C-01) |
| Phase 2A | C-03, C-04 | its own test battery |
| Phase 2B | C-05, C-06, C-08, C-18 | Phase 2C (C-06), Phase 2E (C-05), this phase (C-08) |
| Phase 2C | C-05, C-07, C-17 | Phase 2D/2E (C-07), a reference implementation (C-17) |
| Phase 2D | C-09, C-10, C-11 | this phase (C-09), its own README (C-10, C-11) |
| Phase 2E | C-12, C-14, C-15 | this phase |
| Phase 2F | C-16 | this audit, on itself |

**Seven of the eighteen corrections were caught by a later phase, and three by
this consolidation.** That is the argument for doing consolidation at all: the
error rate in the corpus is not zero, and the corrections found here (C-08,
C-09, C-12, C-15) were all present in shipped, tested packages.
