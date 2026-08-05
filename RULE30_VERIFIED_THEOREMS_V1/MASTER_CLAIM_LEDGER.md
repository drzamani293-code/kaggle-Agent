# Master Claim Ledger

Every substantive claim in the corpus appears in **exactly one** category.

```
A  FORMALLY PROVED THEOREM                   T-nn
B  FORMALLY PROVED CONDITIONAL THEOREM       CT-nn
C  FINITE COMPUTATIONAL CERTIFICATE          CC-nn
D  BOUNDED OBSERVATION                       BO-nn
E  CONJECTURE                                CJ-nn
F  REFUTED CLAIM                             RF-nn
G  VACUOUS OR EQUIVALENT-TO-TARGET CLAIM     VQ-nn
H  OPEN BRIDGE                               BR-nn
```

**No claim is in category A because it passed tests.** Category A requires a
complete proof in `VERIFIED_THEOREMS.md` that has been reconstructed
independently in `PROOF_AUDIT_REPORT.md`. Computational agreement is recorded
in the *external verification* column and never promotes a claim.

Column key — **Seed-free?** `yes` = the proof holds for every configuration
(ANY) or every finitely supported one (FIN); `no` = it uses the single-cell
orbit's actual values. **Region** — which part of the object the claim is
about: `local` (patches/rule), `orbit` (the real space-time diagram), `cycle`
(the eventually periodic region of the prefix tower), `transient`, `diagonal`.

---

## A. Formally proved theorems

| ID | Statement (quantified) | Assumptions | Proof source | Comp. deps | Seed-free? | Region | Limitations | External verification |
|---|---|---|---|---|---|---|---|---|
| **T-01** | `∀ c,r`: `l ↦ l XOR (c OR r)` is a bijection; `∀` config `∀t,j`: `x_t(j-1) = x_{t+1}(j) XOR (x_t(j) OR x_t(j+1))`; `f` is not right-permutive | none | VT T1 | none | **yes** (ANY) | local | none | exhaustive over 8 neighbourhoods; 202 000 space-time points re-simulated |
| **T-02** | `∀` finitely supported `x_0`, `∄ j`: `col_j` and `col_{j+1}` both eventually periodic | finite support | VT T2 | none | **yes** (FIN) | orbit | holds for every FIN seed ⇒ cannot decide Problem 1 | proof-only; mechanism re-simulated on columns −1…−200 |
| **T-03** | `∀` finitely supported `x_0`: `#{ j : col_j eventually periodic } ≤ 1` | finite support | VT T3 | none | **yes** (FIN) | orbit | the centre may be that one column | proof-only |
| **T-04** | `∀` sequence `u` eventually periodic with preperiod `T`, period `p`, `∀n ≥ 1`: `u` has at most `T+p` distinct factors of length `n` | none | VT T4 (lemma part) | none | **yes** (any sequence) | — | the *bound* `T+p ≥ 998140` is **not** here — it needs `CC-02` and is `CT-01` | proof-only |
| **T-05** | `∀p ∀`config`∀t,j`: the exact defect step equation (XOR/OR and ANF forms) | none | VT T5 | none | **yes** (ANY) | local | an identity; constrains nothing alone | exhaustive over all 64 assignments; re-checked cell-by-cell on the orbit at 5 lags |
| **T-06** | W1/W2/W3/W6 in their strongest proved forms, each on a stated finite range | a wall on that range | VT T6 | none | **yes** (ANY) | orbit | hypotheses are met only on short ranges in reality (longest real wall 18) | each checked wherever its hypothesis occurs, 5 lags |
| **T-07** | `∀`config`∀t,k`: `w_{t+1}(k) = w_t(k-2) XOR (w_t(k-1) OR w_t(k))` | none | VT T7 | none | **yes** (ANY) | local | a change of variables | cell-by-cell vs the original engine |
| **T-08** | `∀K ∀t ≥ 0`: `W_{t+1}^K = F_K(W_t^K)`; hence `(W_t^K)` eventually periodic with `T(K)+P(K) ≤ 2^{K+1}` | left boundary | VT T8 | none | **yes** | cycle+transient | says nothing about growing `K` | verified at `K = 40`, all `t ∈ [0,78]`; `T+P ≤ 2^{K+1}` for `K ≤ 20` |
| **T-09** | `∀K ∀z`: `π_K(F_{K+1}(z)) = F_K(π_K(z))`; hence `T(K-1) ≤ T(K)` and `P(K-1) | P(K)` | none | VT T9 | none | **yes** | cycle | organises fixed levels only | monotonicity: 0 violations, `K ≤ 30000` |
| **T-10** | `∀K ≥ 2`: the one-period fibre map is constant / identity / negation (COLLAPSING / NEUTRAL / DOUBLING) | base settled | VT T10 | none | **yes** (any base) | cycle | post-settling only | classification matches the measured period at every `K ≤ 30000` |
| **T-11** | `∀K`: `P(K) ∈ {P(K-1), 2P(K-1)}`; `P(K) = 2^{m(K)}`, `m` non-decreasing with steps in `{0,1}` | — | VT T10.1 | none | **yes** | cycle | — | 0 counterexamples, `K ≤ 30000` |
| **T-12** | `∀K ≥ 1`: `T(K-1) ≤ T(K)` | — | VT T9.1 | none | **yes** | transient | — | 0 violations, `K ≤ 30000` |
| **T-13** | `T(K) ≤ T(K-1)+P(K-1)`; if COLLAPSING, `T(K) ≤ τ(K)+1`; exactly, T-18 | base settled | VT T13 | none | **yes** | transient | all **upper** bounds; the open problem needs a lower bound | 0 violations, `K ≤ 3000` (S2) and `K ≤ 30000` (R3) |
| **T-14** | `∀K`: level `K` not COLLAPSING **iff** coordinate `K-1` is zero throughout its cycle | — | VT T14 | none | **yes** | cycle | — | consistent at all `K ≤ 30000` |
| **T-15** | If coordinate `m` is zero on its cycle then `w_t(m-2) = w_t(m-1)` for `t` on that cycle | — | VT T15 | none | **yes** | cycle | **index is the coordinate, not the level** | holds at all four computed opportunities `m ∈ {2,7,28,399}` |
| **T-16** | R1 erasure `u_{τ+1} = 1 XOR a_τ`; R2 the constant `Φ_K`; R3 `T(K) ≤ τ(K)+1` | COLLAPSING, base settled | VT T16 | none | **yes** | cycle | — | 0 mismatches / 2995 levels |
| **T-17** | The cycle word `C_K` is the unique `P(K)`-periodic solution driven by `C_{K-2}, C_{K-1}`; hence a `2P`-bit transducer; a chain `> 2^{2P}` forces `K`-periodicity | COLLAPSING | VT T17 | none | **yes** | cycle | **T17.2's hypothesis has never been met** (needs 4.3×10⁹ levels vs 29 600) | 2598/2598 unique |
| **T-18** | The exact recurrence for `T(K)` via the defect bit `D(K)`, with the DOUBLING and NEUTRAL cases; and the telescoping identity | base settled | VT T18 | none | **yes** | transient | describes the frontier, not the diagonal's values | 29999/29999, 0 mismatches |
| **T-19** | If `T(K) > T(K-1)` then `T(K) = ρ(K)+1 = τ(K)+1`; in general `T(K) = ρ(K)+1` iff `w_{T(K)-1}(K-1) = 1` | `ρ` defined | VT T19 | none | **yes** | transient | on inheriting levels the equality is a coincidence (47.6 %) | 16394/16394 |
| **T-20** | `∀t`: `|Sk(t)| ≥ ⌊t/2⌋ + 1` (XOR spine) | — | VT T20 | none | spine argument **yes**; the skeleton itself **no** | diagonal | unbounded size is **not** evidence of aperiodicity | spine contained at every `t` tested |
| **T-21** | Auxiliary theorems T21.1–T21.12: coordinate decomposition of `T`; lag monotonicity; the support lower bound; distinctness of `σ,τ,ρ`; `D(K) = w_{T(K-1)}(K) XOR Φ_K`; the strip identity; no strip is autonomous; the strip automaton excludes nothing at every radius; strip lookahead exactly `R`; the diagonal's reset schedule is the centre column; Phase 2A's Lemma T and Model A soundness | as stated per item | VT T21.* | none | mixed | mixed | T21.5's scope is restricted (C-12) | per item; see VT |
| **T-22** | `∀t`: `w_t(7) = 0` | SEED | VT T22 | none | **no** | orbit | **uniqueness of 7 is NOT claimed** | explicit 2-cycle, hand-checkable |
| **T-23** | `∀t ∀s ≤ t`: the ancestors of `w_t(t)` at time `s` are exactly `[max(0,2s-t), min(t,2s)]` | — | VT T23 | none | **yes** | diagonal | `Θ(t²)` cells | 0 violations |

## B. Formally proved conditional theorems

| ID | Statement | Assumptions | Proof source | Comp. deps | Seed-free? | Region | Limitations | External verification |
|---|---|---|---|---|---|---|---|---|
| **CT-01** | If the centre column is eventually periodic with preperiod `T`, period `p`, then `T + p ≥ 998140` | the factor-counting lemma **and** certificate `CC-02` | VT T4 | **CC-02** | no | orbit | the method's ceiling is the length of the computed prefix; it can never prove aperiodicity | factor count by 3 independent algorithms; sequence by 4 independent engines |
| **CT-02** | If the centre column is eventually periodic with `p ≤ 20000` then `T > 979998` | certificate `CC-03` | Phase 1 D2 | **CC-03** | no | orbit | only `p ≤ 20000` | fast big-integer scan cross-checked against a naive `O(Np)` scan for `p ≤ 64` |
| **CT-03** | If `d_t(0) = 0` for all `t ≥ T` (the wall), the consequences T6.1–T6.4 hold globally | (H) | VT T6 | none | yes | orbit | **vacuous if Problem 1 has the expected answer** | untestable by construction |
| **CT-04** | Under (H), for every `r ≠ 0` the column `x_t(r)` is **not** eventually periodic | (H) + T-03 | Phase 2E Thm 8.4 | none | no | orbit | conditional on a hypothesis believed false | proof-only |
| **CT-05** | Under (H) with period `p`, on `{t ≥ T* : x_t(0)=0}` column `+1` is determined by column `−1` and periodic data | (H) | Phase 2E Cor. 8.5 (= Phase 1 Prop. 6) | none | yes | orbit | this **is** Phase 1's Proposition 6; no new content | proof-only |
| **CT-06** | If coordinate `K` is not eventually zero then `T(K+1) - T(K) ≤ P(K)` | non-eventually-zero, i.e. **BO-04** on the computed range | Phase 2E Thm 3.4 | none | yes | transient | the hypothesis is a bounded observation beyond `K = 30000` | 0 violations; max observed step 15 vs bound 16 |

## C. Finite computational certificates

Full details, hashes and reproduction commands in
`COMPUTATIONAL_CERTIFICATES.md`.

| ID | What is certified | Range | Independent implementations |
|---|---|---|---|
| **CC-01** | the first `10⁶` (and `200001`) bits of the centre column, with SHA-256 | `t < 10⁶` | 4 engines (list+table, numpy+table, big-integer, rule-86 mirror) |
| **CC-02** | that prefix has exactly **998140** distinct factors of length 28 | `n = 28`, `N = 10⁶` | 3 algorithms (hash-set, numpy sort-unique, string-set) |
| **CC-03** | every candidate period `p ≤ 20000` fails within the first `10⁶` bits, and `p ≤ 10000` fails within the first `200001` | as stated | fast scan + naive `O(Np)` scan for `p ≤ 64` |
| **CC-04** | eight explicit real-orbit MB1-loc witnesses at `a = 0,4,…,28` | `N ≤ 10⁶` | re-simulated twice, in separate processes |
| **CC-05** | Model A is SAT on all **832** instances `a ∈ {0,…,48} × p ∈ [1,64]` | as stated | 6 PySAT configurations (2 encodings × 3 solvers) + Z3 for `p ≤ 12` + solver-free enumeration on 18 small instances |
| **CC-06** | zero-wall automaton: `\|Inv(r)\|` = 16, 128, 992, 7616, 59136 for `r = 1..5` | `r ≤ 5` | exhaustive; `r ≥ 6` **not computed** |
| **CC-07** | exact `T(K), P(K)` for `K ≤ 30000` | as stated | 3 algorithms (first-repeat two-pointer, hash reference, incremental skew-product) + full periodicity/minimality verification at sampled `K` |
| **CC-08** | collapse-chain classification of every level `K ≤ 30000` | as stated | orbit re-derivation + stored records |
| **CC-09** | Phase 2E transient classification of every level (2-way and 5-way) with `σ, τ, ρ, D` | `K ≤ 30000` | two independent derivations per level; 0 disagreements |
| **CC-10** | strip-lookahead counterexamples: same radius-`R` state, different `q_{t+R+1}(0)`, for `R = 1,2,3,4,6` | `t ≤ 20000` | explicit witness pairs |
| **CC-11** | the five-way classification counts: INHERITED 13601, RESET_IMMEDIATE 4386, PHASE_DELAY 12008, PERIOD_DOUBLING 4, ZERO_PREDECESSOR 0 | `K ≤ 30000` | as CC-09 |
| **CC-12** | the centre column is not determined by its own last `m` bits, `m ≤ 20`, with explicit witness pairs | `t ≤ 50000` | explicit witnesses |

## D. Bounded observations

| ID | Observation | Range | Why it is not a theorem |
|---|---|---|---|
| **BO-01** | no MB1-loc witness for `a ≥ 32` | `N = 200001`, `p ≤ 20000` | **data-limited**: only 1 position even satisfies the hypotheses at `a = 32`, 0 at `a ≥ 36` |
| **BO-02** | the zero-wall automaton's invariant set is non-empty at every computed radius | `r ≤ 5` | `r ≥ 6` not computed; growth `≈ C·7.7^r` is itself a conjecture |
| **BO-03** | `T(K) > K` | `18 ≤ K ≤ 30000` | **false for `K ≤ 17`**; no proof for large `K` |
| **BO-04** | the eventually-zero coordinates are exactly `{2,7,28,399}`; the doubling points exactly `{3,8,29,400}`; no NEUTRAL level occurs; every level in `[401,30000]` is COLLAPSING | `K ≤ 30000` | complete **only in the computed range**. The NEUTRAL evidence is **four opportunities**, not 30 000 confirmations |
| **BO-05** | the diagonal's backward cone is entirely transient from `s ≈ 0.79–0.83 t`; ≈69 % of it never settles | `t ≤ 3000` | four samples, non-monotone, no constant fitted |
| **BO-06** | diagonal ANF monomial counts `1,3,6,10,19,44,81,150,256,545,1100,2183,4297,8701` | `t ≤ 13` | growth is not extrapolated |
| **BO-07** | reset erasure + periodicity rewrite reduce the diagonal's derivation by ≈2.1×; DAG `≈ 0.24–0.35 t²` | `t ≤ 3000` | constants measured at stated `t`, not fitted |
| **BO-08** | the regular left region is invariant under `(t,i) → (t+16,i−16)` out to a band `≈ 0.74 t` | `t ≤ 34000` | **slope has not converged** (0.7645 at `t=2000` → 0.7444 at `t=34000`); no constant claimed |
| **BO-09** | `T(K)/K = 1.3399` at `K = 30000`, and `T(K)/K` is **not monotone** (1.3400 at `K=100`, 1.2525 at `K=400`) | `K ≤ 30000` | no limit is claimed to exist |
| **BO-10** | resetting density `0.546485`, mean increment `2.451873`, product `1.339911` | `K ≤ 30000` | measured at one `K`; the factorisation (T-18) is exact, the numbers are not extrapolated |
| **BO-11** | the maximal run of consecutive INHERITING levels is 12; the maximal `T(K+1)-T(K)` is 15 | `K ≤ 30000` | nothing bounds these for larger `K` |
| **BO-12** | the longest real zero wall over 83 lags × 30000 steps is **18**, at `p = 59` | as stated | — |

## E. Conjectures

| ID | Conjecture | Evidence | Falsification test |
|---|---|---|---|
| **CJ-01** | Model A is satisfiable for every `a ≥ 0`, `p ≥ 1` | 832/832 | extend the grid; a single UNSAT refutes it **and** is a genuine theorem about MB1-loc |
| **CJ-02** | `liminf_K T(K)/K > 1` (Phase 2B G5); equivalently `liminf d(K)·m(K) > 1` (Phase 2E F1′) | `BO-09`, `BO-10` | a single `K > 17` with `T(K) ≤ K` |
| **CJ-03** | a positive density of levels has `D(K) = 1` (Phase 2E F2′) | 0.546485 at `K = 30000`; longest `D=0` run is 12 | an unbounded run of `D = 0` |
| **CJ-04** | the RESETTING/INHERITING word is not eventually periodic in `K` (Phase 2E F3′) | no repeated block of length 30 in `K ≤ 30000` | — |
| **CJ-05** | every level `K > 400` is COLLAPSING (Phase 2E F4′) | `BO-04` — but only **four** opportunities existed | a fifth eventually-zero coordinate |
| **CJ-06** | NEUTRAL never occurs in the real prefix tower | four opportunities, four odd parities | a NEUTRAL level |
| **CJ-07** | `\|Inv(r)\| ≈ C·7.7^r` for the zero-wall automaton | `r ≤ 5` | compute `r = 6` |

## F. Refuted claims

| ID | Claim, as originally made | How it was refuted | Kept because |
|---|---|---|---|
| **RF-01** | the support-edge defects obstruct the wall (Phase 2B G2) | the edges sit at `±(t+p)` and recede from column 0 at speed 1; the geometry sweep found no interaction | it is the natural first idea and must not be re-attempted |
| **RF-02** | a wall forces a closed finite-state cycle incompatible with the seed (Phase 2B G4) | the automaton is **nondeterministic**, so an infinite wall gives a path, not a cycle; and the automaton is seed-blind by construction | ditto |
| **RF-03** | collapse corresponds to variable elimination in the free-variable ANF (Phase 2D) | measured: it does not | ditto |
| **RF-04** | there is a recursive family of proof DAGs at `2^n`, `2^n−1`, doubling points, `T(K)` | ratios `4.582, 4.393, 3.680` and differences `4030, 17492, 60704` — neither constant | only two candidate shapes were tested; **absence is not a proof that none exists** |
| **RF-05** | equal centre value ⟹ equal derivation | `t = 300` and `t = 329` both give `c = 0` with DAGs of 28 553 and 33 843 nodes | it is the natural intermediate step for `VQ-02` |
| **RF-06** | "every MB1-loc witness has `a_t(1)=0, a_{t+p}(1)=1`" (Phase 1 draft) | false at `a = 16, 20` in the 10⁶-bit run (polarity `1→0`) | the refutations themselves are unaffected; see C-02 |
| **RF-07** | "the backward-transfer probability saturates around 0.755, not at 1" (Phase 1 draft) | still rising at `a = 28` (0.9583) | see C-01 |

## G. Vacuous or equivalent-to-target claims

| ID | Claim | Why it is in this category |
|---|---|---|
| **VQ-01** | (H) forces the diagonal's reset schedule to be eventually periodic (Phase 2E F3) | by **T21.10** the diagonal's reset schedule **is** the centre column shifted by one, so this reads "(H) ⟹ (H)" |
| **VQ-02** | equal diagonal tails force equal prefix states, contradicting minimality of `T(K)` (Phase 2E F5) | its conclusion is impossible for a deterministic map (a repeat at `t < t'` forces `T(K) ≤ t`), so the statement **is** `¬(H)`, i.e. Problem 1 |
| **VQ-03** | MB1 (Phase 1 registry C3) | vacuously true if Problem 1 has the expected answer; **untestable by computation** |
| **VQ-04** | every defect component eventually meets every width-2 strip (Phase 2B G3) | its weakest useful form is *equivalent* to `¬(H)`; circular as stated |
| **VQ-05** | Phase 2C Theorem D1, `x_{t+P(K)}(0) = x_t(P(K))` on `T(K) ≤ t ≤ K-P(K)` | that range is **empty** on the real orbit for every `p ≥ 7`; the theorem is true and carries no information |

## H. The open bridge

| ID | Statement | Equivalent formulations | Status |
|---|---|---|---|
| **BR-01** | **The Bridge.** If the centre column is eventually periodic then some **second** column is eventually periodic. | Phase 1 C2 (MB1: `col_{−1}` or `col_1` eventually periodic); Phase 2B G1 (a zero wall forces a second periodic column); Phase 2E F4 (a periodic reset schedule forces a second fixed periodic column) — all the same statement | **OPEN.** With **T-03** it would settle Problem 1 immediately. Untouched by Phases 2A, 2B, 2C, 2D, 2E. |

**Why BR-01 is hard, in one paragraph.** By **T-03** the centre column may be
the unique eventually periodic column, so no argument that is blind to the seed
can close the gap (Phase 1 §10.3, and `T-02`/`T-03` hold for every finitely
supported seed). Every seed-aware structure the corpus has found — the prefix
tower, the collapse chain, the exact frontier recurrence — lives in the
**cycle** region or describes the **frontier**, while the centre column is the
**diagonal**, which by `BO-03` never enters the cycle region on the computed
range. The two halves of the corpus do not meet.

---

## Counts

| category | count |
|---|---|
| A — proved theorems | 23 (with ~35 sub-results) |
| B — conditional theorems | 6 |
| C — certificates | 12 |
| D — bounded observations | 12 |
| E — conjectures | 7 |
| F — refuted | 7 |
| G — vacuous / equivalent | 5 |
| H — open bridge | 1 |

**Claims explicitly NOT made:** that the centre column is aperiodic; that any
finite computation supports aperiodicity; that MB1 or MB1-loc is true for any
`a`; that any bounded search returned UNSAT in the mathematical sense; that any
result here is novel (see `NOVELTY_STATUS.md`); that Problem 1 is solved or
partially solved.
