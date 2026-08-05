# Source Manifest

Every Rule 30 result present in the repository, traced to its origin, with its
original wording and its final status. Nothing is silently repaired: where a
statement changed, both versions appear, here or in
`CORRECTIONS_AND_RETRACTIONS.md`.

---

## 0. Packages ingested

| package | files | ZIP SHA-256 | tests |
|---|---|---|---|
| `RULE30_PHASE1_HANDOFF` | 38 | `1556689077c7534300057f34a2e1d7dcb4c2c106f8cb0f12a70d91e284919c33` | 51/51 |
| `PHASE2A_SAT_MB1LOC` | 56 | `7ffac2a0a10fe2d629888046cc8a6699323894a2a1511082d6662f51ed2f9294` | 37/37 |
| `PHASE2B_GLOBAL_DEFECT` | 21 | `d7f7132b1397450c631018d0ce603589c306bc82d87c0ee162df7357b50af2cc`* | 60/60 |
| `PHASE2C_EDGE_PREFIX` | 15 | `17fe1b30e71e647e44c59ddb63b075c3b240334828fbcac3e735825abd921fe3` | 35/35 |
| `PHASE2D_COLLAPSE_CHAIN` | 17 | `d391ec21a93721171350ba2918012dcfca5255d4ac33fc7de7f1b0f68a1e54b5` | 34/34 |
| `PHASE2E_TRANSIENT_FRONTIER` | 21 + results | `0ad0112cfa7fd81228284896f54f08e6c5c56523c6c70802bf4312286e26c566` | 100/100 |

\* The Phase 2B hash is transcribed from the delivery record; the test suite
in this package re-computes all six from the ZIPs on disk and reports any
mismatch rather than trusting this table.

Also present at repository root, predating the packages: `PHASE1_AUDIT.md`,
`WIDTH2_PROOF_RECONSTRUCTION.md`, `CONJECTURE_REGISTRY.md`,
`FORMAL_PROOF_AT_MOST_ONE_COLUMN.md`, `FACTOR_COMPLEXITY_BOUND.md`,
`CLAIM_LEDGER.md`, `PROOF_DEPENDENCY_GRAPH.md`, `REPRODUCE_ALL.md`,
`research_notes.md`, and the Phase 1 scripts. These are the same documents that
the Phase 1 handoff package contains.

---

## 1. Phase 1 — foundations

| origin | original statement (abridged, as written) | later fate | final ID / status |
|---|---|---|---|
| `PHASE1_AUDIT.md` §0 | "the repository contained no Rule 30 file in any commit on any branch"; the three files the brief referenced never existed | unchanged | meta, not a mathematical claim |
| `rule30_lab.py` `verify_local_rule` | rule 30 table `(0,1,1,1,1,0,0,0)` equals `l XOR (c OR r)` on all 8 neighbourhoods | unchanged | basis of everything; **unverified against a primary source** (see audit D) |
| `FORMAL_PROOF...` F3/F4 | rule 30 is left-permutive and not right-permutive | unchanged | **T1** |
| `FORMAL_PROOF...` Lemma 1 | `a(t,i−1) = a(t+1,i) XOR (a(t,i) OR a(t,i+1))` | unchanged | **T1(b)** |
| `FORMAL_PROOF...` Lemma 2 | two eventually periodic sequences are both `(max T, lcm p)`-periodic | unchanged | **T2.3** |
| `FORMAL_PROOF...` Lemma 3 | leftward propagation **with the same `T` and `p`** | unchanged | **T2.4** |
| `FORMAL_PROOF...` Thm W2 | no two adjacent columns are both eventually periodic | unchanged | **T2** |
| `FORMAL_PROOF...` Lemma 4/5 | boundary forcing; interior inherits EP with period `≤ 2^{j−i−1}·p` | unchanged | **T3.1 / T3.2** |
| `FORMAL_PROOF...` Thm W2′ | at most one column of the diagram is eventually periodic | unchanged | **T3** |
| `WIDTH2_...` Prop. 6 | if `col_0` is `(T,p)`-periodic then `a_{t+p}(−1) XOR a_t(−1) = [a_t(0)=0]·(a_t(1) XOR a_{t+p}(1))` | restated in defect variables as Phase 2B W1 | **T6.1** |
| `FACTOR_COMPLEXITY_BOUND.md` Lemma A | an eventually periodic sequence has `≤ T+p` factors of each length | unchanged | **T4** (lemma part) |
| `PHASE1_AUDIT.md` §7 Lemma B | such a sequence satisfies a GF(2) recurrence with char. poly `X^{T+p}+X^T` | unchanged, but its only consequence (`T+p ≥ 100001`) is strictly weaker than T4 | retained, unused |
| Phase 1 D1 | "if the centre column is eventually periodic then `T+p ≥ 998140`" | **downgraded to conditional** in this phase (audit A-02) | **CT-01** + `CC-02` |
| Phase 1 D2 | "if eventually periodic with `p ≤ 20000` then `T > 979998`" | unchanged | **CC-03** |
| Phase 1 D5 | MB1-loc(`a`) is false for `a = 0,4,…,28` with explicit witnesses | unchanged; **`a ≥ 32` untested** | **CC-04** |
| Phase 1 E1 | MB1 implies Problem 1 | unchanged | **BR-01** (the Bridge) |
| Phase 1 E2 | MB1 is vacuously true if Problem 1 has the expected answer, hence untestable | unchanged | **VQ-03** |
| Phase 1 E4 | W2/W2′ hold for every finitely supported seed, so cannot distinguish the single-cell seed | unchanged | **seed-blindness obstruction**; governs every later phase |
| Phase 1 F7 | left region invariant under `(t,i) → (t+16, i−16)` to a band `≈ 0.74t` | slope **not converged**; no constant claimed | **BO-08** |
| Phase 1 F9/F10 | the backward-transfer probability rises with look-back; its limit is **undetermined** | an earlier draft claimed it "saturates around 0.755" — **retracted** | **C-01** |
| Phase 1 H9 | "every MB1-loc witness has `a_t(1)=0, a_{t+p}(1)=1`" | **retracted**: false at `a = 16, 20` in the 10⁶-bit run (polarity `1→0`) | **C-02** |

## 2. Phase 2A — SAT/SMT for MB1-loc

| origin | original statement | later fate | final ID / status |
|---|---|---|---|
| `SAT_MODEL_SPECIFICATION.md` Lemma T | Model A satisfiability is independent of the base time `t` | unchanged | **T21.11** |
| ibid., Theorem A-sound | a Model A patch refutes any seed-free local proof of MB1-loc at that `(a,p)` | unchanged | **T21.12** |
| ibid., Theorem A-incomplete | a Model A patch is **not** a counterexample to MB1-loc | unchanged; the asymmetry is the whole point | **T21.12** note |
| `PHASE2A_RESULTS.md` §1 | Model A satisfiable for all 832 instances `a ∈ {0,…,48}, p ∈ [1,64]`; no UNSAT | unchanged | **CC-05** |
| ibid., headline 2 | therefore no seed-free local argument at depth `a ≤ 48`, lag `p ≤ 64` can establish MB1-loc | unchanged | **T21.12** applied to `CC-05` |
| ibid., §1.1 conjecture | Model A is satisfiable for every `a, p` | unchanged, unproved | **CJ-01** |
| ibid., headline 7 | "no new witness for `a ≥ 32`" — **data-limited, not method-limited** | unchanged | **BO-01** |
| Phase 2A §5.2 | a `SKIPPED` enumeration verdict was folded into the agreement set, failing a test at `p = 13` | **corrected**: `enumeration_checked` flag added | **C-03** |
| README claim | "Model A rules out every possible local proof" (loose phrasing in an early draft) | **corrected**: it rules out seed-free local proofs at the grid's depths and lags only | **C-04** |

## 3. Phase 2B — global defect field

| origin | original statement | later fate | final ID / status |
|---|---|---|---|
| `DEFECT_DYNAMICS.md` | the exact defect step equation, XOR/OR and ANF forms | unchanged | **T5** |
| ibid. | a black centre cell with no defect masks the defect to its right | unchanged | **T5.1** |
| `ZERO_WALL_LEMMAS.md` W1 | `d_t(-1) = d_t(1)·(1 XOR x_t(0))` | ranges made explicit | **T6.1** |
| ibid. W2 | the wall makes the right half-plane's defect dynamics closed | ranges made explicit | **T6.2** |
| ibid. W3 | a run of `k` ones forces a zero triangle of height `k` | ranges made explicit | **T6.3** |
| ibid. W6 | runs of zeros: `d_s(-1) = d_s(1)` | ranges made explicit | **T6.4** |
| ibid. W4, W5, W7 | corollaries / summary forms | **not restated as independent theorems** | folded into T6 |
| ibid. D7 | `d_t(±(t+p)) = 1` always — the defect field is never empty | unchanged | **T5.2** |
| `EDGE_ALIGNED_DYNAMICS.md` EA1 | the one-sided recurrence | unchanged | **T7** |
| ibid. EA2 | every fixed prefix is eventually periodic, "for all `t ≥ K/2`" | **hypothesis removed** (audit A-01) | **T8**, correction **C-08** |
| ibid. EA3 | `x_t(0) = w_t(t)` | unchanged | **T21.6** |
| ibid. §5 | "preperiod `T(K) ≈ 1.29K`" | **not an all-`K` theorem**; superseded by exact tables and by the non-monotonicity finding | **C-05** |
| `GLOBAL_LEMMA_REGISTRY.md` G1 | an infinite zero wall forces a second periodic column | unchanged, open | **BR-01** |
| ibid. G2 | the support-edge defects obstruct the wall | **refuted as a route**: the edges recede from column 0 | **RF-01** |
| ibid. G3 | every defect component meets every width-2 strip | **circular as stated** | **VQ-04** |
| ibid. G4 | a wall forces a closed finite-state cycle incompatible with the seed | **first half false as stated** (the automaton is nondeterministic); second half seed-blind | **RF-02** |
| ibid. G5 | `liminf T(K)/K > 1` | unchanged, open; factorised in Phase 2E | **CJ-02** |
| `ZERO_WALL_AUTOMATON.md` | maximal invariant set non-empty at every radius `r ≤ 5` (sizes 16, 128, 992, 7616, 59136) | unchanged; `r ≥ 6` **not computed** | **CC-06**, **BO-02** |
| `LITERATURE_GAP_MAP.md` | every primary source returned HTTP 403 | unchanged | **literature unresolved**; nothing depends on it |
| Phase 2B G5 framing | `T(K)/K > 1` presented as the favourable direction | **retracted** in Phase 2C: `T(K) ≤ K-P` eventually is what would prove aperiodicity via D1 | **C-06** |

## 4. Phase 2C — edge-aligned prefix

| origin | original statement | later fate | final ID / status |
|---|---|---|---|
| `PREFIX_MAP_THEORY.md` P1 | `π_K ∘ F_{K+1} = F_K ∘ π_K` | unchanged | **T9** |
| ibid. P2 | the new coordinate is a one-bit fibre over the `K`-prefix | unchanged | **T10** setup |
| `SKEW_PRODUCT_ANALYSIS.md` S1 | `P(K) ∈ {P(K-1), 2P(K-1)}` | unchanged | **T10.1 / T11** |
| ibid. S2 | `T(K) ≤ T(K-1) + P(K-1)` | superseded (not refuted) by T13.2 and T18 | **T13.1** |
| ibid. S3 | `T(K-1) ≤ T(K)` | unchanged | **T9.1 / T12** |
| ibid. S4 | `T(K)+P(K) > ⌊K/2⌋` if coordinate `K` is ever 1 | unchanged | **T21.3** |
| ibid. | `w_t(7) = 0` for every `t` | unchanged — **proved** via the explicit 2-cycle | **T-perm7**, see ledger `T-22` |
| ibid., fibre classes | COLLAPSING / NEUTRAL / DOUBLING trichotomy | unchanged | **T10** |
| `PREFIX_PERIOD_RESULTS.md` E1–E5 | E1 periods are powers of two; E2 `P(K)` divides and at most doubles; E3 `T` non-decreasing | E1–E3 proved | folded into T10.1, T9.1 |
| ibid. E4 | "`T(K) > K`" | **false for `K ≤ 17`**; holds for `18 ≤ K ≤ 30000` as a bounded observation | **C-07**, **BO-03** |
| ibid. | "`T(K) ≈ 1.34K`" | **not an all-`K` theorem**, and `T(K)/K` is **not monotone** | **C-05** |
| `DIAGONAL_BRIDGE_ATTEMPT.md` D1 | `x_{t+P(K)}(0) = x_t(P(K))` for `T(K) ≤ t ≤ K - P(K)` | **vacuous** on the real orbit for every `p ≥ 7` | **VQ-05** |
| suffix-minimum algorithm | an early exact-`(T,P)` method | **failed** (empty-suffix sentinel bug), replaced by first-repeat | preserved as a failed method |

## 5. Phase 2D — collapse chain

| origin | original statement | later fate | final ID / status |
|---|---|---|---|
| `COLLAPSE_RESET_THEORY.md` R1 | erasure at `τ+1` with value `1 XOR a_τ` | unchanged | **T16.1** |
| ibid. R2 | one-period reset constant `Φ` | unchanged | **T16.2** |
| ibid. R3 | `T(K) ≤ max(T(K-1), τ_K+1)` | **`max` redundant** (audit A-03) | **T16.3**, correction **C-09** |
| `COLLAPSE_CHAIN_THEOREMS.md` CH1 | the cycle word `C_K` is the unique `P`-periodic solution | unchanged | **T17** |
| ibid. CH2 | `2P = 32`-bit transducer; prefix-direction cone of width 2 | unchanged | **T17.1** |
| ibid. CH3 | a chain longer than `2^{2P}` forces `K`-periodicity | **hypothesis never met** (needs `4.3×10⁹` levels) | **T17.2**, labelled |
| `NEUTRAL_EXCLUSION.md` N1 | non-COLLAPSING ⟺ predecessor coordinate eventually zero | unchanged | **T14** |
| ibid. N2 | on a zero cycle, `w_t(m-2) = w_t(m-1)` | unchanged; **indexing warning added** (audit A-04) | **T15** |
| ibid. | "NEUTRAL never occurs" | **four opportunities, four odd parities** — much weaker than it sounds | **BO-04** |
| `DIAGONAL_DEPENDENCY.md` DD1 | ancestors of `(t,t)` at time `s` are `[max(0,2s-t), min(t,2s)]` | unchanged | **T-cone**, ledger `T-23` |
| ibid. | the cone is entirely transient for `s ≳ 0.81t` | bounded | **BO-05** |
| `DIAGONAL_ANF_RESULTS.md` | ANF monomial counts `1,3,…,8701` for `t ≤ 13`; collapse ≠ variable elimination | unchanged | **BO-06**, **RF-03** |
| ibid. | eventually-zero coordinates `{2,7,28,399}`; doubling points `{3,8,29,400}` | **complete only in the computed range** | **BO-04**, **C-10** |
| `COLLAPSE_BRIDGE_REGISTRY.md` C2 | hypothesis verified on the whole range, conclusion still does not follow | unchanged | recorded |

## 6. Phase 2E — transient frontier

| origin | original statement | later fate | final ID / status |
|---|---|---|---|
| `TRANSIENT_DEFINITIONS.md` Thm A | `T(K) = max_{k≤K} r_{P(K)}(k)` | unchanged | **T21.1** |
| ibid. 1.2, 1.4 | `R(K) ≤ T(K)`; lag monotonicity | unchanged | **T21.2** |
| ibid. 4.1 | `R_dep(K) = σ(K)` (the fibre dependence horizon is the first reset) | unchanged | **T21.4** |
| `TRANSIENT_RECURRENCE.md` Lemma 2.1 | defect propagation at a level | unchanged | **T18.1** |
| ibid. B1–B4 | the exact recurrence and the RESETTING/INHERITING dichotomy | unchanged | **T18** |
| ibid. B5 | telescoping; `T(K)/K = d(K)·m(K) + O(1/K)` | unchanged | **T18.3** |
| ibid. §7 | five-way classification | unchanged | **CC-11** |
| `RESET_SCHEDULE_THEORY.md` 3.1 | `T(K) = ρ(K)+1` on resetting levels | unchanged | **T19** |
| ibid. 3.4 | `T(K+1)-T(K) ≤ P(K)` given a non-eventually-zero coordinate | unchanged | **T13**, conditional on `BO-04` |
| `FRONTIER_DYNAMICS.md` C1, C2, C3 | increment law; `D(K) = w_{T(K-1)}(K) XOR Φ_K`; no finite-state description | C3's **scope restricted** (audit A-05) | **T21.5**, correction **C-12** |
| `COMOVING_STRIP_DYNAMICS.md` 5.1 | `q_t(r) = x_t(r)` | unchanged | **T21.6** |
| ibid. 5.4, D1 | no strip is autonomous; cone opens one column per side per step | unchanged | **T21.7** |
| ibid. D2 | the strip automaton's recurrent core is the whole state space, every radius | unchanged | **T21.8** |
| ibid. D4 | a radius-`R` strip determines the centre bit exactly `R` steps ahead | unchanged | **T21.9** |
| ibid. §6.6 | the centre column is not determined by its own last `m` bits, `m ≤ 20` | unchanged; refutations only | **CC-12** |
| `DIAGONAL_RESET_SKELETON.md` E1 | the XOR spine gives `|Sk(t)| ≥ ⌊t/2⌋+1` | unchanged | **T20** |
| `DIAGONAL_PROOF_DAG.md` | reset + periodicity reduce the DAG by only ≈2.1×; no recursive family | unchanged | **BO-07**, **RF-04** |
| `TRANSIENT_PERIODICITY_CONSEQUENCES.md` G1 | the diagonal's reset schedule is the centre column shifted | unchanged | **T21.10** |
| ibid. §5.3 | "equal centre value ⟹ equal derivation" | **refuted** (`t = 300`, `t = 329`) | **RF-05** |
| `TRANSIENT_LEMMA_REGISTRY.md` F1 | the frontier has an exact recursive description | **proved**; finite-state form refuted | **T18** + **T21.5** |
| ibid. F2 | the skeleton has unbounded size | **proved** | **T20** |
| ibid. F3 | (H) forces the diagonal's reset schedule to be eventually periodic | **vacuous** | **VQ-01** |
| ibid. F4 | a periodic reset schedule forces a second periodic column | **open — it is the Bridge** | **BR-01** |
| ibid. F5 | equal diagonal tails force equal prefix states | **equivalent to Problem 1** | **VQ-02** |

---

## 7. What was found to be missing

* **No literature was reconstructed** in any phase. Every primary source
  attempted (arXiv:2202.13809, its TCS version, Jen 1990, oeis.org, Semantic
  Scholar) returned HTTP 403 through this environment's proxy. Nothing in the
  corpus depends on a literature claim, and `NOVELTY_STATUS.md` says so
  explicitly rather than approximating.
* **Wolfram's rule-numbering convention is not verified against a primary
  source.** The only external anchor is a 14-bit centre-column prefix matched
  to a secondary transcription of OEIS A051023.
* **Zero-wall automaton radii `r ≥ 6` were never computed** (`2^{4r+1}` states).
  Recorded as NOT COMPUTED, never as "no result".
