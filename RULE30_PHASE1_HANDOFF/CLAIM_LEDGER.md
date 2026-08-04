# Claim Ledger

One row per substantive claim made anywhere in this package. Line references
are to the files as shipped in `RULE30_PHASE1_HANDOFF/`.

**Type key** — `THM` proved theorem · `LEM` proved lemma · `FACT` one-line
proved fact · `COMP` result of a finite computation · `REF` refutation of a
universally quantified statement by explicit counterexample · `OBS` measurement
over a finite sample · `HEUR` modelling assumption · `CONJ` conjecture ·
`META` statement about the work itself.

**Independent verification key** — `3-alg` three independent algorithms agree ·
`4-impl` four independent implementations agree · `2-proc` re-verified in a
separate clean process · `re-sim` re-verified by re-simulating from scratch ·
`replicated` re-tested on a disjoint sample · `proof-only` no computational
content · `single` one code path only.

---

## A. Provenance and method

| ID | Claim | Type | Support | Assumptions | Independent verification | Known weaknesses |
|---|---|---|---|---|---|---|
| **A1** | The repository contained no Rule 30 file in any commit on any branch before this work; `research_notes.md`, `rule30_lab.py`, `rule30_experiments.ipynb` never existed. | META | `PHASE1_AUDIT.md:8-35` (commands and output) | git history is complete and not rewritten | `single` — the two search commands are shown and can be rerun | If history had been force-pushed before this session, an earlier version could exist upstream and be invisible locally. |
| **A2** | No pre-existing repository file was modified. | META | `git show --stat b42ce46` | — | `2-proc` (`git diff` against the parent commit) | — |
| **A3** | All computations are deterministic; no RNG, no sampling, no parallel non-determinism. | META | `REPRODUCE_ALL.md:5-9`; no `random`/`np.random` import anywhere | — | `single` — verifiable by `grep -rn "random" *.py` | Big-integer performance varies by CPython version; results do not. |

## B. Implementation correctness

| ID | Claim | Type | Support | Assumptions | Independent verification | Known weaknesses |
|---|---|---|---|---|---|---|
| **B1** | The local rule table of rule 30, indexed by `4l+2c+r`, is `(0,1,1,1,1,0,0,0)`, and equals `f(l,c,r) = l XOR (c OR r)` on all 8 neighbourhoods. | COMP (exhaustive) | `rule30_lab.py:233` (`verify_local_rule`), `rule30_lab.py:83` (`rule_table`), `rule30_lab.py:91` (`rule30_formula`) | Wolfram's numbering convention as stated in `PHASE1_AUDIT.md:37-60` | exhaustive over the entire domain — no sampling | If Wolfram's numbering convention were misremembered, the whole project would concern a different rule. Guarded only by A5/B6 (external prefix match). |
| **B2** | Rule 30 is left-permutive and **not** right-permutive. | FACT | `rule30_lab.py:233`; proved `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:88` (F3), `WIDTH2_PROOF_RECONSTRUCTION.md:143` (F4) | B1 | exhaustive | — |
| **B3** | The left–right mirror of rule 30 is rule 86. | COMP (exhaustive) | `rule30_lab.py:96` (`mirror_rule_number`), test `mirror_is_86` | B1 | exhaustive | — |
| **B4** | Four implementations produce identical output: list+table, numpy+table, big-integer bit-parallel, and the rule-86 mirror. | COMP | `rule30_lab.py:322` (`verify_implementations`), `:370` (`verify_mirror`); engines at `:113`, `:148`, `:196` | B1, B3 | `4-impl`; complete rows for `t ≤ 400`, centre column for `t ≤ 5000`, mirror for `t ≤ 2000` | Agreement rules out coding slips, **not** a shared misreading of the definition (see B1 weakness). |
| **B5** | Rows `t = 0..3` over sites `−3..+3` match values computed by hand before any code was run. | COMP | `rule30_lab.py:296` (`HAND_ROWS`), `:306` | — | independent of every line of engine code | Only 4 rows; catches gross errors, not subtle ones. |
| **B6** | The first 14 centre bits are `1,1,0,1,1,1,0,0,1,1,0,0,0,1`, matching the reported first terms of OEIS A051023. | OBS | `PHASE1_AUDIT.md:150-163`; `phase1_results.json: external_reference_check` | the search-snippet transcription is correct | `single` and **weak** | `oeis.org` returned HTTP 403 through the proxy, so this is a **secondary source**, not the b-file. This is the only external anchor and it is 14 bits long. |
| **B7** | Zero-padded finite arrays of radius `steps+2` reproduce the infinite-lattice evolution exactly for all reported cells. | FACT + COMP | proved via light cone `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:72` (F1); tested `rule30_lab.py:422` (`verify_boundary_independence`) | — | `2-proc` — re-run at radius +500, identical | — |
| **B8** | Structural invariants hold: `a_t(−t) = 1`, `a_t(t) = 1`, `a_{t+1}(t) = a_t(t−1) XOR 1`. | FACT + COMP | proved `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:80` (F2); tested `rule30_lab.py:395` | B1 | verified `t ≤ 3000` | — |
| **B9** | All 21 verification checks pass. | COMP | `rule30_lab.py:443` (`run_all_verifications`); `run_all_tests.py:60` | — | `2-proc` | — |
| **B10** | The stored 10⁶-bit sequence has SHA-256 `0bb02e4ed6c3d80bd832eed0b6991cc99f6de01a5f32fa795281d3efb2e9104e`; the 200001-bit sequence `cf018b09db932412d5f42c3d03076181a34d806173fbee749e30c871fe5d994a`. | COMP | `gen_center_column.py:26`; `phase1_results/SHA256SUMS.txt` | — | `2-proc` (`run_all_tests.py` step 2) | — |
| **B11** | The stored sequences are reproduced bit-for-bit by the mirror engine (reversed shifts) and, on a prefix, by the numpy engine. | COMP | `verify_sequence_independent.py:46` (mirror engine), `:88` (numpy) | B3 and the mirror-symmetry of the seed | `4-impl` | numpy cross-check covers only a prefix (20000–50000 steps), being `O(N²)` in cells. |

## C. Proved mathematics

| ID | Claim | Type | Support | Assumptions | Independent verification | Known weaknesses |
|---|---|---|---|---|---|---|
| **C1** | **Lemma 1 (local inversion).** `a(t,i−1) = a(t+1,i) XOR (a(t,i) OR a(t,i+1))` for all `t ∈ N`, `i ∈ Z`. | LEM | proof `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:104`; test `rule30_lab.py:1102` | B1 | `re-sim` at 202,000 space-time points | — |
| **C2** | **Lemma 2 (normalisation).** Two eventually periodic sequences are both `(max T, lcm p)`-periodic. | LEM | proof `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:123` | — | `proof-only` | — |
| **C3** | **Lemma 3 (leftward propagation).** If `col_i`, `col_{i+1}` are `(T,p)`-periodic then so is `col_{i−k}` for every `k`, **with the same `T` and `p`**. | LEM | proof `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:152`; mechanism tested `rule30_lab.py:1124` | C1, C2 | `re-sim` — columns `−1..−200` rebuilt from columns 0,1 | Uniformity of `T` is essential and is the step most likely to be misread; it is proved explicitly at `FORMAL_PROOF...:184`. |
| **C4** | **Theorem W2.** No two adjacent columns are both eventually periodic. | THM | proof `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:190` | C1–C3, F1, F2 | `proof-only` — a proof by contradiction about infinite sequences has no computational content | Uses only the light cone and the frozen left edge, hence holds for *every* finitely supported seed — see E4. |
| **C5** | **Lemma 4 (boundary forcing).** The interior of a strip `[i,j]` evolves as a function of its own state and the two boundary columns. | LEM | proof `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:217`; test `rule30_lab.py:1150` | radius-1 locality of the rule | `re-sim` — interior `−5..5` driven 3000 steps by boundaries `−6, 6` | — |
| **C6** | **Lemma 5 (interior inherits EP).** If `col_i` and `col_j` are eventually periodic then so is every column strictly between them; period `≤ 2^{j−i−1}·p`. | LEM | proof `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:248` | C2, C5, pigeonhole on `Σ^m` | `proof-only` | The period bound is exponential in the gap, blocking any quantitative version. |
| **C7** | **Theorem W2′.** At most one column of the whole space-time diagram is eventually periodic. | THM | proof `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:322` | C4, C6 | `proof-only` | Does **not** decide Problem 1: the centre column may be the unique such column (`FORMAL_PROOF...:341`). |
| **C8** | The proof of W2′ uses no compactness, no König's lemma, no choice, no reversibility of the global map, and no bi-infinite time. | THM (meta, argued) | `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:348-403` | — | `proof-only` | Bi-infinite **space** *is* used (arbitrarily negative sites); §7.3 argues this is eliminable via the light cone but does not carry out the finite rewriting. |
| **C9** | **Proposition 6 / MB1 equivalence.** If `col_0` is `(T,p)`-periodic then, for `t ≥ T`, `a_{t+p}(−1) XOR a_t(−1) = [a_t(0)=0] · (a_t(1) XOR a_{t+p}(1))`; hence `col_{−1}` is `p`-periodic **iff** `col_1` agrees at lag `p` on the zero set of `col_0`. | THM | proof `WIDTH2_PROOF_RECONSTRUCTION.md:452-486` | C1 | `proof-only`; the identity's ingredients are C1 (tested) | The equivalence is for the *same* `p`; `col_{−1}` could in principle be eventually periodic with a different period, which the stated `iff` does not cover (only the sufficient direction is used). |
| **C10** | **Lemma A (factor counting).** An eventually periodic sequence with preperiod `T` and period `p` has at most `T+p` distinct factors of each length. | LEM | proof `FACTOR_COMPLEXITY_BOUND.md:29-70` | — | `proof-only` | — |
| **C11** | **Lemma B (linear recurrence).** Such a sequence satisfies the GF(2) recurrence with characteristic polynomial `X^{T+p} + X^T`, so every prefix has linear complexity `≤ T+p`. | LEM | proof `PHASE1_AUDIT.md:376-389` | — | `proof-only` | Proof is stated compactly; it is elementary but is not written at the level of detail of C10. |

## D. Computational results (rigorous)

| ID | Claim | Type | Support | Assumptions | Independent verification | Known weaknesses |
|---|---|---|---|---|---|---|
| **D1** | If the centre column is eventually periodic then `T + p ≥ 998140`. | COMP + C10 | `FACTOR_COMPLEXITY_BOUND.md`; `verify_factor_bound.py`; `phase1_results/verify_factor_1M.log` | B1, B10, B11, C10 | `3-alg` (hash-set, numpy sort-unique, string-set) **and** `4-impl` on the underlying sequence | Ceiling: this method can never exceed `N−n+1 ≈ N`. Bound scales only linearly with computation. |
| **D2** | If the centre column is eventually periodic with `p ≤ 20000` then `T > 979998`. | COMP | `run_anomaly_followup.py:110`; `anomaly_followup.json: period_scan` | B1, B10 | `2-proc`; fast big-integer scan cross-checked against a naive `O(Np)` scan for `p ≤ 64` (`run_phase1.py:150`) | Covers only `p ≤ 20000`. Says nothing about larger periods. |
| **D3** | If the centre column is eventually periodic then `T + p ≥ 100001` (linear-complexity route). | COMP + C11 | `phase1_results.json: linear_complexity`, `rigorous_lower_bounds` | C11 | `single` — one Berlekamp–Massey implementation | Strictly weaker than D1; retained only because it is a logically independent route. **The BM implementation is not independently cross-checked** — a bug would silently weaken (not invalidate) D1/D2, which do not depend on it. |
| **D4** | Every candidate period `p ≤ 10000` fails within the first 200001 bits, and the largest first mismatch over all such `p` is 11 (at `p = 9988`). | COMP | `phase1_results.json: period_scan` | B1 | `2-proc` + naive cross-check for `p ≤ 64` | — |
| **D5** | **MB1-loc(`a`) is false for `a = 0, 4, 8, 12, 16, 20, 24, 28`**, with the explicit witnesses listed in `PHASE1_AUDIT.md:576-586`. | REF | `find_mb1loc_witnesses.py`; `phase1_results/mb1loc_witnesses.json` | B1 | `re-sim` twice — inside the search script and again by `run_all_tests.py:151` in a separate process | Refutation is per-depth. **`a ≥ 32` is untested** (3 eligible positions in 10⁶ bits), so "false for all `a`" is *not* established. A draft of `PHASE1_AUDIT.md` §8 also mis-stated the witness polarity as uniformly `a_t(1)=0 → a_{t+p}(1)=1`; it is `1 → 0` at `a = 16, 20`. Corrected; see H9. |

## E. Interpretive claims about the gap

| ID | Claim | Type | Support | Assumptions | Independent verification | Known weaknesses |
|---|---|---|---|---|---|---|
| **E1** | MB1 (Registry C3) implies Prize Problem 1. | THM | `WIDTH2_PROOF_RECONSTRUCTION.md:488-503` | C4, C9 | `proof-only` | — |
| **E2** | MB1 cannot be confirmed or refuted by computation: it is vacuously true if Problem 1 has the expected answer. | THM (logic) | `WIDTH2_PROOF_RECONSTRUCTION.md:519-533`; `CONJECTURE_REGISTRY.md:90` | — | `proof-only` | This is the single most important caveat in the package and the easiest to lose in summary. |
| **E3** | No soft/dimension-counting argument can bridge width 1 to width 2: without the seed constraint, the pairs `(col_i, col_{i+1})` consistent with a prescribed `col_i` form a set of cardinality `2^N`. | THM (sketch) | `WIDTH2_PROOF_RECONSTRUCTION.md:392-408` | C1 (leftward determinism defines the orbit from any pair) | `proof-only` | Written as an argument, not a fully formal proof; the bijection is asserted rather than constructed in detail. |
| **E4** | Theorem W2/W2′ holds verbatim for every finitely supported seed, hence cannot distinguish the single-cell seed; any proof of Problem 1 must use more about the seed than F1 and F2. | THM | `WIDTH2_PROOF_RECONSTRUCTION.md:410-428`; `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:395` | C4, C7 | `proof-only` | — |

## F. Measurements (no rigorous status)

| ID | Claim | Type | Support | Assumptions | Independent verification | Known weaknesses |
|---|---|---|---|---|---|---|
| **F1** | Bit frequency is `0.500362` at `N = 200001` and `0.500767` at `N = 10⁶`; no prefix deviates significantly. | OBS | `phase1_results.json: frequencies`; `anomaly_followup.json: global_frequency` | — | `2-proc` | `z`-scores assume an i.i.d. fair coin — a **heuristic null with no justification** for a deterministic sequence. |
| **F2** | Berlekamp–Massey linear complexity is `≈ N/2` at every checkpoint; `L(200001) = 100001`. | OBS | `phase1_results.json: linear_complexity` | — | `single` (see D3 weakness) | One implementation only. |
| **F3** | Every binary word of length `≤ 14` occurs in the first 200001 bits; factor complexity saturates the sample bound from `n = 15`. | OBS | `phase1_results.json: subword_complexity` | — | `3-alg` at `n = 20..28` | — |
| **F4** | The families `c(2^k−1)`, `c(2^k)`, `c(2^k+1)` for `k = 0..17` show no signal (smallest two-sided binomial `p` = 0.238). | OBS | `phase1_results.json: powers_of_two_summary` | fair-coin null | `2-proc` | 18 points per family, 3 families examined. Under-powered by construction; a real effect of moderate size would not be detected. |
| **F5** | Longest runs are 19 zeros and 21 ones in 200001 bits; run-length counts halve cleanly. | OBS | `phase1_results.json: run_lengths` | — | `2-proc` | — |
| **F6** | Autocorrelation at lags 1–32 shows nothing beyond `\|z\| = 2.42` at lag 12. | OBS | `phase1_results.json: autocorrelation` | fair-coin null | `2-proc` | 32 lags examined; one value near 2.4 is expected. |
| **F7** | The regular left region is invariant under the diagonal shift `(t,i) → (t+16, i−16)` out to a band of width `≈ 0.74 t`; `τ = 4` and `τ = 8` give bands of *constant* width 29 and 400. | OBS | `phase1_results.json: left_region`; `rule30_lab.py:737` | — | `single` | Slope has **not converged**: `W(t)/t` falls from 0.7645 at `t = 2000` to 0.7444 at `t = 34000`. No constant should be quoted; proximity to `1/4` may be coincidence. |
| **F8** | `P[col_1 agrees at lag p \| col_0 agrees on [t−a, t+b], a_t(0)=0]` depends on `a` and is flat in `b` (0.5000 for all `b` at `a = 0`). | OBS | `bridge_probe.json: two_sided_table` | — | `2-proc`; and the asymmetry is *explained* by the light cone (`PHASE1_AUDIT.md:537-546`) | — |
| **F9** | That probability rises with look-back: `0.5000, 0.7031, 0.7401, 0.7545, 0.7627, 0.7829, 0.8082, 0.9583` at `a = 0,4,8,12,16,20,24,28` (10⁶ bits). | OBS | `phase1_results/mb1loc_1M.log`; `bridge_probe.json: backward_transfer` | — | two independent runs at different lag ranges (`p ≤ 4096`, `p ≤ 20000`) agree where they overlap | Sample sizes collapse: 7.6e4 at `a = 16`, 4662 at `a = 20`, 292 at `a = 24`, **24** at `a = 28`. The last two points are nearly meaningless individually. |
| **F10** | The limit of that probability as `a → ∞` is **undetermined**; both `< 1` and `= 1` are consistent with the data, and they imply opposite research programmes. | OBS (negative) | `PHASE1_AUDIT.md:664-677`; `CONJECTURE_REGISTRY.md:157` | — | — | An earlier draft asserted "saturates around 0.755, not at 1". **That claim was wrong and was retracted** (`research_notes.md`). |

## G. Anomalies (both closed negative)

| ID | Claim | Type | Support | Assumptions | Independent verification | Known weaknesses |
|---|---|---|---|---|---|---|
| **G1** | The length-11 word `10000010011`, absent from the first 20001 bits, occurs 458 times in the first 10⁶ — a sampling artefact, not a forbidden factor. | COMP | `anomaly_followup.json: missing_factor_retest` | B10 | `replicated` on a larger sample | — |
| **G2** | The `+3.311σ` excess of ones at `t ≡ 1 (mod 3)` **did not replicate**: `z = +1.408` on the disjoint fresh sample `t ∈ [200001, 10⁶)`. | COMP | `anomaly_followup.json: mod3_discovery_sample`, `mod3_fresh_sample` | — | `replicated` on a **disjoint** sample | Discovered by scanning 51 residue classes; nominal `p` was never meaningful. |
| **G3** | Two bridging probes reporting 0.654 and 0.500 were both correct — they conditioned on different events (two-sided vs forward-only windows); resolved by a third naive recomputation reproducing 0.654059. | META | `PHASE1_AUDIT.md:444-453` | — | third independent list-based implementation | Recorded because the disagreement looked like a bug and was not. |

## H. Claims deliberately **not** made

| ID | Statement | Status |
|---|---|---|
| **H1** | "The centre column is not eventually periodic." | **NOT CLAIMED.** Open (Registry C1). |
| **H2** | "Finite computation supports non-periodicity." | **REJECTED.** `PHASE1_AUDIT.md:610-616`. Only D1–D4 are rigorous, and they are lower bounds. |
| **H3** | "MB1 is true / false." | **NOT CLAIMED.** Untestable and unproved (E2). |
| **H4** | "MB1-loc is false for every `a`." | **NOT CLAIMED.** Established only for `a ≤ 28` (D5). |
| **H5** | "The transfer probability saturates below 1." | **RETRACTED.** Was asserted in a draft; corrected (F10). |
| **H6** | "The left-region boundary speed is 1/4." | **NOT CLAIMED.** Not converged (F7). |
| **H7** | "Wolfram's partial-knowledge extension holds." | **NOT VERIFIED.** Unreconstructed literature claim (Registry C7). |
| **H8** | "The powers-of-two families show a pattern." | **REJECTED.** No signal (F4). |
| **H9** | "Every MB1-loc witness has `a_t(1) = 0` and `a_{t+p}(1) = 1`." | **RETRACTED.** True for the 200001-bit witnesses, false for the 10⁶-bit ones at `a = 16, 20` (polarity `1 → 0`). The refutations themselves are unaffected: a violation requires only `a_t(1) ≠ a_{t+p}(1)`. Caught by `run_all_tests.py` step 5. |
