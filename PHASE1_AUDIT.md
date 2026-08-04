# Phase 1 Audit — Rule 30 Center Column

**Date:** 2026-08-04 · **Branch:** `claude/rule30-prize-problem-1-qr1omb` ·
**Repo commit at run time:** `602d9fc773759972e27690550498d5c7752092d5`

---

## 0. Provenance warning: the referenced prior work does not exist

The brief asks to read `README.md`, `research_notes.md`, `rule30_lab.py` and
`rule30_experiments.ipynb`, and to preserve all previous research history.

**Finding: none of the Rule 30 files exist, and never did.** The repository
`drzamani293-code/kaggle-agent` is an unrelated LLM agent for Kaggle
competitions (LangGraph planner/executor; `README.md` documents "KagGooLY").
Checked exhaustively:

```
$ find . -iname "*rule30*" -o -iname "*research_notes*" -o -iname "*rule_30*"   # empty
$ git log --all --format=%H | while read c; do git ls-tree -r --name-only $c; done \
    | sort -u | grep -icE "rule30|rule_30|research_note|wolfram|cellular"
0
```

Zero matching files in **any** commit on **any** branch. There is therefore no
prior research history to preserve, no previous implementation to audit against,
and **no earlier results to compare with**. Everything below was built from
scratch in this session. `README.md` and all pre-existing files are untouched.

**Consequence for the audit.** Rule 3 of the brief ("verify the Rule 30
implementation independently before using its outputs") cannot be satisfied by
comparison with a prior implementation. It is satisfied instead by four
mutually independent constructions plus one external reference (§2).

---

## 1. Conventions (fixed, and the source of most possible indexing errors)

| Item | Convention used here |
|---|---|
| Lattice | `Z`, site index `i` increasing to the **right** |
| Time | `t = 0, 1, 2, ...`, increasing **downward** |
| Seed | `a_0(0) = 1`, `a_0(i) = 0` otherwise |
| Local rule | `a_{t+1}(i) = f_30(a_t(i-1), a_t(i), a_t(i+1))`, `f_30(l,c,r) = (30 >> (4l+2c+r)) & 1` |
| Closed form | `f_30(l,c,r) = l XOR (c OR r)` (proved by exhaustion, §2.1) |
| Center column | `c(t) = a_t(0)`, **`t` starts at 0**, so `c(0) = 1` |

**Indexing hazard, stated explicitly.** OEIS A051023 uses offset 1. Our `c(0)`
is their first term. Since eventual periodicity is invariant under deleting
finitely many initial terms, the offset is **irrelevant to Prize Problem 1** —
but it is **not** irrelevant to any claim indexed by `t`, in particular the
`t = 2^k - 1, 2^k, 2^k + 1` experiment in §5.6. Every index in this document is
in our convention.

**Orientation hazard.** Rule 30 is *not* mirror-symmetric, so a left/right flip
silently produces rule 86 instead. This is guarded by an explicit mirror check
(§2.4), not by inspection.

---

## 2. Implementation verification

Four constructions of the same object, sharing as little as possible:

| # | Name | Mechanism | Independent of |
|---|---|---|---|
| A | `step_naive` | pure-Python list; 8-entry table extracted bit-by-bit from the *rule number* 30 | B (different code path), C (different mechanism) |
| B | `step_numpy` | vectorised numpy table lookup with shifted arrays | A, C |
| C | `center_column_bitwise` | big-integer bit-parallel evaluation of `(row << 1) ^ (row \| (row >> 1))` | shares **no** code with A/B; derived from the boolean formula, not the table |
| D | rule 86 mirror | evolve the mirror automaton and read its center column | structurally different automaton |

### 2.1 Local rule

* `rule_number_bits` — table of rule 30 indexed by `4l+2c+r` is
  `(0,1,1,1,1,0,0,0)` = binary `00011110` = 30. **PASS**
* `table_equals_xor_or_formula` — exhaustive over all 8 neighbourhoods:
  `000→0, 001→1, 010→1, 011→1, 100→1, 101→0, 110→0, 111→0`, matching
  `l XOR (c OR r)` in every case. **PASS**
* `left_permutive` — for every `(c,r)`, `l ↦ f(l,c,r)` is a bijection. **PASS**
* `not_right_permutive` — `f(l,1,0) = f(l,1,1)`, so `r ↦ f(l,c,r)` is not
  injective. **PASS** (this asymmetry is the whole subject of
  `WIDTH2_PROOF_RECONSTRUCTION.md`)
* `mirror_is_86` — the left-right mirror of rule 30 is rule 86. **PASS**

### 2.2 Hand computation

Rows `t = 0..3` were computed by hand before running any code, over positions
`-3..+3`:

```
t=0   0 0 0 1 0 0 0
t=1   0 0 1 1 1 0 0
t=2   0 1 1 0 0 1 0
t=3   1 1 0 1 1 1 1
```

All four match the simulation (`hand_row_t0..t3`). **PASS**

### 2.3 Cross-implementation agreement

* `A_vs_B_full_rows` — all 401 complete rows (width 805) identical. **PASS**
* `A_vs_C_full_rows` — all 401 complete rows identical. **PASS**
* `B_vs_C_center_column` — center columns identical for `t = 0..5000`. **PASS**

### 2.4 Mirror check

* `mirror86_center_column` — rule 86 from a single cell yields the *same*
  center column for `t = 0..2000`. **PASS**
* `mirror86_rows_reversed` — every rule-30 row is the exact reversal of the
  corresponding rule-86 row. **PASS**

This is the check that would catch a left/right orientation error: if the
implementation had silently computed rule 86, this test would compare rule 86
against rule 30 and fail.

### 2.5 Structural invariants (proved facts used as tests)

| Invariant | Proof | Verified |
|---|---|---|
| `a_t(-t) = 1` for all `t` | `a_{t+1}(-(t+1)) = f(0,0,a_t(-t)) = a_t(-t)` | `t = 0..3000` **PASS** |
| `a_t(+t) = 1` for all `t` | mirror argument | `t = 0..3000` **PASS** |
| `a_{t+1}(t) = a_t(t-1) XOR 1` | `f(l,1,0) = l XOR 1` on the right edge | `t = 1..2999` **PASS** |

### 2.6 Boundary independence

All simulations use a zero-padded array of radius `R = steps + 2`. Because the
light cone grows at speed exactly 1, this is *exact*, not approximate.
Verified empirically: `boundary_independence` re-runs `t = 0..2000` with the
radius increased from 2002 to 2502 — center column unchanged. **PASS**

### 2.7 Lemmas of the width-2 proof, verified mechanically

| Check | What it verifies | Result |
|---|---|---|
| `lemma1_left_reconstruction` | `a_t(i-1) = a_{t+1}(i) XOR (a_t(i) OR a_t(i+1))` at 202,000 space-time points | **PASS** |
| `lemma2_leftward_determinism` | columns `-1..-200` reconstructed from columns 0 and 1 alone, compared to truth | **PASS** |
| `lemma3_strip_determinism` | interior columns `-5..5` driven for 3000 steps by boundary columns `-6`, `6` only | **PASS** |

### 2.8 External reference

`oeis.org` is blocked by this environment's network proxy (HTTP 403 for both
`curl` and the fetch tool), so the authoritative b-file could not be
downloaded. A web-search snippet reports A051023 beginning

```
1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, ...
```

Our first 14 bits are **identical**. This is a *secondary* source and is
recorded as weak external corroboration, not as verification. The hand
computation of §2.2 is the stronger external check, because it is independent
of every line of code.

### 2.9 Verification summary

**21 / 21 checks pass.** Reproduce with:

```
python3 rule30_lab.py verify          # full suite, ~3 s
python3 rule30_lab.py verify --fast   # reduced sizes, ~0.5 s
```

---

## 3. Exact experimental settings

| Setting | Value |
|---|---|
| Center column length | `N = 200001` bits (`t = 0..200000`) for the main run; `N = 1000001` for the follow-up run |
| Array radius | `steps + 2` (exact, see §2.6) |
| Candidate-period scan | `p = 1 .. 10000` (main), `p = 1 .. 20000` (1M run) |
| Subword complexity | lengths `n = 1..24` |
| Berlekamp–Massey | over GF(2), checkpoints `16, 64, 256, 1024, 4096, 16384, 65536, 200001` |
| Block entropy | first 100000 bits, `n = 1..16` |
| Bridge probe | `N = 200001`, lags `p ≤ 4096` (profiles) and `p ≤ 20000` (witness search) |
| Left-region geometry | `steps = 4000` (in-script) and `steps = 6000, 40000` (ad hoc, §7.3) |
| Python / numpy | 3.11.15 / 2.4.6, Linux x86-64 |
| Randomness | **none** — every computation is deterministic and bit-reproducible |

---

## 4. Reproducible commands

```bash
# verification suite (21 checks)
python3 rule30_lab.py verify

# main Phase 1 experiment set  (~57 s)
python3 run_phase1.py --steps 200000 --max-period 10000 --subword-max 24
#   -> phase1_results/phase1_results.json
#   -> phase1_results/phase1_report.txt
#   -> phase1_results/center_column_200001.txt

# larger independent-sample re-test of the anomalies  (~7 min)
python3 run_anomaly_followup.py --steps 1000000
#   -> phase1_results/anomaly_followup.json

# width-2 -> width-1 bridging-lemma probes  (~4 min)
python3 run_bridge_probe.py --steps 200000 --max-period 4096
#   -> phase1_results/bridge_probe.json

# explicit refutation of the local bridging lemma  (~5 min)
python3 find_mb1loc_witnesses.py --steps 200000 --max-period 20000
#   -> phase1_results/mb1loc_witnesses.json

# ad hoc
python3 rule30_lab.py center --steps 255      # first 256 center bits
python3 rule30_lab.py stats  --steps 10000 --json
```

---

## 5. Results

### 5.1 First 256 center bits (`t = 0 .. 255`)

```
t=  0  1101110011000101100100111010111001110101011000011001010110101011
t= 64  1111000011110001010111000001001011000111000110110110100000001000
t=128  1111101110100111000111010111000001100100011001111001111110000001
t=192  1111110110010110111000001011000110110001100011101101100101011111
```

### 5.2 Frequencies at several prefix lengths

| `N` | ones | zeros | freq(1) | ones − zeros | z (fair-coin null) |
|---|---|---|---|---|---|
| 16 | 9 | 7 | 0.562500 | +2 | +0.500 |
| 64 | 35 | 29 | 0.546875 | +6 | +0.750 |
| 256 | 135 | 121 | 0.527344 | +14 | +0.875 |
| 1024 | 490 | 534 | 0.478516 | −44 | −1.375 |
| 4096 | 2028 | 2068 | 0.495117 | −40 | −0.625 |
| 16384 | 8277 | 8107 | 0.505188 | +170 | +1.328 |
| 65536 | 32877 | 32659 | 0.501663 | +218 | +0.852 |
| 200001 | 100073 | 99928 | 0.500362 | +145 | +0.324 |
| 1000001 | 500768 | 499233 | 0.500767 | +1535 | +1.535 |

No prefix deviates significantly from balance. The `z` column assumes an i.i.d.
fair coin, which is a **heuristic null with no theoretical justification** for a
deterministic sequence; it is reported as a descriptive statistic only.

### 5.3 Subword (factor) complexity

`p_obs(n)` = number of distinct length-`n` factors in the observed prefix
(`N = 200001`).

| `n` | `p_obs(n)` | `2^n` | `N−n+1` |
|---|---|---|---|
| 1–14 | `2^n` (saturated) | `2^n` | ~200000 |
| 15 | 32683 | 32768 | 199987 |
| 16 | 62377 | 65536 | 199986 |
| 17 | 102485 | 131072 | 199985 |
| 18 | 139865 | 262144 | 199984 |
| 20 | 181994 | 1048576 | 199982 |
| 22 | 195180 | 4194304 | 199980 |
| 24 | 198744 | 16777216 | 199978 |
| **28** (1M run) | **998140** | — | 999974 |

Every binary word of length ≤ 14 occurs. From `n = 15` on, `p_obs` is limited
by the sample, not by the sequence: `p_obs(24) = 198744` out of a maximum
possible `199978`.

Compare the Sturmian threshold `n + 1`: an aperiodic sequence must satisfy
`p(n) ≥ n + 1`; the observed complexity exceeds this by four orders of
magnitude. That is *consistent* with aperiodicity and *proves nothing about it*
(see §8.1).

### 5.4 Berlekamp–Massey linear complexity over GF(2)

| `N` | `L(N)` | `L/N` |
|---|---|---|
| 16 | 9 | 0.5625 |
| 64 | 33 | 0.5156 |
| 256 | 128 | 0.5000 |
| 1024 | 513 | 0.5010 |
| 4096 | 2049 | 0.5002 |
| 16384 | 8192 | 0.5000 |
| 65536 | 32771 | 0.5001 |
| 200001 | 100001 | 0.5000 |

`L(N) ≈ N/2` throughout — the profile of a maximally linearly complex
sequence. The center column has no short GF(2) linear recurrence.

### 5.5 Block entropy (first 100000 bits)

`H_n/n` = 1.00000 through `n = 3`, decaying only through sampling saturation:
0.99926 at `n = 10`, 0.99099 at `n = 14`, 0.96539 at `n = 16`. The decay is
consistent with finite-sample bias (`n = 16` has 65536 possible words and only
~10^5 samples) and must not be read as a measurement of entropy rate.

### 5.6 Behaviour at `t = 2^k − 1`, `2^k`, `2^k + 1`

| `k` | `2^k−1` | `c` | `2^k` | `c` | `2^k+1` | `c` |
|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 1 | 1 | 2 | 0 |
| 1 | 1 | 1 | 2 | 0 | 3 | 1 |
| 2 | 3 | 1 | 4 | 1 | 5 | 1 |
| 3 | 7 | 0 | 8 | 1 | 9 | 1 |
| 4 | 15 | 1 | 16 | 1 | 17 | 0 |
| 5 | 31 | 0 | 32 | 0 | 33 | 1 |
| 6 | 63 | 1 | 64 | 1 | 65 | 1 |
| 7 | 127 | 0 | 128 | 1 | 129 | 1 |
| 8 | 255 | 1 | 256 | 1 | 257 | 1 |
| 9 | 511 | 0 | 512 | 0 | 513 | 1 |
| 10 | 1023 | 0 | 1024 | 1 | 1025 | 0 |
| 11 | 2047 | 1 | 2048 | 0 | 2049 | 1 |
| 12 | 4095 | 1 | 4096 | 1 | 4097 | 0 |
| 13 | 8191 | 1 | 8192 | 1 | 8193 | 0 |
| 14 | 16383 | 0 | 16384 | 0 | 16385 | 0 |
| 15 | 32767 | 0 | 32768 | 1 | 32769 | 0 |
| 16 | 65535 | 1 | 65536 | 1 | 65537 | 1 |
| 17 | 131071 | 1 | 131072 | 0 | 131073 | 1 |

| family | ones / n | two-sided exact binomial `p` (fair-coin null) |
|---|---|---|
| `c(2^k − 1)` | 11 / 18 | 0.4807 |
| `c(2^k)` | 12 / 18 | 0.2379 |
| `c(2^k + 1)` | 11 / 18 | 0.4807 |

**No signal.** With 18 data points per family and three families examined, none
of these is remotely significant, and the smallest `p` (0.238) is exactly what
one expects from looking at three families. Any claim of a pattern at powers of
two based on this data would be pure over-fitting.

### 5.7 Candidate periods — first and last mismatches

For each candidate period `p`, `first_mismatch(p) = min{t : c(t) ≠ c(t+p)}`
and `last_mismatch(p) = max{t : c(t) ≠ c(t+p)}` within the observed window.

| `p` | first | last | window end | slack |
|---|---|---|---|---|
| 1 | 1 | 199999 | 199999 | 0 |
| 2 | 0 | 199997 | 199998 | 1 |
| 3 | 2 | 199995 | 199997 | 2 |
| 4 | 3 | 199996 | 199996 | 0 |
| 5 | 1 | 199995 | 199995 | 0 |
| 6 | 0 | 199994 | 199994 | 0 |
| 7 | 0 | 199993 | 199993 | 0 |
| 8 | 3 | 199989 | 199992 | 3 |
| 16 | 1 | 199983 | 199984 | 1 |
| 32 | 0 | 199966 | 199968 | 2 |

(full table for `p ≤ 32` in `phase1_results/phase1_report.txt`)

Aggregate results:

* **Every** `p ≤ 10000` has at least one mismatch; none survives the window.
* The **largest first mismatch over all `p ≤ 10000` is 11** (attained at
  `p = 9988`) — i.e. no candidate period matches even the first dozen bits
  before failing.
* `min over p ≤ 10000 of last_mismatch(p) = 189998` (at `p = 10000`).
  Since eventual period `p` would force the preperiod `T > last_mismatch(p)`,
  this says: **if the center column has any eventual period `p ≤ 10000`, its
  preperiod exceeds 189998.**
* At `N = 1000001` with `p ≤ 20000`: `min last_mismatch = 979998`, so any
  eventual period `p ≤ 20000` has preperiod `> 979998`.
* Cross-check: the big-integer scan and a naive `O(Np)` scan agree exactly on
  `p ≤ 64` over the first 20000 bits.

### 5.8 Two rigorous lower bounds on `T + p`

These are the only conclusions in this document that finite computation
*establishes* rather than merely suggests. Both are elementary.

> **Lemma A (factor counting).** If `x` is eventually periodic with preperiod
> `T` and period `p`, then for every `n`, `x` has at most `T + p` distinct
> factors of length `n`.
> *Proof.* Factors starting at positions `< T`: at most `T`. A factor starting
> at `i ≥ T` is determined by `(i − T) mod p`: at most `p`. ∎

> **Lemma B (linear recurrence).** If `x` is eventually periodic with preperiod
> `T` and period `p`, then `x` satisfies the GF(2) linear recurrence with
> characteristic polynomial `X^{T+p} + X^T`, so every prefix has linear
> complexity at most `T + p`.
> *Proof.* `(S^p + 1)x = 0` on indices `≥ T`, hence `S^T(S^p + 1)x = 0`
> identically, where `S` is the shift. ∎

Applying these to the observed data:

| Source | Bound |
|---|---|
| Lemma A, `N = 200001`, `n = 24` | `T + p ≥ 198744` |
| Lemma B, `N = 200001` | `T + p ≥ 100001` |
| Lemma A, `N = 1000001`, `n = 28` | **`T + p ≥ 998140`** |

So: *if* the center column of rule 30 is eventually periodic, then its
preperiod plus its period is at least **998140**. Note the ceiling: from `N`
observed bits, Lemma A can never give more than about `N`, so this is close to
the best obtainable at this sample size, and further computation buys a
proportional — not qualitative — improvement.

### 5.9 Runs and autocorrelation

* Longest run of 0s: 19; longest run of 1s: 21 (in 200001 bits; a fair coin
  gives ~17–18 typical, so nothing unusual).
* Run counts halve cleanly: 25069 / 12538 / 6195 / 3126 / 1573 / 765 (zeros)
  and 25041 / 12484 / 6323 / 3070 / 1523 / 803 (ones) for lengths 1..6.
* Autocorrelation of the `±1` sequence at lags 1..32: all `|z| < 2.5` except
  lag 12 (`z = +2.42`). With 32 lags examined, one value near 2.4 is expected.

---

## 6. Anomalies found, and what happened to them

Two candidate anomalies surfaced. **Both were found by searching, and both were
re-tested on data that played no part in their discovery.** Neither survived.

### 6.1 A missing length-11 factor — resolved as a sampling artefact

At `N = 20001` the word `10000010011` was the unique binary word of length 11
absent from the sequence (`p_obs(11) = 2047` instead of 2048). A genuinely
forbidden factor would be a major structural fact.

Re-test: the word occurs **458 times** in the first 1000001 bits, first
appearing before `N = 50000`. At `N = 200001` every word of length ≤ 14 occurs.
**Artefact. Closed.**

### 6.2 A mod-3 residue bias — did not replicate

At `N = 200001`, scanning residue classes modulo 2,3,4,5,6,7,8,16 (51 classes),
exactly one exceeded `|z| > 3`:

```
t ≡ 1 (mod 3):  n = 66667, ones = 33761, freq = 0.506412, z = +3.311
```

Nominal `p = 0.0009`, but 51 classes were examined, so the multiplicity-adjusted
expectation of at least one such hit is ≈ 13%. Re-test on the **disjoint fresh
sample** `t ∈ [200001, 1000001)`:

| class | n | freq(1) | z | p |
|---|---|---|---|---|
| `t ≡ 0 (mod 3)` | 266667 | 0.501281 | +1.323 | 0.186 |
| `t ≡ 1 (mod 3)` | 266667 | 0.501363 | +1.408 | 0.159 |
| `t ≡ 2 (mod 3)` | 266666 | 0.499962 | −0.039 | 0.969 |

The effect **does not replicate** (`z` falls from 3.31 to 1.41 on four times the
data — a real effect would have grown as `√n`). **Closed as a multiple-testing
artefact.** No other modulus in `{2,4,5,7,8,16}` shows anything on the fresh
sample.

### 6.3 An analysis bug found and fixed during the audit (recorded for honesty)

An early version of the bridging-lemma probe reported a conditional agreement
probability of 0.654 where a second probe reported 0.500. Neither was a coding
error — they conditioned on different events (two-sided vs forward-only windows)
— but the discrepancy was only resolved by writing a third, deliberately naive
list-based recomputation, which reproduced 0.654059 exactly. The resolution is
in §7.4. **Two probes disagreeing is not evidence that one is broken; it was
worth the third implementation to find out which question each was answering.**

---

## 7. Possible mathematical patterns

Everything in this section is a **computational observation**. Nothing here is
a theorem, a lemma, or a conjecture with evidence sufficient to act on; formal
statements live in `CONJECTURE_REGISTRY.md`.

### 7.1 Statistical structure: none detected

At every measure applied — bit frequency, block entropy, factor complexity,
linear complexity, run-length distribution, autocorrelation, residue-class bias
— the center column is indistinguishable from a fair coin at the sample sizes
used. This is the expected outcome and is worth stating precisely because it
constrains proof strategies: **no proof of aperiodicity will come from a
statistical deviation, because there is no measured deviation to exploit.**

### 7.2 Left/right asymmetry of the space-time diagram (structural, real)

The rule is left-permutive and not right-permutive (§2.1). The consequences are
visible and were verified: the left light-cone edge is frozen at 1, the right
edge is frozen at 1, and the sub-diagonal `a_t(t−1)` alternates exactly. These
are proved facts (used as implementation tests), not observations.

### 7.3 The regular left region: a diagonal period-16 wedge

Measured by searching for space-time shift symmetries `(t,i) → (t+τ, i−τ+σ)`
and recording the width of the band, measured from the left light-cone edge, on
which the symmetry holds:

| `τ` | `σ` | band width behaviour |
|---|---|---|
| 4 | 0 | constant **29** cells at every `t` sampled |
| 8 | 0 | constant **400** cells at every `t` sampled |
| 12 | 0 | constant 29 |
| 16 | 0 | **grows linearly**: ≈ `0.74 t` to `0.77 t` |
| 32, 48, 64, 128, 256 | 0 | identical to `τ = 16` |

So the minimal diagonal period of the regular left region is **16**, and
sub-periods 4 and 8 hold only in fixed finite bands (29 and 400 cells) hugging
the light-cone edge — a crisp, reproducible observation.

The wedge boundary sits at lattice position ≈ `−(1 − s)·t`, so the boundary
between the regular region and the chaotic bulk drifts leftwards at speed
`1 − s`. Measured slopes `s`:

| sample window | slope `s` | implied boundary speed |
|---|---|---|
| `t ∈ [2000, 3935]` | 0.7676 | 0.2324 |
| `t ∈ [3000, 5900]` | 0.7434 | 0.2566 |
| `t ∈ [2000, 34000]`, fitted | 0.7423 | 0.2577 |
| `t ∈ [18000, 34000]` (late) | 0.7416 | 0.2585 |

**Warning:** the ratio `W(t)/t` is still decreasing at `t = 34000`
(0.7645 at `t = 2000`, 0.7444 at `t = 34000`). The slope has **not converged**;
no asymptotic constant should be quoted from this data, and in particular the
proximity to 1/4 may be a coincidence.

*Relation to the theory.* Theorem W2′ (at most one eventually periodic column)
requires that no fixed column stays inside a temporally periodic region forever.
The measurement is consistent with this: a fixed column `j < 0` sits inside the
wedge only for `t ∈ [|j|, |j|/(1−s)]`, a bounded window. Note carefully that
diagonal-shift invariance alone would **not** make any column eventually
periodic (the symmetry relates *different* columns), so this consistency check
is weak — it would not have detected a contradiction.

### 7.4 Local periodicity transfer between columns (the substantive finding)

Motivated by `WIDTH2_PROOF_RECONSTRUCTION.md` §11: the bridging lemma MB1
reduces to whether lag-`p` agreement in column 0 forces lag-`p` agreement in
column 1 on the zero set of column 0. Measured

```
P[ a_t(1) = a_{t+p}(1)  |  col 0 agrees at lag p on [t−a, t+b],  a_t(0) = 0 ]
```

over lags `p ≤ 256`, `N = 200001`:

| | `b=0` | `b=1` | `b=2` | `b=4` | `b=8` |
|---|---|---|---|---|---|
| `a=0` | 0.5009 | 0.5008 | 0.5007 | 0.5004 | 0.5079 |
| `a=1` | 0.6239 | 0.6239 | 0.6237 | 0.6233 | 0.6317 |
| `a=2` | 0.6566 | 0.6564 | 0.6566 | 0.6563 | 0.6674 |
| `a=4` | 0.7048 | 0.7052 | 0.7066 | 0.7073 | 0.7162 |
| `a=8` | 0.7410 | 0.7428 | 0.7427 | 0.7368 | 0.7130 |

Two clean facts:

1. **The columns depend on `a` (past) and not on `b` (future).** Conditioning
   on the *future* of column 0 gives exactly 0.500 — no information at all.
   This has an exact explanation: `a_t(1)` is a function of the cells at time
   `t−k` in positions `[1−k, 1+k]`, a set containing `a_{t−1}(0), a_{t−2}(0),
   …` and **no** cell at any time later than `t`. Conditioning on column 0's
   past matches part of `a_t(1)`'s own dependency cone; its future matches
   nothing. This is a light-cone fact, not a discovery about rule 30.

2. **The probability rises but stays far from 1** (`p ≤ 4096`, `b = 0`):

   | `a` | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
   |---|---|---|---|---|---|---|---|---|---|---|---|
   | `P` | .50002 | .62287 | .65468 | .69387 | .70268 | .72570 | .74018 | .74906 | .75559 | .75762 | .75544 |
   | `n` | 2.0e8 | 1.0e8 | 5.1e7 | 2.5e7 | 1.3e7 | 3.2e6 | 7.9e5 | 2.0e5 | 5.0e4 | 1.2e4 | 3.1e3 |

   Homogeneous across lag bands (`p ∈ [1,64]`, `[65,512]`, `[513,4096]` give
   0.7095 / 0.7033 / 0.7025 at `a = 4` and 0.7453 / 0.7408 / 0.7400 at `a = 8`).

   **The limit is not determined by this data.** The values still creep upward
   at the largest look-backs, where the samples are small. See §8.5.

---

## 8. Explicit refutation of the local bridging lemma

`WIDTH2_PROOF_RECONSTRUCTION.md` §12 defines

> **MB1-loc(`a`)**: for all `t > a`, `p ≥ 1`: if `a_s(0) = a_{s+p}(0)` for all
> `s ∈ [t−a, t]` and `a_t(0) = 0`, then `a_t(1) = a_{t+p}(1)`.

MB1-loc(`a`) for any fixed `a` would immediately imply MB1, hence (with Theorem
W2) settle Problem 1. It is universally quantified, so **a single counterexample
refutes it** — the one direction in which finite computation is logically
sufficient.

Search over `N = 1000001` bits, lags `p ≤ 20000`. Each witness was re-verified
against an independently re-simulated diagram
(`find_mb1loc_witnesses.py --steps 1000000 --max-period 20000`):

| `a` | eligible positions | violations | violation rate | agreement `1 − rate` | witness `(t, p)` | re-verified |
|---|---|---|---|---|---|---|
| 0 | 4934666671 | 2467238826 | 0.5000 | 0.5000 | `t=17, p=1` | ✔ |
| 4 | 308412494 | 91563295 | 0.2969 | 0.7031 | `t=15363, p=1` | ✔ |
| 8 | 19271349 | 5007728 | 0.2599 | 0.7401 | `t=3189, p=2` | ✔ |
| 12 | 1206822 | 296286 | 0.2455 | 0.7545 | `t=94693, p=2` | ✔ |
| 16 | 75728 | 17970 | 0.2373 | 0.7627 | `t=850603, p=6` | ✔ |
| 20 | 4662 | 1012 | 0.2171 | 0.7829 | `t=322343, p=27` | ✔ |
| 24 | 292 | 56 | 0.1918 | 0.8082 | `t=82630, p=1324` | ✔ |
| 28 | 24 | 1 | 0.0417 | 0.9583 | `t=529506, p=2750` | ✔ |
| 32 | 3 | 0 | — | — | **none found** | — |

In every witness, `a_t(0) = 0`, `a_t(1) = 0`, `a_{t+p}(1) = 1`, and column 0
agrees at lag `p` throughout `[t−a, t]`.

**Conclusion: MB1-loc(`a`) is FALSE for `a = 0, 4, 8, 12, 16, 20, 24, 28`.**
For `a ≥ 32` the sample contains only 3 eligible positions — far too few to
decide; that case is **untested, not verified** — see §9.7.

**A caution that cuts against the convenient reading.** The violation rate is
*decreasing* with `a` (0.297 → 0.192 → 0.042 over `a = 4..28`), i.e. the
agreement probability is still climbing, and the last two rows rest on 292 and
24 samples. This is entirely consistent with a probability that tends to 1,
which would make MB1-loc(`a`) true for some large `a` and would be very good
news for the proof programme. The refutations above are solid; the extrapolation
"MB1-loc fails for every `a`" is **not** supported. See §9.8.

---

## 9. Warnings: conclusions that cannot be drawn from any of this

These are ranked by how tempting the mistake is.

**9.1 Finite computation cannot prove non-periodicity — of anything, ever.**
200001 bits, or 10^6, or 10^12, are consistent with eventual periodicity with
any period exceeding the sample. The *only* rigorous outputs here are the
lower bounds of §5.8 (`T + p ≥ 998140`) and the refutations of §8. Nothing in
this repository is a step towards a computational proof of Problem 1, and no
extension of the sample size changes that in kind.

**9.2 "It looks random" is not evidence of aperiodicity in the logical sense.**
The statistics of §5 would be reproduced exactly by a sequence that is
eventually periodic with a period of 10^100. They constrain nothing beyond the
bounds already stated.

**9.3 The fair-coin `z`-scores and `p`-values are descriptive only.** The center
column is deterministic; there is no random experiment, so a `p`-value has no
inferential meaning here. They are used as a *search heuristic* for structure —
and §6.2 shows how the one hit that heuristic produced evaporated on
replication.

**9.4 Do not report the powers-of-two families as a pattern.** §5.6 has 18
points per family and three families; the smallest `p` is 0.238. There is
nothing there.

**9.5 Do not quote the left-region boundary speed as a constant.** §7.3: the
slope is still drifting at `t = 34000`. "≈ 0.25" is an observation over a
finite window, not a measured constant, and its closeness to 1/4 is not
evidence of anything.

**9.6 The refutation of MB1-loc does NOT refute MB1.** MB1 is a conditional
whose hypothesis (the center column is eventually periodic) is conjecturally
false, and it is therefore *vacuously true* if Problem 1 has the expected
answer. **MB1 can never be tested computationally.** §8 refutes only the
finitary statement MB1-loc, which is a different proposition. What §8
establishes is a fact about *proof strategies*, not about MB1's truth.

**9.7 MB1-loc(`a`) for `a ≥ 28` has not been refuted.** The table in §8 runs out
of data, not out of counterexamples. Claiming "false for all `a`" would
overstate the computation by exactly the gap between `a = 24` and `a = ∞`. The
extrapolation is plausible; it is not established.

**9.8 The transfer probability's limit is unknown, and the trend runs the other
way.** §7.4 shows values flattening near 0.75 for `a ≤ 16`, but the deeper
1M-bit measurement of §8 keeps climbing — 0.7627 (`a=16`), 0.7829 (`a=20`),
0.8082 (`a=24`), 0.9583 (`a=28`) — on samples of 75728, 4662, 292 and 24. Both
readings are open:

* limit `< 1` ⟹ MB1-loc fails at every depth, and MB1 needs a global proof;
* limit `= 1` ⟹ MB1-loc(`a`) might hold for some large `a`, which would be a
  *route to a proof*, not an obstruction.

Do **not** report "saturates below 1" as established. Distinguishing these is
the highest-value next computation (Registry C5).

**9.9 Cross-implementation agreement is not correctness.** Four implementations
agreeing rules out coding slips, not a shared misunderstanding of the
definition. The defence against that is §2.1 (exhaustive check against the rule
*number*), §2.2 (hand computation), §2.4 (mirror check) and §2.8 (external
reference) — and the external reference is a search snippet, not the OEIS
b-file, because the proxy blocks `oeis.org`.

**9.10 Theorem W2/W2′ says nothing about the center column by itself.** By
Theorem W2′ at most one column of the diagram is eventually periodic. The
center column could a priori be that one column. Every result in this audit is
compatible with that scenario; ruling it out is the whole problem.

---

## 10. Files produced

| File | Contents |
|---|---|
| `rule30_lab.py` | four implementations, 21-check verification suite, all statistics, bridging-lemma probes |
| `run_phase1.py` | main experiment driver → `phase1_results/phase1_results.json`, `phase1_report.txt`, `center_column_200001.txt` |
| `run_anomaly_followup.py` | 10^6-bit independent-sample re-tests → `anomaly_followup.json` |
| `run_bridge_probe.py` | transfer-profile experiments → `bridge_probe.json` |
| `find_mb1loc_witnesses.py` | explicit MB1-loc counterexamples → `mb1loc_witnesses.json` |
| `PHASE1_AUDIT.md` | this file |
| `WIDTH2_PROOF_RECONSTRUCTION.md` | self-contained proof of Theorems W2 and W2′, and the gap analysis |
| `CONJECTURE_REGISTRY.md` | formal statements, evidence, falsification tests |

No pre-existing file was modified.
