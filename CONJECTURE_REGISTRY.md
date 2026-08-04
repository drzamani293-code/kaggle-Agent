# Conjecture Registry — Rule 30 Prize Problem 1

Companion to `PHASE1_AUDIT.md` and `WIDTH2_PROOF_RECONSTRUCTION.md`.

## Epistemic labels used in this project

| Label | Meaning | May be cited as established? |
|---|---|---|
| **THEOREM** | proved here from the definition of rule 30, full proof written out | yes |
| **LEMMA** | proved here, used only in service of a theorem | yes |
| **FACT** | proved here in one or two lines (induction / exhaustion) | yes |
| **REFUTED** | a universally quantified statement with an explicit, re-verified counterexample | yes, as *false* |
| **CONJECTURE** | precise statement, not proved, with stated evidence | **no** |
| **HEURISTIC** | a modelling assumption (e.g. "behaves like a fair coin") used to generate predictions | **no** |
| **OBSERVATION** | a measured quantity over a finite sample | only as a measurement, with its sample size |

Standing rule: **no finite computation proves non-periodicity.** Finite
computation can *refute* universally quantified claims, and can *lower-bound*
a hypothetical preperiod + period. Nothing else.

---

## Established results (for reference — not conjectures)

| Id | Statement | Label | Where |
|---|---|---|---|
| T1 | No two adjacent columns of the rule-30 single-cell diagram are both eventually periodic | **THEOREM** | WIDTH2 §7 |
| T2 | At most **one** column of the diagram is eventually periodic | **THEOREM** | WIDTH2 §8 |
| T3 | If `col_0` is e.p. with period `p`, then `col_{-1}` is `p`-periodic ⟺ `col_1` agrees at lag `p` on the zero set of `col_0` | **THEOREM** (Prop. 6) | WIDTH2 §11 |
| B1 | If the center column is e.p. with preperiod `T` and period `p`, then `T + p ≥ 998140` | **THEOREM** (Lemma A + computation) | AUDIT §5.8 |
| B2 | If the center column is e.p. with `p ≤ 20000`, then `T > 979998` | **THEOREM** (period scan) | AUDIT §5.7 |
| R1 | MB1-loc(`a`) is false for `a = 0, 4, 8, 12, 16, 20, 24, 28` | **REFUTED** | AUDIT §8 |

---

# C1 — Prize Problem 1 (the target)

**Formal statement.** Let `a_t` be the rule-30 orbit of the single-cell seed
(`a_0 = δ_0`). The sequence `c(t) = a_t(0)`, `t ≥ 0`, is **not** eventually
periodic: there are no `T ≥ 0`, `p ≥ 1` with `c(t + p) = c(t)` for all `t ≥ T`.

**Motivation.** Wolfram's Rule 30 Prize Problem 1. Eventual periodicity of the
center column would make rule 30 useless as a randomness source and would be a
structural fact of the first order.

**Evidence.**
* `T + p ≥ 998140` (B1); no period `p ≤ 20000` survives with preperiod
  `≤ 979998` (B2).
* Linear complexity `≈ N/2` at every checkpoint up to `N = 200001`.
* Factor complexity saturating the sample bound (`p_obs(28) = 998140` of a
  possible 999974).
* No detected deviation from fair-coin statistics on any measure applied
  (AUDIT §5, §7.1).

**Possible counterexample.** A period `p > 10^6` with a comparable preperiod.
Nothing measured excludes this, and the bound B1 grows only linearly in the
computation length.

**Computational falsification test.** Extend the center column and search for
`(T, p)` with `T + p ≤ N`. Falsified if a matching `(T, p)` is found and holds
to `N`; **cannot be confirmed** by any such run (AUDIT §9.1).

**If proved.** Resolves Prize Problem 1.

---

# C2 — The Bridge (what T2 needs)

**Formal statement.** If `col_0` is eventually periodic, then `col_j` is
eventually periodic for some `j ≠ 0`.

**Motivation.** By T2 at most one column is eventually periodic, so C2 is
*exactly* the missing input: C2 + T2 ⟹ C1.

**Evidence.** None direct. C2 is a conditional whose hypothesis is
conjecturally false, so it has no observable consequences (see the vacuity
warning under C3).

**Possible counterexample.** A rule-30 diagram in which `col_0` is eventually
periodic and every other column is aperiodic. T2 says this is the *only*
shape a counterexample to C1 can take, so C2 is not merely sufficient — it is
the natural dividing line.

**Computational falsification test.** None exists. C2 is vacuous if C1 holds.

**If proved.** C2 + T2 ⟹ C1 (Prize Problem 1).

---

# C3 — MB1, the nominated smallest missing lemma ★

**Formal statement.** Suppose `col_0` is eventually periodic with preperiod `T`
and period `p`. Let

```
Z = { t ≥ T : a_t(0) = 0 }.
```

Then there exists `T'` such that

```
a_{t+p}(1) = a_t(1)     for every t ∈ Z with t ≥ T'.
```

**Motivation and exact status.** By Theorem T3 (Proposition 6 of the width-2
reconstruction), MB1 is *equivalent* to "`col_{-1}` is `p`-periodic from some
time on". Hence MB1 ⟹ `col_{-1}` eventually periodic ⟹ `col_{-1}` and `col_0`
are adjacent and both eventually periodic ⟹ contradiction with T1. So

> **MB1 ⟹ C1.**

MB1 is strictly weaker than the obvious candidate "`col_0` e.p. ⟹ `col_1`
e.p.": it demands agreement only at the single lag `p`, only on the zero set of
`col_0` (density ≈ 1/2), and only eventually. This is why it is nominated as
*the smallest* bridging lemma: it is the weakest fully explicit statement in
the chain `MB0 ⟹ MB1 ⟺ MB1′ ⟹ C2 ⟹ C1`.

**Evidence.** None, and none is obtainable — see the warning below.

**Possible counterexample.** A diagram with `col_0` eventually `p`-periodic and
infinitely many `t ∈ Z` at which `col_1` disagrees at lag `p`. By T3 such a
configuration has `col_{-1}` non-`p`-periodic, which is consistent with T1 and
therefore not excluded by anything proved here.

**Computational falsification test.** **None. MB1 is not computationally
testable, in either direction.** Its hypothesis is conjecturally false, so if
C1 holds then MB1 is *vacuously true*. Any experiment claiming to test MB1 is
testing a different statement — usually C5 below. This vacuity is a property
of the logical form, not a limitation of the hardware.

**If proved.** MB1 + T1 ⟹ C1 (Prize Problem 1). MB1 is the shortest path from
the proved width-2 result to the target that we have been able to identify.

---

# C4 — MB0, the stronger neighbour lemma

**Formal statement.** If `col_0` is eventually periodic then `col_1` is
eventually periodic.

**Motivation.** The obvious bridge; recorded to make explicit that it is
*strictly stronger* than MB1 and therefore a worse proof target.

**Evidence.** Same vacuity as C3.

**Possible counterexample.** Any configuration witnessing MB1 but not MB0 —
i.e. `col_1` agreeing at lag `p` on `Z` but not on the complement of `Z`.

**Computational falsification test.** None (vacuous, as C3).

**If proved.** MB0 ⟹ MB1 ⟹ C1. But there is no reason to attack MB0 rather
than MB1: it asks for twice as much on a set where the extra information is,
by Fact F4, invisible to `col_0` anyway.

---

# C5 — The transfer-probability limit ★ (the testable proxy)

**Formal statement.** For `a ≥ 0` define, over the actual rule-30 diagram,

```
              #{ (t,p) : col_0 agrees at lag p on [t-a, t], a_t(0)=0, a_t(1)=a_{t+p}(1) }
   P(a)  =   ------------------------------------------------------------------------
              #{ (t,p) : col_0 agrees at lag p on [t-a, t], a_t(0)=0 }
```

in the limit of large sample (`N → ∞`, `p` ranging over `1..P_max`). **C5 is
the question: does `P(a) → 1` as `a → ∞`, or does `P(a)` converge to a limit
strictly below 1?**

**Motivation.** This is the *finitary shadow* of MB1 — the thing that can
actually be measured. The two answers point in opposite directions:

* `P(a) → L < 1`: local periodicity of the center column never forces the
  neighbour to repeat. Then MB1-loc(`a`) is false for every `a` and any proof
  of MB1 must be global; local/compactness strategies are dead.
* `P(a) → 1` fast enough: MB1-loc(`a`) may become *true* at some finite depth
  `a`, which would be a direct route to a proof of C1.

**Evidence (OBSERVATION, `N = 10^6`, `p ≤ 20000`).**

| `a` | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|---|
| `P(a)` | .5000 | .7031 | .7401 | .7545 | .7627 | .7829 | .8082 | .9583 |
| samples | 4.9e9 | 3.1e8 | 1.9e7 | 1.2e6 | 7.6e4 | 4.7e3 | 292 | 24 |

Monotone increasing; the last two points are statistically weak (±0.023 and
±0.041 respectively at one standard error). The forward analogue (conditioning
on `col_0` agreeing at times *after* `t`) is flat at 0.5000 — explained exactly
by the light cone (AUDIT §7.4), and not evidence of anything.

**Possible counterexample / resolution.** Either behaviour is possible on the
current data. The value at `a = 28` (0.958) rests on 24 samples and must not be
taken at face value.

**Computational falsification test (the highest-value next experiment).**
Extend the sequence to `N = 10^7` and the lag range to `p ≤ 10^6`; eligible
sample sizes scale roughly like `N · P_max · 2^{-(a+1)}`, which would give
`~10^4` samples at `a = 32` and `~10^2` at `a = 40`. Then:

* If a re-verified MB1-loc counterexample is found at `a = 32, 36, 40, ...`,
  each such witness **refutes** MB1-loc at that depth (rigorous, one witness
  suffices).
* If `P(a)` is measured to be within `1 − ε` for growing `a` with adequate
  samples and *no* counterexample is found at some depth despite ≥ 10^3
  eligible positions, that is evidence for `P(a) → 1` — evidence only, since
  absence of a counterexample in a finite scan proves nothing.

Command shape: `python3 find_mb1loc_witnesses.py --steps 10000000
--max-period 1000000` (note: the current implementation is `O(P_max · N/64)`
and will need the sequence generated once and cached — generation is
`O(N^2/64)` and takes ~6 min at `N = 10^6`, so `N = 10^7` needs a different
generator, e.g. light-cone-restricted evolution).

**If resolved.**
* `P(a) → 1` with an explicit rate ⟹ candidate proof of MB1-loc(`a`) for
  large `a` ⟹ C1.
* `P(a) → L < 1` ⟹ MB1 must be proved globally; rules out a class of
  strategies and redirects Phase 2.

---

# C6 — The exact value of the transfer plateau

**Formal statement.** `lim_{a→∞} P(a) = 3/4`, where `P(a)` is as in C5.

**Motivation.** The naive HEURISTIC — "the center cell is 1 about half the
time, in which case the neighbour is irrelevant by Fact F4; otherwise the
neighbour is a fair coin" — predicts a per-step survival of
`1/2 · 1 + 1/2 · 1/2 = 3/4`.

**Evidence.** `P(a)` passes through 0.7545 at `a = 12` (1.2e6 samples), close
to 3/4 but ~2.5 standard errors above it, and keeps rising thereafter. So the
heuristic value is close but the data are **already inconsistent** with `3/4`
as an exact limit unless the approach is non-monotone.

**Possible counterexample.** The measured monotone rise past 0.75 (0.7627,
0.7829, 0.8082 at `a = 16, 20, 24`) is itself evidence against C6.

**Computational falsification test.** The same experiment as C5. C6 is
essentially already falsified by the `a ≥ 16` values, subject to their sample
sizes; it is recorded because the `3/4` heuristic is the natural first guess
and should not be quietly adopted.

**If proved.** Would fix the plateau and settle C5 in the `L < 1` direction,
hence establish that MB1 requires a global proof.

---

# C7 — Wolfram's partial-knowledge extension (unverified literature claim)

**Formal statement (as claimed, not as proved).** In *Announcing the Rule 30
Prizes* (2019) it is asserted that the two-column argument "also works if the
columns are not adjacent, and if one doesn't know every cell in both columns."

**Status here.**
* The *non-adjacent* half **we have proved independently** — it is Theorem T2
  (WIDTH2 §8), via strip determinism plus finiteness of `{0,1}^m`. Our proof
  may or may not be the one alluded to.
* The *partial-knowledge* half is **NOT reconstructed and NOT verified**. We do
  not use it anywhere and take no position on it.

**Motivation for reconstructing it.** A version of T1/T2 requiring only partial
knowledge of the two columns could conceivably be applied with `col_0` known
and `col_1` known only on a density-1/2 subset — which is precisely the data
MB1 (C3) provides. If the partial-knowledge theorem is strong enough, the gap
might be narrower than it currently appears.

**Possible counterexample.** Not applicable until the statement is made
precise. Making it precise is the first task.

**Computational falsification test.** Once a precise statement exists, its
finitary consequences can be probed the way MB1-loc was in AUDIT §8.

**If proved.** Unknown until formalised — potentially the shortest route,
which is why it is registered rather than ignored.

---

# C8 — Left-region wedge geometry

**Formal statement.** There is a constant `s ∈ (0,1)` such that the maximal
width `W(t)`, measured from the left light-cone edge, of the band on which
`a_t(i) = a_{t+16}(i-16)` satisfies `W(t)/t → s`; and `16` is the minimal
`τ > 0` for which the corresponding band width is unbounded.

**Motivation.** Quantifies the visible "regular left region". Theorem T2
requires that no fixed column stays inside a temporally periodic region
forever; a wedge with `s < 1` is consistent with that.

**Evidence (OBSERVATION).** `τ = 16` is minimal with unbounded width;
`τ = 4` and `τ = 8` give bands of *constant* width 29 and 400 respectively at
every sampled `t`; `τ = 32, 48, 64, 128, 256` give exactly the same band as
`τ = 16`. Slope estimates: 0.7676 over `t ∈ [2000, 3935]`, 0.7434 over
`[3000, 5900]`, 0.7423 over `[2000, 34000]`, 0.7416 over `[18000, 34000]`.

**Possible counterexample.** `W(t)/t` has **not converged** — the ratio falls
monotonically from 0.7645 at `t = 2000` to 0.7444 at `t = 34000`. The limit
could be any value in roughly `[0.70, 0.75]`, or `W(t)` could fail to be
asymptotically linear. **Do not quote a constant** (in particular, do not quote
`1 − s ≈ 1/4`).

**Computational falsification test.** Measure `W(t)` at `t = 10^5, 10^6` with a
rolling 17-row buffer (memory `O(t)`, time `O(t^2)`), and fit `W(t) = s t + c`
on the late window only. Falsified if the local slope keeps drifting or if the
constant-width bands at `τ = 4, 8` are not exactly 29 and 400 at large `t`.

**If proved.** A quantitative description of the regular region. Note it would
**not** bear directly on C1: diagonal-shift invariance relates *different*
columns, so it never makes any single column eventually periodic. Recorded to
prevent the mistake of treating it as progress on the main problem.

---

# C9 — Null conjecture: statistical indistinguishability

**Formal statement.** For every fixed `k`, the empirical distribution of
length-`k` blocks in `c(0..N-1)` converges to uniform on `{0,1}^k` as
`N → ∞`; equivalently the center column is normal in base 2.

**Motivation.** The default reading of all statistics in AUDIT §5.

**Evidence (OBSERVATION).** Frequency `0.500767` at `N = 10^6`; all `2^14`
words of length 14 occur; block entropy `H_n/n > 0.999` up to `n = 13`;
autocorrelation flat to lag 32; the one residue-class hit (`t ≡ 1 mod 3`,
`z = +3.31`) did **not** replicate on a disjoint fresh sample (`z = +1.41`,
AUDIT §6.2).

**Possible counterexample.** Any block-frequency bias that grows like `√N`.
The mod-3 candidate is the one that was checked and killed.

**Computational falsification test.** Chi-square on `k`-block counts at
increasing `N`, with the discovery/replication discipline of AUDIT §6:
any hit must be re-tested on a disjoint sample before being reported.

**If proved.** Would imply C1 immediately (an eventually periodic sequence has
at most `T + p` distinct factors of each length, so it is not normal).
**Warning: this is far harder than C1**, and is registered mainly as a
statement of what the statistics do *not* show. Chasing normality is not a
route to the prize.

---

## Phase 2 priority ordering

1. **C5** — the only registered item with a decisive, runnable experiment whose
   two outcomes point to genuinely different proof programmes.
2. **C7** — formalise the partial-knowledge claim; it is the only registered
   route that might shorten the gap rather than characterise it.
3. **C3 / C2** — proof targets, not experiments. Attack only with a global
   argument that uses the single-cell seed (WIDTH2 §10.3 shows any argument
   blind to the seed is necessarily wrong).
4. **C8, C9** — descriptive; do not confuse progress here with progress on C1.
