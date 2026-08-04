# NEUTRAL Exclusion Attempt

Why NEUTRAL has not appeared, what would be needed to exclude it, and an
explicit counterexample showing no seed-free exclusion theorem exists.

---

## 1. Exactly what NEUTRAL requires

At level `K` the fibre is driven by `a_t = w_t(K-2)`, `c_t = w_t(K-1)`. From
the trichotomy (Phase 2C Lemma C), NEUTRAL holds iff

1. `c_t = w_t(K-1) = 0` for **every** phase of the base cycle, i.e. coordinate
   `K-1` is **eventually zero**; and
2. `XOR_t a_t = XOR_t w_t(K-2) = 0` over the cycle (even parity).

## 2. The opportunity characterisation — **THEOREM N1**

> Level `K` is **not** COLLAPSING **iff** coordinate `K-1` is eventually zero
> (zero throughout its own cycle).

*Proof.* Immediate from the definitions: COLLAPSING is `some c_t = 1` on the
cycle, and `c_t = w_t(K-1)`. ∎

This reframes the question entirely. NEUTRAL is not rare because some mechanism
suppresses it; it is rare because **opportunities are rare** — and at each
opportunity it is a parity condition that decides NEUTRAL versus DOUBLING.

**Measured (`K <= 30000`, exact):** the eventually-zero coordinates are

```
    { 2, 7, 28, 399 }        (exactly four)
```

so the non-COLLAPSING levels are `{3, 8, 29, 400}` — **precisely the four
doubling points**. Verified: `non_collapsing_levels == {k+1 : k eventually
zero}` is `True`.

Of the four coordinates, only `K = 7` is **permanently** zero (zero for all
`t`, including the transient); `2, 28, 399` are nonzero somewhere in their
transient and zero thereafter.

> **So the correct statement of "NEUTRAL never occurs" is: there were exactly
> four opportunities, and all four resolved as DOUBLING (odd parity).**
>
> Four coin-flips' worth of evidence. This is much weaker than "NEUTRAL is
> excluded", and the earlier Phase 2C phrasing ("NEUTRAL never occurs in the
> real orbit") should be read with that in mind.

## 3. A derived identity at every opportunity — **THEOREM N2**

> If coordinate `m` is zero throughout its cycle, then
> ```
>     w_t(m-2) = w_t(m-1)      for every t in that cycle.
> ```

*Proof.* Apply the rule at coordinate `m` for `t` on the cycle:
`0 = w_{t+1}(m) = w_t(m-2) XOR ( w_t(m-1) OR w_t(m) ) = w_t(m-2) XOR
( w_t(m-1) OR 0 ) = w_t(m-2) XOR w_t(m-1)`. ∎

*Verification.* Holds at all four opportunities `m = 2, 7, 28, 399`.

**Consequence for the parity.** At an opportunity `K = m+1`, the parity that
decides NEUTRAL vs DOUBLING is `XOR_t w_t(K-2) = XOR_t w_t(m-1)`, and by N2
this equals `XOR_t w_t(m-2)`. Measured at all four: **parity = 1 in every
case**, for both `m-1` and `m-2`.

N2 converts the parity question one level down but **does not resolve it**. No
identity we found forces the parity to be odd.

## 4. Exclusion attempts, and why each fails

| tool | attempt | outcome |
|---|---|---|
| **support edge** | `w_t(k) = 0` for `k > 2t` forces zeros only in the transient, never on the cycle | gives nothing on the cycle — **fails** |
| **left-permutivity** | in edge-aligned coordinates the rule is one-sided; permutivity is already used to get the tower and the fibre trichotomy | already spent — **no further leverage** |
| **projection tower** | `P(K-1) \| P(K)` constrains periods, not the parity of a cycle word | **fails** |
| **cycle parity identities** | N2 relates the parity at `m-1` to the one at `m-2` | shifts the question, does not close it — **fails** |
| **the permanently-zero `K = 7`** | is one of the four opportunities; gave odd parity, hence DOUBLING at `K = 8` | consistent, but a single data point — **no general statement** |

**No exclusion theorem was obtained.**

## 5. NEUTRAL is realisable in the relaxed model — explicit counterexamples

Any seed-free exclusion attempt is doomed, because NEUTRAL occurs freely among
abstract forcing words. Two minimal explicit counterexamples:

```
    period 1:   (a, c) = (0, 0)          ->  Phi = identity       -> NEUTRAL
    period 2:   (a, c) = (1, 0), (1, 0)  ->  Phi = identity       -> NEUTRAL
```

Both satisfy the two NEUTRAL conditions exactly. Among the 43,690 abstract
forcing words of lengths 1–9 enumerated in Phase 2C, **156 are NEUTRAL** — the
class is far from empty. Any proof that NEUTRAL never occurs in the rule-30
tower must therefore use the single-cell seed, exactly as Phase 1 §10.3
predicted for every seed-blind route.

## 6. Labels

| claim | label |
|---|---|
| `w_t(7) = 0` for all `t` | **THEOREM** (2-cycle exhibited, Phase 2C) |
| no other permanently-zero coordinate exists for `K <= 3000` | **BOUNDED OBSERVATION** |
| eventually-zero coordinates are `{2,7,28,399}` for `K <= 30000` | **BOUNDED OBSERVATION** (exact within the range) |
| N1 (non-COLLAPSING iff predecessor eventually zero) | **THEOREM** |
| N2 (`w_t(m-2) = w_t(m-1)` on a zero cycle) | **THEOREM** |
| NEUTRAL does not occur for `K <= 30000` | **BOUNDED OBSERVATION** — four opportunities, four odd parities |
| NEUTRAL never occurs in the real tower | **CONJECTURE**, weakly supported |
| NEUTRAL is impossible in general | **FALSE** — counterexamples in §5 |
