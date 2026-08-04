# Forced-Bit Classification

Once the `(K-1)`-prefix has entered its cycle, coordinate `K` is a one-bit
Boolean recurrence driven by a **purely periodic** forcing word. This document
classifies every possible forcing word and reports which classes actually occur
in the rule-30 orbit.

---

## 1. The setting

For `t >= T(K-1)` the driving pair `(a_t, c_t) = (w_t(K-2), w_t(K-1))` is
periodic with period `P = P(K-1)`, and

```
    u_{t+1} = a_t XOR ( c_t OR u_t ),          u_t = w_t(K).
```

The **forcing word** is `((a_t, c_t))_{t=T}^{T+P-1} ∈ ({0,1}^2)^P`.

## 2. Complete classification — **THEOREM** (Lemma C)

Every forcing word falls into exactly one of three classes, determined by two
scalars: whether any `c_t = 1`, and the parity `XOR_t a_t`.

| class | condition | composition `Phi` | fibre period | extra transient | initial condition matters? |
|---|---|---|---|---|---|
| **COLLAPSING** | some `c_t = 1` | constant | `P` | up to **one full base period** `P` | **no** — the value is forgotten |
| **NEUTRAL** | all `c_t = 0`, `XOR a_t = 0` | identity | `P` | **none** | **yes** — two invariant fibres, both of period `P` |
| **DOUBLING** | all `c_t = 0`, `XOR a_t = 1` | negation | **`2P`** | **none** | selects the phase only; both values lie on one orbit of length `2P` |

*Proof.* In `SKEW_PRODUCT_ANALYSIS.md` §2. *Verification:* 43,690 forcing words
of lengths 1–9 (`verify_classification(9)`); the structural rule predicts the
composed map in **every** case.

### 2.1 The four answers to the brief's questions

* **When does coordinate `K` become periodic?** COLLAPSING: at
  `T(K-1) + P(K-1)` at the latest — and never later, because one constant map
  erases the past. NEUTRAL and DOUBLING: immediately at `T(K-1)`.
* **Does its period double?** Only in the DOUBLING class, and then by exactly a
  factor 2. No other multiplier is possible (Theorem S1).
* **How much additional transient is possible?** At most `P(K-1)`, attained in
  the COLLAPSING class. This is exactly the bound `T(K) <= T(K-1) + P(K-1)`
  (Theorem S2), and it is tight.
* **Does the initial condition select a special branch?** COLLAPSING: no.
  NEUTRAL: **yes** — the fibre splits into two disjoint invariant branches, and
  which one you are on is decided by the value at time `T(K-1)`; this is the
  only class where the seed's history survives into the cycle. DOUBLING: the
  two values are two phases of one cycle, so the branch is not "special".

## 3. Census on the real orbit

`forcing_word_census` reads the actual base cycle at each `K` and classifies it
(`K` from 2 to 5000):

| class | count |
|---|---|
| COLLAPSING | **4995** |
| NEUTRAL | **0** |
| DOUBLING | **4** |

* **DOUBLING occurs at exactly `K = 3, 8, 29, 400`** — precisely the four
  period-doubling points of `PREFIX_PERIOD_RESULTS.md` §4. The classification
  predicts the measured period at **every** `K`
  (`classification_matches_measured_period: true`), which is a strong internal
  consistency check: an independent structural computation reproduces the
  measured `P(K)` sequence exactly.

* **NEUTRAL never occurs.** *(COMPUTATIONAL OBSERVATION, `K <= 5000`.)* This is
  the most interesting entry in the table. NEUTRAL requires `c_t = w_t(K-1) = 0`
  for the whole base period *and* an even number of `a_t = 1`. The first
  condition alone is already rare: a cycle in which coordinate `K-1` is
  identically zero. Since NEUTRAL is the only class in which the initial
  condition survives into the cycle, its absence means:

  > In the rule-30 orbit, **every** coordinate's eventual behaviour at every
  > `K <= 5000` is either forced by the base (COLLAPSING) or is a phase of a
  > doubled cycle (DOUBLING). The seed never selects between two coexisting
  > invariant branches.

  **Not proved, and not extrapolated.** NEUTRAL words exist in abundance among
  abstract words (156 of the 43,690 sampled); they simply do not arise as
  rule-30 base cycles in the measured range.

## 4. The permanently zero coordinate

`coordinate_ever_one(3000, 8000)` finds exactly one `K <= 3000` for which
`w_t(K) = 0` for **every** `t`:

> **`K = 7`. THEOREM.** `T(7) = 2`, `P(7) = 2`, and the 7-prefix cycles between
> `11001000` and `11011110` (bits `k = 0..7`). Both cycle states have bit 7
> equal to 0, and `w_0(7) = w_1(7) = 0`, so `w_t(7) = 0` for all `t`.

This is a genuine exception to the hypothesis of Theorem S4 (the support lower
bound), and is the reason S4 is stated conditionally. Preserved as a
counterexample to the unconditional version.

## 5. What the classification does not settle

* It gives no way to predict *which* class occurs at a given `K` without
  computing the base cycle — the class depends on the cycle word, which is
  itself the object being computed.
* It therefore yields no formula for the doubling points `3, 8, 29, 400`, and
  no prediction of the next one.
* COLLAPSING dominating (4995 of 4999) explains why `P(K)` grows so slowly, but
  gives no lower bound on how often DOUBLING recurs — hence no proof that
  `P(K) -> infinity`, and none that it is bounded.
