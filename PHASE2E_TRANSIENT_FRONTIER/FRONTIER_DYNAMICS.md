# Frontier Dynamics — Phase 2E §4

How fast the periodic frontier `K_per(t) = max{K : T(K) ≤ t}` advances, and how
fast `T(K)` climbs. Everything provable here is a **bound**; the measured rates
are reported with their range and **no asymptotic constant is fitted**.

---

## 1. Bounds on `T(K+1) - T(K)`

### THEOREM 4.1 (lower bound)

> `T(K+1) - T(K) ≥ 0` for every `K`.

Theorem 1.3.

### THEOREM 4.2 (upper bound)

> If coordinate `K` is not eventually zero, `T(K+1) - T(K) ≤ P(K)`.
> If coordinate `K` is eventually zero, `T(K+1) = T(K)`.

*Proof.* First case: Theorem 3.4. Second case: an eventually-zero coordinate `K`
makes level `K+1` non-COLLAPSING, hence INHERITING by Corollary B4 (via B2/B3);
in fact Phase 2D's Theorem N1 is exactly the equivalence
"level `K+1` non-COLLAPSING ⟺ coordinate `K` eventually zero". ∎

*Verified:* `K ≤ 30000`, **0 violations**. Distribution of `T(K+1)-T(K)`:

```
    step    0     1     2     3     4    5    6    7   8   9  10  11  12  13  14  15
    count 13605 4387  6228  2825  1458  733  375  200  84  48  32  11   2   6   5   1
```

Maximum observed step `15`; the bound is `P = 16` for all `K ≥ 400`. **The
bound is attained to within one and is therefore essentially sharp on this
range.**

### COROLLARY 4.3 (a rigorous linear upper bound on `T`)

> For `K ≥ 400`, `T(K) ≤ T(400) + 16·(K - 400) = 501 + 16(K-400)`.

*Proof.* Sum Theorem 4.2 with `P(K) = 16` (Phase 2C: no period doubling occurs
after `K = 400` on the computed range — see the caveat below). ∎

**Caveat, stated because it matters.** `P(K) = 16` for `400 ≤ K ≤ 30000` is a
**BOUNDED OBSERVATION**. If a fifth doubling point exists beyond `30000`, the
constant `16` in Corollary 4.3 becomes `32` from that point on. The corollary is
therefore stated for the computed range; the *shape* of the bound (linear, with
slope the current period) holds unconditionally on any range with no doubling.

### What is NOT proved

* No lower bound of the form `T(K) ≥ cK` with `c > 1/2` is proved anywhere in
  this package. The best proved lower bound remains Phase 2C's
  `T(K) > ⌊K/2⌋ - P(K)`, which comes from the support bound `w_t(k) = 0` for
  `k > 2t` and is far below the measured `≈ 1.34 K`.
* The measured ratio `T(K)/K` is **not** monotone and **not** fitted:

  ```
        K      18     100    400    1000   5000   10000   20000   30000
      T(K)     20     134    501    1275   6646   13315   26839   40197
      T/K    1.1111  1.3400 1.2525 1.2750 1.3292  1.3315  1.3419  1.3399
  ```

  It falls from `1.3400` at `K = 100` to `1.2525` at `K = 400` (the last
  period-doubling level) and rises again. **No limit is claimed to exist**, and
  the phrase "`T(K) ≈ 1.34 K`" used loosely in earlier phases should be read as
  "the measured value at the largest `K` computed", nothing more.

---

## 2. Bounds on `K_per(t+1) - K_per(t)`

### PROPOSITION 4.4

> `K_per(t+1) - K_per(t) = #{ K : T(K) = t+1 }`, and this set is an interval of
> consecutive levels.

*Proof.* The first equality is the definition plus Theorem 1.3; the interval
claim is monotonicity of `T`. ∎

### COROLLARY 4.5

> `K_per(t+1) - K_per(t)` equals `1 +` (the number of INHERITING levels
> immediately following the RESETTING level that lands on `t+1`), or `0` if no
> level has `T(K) = t+1`.

So the frontier's step distribution and the gap distribution between RESETTING
levels are the same statistic viewed from the two sides. Measured, `t ≤ 40197`:

```
    step     0     1     2    3    4    5   6   7  8  9 10 12
    count 23802  5930  8786  830  420  325  62  26  7  3  3  3
```

maximum step **12** — exactly the maximal run of consecutive INHERITING levels
(§3.4), as Corollary 4.5 predicts.

### COROLLARY 4.6 (rigorous lower bound on the frontier's speed)

> On any range where `P(K) = P` and no coordinate is eventually zero,
> `K_per(t) ≥ K_0 + (t - T(K_0))/P`.

*Proof.* Invert Corollary 4.3. ∎

For `400 ≤ K ≤ 30000` this gives slope `≥ 1/16 = 0.0625`; the measured slope is
`≈ 1/1.34 = 0.746`. **The proved bound is a factor of 12 away from the
measurement, and closing that gap is the whole difficulty.**

---

## 3. The diagonal against the frontier

The diagonal reads level `K = t` at time `t`. So it is inside the periodic
region iff `t ≥ T(t)`, i.e. iff `A(t) = T(t) - t ≤ 0`.

```
        t        18     100    400    1000   5000   10000   20000   30000
      A(t)        2      34    101     275   1646    3315    6839   10197
```

* `T(K) ≤ K` holds **exactly** for `K ∈ {0,…,17}` — those 18 levels are the
  only ones where the diagonal reads settled data, and they are recorded as
  counterexamples to the unqualified statement, not hidden.
* For `18 ≤ K ≤ 30000` the diagonal is strictly inside the transient, with a
  deficit growing roughly linearly. **BOUNDED OBSERVATION.**
* **This is not evidence of non-periodicity of the centre column.** It says the
  centre column is read out of the transient; a transient can perfectly well
  emit an eventually periodic sequence. Phase 2B's registry entry G5 makes the
  same point and is repeated here because the temptation to over-read it is
  strong.

---

## 4. Controls

* Theorems 4.1, 4.2, Propositions 4.4 and Corollaries 4.5, 4.6 hold for all
  `K` / all `t` **on any range satisfying their stated hypotheses**. The
  hypotheses ("`P(K) = 16`", "coordinate `K` not eventually zero") are BOUNDED
  OBSERVATIONS and are always written out.
* **No asymptotic constant is fitted.** `1.3399`, `0.746`, `1.8299` are values
  measured at `K = 30000` and are reported as such. No regression, no
  extrapolation, no claimed limit.
* The non-monotonicity of `T(K)/K` is reported prominently precisely because it
  is the kind of thing a fitted constant would hide.

---

## 5. Is there a finite-state description of the frontier increments?

The brief asks for a finite-state or recursive description of the increments
`δ(K) = T(K) - T(K-1)`. The answer splits cleanly, and both halves are proved.

### THEOREM C1 (the increment IS a finite-state function — given one extra bit)

> Let `P(K) = P(K-1) = P` and let `c` be the base cycle word of coordinate
> `K-1` read from phase `T(K-1)` (a `P`-bit word). Then
>
> ```
>     δ(K)  =  0                                    if D(K) = 0
>     δ(K)  =  1 + (index of the lowest set bit of c)   if D(K) = 1
> ```

*Proof.* Theorem B1 plus the definition of `τ_K` as the first `1` of `c`. ∎

*Verified:* 29995 levels, `K ≤ 30000`, **0 mismatches**.

So a machine holding `(c, D)` — 17 bits at `P = 16` — outputs the increment
exactly. The whole question is whether such a machine can be *driven*, i.e.
whether `D` is itself computable from cycle data.

### THEOREM C2 (the defect bit is one transient bit away from cycle data)

> For a COLLAPSING level,
>
> ```
>     D(K)  =  w_{T(K-1)}(K)   XOR   Φ_K
> ```
>
> where `Φ_K` — Phase 2D's one-period reset constant — depends only on the base
> cycle word.

*Proof.* Phase 2D's R2 says the one-period fibre map from `T(K-1)` is the
constant `Φ_K`, so `w_{T(K-1)+P}(K) = Φ_K` whatever `w_{T(K-1)}(K)` is. Then
`D(K) = w_{T(K-1)}(K) XOR w_{T(K-1)+P}(K) = w_{T(K-1)}(K) XOR Φ_K`. ∎

*Verified:* 29995 levels, **0 mismatches**.

### COROLLARY C3 (no finite-state description from cycle data)

> `D(K)` is **not** a function of the base cycle words at any depth: flipping the
> single transient bit `w_{T(K-1)}(K)` flips `D(K)` while leaving every cycle
> word below unchanged.

*Proof.* Immediate from C2, since `Φ_K` is a function of the base cycle word
alone. ∎

*Verified directly*, not only through C2: re-running the level-`K` fibre from
`T(K-1)` with the fibre bit flipped, holding the base orbit fixed, flips the
defect bit at **every** level tested (400 levels, 0 failures).

**Consequence.** The frontier increment sequence is generated by a machine that
consumes **exactly one new transient bit per level**. No fixed-width summary of
the cycle structure — however deep — reproduces it. This is a theorem, not a
failed search.

### The corresponding search on the real orbit, and its limits

For completeness the collision search the brief's phrasing suggests was also
run, comparing levels with identical windows of cycle words:

| window depth | distinct windows over 29999 levels | levels sharing a window with a different `D` |
|---|---|---|
| 1 (predecessor cycle word only) | 22973 | **2545 counterexamples** — refuted |
| 2 | 29995 | 0 |

**The depth-2 row is not evidence of anything.** With 32-bit windows and only
30000 levels, essentially no two levels share a window, so the test has no
power at depth ≥ 2. It is reported to make that explicit. Corollary C3 is the
statement that actually settles the question, and it settles it negatively for
every depth.

### Is the increment word itself eventually periodic in `K`?

No repeated block of length 30 occurs anywhere in the increment word for
`K ≤ 30000` (29969 windows, all distinct). **This is a statistic, not a proof**:
a repeated block would not have implied eventual periodicity either, and its
absence implies nothing about large `K`. Registered as F3 in
`TRANSIENT_LEMMA_REGISTRY.md`.
