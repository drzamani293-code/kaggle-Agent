# Transient Lemma Registry — Phase 2E §9

Target lemmas **F1 – F5**. Same conventions as the Phase 1 and Phase 2B
registries: **THEOREM** (proved, all `K`), **CONJECTURE** (precise, unproved),
**BOUNDED OBSERVATION** (measured, with range), **STRATEGY ASSESSMENT** (a
judgement about a route). Failed and refuted items are kept, never deleted.

Prior registries this one sits beside — nothing here supersedes them:

* Phase 1 `CONJECTURE_REGISTRY.md`: C2 **the Bridge** (open), C3 (vacuity).
* Phase 2B `GLOBAL_LEMMA_REGISTRY.md`: G1 (= the Bridge, open), G2 (refuted as a
  route), G3 (circular), G4 (partially refuted), G5 (open).
* Phase 2D `COLLAPSE_BRIDGE_REGISTRY.md`: C1–C4.

---

## F1 — the transient outruns the diagonal, in reset form

**Formal statement.**
```
    liminf_{K → ∞}  d(K) · m(K)  >  1 ,
```
where `d(K)` is the fraction of levels `K' ≤ K` that are RESETTING and `m(K)`
is the mean reset increment `δ(K') = τ_{K'} + 1 - T(K'-1)` over those levels.

**Relation to earlier work.** By Corollary B5, `T(K)/K = d(K)·m(K) + O(1/K)`,
so **F1 is exactly Phase 2B's G5** (`liminf T(K)/K > 1`), re-expressed in terms
of the two quantities Phase 2E can actually measure and classify. The
equivalence is a **THEOREM** (Corollary B5); the statement itself is **OPEN**.

**What Phase 2E contributes.** Not a proof, and not a reduction — a
factorisation. G5 was one number with no internal structure; F1 splits it into
a **density** (how often the defect bit `D(K)` is 1) and a **wait** (how long
after `T(K-1)` the previous coordinate first shows a 1). Each factor is now
exactly computable level by level, and each has an exact characterisation
(Theorem B1, Theorem 3.4).

**Current evidence.** BOUNDED OBSERVATION, `K ≤ 30000`:
`d = 0.546485`, `m = 2.451873`, `d·m = 1.339911`, against `T(K)/K = 1.339900`.

**What is proved about the factors.**
* `m(K) ≤ P(K)` — **THEOREM 3.4**, sharp to within one (max observed 15 vs 16).
* `m(K) ≥ 1` — trivial.
* `d(K) ≤ 1` — trivial.
* **No lower bound on `d(K)` is proved.** This is the gap.

**Falsification test.** A single `K` with `T(K) ≤ K` refutes F1's weak form; a
long enough run of INHERITING levels would drive `d` down. Both are cheap to
test and neither has occurred for `18 ≤ K ≤ 30000`.

**Caveat, repeated from Phase 2B.** F1 is a reformulation with one half proved,
not progress on Problem 1. Proving F1 is not obviously easier than the problem,
and **no route from F1 to a contradiction with (H) exists** — see
`TRANSIENT_PERIODICITY_CONSEQUENCES.md` Route A.

**Status: OPEN. Inherits G5's rank.**

---

## F2 — a positive density of levels has defect bit 1

**Formal statement.** `liminf_{K→∞} (1/K)·#{ K' ≤ K : D(K') = 1 } > 0`, where
`D(K) = w_{T(K-1)}(K) XOR w_{T(K-1)+P(K)}(K)`.

**Why it matters.** By Theorem B1, `D(K) = 1` is *exactly* the condition for
level `K` to add transient. F2 is the first factor of F1 and is strictly weaker
than F1. It is the smallest statement in this package whose proof would be new
information.

**Current evidence.** BOUNDED OBSERVATION, `K ≤ 30000`: 16394 of 29999 levels,
density `0.546485`. Longest observed run of `D = 0` is **12** levels.

**Known counterexamples in relaxed models.** None constructed. But note the
warning: `D(K)` is a single bit read off the real orbit at two explicit times,
and its definition uses `T(K-1)` — so it is intrinsically seed-dependent, like
G5/F1 and unlike G1–G4. By Phase 1 §10.3 that is a **necessary** feature of any
route that can work.

**Falsification test.** An infinite run of `D = 0` would refute it. Cheap to
monitor: the RESETTING gap histogram in `RESET_SCHEDULE_RESULTS.csv`.

**Dependencies.** Theorem B1 (proved). No literature dependency.

**Status: OPEN. New in Phase 2E. Recommended as the smallest non-trivial
target, while being explicit that it is a step towards F1/G5, not towards the
Bridge.**

---

## F3 — the reset word is not eventually periodic in `K`

**Formal statement.** The infinite word `Ω ∈ {R, I}^ℕ` with `Ω_K = R` iff level
`K` is RESETTING is not eventually periodic.

**Current evidence.** BOUNDED OBSERVATION only. Over `K ≤ 30000` the word has
16394 `R`s; the gap histogram is `{1: 5928, 2: 8786, 3: 830, 4: 420, 5: 325,
6: 62, 7: 26, 8: 7, 9: 3, 10: 3, 12: 3}`, max gap 12.

**Why it is registered, and why it is ranked low.**
* If `Ω` *were* eventually periodic with mean increment structure also periodic,
  `T(K)` would be eventually affine in `K`, so `T(K)/K` would converge — which
  is not itself a contradiction with anything.
* More importantly, **F3 does not connect to the diagonal**. Route B in
  `TRANSIENT_PERIODICITY_CONSEQUENCES.md` §3 is the only route from F3 to
  Problem 1, and its second dependency is *false* (§7: the diagonal is not
  determined by a finite summary of reset events).

**STRATEGY ASSESSMENT: F3 is precise, open, and — as currently connected —
useless for Problem 1. Registered so it is not rediscovered as promising.**

---

## F4 — every level beyond 400 is COLLAPSING

**Formal statement.** For every `K > 400`, the level-`K` fibre is COLLAPSING;
equivalently (Phase 2D **Theorem N1**) no coordinate `k ≥ 399` is eventually
zero.

**Current evidence.** BOUNDED OBSERVATION, `401 ≤ K ≤ 30000`: all COLLAPSING;
the eventually-zero coordinates are exactly `{2, 7, 28, 399}`. Coordinate `7` is
permanently zero (**THEOREM**, Phase 2D, via the explicit 2-cycle
`11001000 / 11011110`); the other three are eventually zero.

**Why it matters here.** F4 is the standing hypothesis of Theorem 3.4 and
Theorem 4.2 (`T(K+1) - T(K) ≤ P(K)`) and hence of Corollary 4.3's linear upper
bound on `T`. Every use of those results in this package is written with the
hypothesis visible.

**Warning inherited from Phase 2D, repeated because it is easy to forget.** The
evidence for "no further non-COLLAPSING level" is much weaker than "29 600
consecutive confirmations" suggests: by N1 there were only **four**
opportunities in the whole range, and all four were resolved by a parity coming
out odd. Four events, not 29 600.

**Status: OPEN. Bounded evidence only. Never to be cited as an all-`K`
theorem.**

---

## F5 — some column other than the centre is eventually periodic

**Formal statement.** There exist `r ≠ 0`, `q ≥ 1`, `T'` with
`x_{t+q}(r) = x_t(r)` for all `t ≥ T'`.

**Why it matters.** By Phase 1 **Theorem W2′** (at most one eventually periodic
column), F5 **immediately refutes (H)** and settles Problem 1 in the expected
direction. It is the only item in this registry that would close the problem,
and it is a *positive* existence statement rather than a negative one — a
different logical shape from every other target in the project.

**Current evidence.** Strongly negative, and honestly so. Proposition 8.6
refutes every `(preperiod, period)` budget with `q ≤ 32` and preperiod below
`≈ 19 990` for all `r ∈ [-4, 4]`; Phase 1's factor-counting bound gives
`T + p ≥ 998140` for the centre column at `p ≤ 20000`. Nobody expects any
column to be eventually periodic.

**STRATEGY ASSESSMENT.** F5 is registered not because it is plausible but
because it is the only *proved* one-step route from a positive finite discovery
to a solution, and because Theorem 8.4 makes its relationship to (H) exact. It
should be treated as a sanity anchor, not a research direction.

**Status: OPEN, believed false, retained deliberately.**

---

## Priority ordering

| Rank | Item | Why | Status |
|---|---|---|---|
| — | **Phase 1 C2 / Phase 2B G1 — the Bridge** | would close the problem with W2′ | **OPEN, and untouched by Phase 2E** |
| 1 | **F2** | smallest genuinely new statement; seed-dependent; exactly characterised by Theorem B1 | OPEN, new |
| 2 | **F1** (= G5, factorised) | equivalent to G5, now with internal structure | OPEN |
| 3 | **F4** | hypothesis of several results used here | OPEN, weak evidence |
| 4 | F3 | precise and open, but **not connected** to the diagonal | OPEN, low value |
| 5 | F5 | one-step route if ever found true; believed false | OPEN, retained as an anchor |

---

## Cross-cutting facts any future attempt must respect

1. **Seed-blindness is fatal** (Phase 1 §10.3). F1, F2 and F4 are
   seed-dependent; F3 is a statement about the real orbit too. Nothing in
   Phase 2E is seed-blind — which is necessary but nowhere near sufficient.
2. **The diagonal is not determined by the settled structure.** §6 (69 % of the
   ancestry never settles) and §7 (the periodicity rewrite saves a factor of
   1.45 and leaves a quadratic DAG) are two independent measurements of the same
   obstruction. Any route that only uses cycle data is blocked before it starts.
3. **The co-moving strip is the original diagram** (Theorem 5.1) and is **not**
   a finite-state system (Theorem 5.4). Re-coordinatising the centre column
   gains nothing by itself.
4. **Conditionals on (H) cannot be tested** — only their finitary shadows, and
   Phase 2A showed those can be refuted without touching the conditional.
5. **The Bridge remains untouched.** Phases 2A, 2B, 2C, 2D and 2E have each
   ended at it from a different direction. Phase 2E's contribution is a sharper
   description of *why* the transient side is where the difficulty lives, not a
   step across.
