# Comparison with Rowland, "Local Nested Structure in Rule 30"

**Complex Systems 16 (3) (2006) 239–258.** DOI `10.25088/ComplexSystems.16.3.239`.

> **The paper was not retrieved.** See `RETRIEVAL_LOG.md`: every fetch returned
> HTTP 403 at the proxy, including from the author's own site. Everything below
> attributed to Rowland is **UNVERIFIED — SECONDARY** (search-engine summary)
> or **UNVERIFIED — FIGURE CAPTION**, never a reading.
>
> Nonetheless this is the most consequential comparison in the corpus, because
> one attributed *numerical* statement was testable — and it matched exactly.

---

## 1. What is attributed to the paper

| # | attributed statement | source status |
|---|---|---|
| R-a | Rule 30 is drawn **left-justified**; in that presentation "the columns are eventually periodic, with period doublings" | **FIGURE CAPTION** (fig. 5) |
| R-b | Rule 86 is used as "a left-right reflected, left-justified image of rule 30" | **FIGURE CAPTION** (fig. 4) |
| R-c | "Each left diagonal of rule 30 is eventually periodic with period length a power of 2"; the left side is characterised by eventual periodicity and loss of information | SECONDARY |
| R-d | **Proposition 2**: "Each period doubling … occur[s] exactly when a diagonal in the pattern eventually becomes a white stripe, and the diagonal to its left has an odd number of black cells in each repeating block", and this "holds for rule 30 beginning from **any** initial condition" | SECONDARY (with the inner clause quoted in the summary) |
| R-e | From the leftmost nonwhite diagonal, the eventual period lengths begin `1,1,1,2,1,2,2,1,4,1,4,4,4,4,4,4,4,4,4,4,4,4,4,2,4,4,4,4,1,8,1,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,8,8` | SECONDARY |
| R-f | At row `2^n` a region of the initial condition reappears on the right side, so the automaton "begins again" locally, producing local nested structure | SECONDARY |
| R-g | The mechanism is that rule 30 is **reversible in time under the condition that the right half of each row is white** | SECONDARY |
| R-h | The order/chaos boundary moves left by **about 0.252 cells per step** on average | SECONDARY (reported in several places, not specifically attributed to this paper) |

## 2. The decisive test — and it passed

R-e is 48 specific integers. We computed the exact minimal eventual period of
each single left diagonal `k` of our edge-aligned tower, `k = 0 … 47`, by two
independent methods (a lag-scan for the last failure, and direct minimal-period
detection on the tail word):

```
ours      : 1 1 1 2 1 2 2 1 4 1 4 4 4 4 4 4 4 4 4 4 4 4 4 2 4 4 4 4 1 8 1 8 8 8 8 8 8 8 8 8 8 8 8 8 8 4 8 8
attributed: 1 1 1 2 1 2 2 1 4 1 4 4 4 4 4 4 4 4 4 4 4 4 4 2 4 4 4 4 1 8 1 8 8 8 8 8 8 8 8 8 8 8 8 8 8 4 8 8
```

**Exact match on all 48 terms** (`verify_against_literature.py`;
**VERIFIED — OUR COMPUTATION**). Also verified: our prefix period `P(K)` is
exactly the running maximum of these diagonal periods, and the positions where
the running maximum first increases are `k = 3, 8, 29` — our doubling points.

### What the match establishes

1. **Our edge-aligned coordinate `w_t(k)` is Rowland's `k`-th left diagonal**,
   indexed from the leftmost nonwhite one. The identification is not a guess.
2. **Our rule-numbering convention is externally corroborated.** This was
   flagged in Phase 2F as "the corpus's single largest unverified assumption"
   (`CC-01`'s caveat, `NOVELTY_STATUS.md` §4 item 5). A 48-term agreement with a
   published sequence retires it — not with certainty, since the sequence
   reached us second-hand, but the probability of a 48-term coincidence under a
   wrong rule is negligible.
3. **R-c, R-d and R-e are about exactly our objects**, so the novelty
   assessments below are not speculative analogies.

A second, weaker corroboration: our measured frontier gives
`T(30000)/30000 = 1.339900`, i.e. an implied boundary speed
`1 − 1/1.3399 = 0.25368` against the attributed `0.252` — a **0.67 %**
difference. Our Phase 1 `BO-08` and Phase 2C/2E `BO-09` are therefore
re-measurements of a **known constant**, not new observations.

## 3. Correspondence table

Labels as required by the brief: **EXACTLY KNOWN** · **IMMEDIATE COROLLARY** ·
**REFORMULATION** · **STRICT STRENGTHENING** · **ORBIT-SPECIFIC REFINEMENT** ·
**APPARENTLY NOT PRESENT** · **UNCERTAIN**.

Because the text was not read, **no cell can be certified**. Each row gives the
label our reconstruction *predicts*, and the confidence in it. A row marked
"predicted EXACTLY KNOWN (high)" should be treated as known until someone with
the paper says otherwise — that is the safe direction, and the brief's
instruction "do not claim discovery of any known structure" requires it.

| our result | Rowland item | predicted label | confidence | note |
|---|---|---|---|---|
| **T-07** edge-aligned one-sided recurrence `w_{t+1}(k)=w_t(k-2) XOR (w_t(k-1) OR w_t(k))` | R-a (left-justified presentation) | **REFORMULATION** of a presentation Rowland already uses | **high** | he draws the left-justified diagram; whether he writes the one-sided *recurrence* is unknown |
| **T-08** every fixed prefix is eventually periodic | R-c | **IMMEDIATE COROLLARY** of R-c (finitely many diagonals, take the lcm) — or EXACTLY KNOWN if he states it for prefixes | **high** | ours is the vector form of his per-diagonal statement |
| **T-11** `P(K) ∈ {P(K-1), 2P(K-1)}`, periods are powers of 2 | R-c + R-d | **EXACTLY KNOWN** | **high** | R-c gives "power of 2"; R-d gives the doubling criterion. **Brief task 4 answered: yes, Rowland has this, at least implicitly and probably explicitly.** |
| **T-10** fibre trichotomy COLLAPSING / NEUTRAL / DOUBLING | R-d | **REFORMULATION** — R-d is the DOUBLING criterion; the three-way split is our packaging | **high** | see §4 |
| **T-14** non-COLLAPSING ⟺ predecessor diagonal eventually zero | R-d ("becomes a white stripe") | **EXACTLY KNOWN** | **high** | see §4 |
| **T-22** `w_t(7)=0` for all `t` | consistent with R-e (period 1 at `k = 7`) and with R-d's "white stripe" | **ORBIT-SPECIFIC REFINEMENT** already implicit in his data | **medium** | our proof (an explicit 2-cycle) is a triviality either way |
| **T-01** left-permutivity, left reconstruction | R-g (reversibility given a white right half) | **EXACTLY KNOWN** — R-g is a sharper, more useful form | **high** | his conditional reversibility is *stronger* than plain left-permutivity |
| **T-09** projection `π_K ∘ F_{K+1} = F_K ∘ π_K` | implicit in any left-justified treatment | **UNCERTAIN**, likely folklore | medium | trivial once R-a is set up |
| **T-12** `T(K-1) ≤ T(K)` | not mentioned in any summary | **UNCERTAIN** | low | trivially provable; likely folklore |
| **T-16** exact reset calculus (R1 erasure, R2 the constant `Φ`, R3 `T(K) ≤ τ+1`) | — | **UNCERTAIN**, plausibly not present | **low–medium** | no summary mentions preperiods at all |
| **T-18** the exact preperiod recurrence via the defect bit `D(K)` | — | **UNCERTAIN**, plausibly not present | **low–medium** | *the* open question for this comparison |
| **T-19** `T(K) = ρ(K)+1` on resetting levels | — | **UNCERTAIN**, plausibly not present | low–medium | same |
| **T-21.5** `D(K) = w_{T(K-1)}(K) XOR Φ_K`; not finite-state in cycle data | — | **UNCERTAIN**, plausibly not present | low–medium | depends on T-18 |
| **T-17** unique cycle word; `2P`-bit transducer | — | **UNCERTAIN** | low | a chain statement about the cycle region |
| **T-02 / T-03** column periodicity (fixed vertical columns, not diagonals) | — | **APPARENTLY NOT PRESENT** in Rowland | medium | different object: he studies *diagonals*, we also study *columns*. See `JEN_COMPARISON.md` |
| **T-05 / T-06** defect field and wall identities | — | **APPARENTLY NOT PRESENT** | medium | different apparatus |
| **T-20** XOR-spine lower bound; **T-23** the diagonal's cone | — | **UNCERTAIN** | low | elementary |
| **BO-08 / BO-09** boundary slope ≈ 0.746, `T(K)/K ≈ 1.34` | R-h | **EXACTLY KNOWN** (a re-measurement of `≈ 0.252`) | **high** | our numbers agree to 0.67 % |
| **R-f / R-g** the `2^n` nested-restart mechanism | — | **APPARENTLY NOT PRESENT IN OURS** | **high** | *Rowland has something we do not.* See §5 |

## 4. The single most important correspondence

Rowland's **Proposition 2**, as quoted second-hand:

> each period doubling occurs exactly when a diagonal eventually becomes a
> white stripe, and the diagonal to its left has an odd number of black cells
> in each repeating block

Our **T-10 + T-14** say: level `K` is DOUBLING (i.e. `P(K) = 2P(K-1)`) **iff**
`w_t(K-1) = 0` throughout its cycle (coordinate `K-1` is eventually zero — "a
diagonal becomes a white stripe") **and** `XOR_t w_t(K-2) = 1` over the cycle
(coordinate `K-2`, which in left-justified indexing is the diagonal *to the
left* of `K-1`, has an odd number of black cells per repeating block).

**These are the same statement**, under the identification established in §2,
including the direction of "to its left". Both are stated for arbitrary initial
conditions.

**Conclusion (predicted, high confidence): T-14 and the DOUBLING half of T-10
are already in the literature, published in 2006.** They must be relabelled, and
they are — `REVISED_NOVELTY_LEDGER.md`. Phase 2F listed T-13–T-19 as "POSSIBLY
NEW as a block"; **that block assessment was wrong and is withdrawn.**

Our Phase 2D framing of this material — "four opportunities, four odd parities"
— now reads differently: Rowland already characterised *when* a doubling
happens; our contribution there was to *count the opportunities* in the
single-cell orbit and observe that all four resolved as doubling. That is an
**ORBIT-SPECIFIC REFINEMENT** of his proposition, not a discovery of it.

## 5. What Rowland has that we do not

R-f and R-g describe a mechanism we never found: at row `2^n` a region of the
initial condition **reappears on the right**, making the automaton restart
locally, and the reason is a **conditional time-reversibility** — rule 30 is
invertible backwards provided the right half of each row is white.

We have nothing like this. Our T-01 gives *leftward* reconstruction from two
adjacent columns; Rowland's is a *global* backward step under a support
condition, and it produces the nested structure on the right — the side our
entire corpus treats as opaque. Our Phase 2B–2E all work on the left/prefix
side precisely because we had no handle on the right.

**This is the most valuable thing this comparison found**, and it is a gap in
our work rather than an overlap. If a future phase is authorised, reading R-f
and R-g properly is the first thing to do.

## 6. Brief task 4, answered explicitly

> *Check whether Rowland already proves `P(K) ∈ {P(K-1), 2P(K-1)}` for binary
> Rule 30, explicitly or implicitly.*

**Predicted answer: yes.** R-c states every left diagonal is eventually periodic
with period a power of 2; R-d gives the exact criterion for each doubling; R-e
tabulates the resulting period sequence, which we reproduced exactly. Taken
together those give `P(K) ∈ {P(K-1), 2P(K-1)}` at least implicitly, and the
existence of a numbered Proposition about the doublings makes an explicit
statement likely.

**Confidence: high. Verification still required** — the label in every table
above remains formally UNCERTAIN until someone reads §§ of the paper.

## 7. Brief task 5, partially answered

> *Check whether Rowland contains an exact analogue of: the `T(K)` recurrence
> using `D(K)`; the `σ/τ/ρ` reset times; non-COLLAPSING iff predecessor cycle
> is zero.*

* **non-COLLAPSING ⟺ predecessor eventually zero** — **yes, predicted present**
  as half of Proposition 2 (§4).
* **the `T(K)` recurrence via `D(K)`, and `σ/τ/ρ`** — **no indication either
  way.** No summary of Rowland mentions preperiods, transients, or reset times;
  every attributed statement is about *periods*. That is weak evidence of
  absence, and it is the reason T-18 is nominated as the strongest plausibly-new
  result (`REVISED_NOVELTY_LEDGER.md` §4) — with the caveat that a paper about
  exactly this structure is the single most likely place for such a recurrence
  to be hiding.
