# Revised Novelty Ledger

Supersedes the assessments in `RULE30_VERIFIED_THEOREMS_V1/NOVELTY_STATUS.md`.
That file is corrected in place and points here.

**Governing rule for this revision:** where evidence points toward a result
being known, it is relabelled as known. The brief's instruction — *do not claim
discovery of any known structure* — makes the asymmetric treatment correct: an
over-cautious "known" costs nothing, an over-confident "new" is a false claim.

**Nothing here is certified.** No paper was read (`RETRIEVAL_LOG.md`). Labels
are predictions with confidence levels, and every one of them is formally
**pending verification**.

---

## 1. Downgrades — results moved toward "known"

| ID | Phase 2F label | **revised label** | conf. | basis |
|---|---|---|---|---|
| **T-07** edge-aligned / left-justified coordinates | LIKELY KNOWN, source not verified | **KNOWN — Rowland 2006 uses the left-justified presentation** | high | figure caption "Rule 30, left-justified"; our 48-term match |
| **T-08** every fixed prefix eventually periodic | LIKELY KNOWN | **KNOWN — immediate from Rowland's "each left diagonal is eventually periodic"** | high | R-c |
| **T-11** `P(K) ∈ {P(K-1), 2P(K-1)}`, powers of two | POSSIBLY NEW as part of a package | **KNOWN — Rowland 2006** | **high** | R-c + R-d + the 48-term period sequence R-e, which we reproduced exactly |
| **T-14** non-COLLAPSING ⟺ predecessor diagonal eventually zero | POSSIBLY NEW (in the T-13–T-19 block) | **KNOWN — this is half of Rowland's Proposition 2** | **high** | R-d quoted verbatim; the indexing matches, including "the diagonal to its left" |
| **T-10** the DOUBLING criterion (odd parity of the `a`-word) | POSSIBLY NEW (same block) | **KNOWN — the other half of Rowland's Proposition 2** | **high** | R-d |
| **T-01** left-permutivity, left reconstruction | KNOWN STANDARD FACT | **KNOWN, and Rowland has a sharper form** — conditional time-reversibility when the right half of each row is white | high | R-g |
| **BO-08** left-region band slope ≈ 0.74 | bounded observation, no constant claimed | **RE-MEASUREMENT OF A KNOWN CONSTANT** — the order/chaos boundary moves left at ≈ 0.252 cells/step | high | R-h; our implied value 0.25368, a 0.67 % difference |
| **BO-09** `T(K)/K ≈ 1.34` | bounded observation | **RE-MEASUREMENT OF A KNOWN CONSTANT** — it is `1/(1 − 0.252)` | high | same |
| **CJ-02 / G5** `liminf T(K)/K > 1` | open conjecture, "new in Phase 2B" | **open, but it is the known statement "the boundary speed is < 1"** in our coordinates | high | same |
| **T-22** `w_t(7) = 0` for all `t` | new only as an orbit certificate | **implicit in Rowland's data** (period 1 at diagonal 7) | medium | R-e |
| **T-03** at most one EP column | POSSIBLY NEW, low confidence | **UNCERTAIN — Jen 1990 and Kopra Thm 3.5 occupy this territory** | — | J-a, J-d, K-b, K-c |
| **T-02** no two adjacent EP columns | LIKELY KNOWN | **UNCERTAIN, leaning KNOWN** | medium | same |

### The block assessment that was wrong

Phase 2F wrote: *"T-13 – T-19 … **POSSIBLY NEW**. This is the most substantial
block."* **That is withdrawn.** T-10 and T-14 are, on this evidence, Rowland's
Proposition 2. The block must be assessed result by result, and it is below.

## 2. What survives as plausibly new

| ID | result | revised label | conf. | why it survives |
|---|---|---|---|---|
| **T-18** | the **exact preperiod recurrence** via the defect bit `D(K)`: `D=0 ⟹ T(K)=T(K-1)`, `D=1 ⟹ T(K)=τ(K)+1` | **POSSIBLY NEW** | medium | every attributed statement about Rowland concerns **periods**; none mentions preperiods, transients or reset times |
| **T-19** | `T(K) = ρ(K)+1` at every resetting level | POSSIBLY NEW | medium | same |
| **T-16** | the reset calculus: erasure at `τ+1`, the one-period constant `Φ_K`, `T(K) ≤ τ+1` | POSSIBLY NEW | medium | same; and `Φ_K` is the ingredient T-18 needs |
| **T-21.5** | `D(K) = w_{T(K-1)}(K) XOR Φ_K`, hence the recurrence is not a function of cycle data at any depth | POSSIBLY NEW | medium | depends on T-16/T-18 being new |
| **T-18.3** | the factorisation `T(K)/K = (density of resetting levels) × (mean increment)` | POSSIBLY NEW | medium–low | but the *quantity* it factorises is a known constant (§1), so the novelty is the decomposition, not the number |
| **T-20** | the XOR-spine bound `\|Sk(t)\| ≥ ⌊t/2⌋+1` | POSSIBLY NEW, **and trivial** | medium | three lines from permutivity; novelty would be worth little |
| **T-06** | the zero-wall identities W1/W3/W6 under a finite wall | UNCERTAIN | low | different apparatus from all three papers |
| **T-21.8** | the strip automaton has in-degree exactly 4, so its recurrent core is everything at every radius | UNCERTAIN, likely folklore | low | standard for permutive rules |
| **CT-01** | `T + p ≥ 998140` for the centre column | **NEW ONLY AS AN ORBIT-SPECIFIC CERTIFICATE** | high | unchanged; the lemma is textbook, the numeral is ours |
| **T-21.11/12** | Phase 2A's Model A soundness and translation invariance | REFORMULATION | high | unchanged; SAT-encoding hygiene |

## 3. What the comparison found that we do not have

Recorded because it is more useful than any novelty claim:

* **Rowland's `2^n` nested-restart mechanism** (R-f) and the **conditional
  time-reversibility** it rests on (R-g): rule 30 is invertible backwards
  provided the right half of each row is white, and at row `2^n` a region of
  the initial condition reappears on the right. **We have no analogue.** Our
  entire corpus works on the left/prefix side because we had no handle on the
  right. This is a gap in our work.
* **Kopra's rapid left expansivity** and the **distribution-modulo-1**
  apparatus (K-a, K-d): a general framework containing rule 30, where we have
  only rule-30-specific arguments.
* **Jen's injectivity characterisation** (J-b): the general condition under
  which columns are aperiodic, where we have one rule.

## 4. The strongest result that remains plausibly new

> **T-18 — the exact recurrence for the preperiod of the left-diagonal prefix
> tower, driven by a single bit.**
>
> With `D(K) := w_{T(K-1)}(K) XOR w_{T(K-1)+P(K)}(K)` and
> `τ(K) := min{t ≥ T(K-1) : w_t(K-1) = 1}`:
> `D(K)=0 ⟹ T(K)=T(K-1)`; `D(K)=1 ⟹ T(K)=τ(K)+1` exactly; and the
> period-doubling and neutral cases both give `T(K)=T(K-1)`.

**Why it is the strongest candidate.** It is the exact counterpart, for
**preperiods**, of what Rowland's Proposition 2 does for **periods** — and
periods are the only thing any attributed statement mentions. If Rowland's
paper stops at periods, T-18 completes the picture; if it does not, T-18 is
also his.

**Why the claim is fragile.** A 2006 paper devoted to precisely this structure
is the most likely place in the literature for such a recurrence to appear, and
we have not read it. **This nomination must not be quoted without the
verification in `RETRIEVAL_LOG.md` §6 item 2 having been carried out.**

Second and third candidates: **T-19** (the `ρ(K)+1` identity) and **T-21.5**
(the one-transient-bit obstruction), both contingent on T-18 in the same way.

## 5. Consequences for the corpus

1. `NOVELTY_STATUS.md` in `RULE30_VERIFIED_THEOREMS_V1` is corrected to point
   here and to carry the downgrades of §1.
2. `PAPER_DRAFT_V1.md` is corrected: §§3–4 must attribute the period structure
   and the doubling criterion to Rowland 2006, and §8's boundary-slope numbers
   must be presented as agreeing with a known constant.
3. **The corpus's largest unverified assumption is retired.** Phase 2F flagged
   the rule-numbering convention as resting on a 14-bit prefix from a secondary
   source. The 48-term agreement with Rowland's published period sequence is a
   far stronger external anchor. `CC-01`'s caveat is downgraded from "largest
   unverified assumption" to "corroborated against a published sequence,
   second-hand".
4. The expert-review request should be **shortened**: questions 1 and 2 of
   `EXPERT_REVIEW_REQUEST.md` are now largely answered (T-03 → ask about
   Jen/Kopra; the prefix tower → Rowland has it). What remains worth an
   expert's time is: *does Rowland treat preperiods?*
