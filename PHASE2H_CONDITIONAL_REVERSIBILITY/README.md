# Phase 2H — Conditional Reversibility Bridge

Rule 30 Prize Problem 1 research corpus. This phase asks whether conditional
backward determinism ("run rule 30 backwards where the right side is white")
can be turned into a constraint on the centre column.

> ## Problem 1 is not solved. No partial claim on it is made here.
>
> The **Bridge** (`BR-01`) — deriving a *second* eventually periodic column
> from a periodic centre column — remains **open**, as it has been since
> Phase 1.

---

## The one-line summary of the phase

**The strongest result is negative.** Rule 90 satisfies every hypothesis used
by T-01/T-02/T-03, and its centre column *is* eventually periodic while its
column 1 provably is not. So no argument built from left-permutivity, the light
cone, the frozen edge and radius 1 — which is every column-based argument this
corpus has produced — can settle Problem 1. That includes this phase's own
targets H1 and H2.

## Contents

| file | brief § | what it contains |
|---|---|---|
| `CONDITIONAL_REVERSIBILITY_THEORY.md` | 1 | the backward step (BW), the row sweep, the backward cone `[b-i, m+1]` and its exact cell count, uniqueness |
| `MINIMAL_BOUNDARY_DATA.md` | 2 | candidates A–E: two refuted, three sufficient; the diagonal recursion; preserved counterexamples |
| `PERIODIC_CENTER_TRIANGLES.md` | 3 | OR-blindness and the **reduced boundary word** `B(t,h,p)` — the smallest missing data |
| `RESTART_EVENT_THEORY.md` | 4 | `W(t)` records at powers of two; the white triangle below `2^n` (bounded observations) |
| `RESTART_CENTER_GEOMETRY.md` | 5 | inequalities I–IV; why the restart events cannot reach the centre |
| `WEAK_BRIDGE_THEOREMS.md` | 6 | targets **H1–H5**, all proved; H5 is limitative |
| `KOPRA_PROOF_ADAPTATION.md` | 7 | **NOT COMPLETED** — the paper is unreadable; where width 2 enters *our* proofs instead |
| `TRIANGLE_TO_TRACE_TRANSFER.md` | 8 | the explicit `L(h, n) = n + h + 1`; the complexity transfer and why it is weaker than `CT-01` |
| `SMALL_TRIANGLE_ANALYSIS.md` | 9 | exhaustive `h ≤ 12` enumeration; Proposition 9.1; conjectures refuted |
| `CONDITIONAL_BRIDGE_REGISTRY.md` | 10 | the registry, what closed, and the single prioritised next target `BR-02` |
| `RETRIEVAL_NOTE.md` | — | every source unreachable; verbatim 403 transcripts |

### Code

| file | role |
|---|---|
| `reversibility_lab.py` | backward step, row sweep, backward cone, white-strip search |
| `boundary_lab.py` | candidates A–E, OR-blindness, factor counts, rule-90 control, bit-parallel orbit |
| `right_tower_lab.py` | the right-aligned tower `v_t(k) = x_t(t-k)`, bijectivity, periods |
| `triangle_enumerator.py` | exhaustive enumeration of all `2^{2h+1}` triangles |
| `run_phase2h.py` | measurement driver → `results/phase2h_results.json` |
| `run_phase2h_tests.py` | validation suite (groups A–D) |

### Results

`results/phase2h_results.json`, `results/phase2h_boundary.json`,
`results/phase2h_right_tower.json`, `SMALL_TRIANGLE_RESULTS.json`, plus the
run logs.

## Reproducing

```bash
python3 run_phase2h.py            # ~1 s   sections 1, 4, 5, 11
python3 boundary_lab.py           # ~10 s  sections 2, 3, 6, 8, 11
python3 right_tower_lab.py        # ~1 s   section 6 (H3)
python3 triangle_enumerator.py --h-max 12    # ~2 min, needs ~2 GB
python3 run_phase2h_tests.py      # validation
python3 run_phase2h_tests.py --fast
```

## What was established

**Proved.**

* The backward cone: from row `t_bot` on `[b, m+1]` plus a white pair at
  `m ≥ t_bot+1`, row `t_bot-i` is determined on exactly `[b-i, m+1]`, giving
  `(h+1)(m-b+2) + h(h+1)/2` cells. Verified against the forward orbit on
  12 296 cells, **0 mismatches**.
* One column determines nothing; two adjacent columns determine exactly
  `h(h-1)/2` further cells and no more.
* The diagonal recursion
  `E_{k+2}(s) = E_k(s+1) XOR ( E_{k+1}(s) OR E_k(s) )`, and that the two
  rightmost diagonals of a backward cone determine its centre column.
* **The reduced boundary word:** if the centre column is eventually periodic
  with period `p`, then the bits of column 1 *at the rows where the centre is
  white* are not eventually `p`-periodic. About half the naive data.
* **The right-aligned tower `v_t(k) = x_t(t-k)` is autonomous and its map is a
  bijection** — preperiod exactly 0 at every level — whereas Phase 2E's
  left-aligned map is not injective at any level `K ≥ 1`. This proves *why*
  Phase 2E's preperiods `T(K)` exist, and closes the gap Phase 2G identified.
* `L(h, n) = n + h + 1`: periodicity transfers to depth `h` only at an overhead
  of `h+1` rows, so no finite window reaches a full column.
* **The rule-90 calibration (H5)**, above.

**Measured, and labelled as bounded observations.**

* Every record of the right-edge white run `W(t)` occurs at a power of two, for
  `t ≤ 1200`; `W(2^n) = 2,3,5,6,8,14,15,23,24,26` for `n = 1..10`.
* The white triangle below `t = 2^n` is full for `3 ≤ n ≤ 10`.
* `N(h) = 2,3,4,5,7,9,12,15,19,24,30,36` for `h = 1..12` (rule 90: `2^h`).
* Centre-column white density `9880 / 20000` at `t = 20000`.

No growth rate, limit or asymptotic constant is claimed for any of these.

## What was not established

* `BR-01` — the Bridge. Open.
* Section 7 of the brief. Not completed; the paper could not be retrieved.
* Any novelty claim. No primary source was readable in any phase.
* Any statement about `W(t)` or `N(h)` beyond the enumerated range.

## Standing rules observed

Finite computation is never presented as proving non-periodicity. Theorem,
lemma, conjecture, heuristic and computational observation are labelled
separately. The rule-30 implementation is checked against a second independent
implementation. Bounded searches are reported as "no witness found in the
searched range", never as UNSAT. Every failed conjecture and correction is
preserved (`X-2H-01` … `X-2H-06`, `C-2H-01`). Rule 90 is used as a mandatory
negative control throughout, and in this phase it is the load-bearing result.
