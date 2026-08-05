# Minimal backward-determining boundary data

**Phase 2H, section 2.**

The question: **what is the smallest boundary word that determines a rule-30
light-cone triangle?** Five candidate shapes A–E are stated exactly, and each
is either **proved sufficient** or **refuted by an explicit counterexample**.
Exhaustive enumeration is used only as a falsification tool; no all-`h` claim
is made from bounded enumeration, and every sufficiency claim below carries a
proof that is independent of the enumeration.

Geometry, fixed for this document (same as `triangle_enumerator.py`):

```
   top row  : columns 0 .. 2h              (2h+1 free bits)
   row s    : columns s .. 2h-s
   apex     : (h, h)

        col:  0  1  2  3  4  5  6                 (h = 3)
   s = 0 :    #  #  #  #  #  #  #
   s = 1 :       #  #  #  #  #
   s = 2 :          #  #  #
   s = 3 :             #                <- apex, column h
```

The **centre word** is `C = ( x_s(h) )_{s=0..h}`, `h+1` bits — the triangle's
copy of the centre column.

---

## 1. The five candidates

| | name | boundary word | size |
|---|---|---|---|
| **A** | centre column alone | `( x_s(h) )_{s=0..h}` | `h+1` bits |
| **B** | width-2 column pair | `( x_s(h), x_s(h+1) )_{s=0..h-1}` | `2h` bits |
| **C** | right cone edge (1 diagonal) | `( x_s(2h-s) )_{s=0..h}` | `h+1` bits |
| **C′** | two rightmost diagonals | `( x_s(2h-s), x_s(2h-s-1) )_{s=0..h-1}` | `2h` bits |
| **D** | one full row, no seed | row `t_bot` on `[b, m]` | `m-b+1` bits |
| **E** | one full row **+ white right tail** | row `t_bot` on `[b,m]`, `x_s(j)=0` for `j ≥ m` | same + a condition |

Candidate D/E are the pair the brief separates; they are the hypothesis of
Theorem 1.3 with and without hypothesis (b).

---

## 2. Candidate A — **REFUTED**

> ### Theorem 2.A. For every `h ≥ 1` the centre word does **not** determine the
> triangle. Exactly `2^h` distinct top rows share each of the `2^{h+1}`
> centre words (verified exhaustively for `1 ≤ h ≤ 12`), and the centre word
> does not even determine the single adjacent column.

**Explicit counterexample at `h = 1`** (`2h+1 = 3` cells, one step):

```
   top row  0 1 0        top row  0 1 1
   →  x_1(1) = 0 XOR (1 OR 0) = 1     →  x_1(1) = 0 XOR (1 OR 1) = 1
   centre word (x_0(1), x_1(1)) = (1,1)     centre word = (1,1)     SAME
   x_0(2) = 0                                x_0(2) = 1             DIFFER
```

So the centre word `11` is compatible with both `x_0(2) = 0` and `x_0(2) = 1`.
The counterexample is preserved verbatim; it is the minimal one.

**Quantitative version** (`SMALL_TRIANGLE_RESULTS.json`, exhaustive over all
`2^{2h+1}` triangles). `N(h)` = the largest number of distinct **width-2**
words (candidate B's word) compatible with one centre word:

| `h` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **rule 30** `N(h)` | 2 | 3 | 4 | 5 | 7 | 9 | 12 | 15 | 19 | 24 | 30 | 36 |
| **rule 90** `N(h)` | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | — | — | — |

Also exhaustively true for `1 ≤ h ≤ 12`: **all `2^{h+1}` centre words occur**,
and **exactly `2^h` triangles carry each of them** — the centre-word map is
perfectly balanced.

**Rule 90 control.** For rule 90, `N(h) = 2^h` exactly, i.e. the centre word
constrains the neighbouring column not at all. Rule 30's `N(h)` is far smaller
over this range. **No growth rate is claimed for rule 30's `N(h)`**; `h ≤ 12`
is twelve data points and the standing rules forbid fitting an asymptotic.
What matters downstream (section 6) is only the *measured values*, used inside
inequalities that hold for each individual `h`.

## 3. Candidate B — **SUFFICIENT, with an exactly sharp region**

> ### Theorem 2.B (width-2 determines a left triangle).
> Let `W2 = ( x_s(h), x_s(h+1) )_{s=0..h-1}`. Then `W2` determines
> ```
>        x_s(h-k)      for   1 ≤ k ≤ h-1   and   0 ≤ s ≤ h-1-k,
> ```
> i.e. exactly `h(h-1)/2` further cells, and **nothing beyond that envelope**:
> for each `k`, the cell `(h-k, h-k)` — one row below the envelope — is not
> determined.

*Proof of sufficiency.* Induction on `k`. For `k = 1`, apply (BW) at row `s`,
`j = h`: `x_s(h-1) = x_{s+1}(h) XOR ( x_s(h) OR x_s(h+1) )`. The right-hand
side needs `x_{s+1}(h)`, which is in `W2` for `s+1 ≤ h-1`, and `x_s(h)`,
`x_s(h+1)`, which are in `W2` for `s ≤ h-1`. Both hold for `s ≤ h-2 = h-1-1`.
Inductive step: suppose column `h-k+1` is known on rows `0..h-k` and column
`h-k+2` on rows `0..h-k+1`. Apply (BW) at row `s`, `j = h-k+1`:
`x_s(h-k) = x_{s+1}(h-k+1) XOR ( x_s(h-k+1) OR x_s(h-k+2) )`, which needs
`s+1 ≤ h-k`, i.e. `s ≤ h-k-1 = h-1-k`. ∎

*Sharpness.* Verified exhaustively for `2 ≤ h ≤ 10`
(`boundary_lab.candidate_B_region`): for every `h` in that range, all
`h(h-1)/2` envelope cells are constant on every `W2`-fibre (**0 violations**),
and every probe cell `(h-k, h-k)` immediately below the envelope takes both
values on some fibre. The envelope is therefore not merely a lower bound on
what `W2` determines — it is exactly what `W2` determines, in the measured
range.

| `h` | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| distinct `W2` words | 12 | 32 | 80 | 200 | 496 | 1208 | 2916 | 6964 | 16476 |
| cells determined | 1 | 3 | 6 | 10 | 15 | 21 | 28 | 36 | 45 |
| `h(h-1)/2` | 1 | 3 | 6 | 10 | 15 | 21 | 28 | 36 | 45 |
| violations | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Degrees of freedom.** Every `W2` fibre has size between **2** and `2^h`. The
lower bound is exact and provable: `x_s(h)` and `x_s(h+1)` for `s ≤ h-1` depend
only on top-row columns `1 … 2h` (the cell `(s, h+1)` sees columns
`h+1-s … h+1+s ⊆ [2, 2h]`, the cell `(s,h)` sees `h-s … h+s ⊆ [1, 2h-1]`), so
**top-row column 0 is always free**, and every fibre has even size ≥ 2. This
matches the measured minimum of 2 at every `h`.

**This is the "width 2" of the literature.** It is the smallest number of
*adjacent columns* from which the backward sweep can start, because (BW) reads
two same-row cells. One column is not enough — that is Theorem 2.A.

## 4. Candidate C — **REFUTED**; candidate C′ — **SUFFICIENT**

### 4.1 One diagonal is not enough

> ### Theorem 2.C. The right cone edge `( x_s(2h-s) )_{s=0..h}` does not
> determine the centre word.

Verified exhaustively for `2 ≤ h ≤ 10`: `2^{h+1}` distinct right-edge words,
each carried by `2^h` top rows, and the centre word (with the shared apex
removed, so the test is not vacuous) is non-constant on the fibres at every
`h`. The reason is structural: (BW) consumes **two same-row cells**, and one
diagonal supplies only one cell per row.

> **Correction C-2H-01.** The first version of `candidate_C_right_edge` asked
> whether the right edge determines *the apex* and reported `True` for every
> `h`. That test is **vacuous**: the apex `(h, h)` is the last letter of the
> right-edge word `x_s(2h-s)` at `s = h`, so the boundary word contains its own
> target. The result was recorded, spotted, and the test replaced with the
> centre word minus the apex. The retracted `True` is preserved here rather
> than deleted.

### 4.2 Two diagonals are enough — the diagonal recursion

> ### Theorem 2.C′ (diagonal recursion). Index the backward cone of a cell
> `(T, 0)` by diagonals measured inward from the right edge:
> ```
>       E_k(s) := x_s( T - s - k ),        0 ≤ k,  0 ≤ s ≤ T - k .
> ```
> Then for all `k ≥ 0` and `0 ≤ s ≤ T - k - 2`,
> ```
>       E_{k+2}(s)  =  E_k(s+1)  XOR  ( E_{k+1}(s)  OR  E_k(s) ) .      (DIAG)
> ```
> Consequently `E_0` and `E_1` on rows `0 … T-1` determine `E_k` on rows
> `0 … T-k` for every `k`, and hence the whole centre column of the cone,
> `x_s(0) = E_{T-s}(s)` for `0 ≤ s ≤ T-1`.

*Proof.* Apply (BW) at row `s`, column `j = T-s-k-1`:
`x_s(j-1) = x_{s+1}(j) XOR ( x_s(j) OR x_s(j+1) )`. Now
`j - 1 = T-s-k-2`, so the left side is `E_{k+2}(s)`;
`x_{s+1}(j) = x_{s+1}(T-(s+1)-k) = E_k(s+1)`;
`x_s(j) = E_{k+1}(s)`; and `x_s(j+1) = x_s(T-s-k) = E_k(s)`. Substituting gives
(DIAG). The row range follows because `E_k(s+1)` requires `s+1 ≤ T-k`. For the
centre column, `x_s(0) = x_s(T-s-(T-s)) = E_{T-s}(s)`, and `E_{T-s}` is
available on rows `0 … T-(T-s) = s`, which includes row `s`. ∎

*Verification.* `boundary_lab.diagonal_recursion` runs (DIAG) inside the real
single-cell orbit and compares every produced cell against the forward
computation. Results are in `results/phase2h_boundary.json`
(`diagonal_recursion`): **0 recursion failures and 0 centre-column failures**
at `T = 40, 120, 300`. Exhaustive triangle enumeration
(`candidate_C_prime`, `2 ≤ h ≤ 10`) independently gives fibres of size exactly
**2** — the one free bit being top-row column 0, which no cell of either
diagonal depends on.

**The catch, stated plainly.** Candidate C′ looks like a compression and is
not one. By the light cone, `E_0(s) = x_s(T-s) = 0` for every `s < T/2`, and
likewise for `E_1`; so the first half of each diagonal is identically zero and
carries no information. The genuinely free part of the boundary is about `T`
bits — the same order as the centre column it produces. **C′ buys structure,
not compression.**

**Why C′ still matters.** It says the centre cell `(T,0)` is a function of the
orbit *along the right boundary of its own backward cone* — and that boundary
is exactly where the white restart regions of section 4 live. The tangency
condition is computed in `RESTART_CENTER_GEOMETRY.md` §3: the level-`n` restart
region touches this boundary precisely at `T = 2^{n+1} - W_n`.

## 5. Candidates D and E — the row-plus-tail pair

> ### Theorem 2.D (one row alone is **not** enough).
> Knowing row `t_bot` on `[b, m]` and nothing else determines no cell of any
> earlier row. The one-step map is not injective:
> `boundary_lab.forward_collisions(8, 30)` exhibits distinct 8-cell rows with
> identical images on the interior, so a row has several predecessors.

> ### Theorem 2.E (row + white tail is enough — this is Theorem 1.3).
> Adding `x_s(j) = 0` for `j ∈ {m, m+1}` on every row `s ∈ [t_bot-h, t_bot]`
> upgrades D to the full backward cone `[b-i, m+1]` of Theorem 1.3. For a
> finite seed the condition is free at any `m ≥ t_bot + 1`.

The gap between D and E is **two bits per row**, and by Theorem 2.A those two
bits cannot be reduced to one.

## 6. Preserved counterexamples and failures

Kept per the standing rule; none of these is silently repaired.

| id | what failed | detail |
|---|---|---|
| **X-2H-01** | first backward reconstruction, `t_bot = 60`, `m = 40` | columns 40, 41 are **not** white on rows 40–60 — the light-cone edge `x_t(t)=1` intrudes. 413 mismatches out of 882 cells. Caught by the independent `white_strip_really_white` flag, **not** by the reconstruction, which failed silently. Fixed by requiring `m ≥ t_bot + 1` (Cor. 1.2b). |
| **X-2H-02** | candidate A conjecture "the centre word pins the neighbour" | refuted at `h = 1` by `010` vs `011`; fibres of size `2^h` at every `h ≤ 12` |
| **C-2H-01** | vacuous candidate-C test (apex ⊂ boundary word) | reported `True` for `h ≤ 10`; retracted, test replaced, original result kept above |
| **X-2H-03** | "two diagonals compress the boundary" | false — half of each diagonal is forced to zero by the light cone, so C′ costs `≈ T` bits, the same order as its output |

## 7. Summary

| candidate | verdict | basis |
|---|---|---|
| **A** centre column alone | **REFUTED** | proof + minimal counterexample + exhaustive `h ≤ 12` |
| **B** width-2 column pair | **SUFFICIENT**, envelope exactly `h(h-1)/2` cells | proved; sharpness exhaustive `h ≤ 10` |
| **C** one diagonal | **REFUTED** | proved (BW needs two same-row cells); exhaustive `h ≤ 10` |
| **C′** two diagonals | **SUFFICIENT** via (DIAG) | proved; verified on the orbit at `T ≤ 300`; no compression |
| **D** one row, no seed | **REFUTED** | non-injectivity of the one-step map |
| **E** one row + white pair | **SUFFICIENT** | Theorem 1.3 |

**The minimal shape, as far as this section establishes it, is two adjacent
columns (B) or two adjacent diagonals (C′) — in both cases `2` cells per row,
never `1`.** Section 3 asks whether the second cell can be reduced *on
average*, and shows that it can: only the rows where the centre is white
matter.
