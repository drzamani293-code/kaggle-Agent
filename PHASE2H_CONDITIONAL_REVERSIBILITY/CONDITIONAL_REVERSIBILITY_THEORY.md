# Conditional Reversibility of Rule 30 — exact statements

**Phase 2H, section 1.**

The brief asks this section to *"reconstruct Rowland's reversibility"*. It does
not do that, because **Rowland's paper could not be read** — see
`RETRIEVAL_NOTE.md`; every fetch of every host returns HTTP 403 at the
proxy's `CONNECT` stage. What follows is therefore **derived from the local
rule alone**, in one fixed coordinate system, with every statement quantified.

Where a statement is *attributed* to Rowland it carries the label
**ATTRIBUTED — SECONDARY, NOT VERIFIED** and is never used as a premise for
anything proved here.

---

## 0. Coordinates, fixed for the whole of Phase 2H

Cells `x_t(j)` for `t ≥ 0` and `j ∈ ℤ`. Original (centre-anchored) columns;
the centre column is `j = 0`. The single-cell seed is

```
x_0(j) = 1 if j = 0, else 0.
```

**Forward rule (FW).**

```
x_{t+1}(j)  =  x_t(j-1)  XOR  ( x_t(j)  OR  x_t(j+1) )
```

This is Wolfram's rule 30 under the convention `(l, c, r) ↦ 4l + 2c + r`;
`CC-01` in `RULE30_VERIFIED_THEOREMS_V1/COMPUTATIONAL_CERTIFICATES.md` records
the table, and Phase 2G corroborated it externally by reproducing 48 terms of a
published diagonal-period sequence.

Ascii convention used throughout: `.` = 0, `#` = 1, columns increase to the
right, time increases downward.

```
        j:  -4 -3 -2 -1  0  1  2  3  4
   t = 0 :   .  .  .  .  #  .  .  .  .
   t = 1 :   .  .  .  #  #  #  .  .  .
   t = 2 :   .  .  #  #  .  .  #  .  .
   t = 3 :   .  #  #  .  #  #  #  #  .
   t = 4 :   #  #  .  .  #  .  .  .  #
```

**Light cone (T-2.1, recalled).** `x_t(j) = 0` whenever `|j| > t`. In
particular `x_t(j) = 0` for every `j ≥ t+1`, on every row `t`. This is the
fact that supplies the white right tail for free; nothing has to be assumed.

---

## 1. The exact bijectivity property

Everything in this document rests on one line.

> ### Lemma 1.0 (the bijection).
> For every fixed pair `(c, r) ∈ {0,1}²`, the map
> ```
> φ_{c,r} : {0,1} → {0,1},        φ_{c,r}(l) = l XOR (c OR r)
> ```
> is a **bijection**, with `φ_{c,r}^{-1} = φ_{c,r}`.

*Proof.* `(c OR r)` is a constant `k ∈ {0,1}` once `c, r` are fixed, and
`l ↦ l XOR k` is translation by `k` in the group `(ℤ/2, XOR)`, hence a
bijection equal to its own inverse. ∎

This is *left-permutivity* (T-01). It is a statement about the **left**
argument only. The corresponding statement about the right argument is
**false**:

> ### Lemma 1.0′ (rule 30 is not right-permutive).
> For `l = 0, c = 1`: `f(0,1,0) = 0 XOR (1 OR 0) = 1` and
> `f(0,1,1) = 0 XOR (1 OR 1) = 1`. So `r ↦ f(l,c,r)` is constant whenever
> `c = 1`, and is not injective. ∎

**Consequence, stated because it governs the whole phase.** Backward
reconstruction can proceed **leftwards only**. There is no mirror-image
theorem. Every "conditional reversibility" statement below is
conditional in exactly the same way: it needs data on the **right** in order to
produce cells on the **left**.

**Negative control.** Rule 90, `f_{90}(l,c,r) = l XOR r`, is permutive in
*both* arguments, so for rule 90 both directions work and no whiteness
condition is ever needed. Every asymmetry recorded below is therefore
specific to rule 30 and is not an artefact of the coordinate system.

---

## 2. The backward step

> ### Theorem 1.1 (backward step, BW).
> For all `s ≥ 0` and all `j ∈ ℤ`,
> ```
> x_s(j-1)  =  x_{s+1}(j)  XOR  ( x_s(j)  OR  x_s(j+1) ).            (BW)
> ```

*Proof.* Apply (FW) at `(t, j) = (s, j)`:
`x_{s+1}(j) = x_s(j-1) XOR (x_s(j) OR x_s(j+1))`. XOR both sides with
`(x_s(j) OR x_s(j+1))` and use `a XOR b XOR b = a`. Equivalently, invert
`φ_{c,r}` of Lemma 1.0 with `c = x_s(j)`, `r = x_s(j+1)`. ∎

**Which cell is reconstructed from which cells.** Exactly three inputs, one
from the later row and two from the *same* row, both strictly to the right:

```
             col:   j-1     j     j+1
   row  s+1     :           O                 O = the one later cell
   row  s       :    X  <-- A --- B           A, B = same row, to the right
                                              X = the cell produced
```

So **within a row the recursion runs right to left**, and it must be *seeded*
on the right. This is the entire content of the word "conditional".

**Rule 30 is not backward-deterministic without that seed.** The one-step map
is not injective: `boundary_lab.forward_collisions(8, 30)` exhibits distinct
8-cell rows with identical images on the interior. "Running rule 30 backwards"
is not defined; only "running it backwards from a seed" is.

---

## 3. One backward row-sweep

> ### Theorem 1.2 (row sweep, weak seed form).
> Let `s ≥ 0`, and let `a ≤ m` be integers. Suppose
>
> * **(i)** `x_{s+1}(j)` is known for every `j ∈ [a, m]`; and
> * **(ii)** `x_s(m)` and `x_s(m+1)` are known.
>
> Then `x_s(j)` is determined for every `j ∈ [a-1, m+1]`, by the explicit
> right-to-left recursion
> ```
> for j = m, m-1, …, a :      x_s(j-1) := x_{s+1}(j) XOR ( x_s(j) OR x_s(j+1) ).
> ```

*Proof.* Downward induction on `j`. **Base** `j = m`: both `x_s(m)` and
`x_s(m+1)` are known by (ii), and `x_{s+1}(m)` is known by (i), so (BW) gives
`x_s(m-1)`. **Step**: assume `x_s(j)` and `x_s(j+1)` known for some
`a ≤ j ≤ m`. Then `x_{s+1}(j)` is known by (i) since `j ∈ [a,m]`, and (BW)
gives `x_s(j-1)`; now `x_s(j-1)` and `x_s(j)` are known, which is the
hypothesis at `j-1`. The induction runs down to `j = a`, producing
`x_s(a-1)`. ∎

> ### Corollary 1.2a (white-seed form — "which half-line must be white").
> Hypothesis (ii) is implied by: **`x_s(j) = 0` for every `j ≥ m`**, i.e. row
> `s` is white on the half-line `[m, ∞)`. Two white cells are enough; a whole
> white half-line is more than is needed, but is what the orbit supplies.

> ### Corollary 1.2b (the seed is free for a finite seed).
> For the single-cell orbit, take **`m = s + 1`**. Then `x_s(m) = x_s(m+1) = 0`
> automatically by the light cone. More usefully, for any `t_bot` and any
> `m ≥ t_bot + 1`, columns `m` and `m+1` are white on **every** row
> `s ≤ t_bot` simultaneously, so one choice of `m` seeds every row of a
> backward sweep of any depth.

**This corollary is where the first attempt of this phase failed.** The
initial run chose `m = 40` for `t_bot = 60`; columns 40 and 41 are *not* white
on rows 40–60, because the light-cone edge `x_t(t) = 1` intrudes, and the
reconstruction produced 413 mismatches out of 882 cells. The failure was caught
by the independent `white_strip_really_white` flag, not by the reconstruction
itself — a reconstruction seeded with false data returns wrong answers
silently. The counterexample is preserved in
`MINIMAL_BOUNDARY_DATA.md` §6 and is the reason the sufficient condition is
stated as an inequality `m ≥ t_bot + 1` rather than "pick a white-looking
column".

---

## 4. The reconstructible region: a left-opening cone

> ### Theorem 1.3 (backward cone; exact index formula).
> Fix `t_bot ≥ 0`, `h ≥ 0`, and integers `b ≤ m` with `m ≥ t_bot + 1`.
> Suppose
>
> * **(a)** row `t_bot` is known on columns `[b, m+1]`;
> * **(b)** `x_s(m) = x_s(m+1) = 0` for every `s ∈ [t_bot - h, t_bot]`
>   (automatic, by Cor. 1.2b, when `m ≥ t_bot + 1`).
>
> Then for every `i` with `0 ≤ i ≤ h`, **row `t_bot - i` is determined on
> exactly the columns**
> ```
>          [ b - i ,  m + 1 ] .
> ```
> The total number of determined cells is
> ```
>   N(h, b, m)  =  (h+1)·(m - b + 2)  +  h(h+1)/2 .                   (★)
> ```

*Proof.* Induction on `i`. `i = 0` is hypothesis (a). Suppose row `t_bot - i`
is known on `[b-i, m+1]`. Apply Theorem 1.2 with `s = t_bot - i - 1`,
`a = b - i`, and this `m`: hypothesis (i) holds because row `s+1 = t_bot - i`
is known on `[b-i, m+1] ⊇ [b-i, m]`; hypothesis (ii) holds by (b). The
conclusion is that row `s` is determined on `[b-i-1, m+1]`, which is the claim
at `i+1`. For (★), row `t_bot - i` contributes `(m+1) - (b-i) + 1 = m-b+2+i`
cells; sum over `i = 0 … h`. ∎

**ASCII picture** (`t_bot = 8`, `h = 4`, `b = 5`, `m = 9`; `?` = not
determined, `=` = determined, `0` = the white seed pair at columns `m`, `m+1`):

```
        col:  0  1  2  3  4  5  6  7  8  9 10 11 12
   t = 4  :   ?  =  =  =  =  =  =  =  =  0  0  ?  ?     row t_bot-4, left edge b-4 = 1
   t = 5  :   ?  ?  =  =  =  =  =  =  =  0  0  ?  ?     left edge b-3 = 2
   t = 6  :   ?  ?  ?  =  =  =  =  =  =  0  0  ?  ?     left edge b-2 = 3
   t = 7  :   ?  ?  ?  ?  =  =  =  =  =  0  0  ?  ?     left edge b-1 = 4
   t = 8  :   ?  ?  ?  ?  ?  =  =  =  =  0  0  ?  ?     the KNOWN row, left edge b = 5
                                         ^m ^m+1
```

Read bottom to top: the region **opens to the left** as time runs backward, at
exactly one column per step, while its right edge stays pinned at `m+1`. Total
cells here: `(h+1)(m-b+2) + h(h+1)/2 = 5·6 + 10 = 40`, which is the row widths
`6, 7, 8, 9, 10` summed.

**The cone is a light cone, run backwards.** Its left boundary has slope 1 in
`(t, j)`, matching the maximum speed of information in (FW); its right
boundary is vertical, because (BW) never reaches right of the seed.

### 4.1 Verification of Theorem 1.3, including the count (★)

`reversibility_lab.verify_backward_on_orbit` reconstructs inside the genuine
single-cell orbit and compares every reconstructed cell against the forward
computation. Results (`results/phase2h_results.json`):

| `t_bot` | `h` | `m` | `b` | cells reconstructed | (★) predicts | mismatches |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 20 | 61 | 31 | 882 | `21·32 + 210 = 882` | **0** |
| 120 | 40 | 121 | 61 | 3362 | `41·62 + 820 = 3362` | **0** |
| 200 | 60 | 201 | 101 | 8052 | `61·102 + 1830 = 8052` | **0** |

The per-row cell counts are also exact: row `t_bot - i` holds `m-b+2+i` cells
in every one of the three runs (`cells_per_row` in the results file), which is
the index formula of Theorem 1.3 and not merely the total.

**What this verification is and is not.** It is a check that the *implementation*
of the proved theorem agrees with the forward orbit on 12 296 cells. It is
**not** evidence for the theorem — the theorem is proved above — and it is not
evidence about periodicity of anything.

---

## 5. Boundary data required for uniqueness

> ### Theorem 1.4 (uniqueness and degrees of freedom).
> Under the hypotheses of Theorem 1.3, the values on the cone
> `{ (t_bot - i, j) : 0 ≤ i ≤ h, b - i ≤ j ≤ m+1 }` are **unique**: there is
> exactly one assignment consistent with (FW), (a) and (b). Conversely, the
> data (a)+(b) is **not redundant**:
>
> 1. dropping either seed cell `x_s(m)` or `x_s(m+1)` on a single row leaves
>    that row's sweep unstarted, and the cone is truncated at that row;
> 2. shortening (a) to `[b', m+1]` with `b' > b` shrinks every row's left
>    endpoint by `b' - b`;
> 3. the cells to the **right** of column `m+1` are entirely free: they are not
>    determined by (a)+(b) at any row, and different choices of them are
>    consistent with the same cone.

*Proof.* Uniqueness is immediate: Theorem 1.2's recursion is an explicit
formula, so the value at each cell is a function of the data. (1) and (2) are
read off the induction, which cannot start without both seed cells and cannot
reach left of `b - i`. For (3): rule 30's dependence cone means `x_s(j)` for
`j > m+1` influences the cone only through cells at columns `≤ m+1`, all of
which are pinned by (a)+(b); an explicit pair of orbits differing only right of
`m+1` and agreeing on the cone is produced by
`boundary_lab.candidate_B_region`, whose group sizes count exactly these free
choices. ∎

**Degrees of freedom, counted.** Given the cone of Theorem 1.3, the free data
are: the cells of row `t_bot` right of column `m+1`. That is the sense in
which the boundary data is minimal *for this geometry*. Section 2
(`MINIMAL_BOUNDARY_DATA.md`) asks the sharper question — what is the smallest
boundary word of **any** shape that determines a triangle — and answers it.

---

## 6. What this does **not** give

Stated explicitly, because the temptation runs the other way.

1. **No statement about periodicity.** Theorems 1.0–1.4 are one-step algebra
   plus an induction. Nothing here constrains the centre column's orbit.
2. **No global inverse.** By Lemma 1.0′ and the collision examples, rule 30 has
   no inverse as a map on configurations. Every theorem above is conditional on
   right-hand data.
3. **No access to the right.** The cone opens *left*. Cells to the right of the
   seed are never produced. Since the centre column sits at `j = 0` and the
   interesting white regions of the orbit sit near `j = t` (section 4,
   `RESTART_EVENT_THEORY.md`), the cone from a white region at column `≈ t`
   reaches column 0 only after a further `≈ t` steps — the exact inequality is
   in `RESTART_CENTER_GEOMETRY.md`.
4. **Nothing is imported from Rowland.** The phrase "the computation begins
   again", which the brief warns against, appears nowhere in this document as a
   premise, and the `2^n` mechanism is treated in section 4 strictly as a
   **bounded computational observation**.

---

## 7. Statement inventory for this section

| id | statement | status | domain |
|---|---|---|---|
| **R-1.0** | `l ↦ l XOR (c OR r)` is a bijection for each `(c,r)` | **THEOREM**, proved | any configuration |
| **R-1.0′** | rule 30 is not right-permutive | **THEOREM**, proved | any |
| **R-1.1** | the backward step (BW) | **THEOREM**, proved | any |
| **R-1.2** | row sweep from a two-cell right seed | **THEOREM**, proved | any |
| **R-1.2a** | white half-line implies the seed | **COROLLARY**, proved | any |
| **R-1.2b** | `m ≥ t_bot + 1` seeds every row, for a finite seed | **COROLLARY**, proved | finite seed |
| **R-1.3** | backward cone `[b-i, m+1]`, count (★) | **THEOREM**, proved; verified on 12 296 cells | finite seed |
| **R-1.4** | uniqueness; the three non-redundancy claims | **THEOREM**, proved | any |
| **R-1.5** | rule 90 needs no whiteness condition (control) | **THEOREM**, proved | any |

Domains follow `RULE30_VERIFIED_THEOREMS_V1/DEFINITIONS_AND_NOTATION.md`:
**ANY** = any configuration, **FIN** = any finite seed, **SEED** = the
single-cell seed only. Nothing in this section is **SEED**-only.
