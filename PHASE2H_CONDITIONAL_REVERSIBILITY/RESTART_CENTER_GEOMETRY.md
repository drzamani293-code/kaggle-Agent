# Where the restart events sit relative to the centre column

**Phase 2H, section 5.** The brief: *"This section must contain exact
inequalities, not only plots."* There are no plots. There are four
inequalities, all proved, and one measured table.

The conclusion is negative and should be read first:

> **The level-`n` restart region cannot influence the centre column before time
> `2^{n+1} - W_n`, and over the measured range that arrival time grows like
> `2^{n+1}`, while `W_n` stays small. The restart events recede from the centre
> exponentially. No mechanism found in this phase brings them back.**

---

## 1. Light-cone inequalities

> ### Lemma 5.1 (speed of influence).
> If two configurations agree at time `t` on every column outside `{j₀}`, then
> their evolutions agree at time `t + Δ` on every column outside
> `[j₀ - Δ, j₀ + Δ]`.

*Proof.* (FW) at `(t, j)` reads only `x_t(j-1), x_t(j), x_t(j+1)`, so a
difference at a single column spreads to at most one extra column on each side
per step; induct on `Δ`. ∎

> ### Corollary 5.2 (earliest arrival at the centre).
> A change made at cell `(t₀, j₀)` cannot alter `x_t(0)` for any
> ```
>        t  <  t₀ + |j₀| .
> ```

> ### Corollary 5.3 (backward-cone membership).
> Cell `(t₀, j₀)` lies in the backward light cone of `(T, 0)` if and only if
> ```
>        |j₀|  ≤  T - t₀ ,    equivalently   T  ≥  t₀ + |j₀| .
> ```

## 2. The restart region, located exactly

By `RESTART_EVENT_THEORY.md` Def. 4.1 and BO-4.5, the level-`n` restart region
occupies, at time `t = 2^n`, the columns

```
        [ 2^n - W_n ,  2^n - 1 ] ,          W_n := W(2^n) ,
```

all white, with the black cone-edge cell at column `2^n` immediately to its
right (Lemma 4.2). Its **leftmost** column is `2^n - W_n`, so by Cor. 5.2 the
earliest time at which any part of it can affect the centre column is

> ### Inequality (I) — the arrival bound.
> ```
>        t  ≥  2^n + ( 2^n - W_n )  =  2^{n+1} - W_n .
> ```

> ### Inequality (II) — the tangency identity.
> The level-`n` restart region **first touches** the right boundary of the
> backward cone of the centre cell `(T, 0)` exactly when
> ```
>        T  =  2^{n+1} - W_n .
> ```
> Precisely: in the diagonal coordinates of `MINIMAL_BOUNDARY_DATA.md`
> Theorem 2.C′, `E_0(s) = x_s(T - s)`; at `s = 2^n` this is the cell
> `(2^n, T - 2^n)`, which is the region's leftmost column `2^n - W_n` exactly
> when `T = 2^{n+1} - W_n`.

*Proof.* Both are Cor. 5.2/5.3 with `t₀ = 2^n`, `j₀ = 2^n - W_n`, together with
the definition of `E_0`. ∎

Inequality (II) is the reason section 2 bothered with candidate C′: the
boundary datum that determines the centre cell is evaluated **exactly along the
diagonal that sweeps through the restart regions.** If the restart structure
were ever to constrain the centre column, this is the channel it would have to
use.

> ### Inequality (III) — the centre is never inside the white triangle.
> For every `n` in the measured range, the white triangle of BO-4.5 lies
> strictly to the right of column 0:
> ```
>        2^n - W_n  >  0        ⟺        W_n  <  2^n .
> ```
> `W_n ≤ 26 < 2^n` for `3 ≤ n ≤ 10` (measured). For `n ≤ 2`, `W_n ≥ 2^n` can
> fail to be excluded — `W(2) = 2 = 2^1`, and indeed `2^1 - W_1 = 0`, so the
> level-1 event touches column 0. **This is the only measured level at which
> it does, and it happens at `t = 2`.**

> ### Inequality (IV) — the receding gap.
> The distance from column 0 to the restart region is `d_n := 2^n - W_n`, and
> the arrival time is `2^n + d_n`. Over the measured range `d_n` is increasing
> and `d_{n+1} > 2 d_n - W_{n+1}`; since `W_n` never exceeded 26,
> `d_{n+1} > 2 d_n - 26` for `n ≤ 9`. **No claim is made for `n > 10`.**

## 3. The measured table

`results/phase2h_results.json`, `restart_geometry`:

| `n` | `t = 2^n` | `W_n` | region columns | distance `d_n = 2^n - W_n` | earliest influence on column 0, `2^{n+1} - W_n` |
|---:|---:|---:|---|---:|---:|
| 3 | 8 | 5 | [3, 7] | 3 | 11 |
| 4 | 16 | 6 | [10, 15] | 10 | 26 |
| 5 | 32 | 8 | [24, 31] | 24 | 56 |
| 6 | 64 | 14 | [50, 63] | 50 | 114 |
| 7 | 128 | 15 | [113, 127] | 113 | 241 |
| 8 | 256 | 23 | [233, 255] | 233 | 489 |
| 9 | 512 | 24 | [488, 511] | 488 | 1000 |
| 10 | 1024 | 26 | [998, 1023] | 998 | 2022 |

`d_n` roughly doubles at each level; the arrival time roughly doubles with it.

## 4. Why this closes the section negatively

Three obstructions, stated so that the next phase does not re-discover them.

1. **Arrival is too late to be a constraint.** By the time the level-`n` region
   could reach column 0 (`t ≥ 2^{n+1} - W_n`), levels `n+1` and beyond have
   already happened, and everything between column 0 and column `2^n` has been
   overwritten `2^n` times. The white region does not travel; only its
   *influence* does, mixed with everything else in the cone.

2. **The region is white, and white is not information.** The restart region is
   a block of zeros. What would have to propagate to column 0 is not the zeros
   but the *conditional structure* they enable — and by Theorem 1.3 the cone
   they seed opens **leftwards from column `2^n - W_n`**, reaching column 0
   after `2^n - W_n` further steps, at which point it is a cone of width
   `≈ 2^{n+1}` containing the entire chaotic bulk. It determines the centre
   column, but only in the sense that the whole row does.

3. **The control has the same geometry and the opposite conclusion.** Rule 90's
   row `2^n` is white *everywhere* between the two cone edges — `W_n = 2^{n+1}-1`,
   so `d_n = 2^n - (2^{n+1}-1) < 0` and inequality (III) **fails**: the white
   region contains column 0 itself, at every level. And rule 90's centre column
   is eventually periodic. So a white region *touching* the centre is
   compatible with periodicity, and a white region *receding from* the centre
   is what rule 30 does. Neither configuration decides anything.

> ### Conclusion 5.4 (recorded as a negative result).
> **The `2^n` restart events observed in section 4 impose no constraint on the
> centre column that this phase could extract.** The geometry (inequalities
> I–IV) shows why: the events occur at distance `≈ 2^n` from the centre and
> their influence arrives no earlier than `≈ 2^{n+1}`, by which time it is
> indistinguishable from the bulk. Sections 6 and 10 therefore do **not** build
> a bridge candidate on this mechanism.

## 5. Statement inventory

| id | statement | status |
|---|---|---|
| **L-5.1** | speed of influence ≤ 1 column/step | **THEOREM**, proved |
| **C-5.2** | earliest arrival `t ≥ t₀ + \|j₀\|` | **COROLLARY**, proved |
| **C-5.3** | backward-cone membership `T ≥ t₀ + \|j₀\|` | **COROLLARY**, proved |
| **I-5.(I)** | arrival bound `t ≥ 2^{n+1} - W_n` | **THEOREM**, proved from L-5.1 + measured `W_n` |
| **I-5.(II)** | tangency at `T = 2^{n+1} - W_n` | **THEOREM**, proved |
| **I-5.(III)** | `2^n - W_n > 0` for `3 ≤ n ≤ 10` | **BOUNDED OBSERVATION** (depends on measured `W_n`) |
| **I-5.(IV)** | receding gap `d_{n+1} > 2d_n - 26`, `n ≤ 9` | **BOUNDED OBSERVATION** |
| **C-5.4** | the restart mechanism yields no constraint on the centre | **NEGATIVE RESULT**, recorded |
