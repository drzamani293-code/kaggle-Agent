# Restart events at powers of two

**Phase 2H, section 4.**

Secondary summaries attribute to Rowland a *nested restart* mechanism in which
"the computation begins again" near row `2^n`. **That paper could not be read**
(`RETRIEVAL_NOTE.md`), so nothing here is a reconstruction of it. What follows
is a measurement made in this phase, reported as a measurement, plus the small
part of it that can be proved.

**Everything in §2–§4 below is a BOUNDED COMPUTATIONAL OBSERVATION for
`t ≤ 1200`.** It is not a theorem, it is not evidence of a theorem, and it is
not used as a premise anywhere in Phase 2H.

---

## 1. Definitions

> ### Definition 4.1 (right-edge white run).
> ```
>      W(t) := max { L ≥ 0 : x_t(j) = 0 for all j ∈ [t-L, t-1] } .
> ```
> `W(t)` is the length of the maximal all-white run of row `t` ending
> immediately to the left of the light-cone edge column `t`.

> ### Lemma 4.2 (the edge is always black).
> `x_t(t) = 1` for every `t ≥ 0`.

*Proof.* Induction. `x_0(0) = 1`. Given `x_t(t) = 1` and `x_t(j) = 0` for
`j > t`, (FW) at `j = t+1` gives
`x_{t+1}(t+1) = x_t(t) XOR ( x_t(t+1) OR x_t(t+2) ) = 1 XOR (0 OR 0) = 1`. ∎

Verified independently for `t ≤ 1200` (`right_edge_always_one: true`). So `W(t)`
measures a white run *strictly inside* the cone, bounded on the right by a black
cell that is there for all time.

## 2. The measurement

`run_phase2h.py` §4, `t ≤ 1200`. Records of `W`:

| `t` | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `W(t)` | 0 | 2 | 3 | 5 | 6 | 8 | 14 | 15 | 23 | 24 | 26 |

> **Observation BO-4.3.** For `1 ≤ t ≤ 1200`, **every record value of `W` occurs
> at a power of two**, and at no other time. (`all_records_at_powers_of_two:
> true`.)

> **Observation BO-4.4.** `W(2^n - 1) = W(2^n + 1) = 0` for every `n` with
> `2^n ± 1 ≤ 1200`. The white run appears at `2^n` and is gone one step either
> side.

The values `2, 3, 5, 6, 8, 14, 15, 23, 24, 26` for `n = 1 … 10` are **not**
claimed to follow any formula. Eleven data points admit many; the standing rule
forbids fitting one.

## 3. The white triangle below a restart

> **Observation BO-4.5.** For `3 ≤ n ≤ 10`, writing `W_n := W(2^n)`, every cell
> in the triangle
> ```
>      { ( 2^n + i , j ) :  0 ≤ i < ⌊W_n/2⌋ ,  2^n - W_n + i ≤ j ≤ 2^n - 1 - i }
> ```
> is **0**. (`white_triangles`, `full_triangle: true` at every `n` in range.)

Picture, schematically (`#` = the light-cone edge column, `.` = verified white):

```
    t = 2^n      :   . . . . . . . . . . . . . #        W_n white cells
    t = 2^n + 1  :     . . . . . . . . . . . ? #
    t = 2^n + 2  :       . . . . . . . . . ? ? #
    t = 2^n + 3  :         . . . . . . . ? ? ? #
                            ...
                  ^                     ^
             col 2^n - W_n         col 2^n - 1
```

The white region narrows by one column on **each** side per step, which is the
maximum rate at which a white run can be eaten by rule 30 from both ends, so
the triangle closes after about `W_n/2` rows.

**What can be proved about this.** Only the trivial half:

> ### Lemma 4.6 (white runs shrink by at most one per side).
> If `x_t(j) = 0` for all `j ∈ [a, b]` with `b - a ≥ 2`, then
> `x_{t+1}(j) = 0` for all `j ∈ [a+2, b]`.

*Proof.* For `a+2 ≤ j ≤ b`, (FW) gives
`x_{t+1}(j) = x_t(j-1) XOR ( x_t(j) OR x_t(j+1) )`. Here `j-1, j ∈ [a,b]` are
white; `x_t(j+1)` is white too when `j+1 ≤ b`, giving `0 XOR (0 OR 0) = 0`. At
`j = b` the term `x_t(b+1)` is unknown, so the identity is applied only for
`j ≤ b-1`; at `j = b` the value may be `1`. Hence the guaranteed white interval
at time `t+1` is `[a+2, b-1]` — one lost on the left (two, in the worst case
counted here) and one on the right. ∎

Lemma 4.6 explains the *shape* of BO-4.5 given its top row. It says **nothing**
about why the top row is white at `t = 2^n`, which is the entire content of the
observation. **That remains unexplained by anything in this corpus.**

## 4. What is not claimed

1. **`W(t)` is not claimed to have records only at powers of two for all `t`.**
   The observation covers `t ≤ 1200`, i.e. `n ≤ 10`, i.e. ten events.
2. **No growth rate for `W_n` is claimed.** The values may be bounded,
   logarithmic, or anything else.
3. **No connection to Rowland is claimed.** Whether this is his `2^n` mechanism
   is unknown, because the paper is unreadable. If it is, the observation is
   **KNOWN** and this section discovers nothing; that possibility is left open
   rather than resolved in our favour.
4. **This is not evidence of aperiodicity.** A structure appearing at powers of
   two is perfectly compatible with an eventually periodic centre column — see
   §5, where the rule-90 control has both.

## 5. Rule 90 negative control

For rule 90 from a single cell (`run_phase2h.py` §11, `t ≤ 600`):

```
   W_90(2^n) = 2^(n+1) - 1        exactly, for every n with 2^n <= 600
```

i.e. row `2^n` of rule 90 is **entirely white** except the two cone-edge cells
at `±2^n`. This is a classical Pascal-mod-2 fact and it is provable:
`x_t(j) = C(t, (t+j)/2) mod 2` when `t+j` is even, and by Kummer's theorem
`C(2^n, k)` is odd only for `k ∈ {0, 2^n}`.

**The control is the point.** Rule 90 has a *far stronger* power-of-two white
structure than rule 30 — a whole white row rather than a run of length 26 — and
rule 90's centre column **is** eventually periodic. So:

> **The existence of power-of-two white structure carries no implication
> whatsoever about periodicity of the centre column.** Any argument that
> concluded otherwise would prove a false statement about rule 90.

This is recorded here so that the observations of §2–§3 are not mistaken for
progress on Problem 1. They are not.

## 6. Statement inventory

| id | statement | status | range |
|---|---|---|---|
| **L-4.2** | `x_t(t) = 1` for all `t` | **THEOREM**, proved | all `t` |
| **L-4.6** | white runs lose ≤1 cell per side per step | **THEOREM**, proved | all `t` |
| **BO-4.3** | every record of `W` is at a power of two | **BOUNDED OBSERVATION** | `t ≤ 1200` |
| **BO-4.4** | `W(2^n ± 1) = 0` | **BOUNDED OBSERVATION** | `n ≤ 10` |
| **BO-4.5** | the white triangle below `2^n` | **BOUNDED OBSERVATION** | `3 ≤ n ≤ 10` |
| **T-4.7** | rule 90: `W(2^n) = 2^{n+1}-1`; centre column eventually periodic | **THEOREM** (Kummer), verified | control |
