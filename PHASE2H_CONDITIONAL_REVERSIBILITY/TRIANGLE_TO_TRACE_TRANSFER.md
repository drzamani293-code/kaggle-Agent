# From triangles to traces: the explicit function `L(h)`

**Phase 2H, section 8.** The brief: *"Derive an explicit function `L(h)`."*

The question this section answers precisely:

> If a boundary word is periodic over a window, **how long must that window be**
> before periodicity can be concluded for a reconstructed cell `h` columns
> deeper?

The answer is an overhead of exactly `h + 1` rows, and it is the reason the
transfer never closes.

---

## 1. The transfer, stated as an identity of windows

Recall Theorem 2.B: from the width-2 boundary `(x_s(0), x_s(1))` the backward
sweep produces column `-1`, then `-2`, and so on, **losing one row per column**.
In orbit coordinates:

> ### Lemma 8.1 (row loss per column).
> If `x_s(0)` is known for `s ∈ [t, t+L]` and `x_s(1)` for `s ∈ [t, t+L-1]`,
> then `x_s(-k)` is determined for
> ```
>       s ∈ [ t ,  t + L - k ] ,        1 ≤ k ≤ L .
> ```

*Proof.* Induction on `k` using (BW) at `j = -k+1`:
`x_s(-k) = x_{s+1}(-k+1) XOR ( x_s(-k+1) OR x_s(-k+2) )`. The term
`x_{s+1}(-k+1)` needs row `s+1` at depth `k-1`, available for
`s+1 ≤ t + L - (k-1)`, i.e. `s ≤ t + L - k`. ∎

## 2. The function `L(h)`

> ### Definition 8.2. `L(h, n)` := the smallest window length such that
> `p`-periodicity of the boundary data on `[t, t + L]` implies `p`-periodicity
> of the reconstructed cell at depth `h` on a window of `n` consecutive rows.

> ### Theorem 8.3 (explicit `L`).
> ```
>        L(h, n)  =  n + h + 1 .
> ```
> Concretely: if `x_s(0)` and `x_s(1)` satisfy `x_{s+p}(·) = x_s(·)` for every
> `s ∈ [t, t + n + h + p]`, then `x_{s+p}(-h) = x_s(-h)` for every
> `s ∈ [t, t + n]`. Writing the overhead alone,
> ```
>        L(h)  :=  L(h, 0)  =  h + 1 .
> ```

*Proof.* By Lemma 8.1 with `L = n + h`, depth `h` is determined on
`[t, t+n]`, and the same holds for the shifted window `[t+p, t+p+n]` provided
the boundary is known up to row `t + n + h + p`. Since every cell at depth `h`
is an explicit function (a composition of (BW)) of boundary cells at rows
`≤ s + h`, equality of the boundary at `s` and `s+p` for all rows in range
forces equality of the depth-`h` cells. The `+1` is the extra row of column 0
that (BW) consumes at every step (`x_{s+1}(0)` in the `j = 0` case). ∎

> ### Corollary 8.4 (the reduced version).
> Under Theorem 3.4 the same statement holds with the full column 1 replaced by
> the reduced boundary word `B(t, n+h)`. `L(h, n)` is unchanged: **the overhead
> is a property of the geometry, not of how many bits the boundary carries.**

## 3. Why the transfer never closes

> ### Proposition 8.5 (the escape).
> `L(h) = h + 1 → ∞`. To conclude `p`-periodicity of **every** column `≤ 0` —
> which is what Prop. 3.1 needs for a contradiction with T-02 — one needs
> `h → ∞`, hence a boundary window of unbounded length, hence
> `p`-periodicity of the boundary **for all sufficiently large `s`**, not on any
> finite window.

This is the precise sense in which the bridge is not a finite computation.
Every finite window of periodic boundary data yields a finite triangle of
periodic cells, and the triangle's depth is always **one less per row spent**.
No amount of finite verification reaches the contradiction.

> ### Corollary 8.6 (what a finite computation *can* certify).
> If the boundary is verified `p`-periodic on `L` rows, the certified periodic
> region is the triangle `{ (s, -k) : 1 ≤ k ≤ L, t ≤ s ≤ t + L - k }` of
> `L(L-1)/2` cells. Its area grows quadratically in the verification effort,
> and it never contains a full column.

## 4. Interaction with the factor-complexity route

The other transfer available is combinatorial rather than geometric. Every
window of the centre column, together with the corresponding width-2 word, is
the `(C, W2)` pair of a locally valid triangle (`SMALL_TRIANGLE_ANALYSIS.md`
§2), so:

> ### Theorem 8.7 (complexity transfer).
> For every `h ≥ 1`,
> ```
>        p_{01}(h)  ≤  p_0(h+1) · N(h) ,
> ```
> where `p_0(n)` is the number of distinct length-`n` factors of the centre
> column, `p_{01}(h)` the number of distinct length-`h` factors of the width-2
> column pair, and `N(h)` the maximum number of width-2 words compatible with
> one centre word.

*Proof.* Map each occurrence of a width-2 factor to (its centre factor, itself).
The first coordinate takes `p_0(h+1)` values; for each, the second takes at
most `N(h)` values by the definition of `N`. ∎

> ### Corollary 8.8. Under EP(T, p), `p_0(n) ≤ T + p` for every `n`, hence
> ```
>        T + p  ≥  p_{01}(h) / N(h)       for every h.
> ```

**This corollary is a genuine theorem and a useless bound.** Recorded as a
failed improvement, per the standing rule:

| `h` | `p_0(h+1)` measured | `p_{01}(h)` measured | `N(h)` | derived `T+p ≥` |
|---:|---:|---:|---:|---:|
| 4 | 32 | 80 | 5 | 16 |
| 8 | 512 | 2893 | 15 | 193 |
| 12 | 8186 | 35808 | 36 | 995 |
| 16 | 47913 | 58061 | — | — |

(measured over a run of 60 000 rows; `N(h)` from
`SMALL_TRIANGLE_RESULTS.json`.)

> ### Failed conjecture X-2H-06.
> *"Counting width-2 factors gives a better lower bound on `T + p` than
> counting centre-column factors."* **False.** `p_{01}(h)` is itself bounded
> by the run length, so `p_{01}(h)/N(h)` is bounded by *run length / `N(h)`*,
> which is **smaller** than the direct bound `p_0(n) ≤ T+p` used by `CT-01`.
> `CT-01` gives `T + p ≥ 998140`; the best value obtainable here is `995`.
> The width-2 route divides the available evidence by `N(h)` instead of
> multiplying it. The conjecture is retained rather than deleted.

At `h = 16, 20, 24` the measured `p_0` and `p_{01}` both saturate near the run
length (47 913, 59 088, 59 932 out of ≈60 000), which is the finite-run
ceiling, not a property of rule 30.

## 5. Statement inventory

| id | statement | status |
|---|---|---|
| **L-8.1** | one row lost per column of depth | **THEOREM**, proved |
| **T-8.3** | `L(h, n) = n + h + 1`; `L(h) = h+1` | **THEOREM**, proved |
| **C-8.4** | unchanged under the reduced boundary | **COROLLARY**, proved |
| **P-8.5** | `L(h) → ∞`: no finite window suffices | **THEOREM**, proved |
| **C-8.6** | certified region has `L(L-1)/2` cells | **COROLLARY**, proved |
| **T-8.7** | `p_{01}(h) ≤ p_0(h+1)·N(h)` | **THEOREM**, proved |
| **C-8.8** | `T + p ≥ p_{01}(h)/N(h)` | **THEOREM**, proved; **weaker than CT-01** |
| **X-2H-06** | width-2 counting beats centre counting | **REFUTED**, preserved |
