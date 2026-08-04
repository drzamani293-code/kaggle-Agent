# Diagonal ANF Results

Algebraic normal forms of the diagonal bit `w_t(t)` in the **free** initial-row
variables `z_0, z_1, ...` (before substituting the single-cell seed).

Because the edge-aligned rule is one-sided, `w_t(k)` depends only on
`z_0, ..., z_k`; in particular `w_t(t)` is a polynomial in `t+1` variables.

Recurrence used (exact over GF(2), from `a OR b = a + b + ab`):

```
    w_{t+1}(k) = w_t(k-2) + w_t(k-1) + w_t(k) + w_t(k-1)·w_t(k).
```

*Verification.* Substituting `z_0 = 1`, `z_j = 0` (`j >= 1`) into every computed
ANF reproduces the real orbit: **130 cells checked, 0 mismatches.**

---

## 1. Measured growth

| `t` | monomials in `w_t(t)` | degree | variables present | contains `z_t` |
|---|---|---|---|---|
| 0 | 1 | 1 | 1 | yes |
| 1 | 3 | 2 | 2 | yes |
| 2 | 6 | 3 | 3 | yes |
| 3 | 10 | 3 | 4 | yes |
| 4 | 19 | 4 | 5 | yes |
| 5 | 44 | 5 | 6 | yes |
| 6 | 81 | 6 | 7 | yes |
| 7 | 150 | 8 | 8 | yes |
| 8 | 256 | 9 | 9 | yes |
| 9 | 545 | 10 | 10 | yes |
| 10 | 1100 | 11 | 11 | yes |
| 11 | 2183 | 12 | 12 | yes |
| 12 | 4297 | 13 | 13 | yes |
| 13 | 8701 | 14 | 14 | yes |

**Observations, all bounded (`t <= 13`), none extrapolated:**

* the monomial count roughly **doubles per step** (ratios 3.0, 2.0, 1.7, 1.9,
  2.3, 1.8, 1.9, 1.7, 2.1, 2.0, 2.0, 2.0, 2.0);
* the **degree is `t+1` or one less** — the polynomial is close to full degree,
  so no low-degree structure is visible;
* **every one of the `t+1` available variables appears**;
* the monomial count is a small fraction of the `2^{t+1}` available monomials
  (8701 of 16384 at `t = 13`), so there *is* cancellation, but not enough to
  suggest structure.

**Nothing is inferred about asymptotics**, per the brief. Thirteen points of a
roughly doubling sequence constrain nothing.

## 2. Collapse does *not* correspond to variable elimination — **NEGATIVE RESULT**

The natural guess was that a COLLAPSING extension at level `k` — which erases
the fibre's dependence on its own earlier value (Theorem R1) — should show up
in the ANF as the disappearance of `z_k` from `w_t(k)` for large `t`.

**It does not.** For every `k <= 10` and every `t <= 13` computed,
`w_t(k)` still contains `z_k`:

```
    first t at which w_t(k) no longer contains z_k :   none, for k = 0..10
```

*Why the guess fails.* Collapse is a property of the **base cycle of the actual
orbit**: it happens when `w_t(k-1) = 1` at some cycle phase. With `z` free, the
value of `w_t(k-1)` is itself a polynomial, not a constant, so the masking term
`c OR u` never becomes an unconditional constant — it becomes
`c + u + cu` with `c` a non-trivial polynomial, which *retains* `u`. Variable
elimination is a seed-specific phenomenon, invisible in the free-variable ANF.

This is worth recording because it closes an attractive-looking route: one
cannot read the collapse structure off the symbolic diagonal formula.

## 3. Cancellation structure

At `t = 13` the ANF has 8701 monomials of a possible 16384 — about 53%. The
sequence of ratios (monomials / `2^{t+1}`) is
`0.50, 0.75, 0.75, 0.63, 0.59, 0.69, 0.63, 0.59, 0.50, 0.53, 0.54, 0.53, 0.52,
0.53` — hovering near 1/2 with no visible trend. **BOUNDED OBSERVATION.**

## 4. Feasibility limit

The computation is `O(M^2)` per multiplication where `M` is the monomial count,
so `t = 13` (8701 monomials) is near the practical limit of this
implementation; `t = 20` would need roughly `10^6` monomials and a
bitset/BDD representation. **Not attempted** — the brief asks not to extend
ranges merely for larger tables, and §1–§3 already show there is no visible
structure to find.
