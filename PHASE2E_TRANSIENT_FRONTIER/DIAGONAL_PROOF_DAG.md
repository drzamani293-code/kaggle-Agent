# Diagonal Proof DAG — Phase 2E §7

Symbolic simplification of the derivation of the centre-column bit
`x_t(0) = w_t(t)`, using hash-consing plus the strongest sound rewrite the
package has: replace any already-settled cell by its earlier representative.

**Result, stated first: the rewrite buys a constant factor of about 1.45, and
the DAG stays quadratic in `t`.** Phase 2D found that collapse does not
correspond to variable elimination in the ANF; this section finds the
structural counterpart — periodicity does not correspond to sharing in the
derivation DAG either.

---

## 1. The DAG and the rewrite

Nodes are space-time cells `(s, k)` in edge-aligned coordinates. The edges come
from the one-sided rule:

```
    (s, k)  ⟶  (s-1, k-2),  (s-1, k-1),  (s-1, k) .
```

Two **constant-folding** rules, both theorems:

* `k < 0`   ⟹ `w_s(k) = 0` (boundary convention).
* `k > 2s`  ⟹ `w_s(k) = 0` (support bound, induction on `s`).
* `s = 0`   ⟹ `w_0(k) = [k = 0]` (the seed).

One **periodicity rewrite**, sound by Theorem EA2 together with the exact
`(T, P)` table:

```
    (R)     w_s(k) = w_{s - P(k)}(k)      whenever   s ≥ T(k) + P(k).
```

Canonical form: apply (R) repeatedly, i.e. map `s ↦ T(k) + ((s - T(k)) mod P(k))`
whenever `s ≥ T(k) + P(k)`. **Hash-consing** identifies nodes with equal
canonical form, so the derivation becomes a DAG rather than a tree.

Rule (R) is the *only* non-trivial rewrite available. It is exactly the
statement that a settled coordinate repeats, and it is the strongest sound use
of everything Phases 2C–2E proved about the cycle structure.

---

## 2. Measurements

`frontier_lab.diagonal_proof_dag`, exact `(T, P)` table for `K ≤ 30000`:

| `t` | DAG nodes | naive cone cells | irreducible cone cells | reduction | nodes / `t²` |
|---|---|---|---|---|---|
| 100 | 3 651 | 5 101 | 3 653 | 1.397 | 0.3651 |
| 300 | 32 520 | 45 301 | 32 543 | 1.393 | 0.3613 |
| 600 | 125 572 | 180 601 | 125 642 | 1.438 | 0.3488 |
| 1200 | 494 886 | 721 201 | 495 039 | 1.457 | 0.3437 |
| 2000 | 1 375 666 | 2 002 001 | 1 375 885 | 1.455 | 0.3439 |
| 3000 | 3 112 401 | 4 503 001 | 3 112 717 | 1.447 | 0.3458 |

*"naive cone cells"* is `|{(s,k) : s ≤ t, k ∈ cone(t,s)}|`, the size of the
derivation with no rewriting at all.
*"irreducible cone cells"* is `|{(s,k) ∈ cone(t,s) : s < T(k) + P(k)}|`, the
cells to which (R) cannot be applied.

### Two things to read off

1. **`nodes ≈ irreducible cone cells`, to within 0.02 %.** The DAG is, to
   measurement precision, exactly the un-rewritable part of the cone. Rewriting
   collapses the settled region to a strip of height at most `P(k) ≤ 16` and
   changes nothing else.
2. **`nodes / t²` sits between 0.344 and 0.365 across a 30× range of `t`, with
   no downward trend.** The reduction factor is a constant, not a growing one.

**No constant is fitted here and no asymptotic claim is made.** The two
statements above are about the six computed values and are labelled **BOUNDED
OBSERVATION**, `t ≤ 3000`.

---

## 3. Why the saving is only constant

The rewrite applies at `(s,k)` iff `s ≥ T(k) + P(k)`, i.e. only to cells whose
level is **low relative to the time**. Using the measured frontier
(`T(k)/k ≈ 1.34` on this range) the rewritable region is roughly
`k ≲ 0.746 s`.

The cone's lower edge is `k = max(0, 2s - t)`. The cone therefore contains a
rewritable cell only while `2s - t ≲ 0.746 s`, i.e. while

```
    s  ≲  t / 1.254  ≈  0.797 t ,
```

which is the crossover measured independently in §6 (`0.79`–`0.83` across four
values of `t`). For `s` beyond it **no cell of the cone is rewritable at all**.
Since the cone's cell count grows like `2(t-s)` on `s > t/2`, the untouched
region `s > 0.8t` alone contains order `(0.2t)²` cells, and the surviving
irreducible part of the earlier region contributes the rest — leaving the total
quadratic. The arithmetic above uses a measured ratio and is therefore an
**explanation of the measurements, not a proof**.

This is a quantitative restatement of §6's conclusion: the settled region and
the diagonal's ancestry overlap only in the early, narrow part of the cone.

---

## 4. Comparison with the ANF route (Phase 2D)

| representation | size of `w_t(t)` | growth |
|---|---|---|
| ANF over the free initial row (Phase 2D) | 1, 3, 6, 10, 19, 44, 81, 150, 256, 545, 1100, 2183, 4297, 8701 monomials for `t = 0…13` | roughly doubling per step, **exponential** |
| hash-consed DAG with rule (R) (here) | 3 651 at `t = 100`; 3 112 401 at `t = 3000` | **quadratic** |

The DAG is enormously better as a *representation*, and no better as an
*argument*: it is a compressed way of saying "simulate the cone", which is what
one could do anyway. Nothing in the DAG's structure is periodic, self-similar,
or bounded-width, and no sub-DAG recurs in a way that would support an
induction on `t`.

**NEGATIVE RESULT, recorded and kept.** Symbolic simplification with the full
strength of the proved periodicity structure does not reduce the diagonal to a
finite object. This closes the "simplify the diagonal formula" route in the same
way Phase 2D's ANF experiment closed the "eliminate variables" route.

---

## 5. Controls

* Rule (R) is sound: it is Theorem EA2 plus the exact `(T,P)` table, and the
  table itself was computed three independent ways (§1).
* Constant folding uses only proved facts (`k>2s ⟹ 0` is proved by induction;
  `k<0 ⟹ 0` is the convention defining the dynamics).
* `nodes ≈ irreducible cone cells` is an **observed near-equality**, not a
  proved identity: the DAG is the set of *reachable* canonical cells, and
  reachability under rewriting is not obviously the same as un-rewritability.
  The two counts are reported separately for that reason.
* No conclusion about non-periodicity is drawn from the DAG's growth. Per the
  standing constraint, **growing complexity is not evidence of aperiodicity**.
