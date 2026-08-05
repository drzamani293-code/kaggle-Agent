# Weak bridge targets H1–H5

**Phase 2H, section 6.**

The **Bridge** (`BR-01`, open since Phase 1) is:

> from *"the centre column is eventually periodic"*, derive *"a second column is
> eventually periodic"* — which contradicts **T-02**, and would settle Problem 1.

`BR-01` is untouched by every phase, including this one. Section 6 states five
*weak* targets — statements strictly short of the Bridge that a conditional
reversibility argument might reach. Four are proved. The fifth is proved and is
**limitative**: it shows the whole family cannot reach the Bridge.

**None of these solves Rule 30 Prize Problem 1, and none is claimed to.**

---

## H1 — reduced boundary transfer · **PROVED**

> If the centre column is eventually periodic with period `p`, then the
> **reduced boundary word** `β = ( x_s(1) )_{s ∈ Z}`, `Z = { s : x_s(0) = 0 }`,
> is not eventually `p`-periodic along `Z`.

Proof: `PERIODIC_CENTER_TRIANGLES.md` Theorem 3.5. Domain **FIN** (any finite
seed).

**Strength.** Strictly sharper than T-02, which says only "column 1 is not
eventually periodic". H1 confines the obstruction to the rows where the centre
is white — measured at 9880 of 20 000 rows, i.e. about half the bits.

**Why it is not the Bridge.** It is a statement *about* the missing data, not a
derivation *of* it. Refuting its hypothesis requires knowing `β`, i.e. knowing
column 1, which is exactly what is unavailable.

## H2 — width-2 envelope · **PROVED**

> The pair of columns `(0, 1)` determines every column `j ≤ 0`, with exactly
> one row lost per column of depth; and one column determines nothing.

Proof: Theorem 2.B (sufficiency and sharpness), Theorem 2.A (one column is not
enough), Lemma 8.1 (row loss). Domain **ANY**.

**Strength.** It fixes the *exact* boundary geometry: `2` cells per row, never
`1`; envelope `h(h-1)/2` cells; overhead `L(h) = h+1` rows (Theorem 8.3).

**Why it is not the Bridge.** `L(h) → ∞` (Prop. 8.5): no finite window of
periodic boundary data reaches a full column.

## H3 — the right-aligned tower is a bijection · **PROVED**

Phase 2G recorded, as the main gap in this corpus, that *"our entire corpus
works on the left/prefix side because we had no handle on the right."* H3 is
the handle. It does not help.

> ### Theorem 6.3. Put `v_t(k) := x_t(t - k)` (right-aligned coordinates, with
> `v_t(k) = 0` for `k < 0`). Then
> ```
>       v_{t+1}(k)  =  v_t(k)  XOR  ( v_t(k-1)  OR  v_t(k-2) )          (RT)
> ```
> so the prefix `(v_t(0), …, v_t(K))` is autonomous, and its one-step map
> `G_K` is a **bijection** for every `K`.

*Proof of (RT).* `v_{t+1}(k) = x_{t+1}(t+1-k)`; apply (FW) at `j = t+1-k`:
`= x_t(t-k) XOR ( x_t(t+1-k) OR x_t(t+2-k) ) = v_t(k) XOR ( v_t(k-1) OR v_t(k-2) )`. ∎

*Proof of bijectivity.* Define `G_K^{-1}` by induction on `k`:
`v(k) := u(k) XOR ( v(k-1) OR v(k-2) )`, the terms on the right already
recovered. This is a well-defined map and inverts `G_K` by construction, so
`G_K` is injective on a finite set, hence bijective. ∎

> ### Corollary 6.3a (pure periodicity). Every right-aligned prefix orbit has
> **preperiod exactly 0**: `v_{P(K)} = v_0` on `[0, K]`.

*Proof.* An orbit of a bijection on a finite set is purely periodic. ∎

> ### Theorem 6.3b (the asymmetry). The left-aligned map
> `F_K : w ↦ ( w(k-2) XOR ( w(k-1) OR w(k) ) )_k` of Phase 2E is **not**
> injective for any `K ≥ 1`.

*Proof.* Inverting `F_K` requires solving for `w(k-2)` from terms at the
*higher* indices `k-1, k` — which the prefix does not determine. Explicit
collision at `K = 3`: `F_3(1,1,0,0) = F_3(1,0,1,0) = (1,1,0,1)`. ∎

**Verification** (`right_tower_lab.py`, `results/phase2h_right_tower.json`):
(RT) reproduces the orbit on 22 300 cells with **0 disagreements**; the explicit
inverse succeeds on all 65 534 states with `K ≤ 14`; exhaustive census for
`K ≤ 12` shows `G_K` bijective and `F_K` not, at every `K ≥ 1`.

Measured (bounded observation, `K ≤ 20`):

| `K` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10–14 | 15 | 16–20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| right period `P_R(K)` | 1 | 2 | 2 | 4 | 8 | 8 | 16 | 32 | 32 | 64 | 64 | 128 | 256 |
| right preperiod | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| left preperiod (Phase 2E) | 0 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 5 | 6–11 | 13 | 16–23 |

**Why this is a real finding.** It explains, from the algebra alone, the
central phenomenon of Phase 2E: the left tower has preperiods `T(K)` growing
linearly *because `F_K` is not injective*, and the right tower has none
*because `G_K` is*. Phase 2E measured `T(K)`; this proves why `T(K)` exists.

**Why it is not the Bridge, and this is the important part.** The centre column
is the moving diagonal of **both** towers:

```
        x_t(0)  =  w_t(t)  =  v_t(t) .
```

The right tower is better behaved in every respect — purely periodic, periods
powers of two, an explicit inverse — and the centre column is *still* its
diagonal, reading a different coordinate at every step. Bounded per-coordinate
periods would not make the diagonal periodic, because the diagonal never stays
in one coordinate. **The obstruction recorded in Phase 2E is not an artefact of
choosing the left side. It is the same obstruction on the good side.**

## H4 — complexity transfer · **PROVED, and weaker than what we had**

> For every `h`: `p_{01}(h) ≤ p_0(h+1) · N(h)`, hence under EP(T,p),
> `T + p ≥ p_{01}(h) / N(h)`.

Proof: `TRIANGLE_TO_TRACE_TRANSFER.md` Theorems 8.7, 8.8.

**Best value obtained: `T + p ≥ 995`** (at `h = 12`). `CT-01` already gives
`T + p ≥ 998140`. H4 is a correct theorem that loses a factor of `N(h)` against
the evidence, and is recorded as failed conjecture **X-2H-06**.

## H5 — the rule-90 calibration · **PROVED, and limitative**

> ### Theorem 6.5. For rule 90 from a single cell:
> **(a)** the centre column is eventually periodic — indeed `x_t(0) = 0` for
> every `t ≥ 1`;
> **(b)** column 1 is **not** eventually periodic — `x_t(1) = 1` exactly at
> `t = 2^n - 1`, `n ≥ 1`;
> **(c)** **T-01, T-02 and T-03 hold for rule 90, with their proofs unchanged.**

*On (c), spelled out because the whole corollary rests on it.* The proofs of
T-01/T-02/T-03 in `RULE30_VERIFIED_THEOREMS_V1/VERIFIED_THEOREMS.md` use
exactly four properties of the rule, and rule 90 has all four:

| ingredient | where used | rule 90 |
|---|---|---|
| left-permutivity: `l ↦ f(l,c,r)` bijective | T1(a),(b); T2.4 | `f_{90} = l XOR r` — bijective in `l` ✔ |
| radius 1 | T3.1 boundary forcing | ✔ |
| light cone T2.1 (`f(0,0,0)=0`) | T2, T3 | `0 XOR 0 = 0` ✔ |
| frozen edge T2.2 (`x_t(-m-t)=1`) | T2's contradiction | `x_{t+1}(-t-1) = x_t(-t-2) XOR x_t(-t) = 0 XOR 1 = 1` ✔ |

Not one step of T-01/T-02/T-03 uses the `OR` term, right-non-permutivity, or
anything else specific to rule 30.

*Proof.* For rule 90 from a single cell, `x_t(j) = C(t, (t+j)/2) mod 2` when
`t+j` is even and `0` otherwise. **(a)** `x_t(0) = C(t, t/2) mod 2` for even
`t = 2m`; by Kummer's theorem `C(2m, m)` is odd iff adding `m + m` in base 2
produces no carry, i.e. iff `m = 0`. **(b)** For `t = 2m+1`,
`x_t(1) = C(2m+1, m+1) mod 2`, odd iff `(m+1) AND m = 0`, i.e. iff
`m = 2^a - 1`, i.e. `t = 2^{a+1} - 1`; the gaps `2^{a+1}` are unbounded, so no
period exists. **(c)** `f_{90}(l,c,r) = l XOR r` is a bijection in `l`. ∎

*Verification.* `boundary_lab.rule90_column1_ones(4000)`: the ones of column 1
occur at exactly `{1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047}` — the closed
form matches with largest gap 1024. The centre column is nonzero only at
`t = 0`.

> ### Corollary 6.6 (what H5 rules out).
> **T-03 ("at most one eventually periodic column") is attained.** Rule 90
> satisfies every hypothesis used by T-01/T-02/T-03 — left-permutive, finite
> seed, infinite lattice — and its centre column **is** the one exceptional
> eventually periodic column.
>
> Therefore **no argument that uses only left-permutivity, the light cone, a
> finite seed, and the T-02/T-03 column machinery can prove that rule 30's
> centre column is aperiodic.** Any such argument would prove a false statement
> about rule 90.

This is the sharpest limitative statement in the corpus. It converts Phase 1's
informal "seed-blindness" worry into a proved obstruction with an explicit
witness, and it applies to **H1 and H2 directly**: both are derived from
exactly that machinery, and rule 90 satisfies both while having a periodic
centre column.

## Summary

| target | status | domain | reaches the Bridge? |
|---|---|---|---|
| **H1** reduced boundary transfer | **PROVED** | FIN | no — H5 shows why |
| **H2** width-2 envelope, `L(h)=h+1` | **PROVED** | ANY | no — `L(h) → ∞` |
| **H3** right tower is a bijection | **PROVED** | SEED / ANY | no — same diagonal obstruction |
| **H4** complexity transfer | **PROVED**, weaker than CT-01 | FIN | no |
| **H5** rule-90 calibration | **PROVED**, **limitative** | control | it proves H1/H2 cannot |

**`BR-01` remains open.** Phase 2H narrowed the missing data (H1), fixed its
exact geometry (H2), closed the right-hand gap Phase 2G identified (H3),
measured the combinatorial route and found it weaker than what we had (H4), and
proved that the machinery used for H1/H2 is provably incapable of settling
Problem 1 (H5). The last of these is the most useful result of the phase, and
it is a negative one.
