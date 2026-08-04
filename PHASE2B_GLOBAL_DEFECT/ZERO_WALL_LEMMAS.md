# Zero-Wall Lemmas

Consequences of the hypothesis

> **(H-WALL)** `d_t(0) = 0` for every `t >= T`, where
> `d_t(j) = x_{t+p}(j) XOR x_t(j)`.

(H-WALL) is exactly what eventual `p`-periodicity of the centre column after
time `T` would give. Every item is labelled **THEOREM**, **CONJECTURE** or
**COMPUTATIONAL OBSERVATION**. Notation and the identities D1–D4 are from
`DEFECT_DYNAMICS.md`.

**Standing caveat.** (H-WALL) is conjecturally false. Everything proved from it
is therefore *conditional*, and — as in Phase 1 — a conditional whose
hypothesis fails is vacuously true. These lemmas are proof material, not
evidence about Rule 30. What *can* be tested is whether they hold at the times
where a finite wall happens to occur in the real orbit, and that is what the
verification columns report.

---

## W1 — the wall equation — **THEOREM**

> Let `t` satisfy `d_t(0) = 0` and `d_{t+1}(0) = 0`. Then
> ```
>     d_t(-1) = d_t(1) · ( 1 XOR x_t(0) ).
> ```
> Equivalently: if `x_t(0) = 1` then `d_t(-1) = 0`;
> if `x_t(0) = 0` then `d_t(-1) = d_t(1)`.

*Proof.* Apply D1 at `j = 0`: `d_{t+1}(0) = d_t(-1) XOR Delta(x_t(0), x_t(1),
d_t(0), d_t(1))`. Since `d_t(0) = 0`, the collapse identity D4 gives
`Delta = d_t(1)·(1 XOR x_t(0))`. Setting `d_{t+1}(0) = 0` and solving for
`d_t(-1)` gives the claim. ∎

*Relation to Phase 1.* W1 **is** Proposition 6 of
`WIDTH2_PROOF_RECONSTRUCTION.md`, rewritten in defect variables. The two
statements are the same identity; the defect form makes the mechanism visible
as "a black centre cell masks the defect on its right" (row 9/13 of the D3
table).

*Verification.* Real orbit, lags 1, 3, 7, 16, 27, `t <= 8000`: hypothesis
occurred 1984–2025 times per lag, **0 violations**.

---

## W2 — the wall decouples the two half-planes — **THEOREM**

> Under (H-WALL), for every `t >= T`:
> 1. the defect field on `j >= 1` evolves autonomously:
>    `d_{t+1}(j)` for `j >= 1` is a function of `d_t(·)` restricted to `j >= 1`
>    (the value `d_t(0) = 0` being known);
> 2. `d_{t+1}(1) = Delta(x_t(1), x_t(2), d_t(1), d_t(2))` — no term from the
>    left half-plane appears;
> 3. the left half-plane is *driven*: `d_t(-1)` is determined by `d_t(1)` and
>    `x_t(0)` through W1.

*Proof.* (1) and (2): D1 at `j >= 1` uses `d_t(j-1), d_t(j), d_t(j+1)`, and for
`j = 1` the term `d_t(0)` vanishes by (H-WALL). (3) is W1. ∎

**Interpretation.** A zero wall makes column 0 a one-way membrane: the right
half-plane's defect dynamics is closed, and it dictates the left. This is the
defect-field counterpart of Phase 1's Lemma 2 (two adjacent columns determine
everything to the left) and is the reason the wall hypothesis is so rigid.

---

## W3 — a run of ones forces a zero triangle — **THEOREM**

> Suppose `d_s(0) = 0` for every `s ∈ [t0, t1+1]` and `x_s(0) = 1` for every
> `s ∈ [t0, t1]`. Then
> ```
>     d_s(-m) = 0    for every m = 1, ..., t1-t0+1
>                    and every s ∈ [t0, t1-(m-1)].
> ```
> The zero region is a **triangle**: it loses one time step per column moved
> left.

*Proof.* Induction on `m`.

*Base `m = 1`.* For `s ∈ [t0, t1]` both `d_s(0) = 0` and `d_{s+1}(0) = 0`, so
W1 applies and `x_s(0) = 1` gives `d_s(-1) = 0`.

*Step.* Assume `d_s(-m) = 0` for `s ∈ [t0, t1-(m-1)]`, and (for `m >= 2`) also
`d_s(-(m-1)) = 0` on the wider window. Apply D1 at `j = -m`:
```
    d_{s+1}(-m) = d_s(-m-1) XOR Delta( x_s(-m), x_s(-m+1), d_s(-m), d_s(-m+1) ).
```
For `m = 1` the two defect arguments are `d_s(-1) = 0` and `d_s(0) = 0`, so
`Delta = 0` (first row of D3). For `m >= 2` they are `d_s(-m) = 0` and
`d_s(-m+1) = 0` by the inductive hypothesis, so again `Delta = 0`. Hence
`d_s(-m-1) = d_{s+1}(-m)`, which is `0` whenever `s+1 ∈ [t0, t1-(m-1)]`, i.e.
whenever `s ∈ [t0-1, t1-m]`. Intersecting with the range on which the
inductive hypothesis is available, `s ∈ [t0, t1-m]`. ∎

*Verification.* Real orbit, wherever the hypothesis occurs:

| lag `p` | windows found | longest run | cells checked | violations |
|---|---|---|---|---|
| 1 | **0** | — | 0 | — |
| 3 | 343 | 2 | 1029 | **0** |
| 7 | 315 | 6 | 1269 | **0** |
| 16 | — | — | 1231 | **0** |
| 27 | — | — | 1362 | **0** |

**Negative result, preserved.** At `p = 1` the hypothesis of W3 **never occurs,
and provably cannot**: if `d_t(0) = 0` with `p = 1` then `x_{t+1}(0) = x_t(0)`,
so `x_t(0) = 1` forces `x_{t+1}(0) = 1`, and the run can only terminate by the
wall breaking — which is exactly the case W3 excludes. So W3 is vacuous for
`p = 1`. At `p = 2` the longest qualifying run in `t <= 20000` is 1, so the
`min_run = 2` form is also unrealised. This is a limitation of what the real
orbit offers, not of the lemma.

---

## W4 — the second left column is slaved to the right — **THEOREM**

> If `d_s(0) = 0` for `s ∈ {t, t+1, t+2}` then
> ```
>     d_t(-2) = d_{t+1}(1)·(1 XOR x_{t+1}(0))  XOR  d_t(1)·(1 XOR x_t(0)).
> ```

*Proof.* D1 at `j = -1` with `d_t(0) = 0` gives, via D4,
`d_{t+1}(-1) = d_t(-2) XOR d_t(-1)·(1 XOR x_t(0))`. Apply W1 at `t` and at
`t+1` to replace `d_{t+1}(-1)` and `d_t(-1)`, and use
`(1 XOR x_t(0))² = (1 XOR x_t(0))`. ∎

*Verification.* Lags 1, 3, 7, 16, 27, `t <= 8000`: hypothesis met ~2000 times
per lag, **0 violations**.

**Corollary W5 (leftward determination) — THEOREM.** By induction on `m`, under
(H-WALL) every `d_t(-m)` is a function of the centre column and of the defect
values `d_s(1)` at times `s ∈ [t, t+m-1]`. The entire left defect half-plane is
determined by column 1's defect sequence together with the centre column.

---

## W6 — runs of zeros — **THEOREM**

> If `d_s(0) = 0` for `s ∈ [t0, t1+1]` and `x_s(0) = 0` for `s ∈ [t0, t1]`,
> then for `s ∈ [t0, t1]`:
> ```
>     d_s(-1) = d_s(1),
> ```
> and, for `s ∈ [t0, t1-1]`,
> ```
>     d_s(-2) = d_{s+1}(1) XOR d_s(1).
> ```

*Proof.* W1 with `x_s(0) = 0`; then W4 with both correction factors equal to 1. ∎

**Contrast with W3.** A run of *ones* in the centre column **kills** defects to
the left (W3); a run of *zeros* **copies** them from the right (W6). The centre
column therefore acts as a programmable gate on the defect field, and which
gate it is depends on the centre bit. This is the sharpest structural statement
Phase 2B produces about the wall.

---

## W7 — what a centre block forces — **THEOREM (summary form)**

> Let the wall hold on `[t0, t1+1]` and let `B = x_{t0}(0) … x_{t1}(0)` be the
> centre block. Then for `s ∈ [t0, t1]`:
> ```
>     d_s(-1) = 0            wherever B has a 1 at position s,
>     d_s(-1) = d_s(1)       wherever B has a 0 at position s.
> ```
> Consequently the sequence `(d_s(-1))` on the block is obtained from
> `(d_s(1))` by **masking with the complement of the block**, and the maximal
> runs of 1s in `B` of length `k` generate zero triangles of height `k` reaching
> `k` columns to the left (W3).

*Proof.* Pointwise application of W1, then W3 on each maximal run of ones. ∎

**COMPUTATIONAL OBSERVATION (not a theorem).** In the real orbit the longest
run of ones in the centre column is 21 within `10^6` steps (Phase 1 §5.9), so
the largest zero triangle any *actual* finite wall could force has height ≤ 21
— and in fact the longest actual wall (a maximal run with `d_t(0)=0`) over all
83 lags sampled here is **18**, at `p = 59`. Finite walls in the real orbit are
short; nothing here bounds the length of a *hypothetical* infinite one.

---

## What these lemmas do and do not give

**They give:** an exact algebraic description of how a zero wall constrains its
neighbourhood — the right half-plane closes on itself (W2), the left is
determined by it (W5), and the centre column acts as a mask that annihilates
(W3) or copies (W6) defects depending on its bits.

**They do not give:** any contradiction. Every lemma above is consistent with a
wall existing. Deriving a contradiction requires an input that none of W1–W7
supplies — a *global* fact about the single-cell orbit, which is precisely what
Phase 1 §10.3 identified as the missing ingredient. The candidate global
statements are registered as G1–G4 in `GLOBAL_LEMMA_REGISTRY.md`.

**In particular** it is *not* claimed that (H-WALL) is impossible, nor that the
lemmas make it unlikely. They are conditional identities, all verified, all
consistent.
