# Skew-Product Analysis

`F_K` is an iterated skew product: coordinate `K` is a **one-bit fibre** driven
by the `(K-1)`-prefix base (Theorem P2). This section turns that into exact
recursive bounds — and converts three of the five E-statements from
conjectures into **theorems**.

---

## 1. The fibre map

With `a_t = w_t(K-2)`, `c_t = w_t(K-1)`, `u_t = w_t(K)` (Theorem P2 at index
`K`):

```
    u_{t+1} = g_t(u_t),        g_t(u) = a_t XOR ( c_t OR u ).
```

**LEMMA F (fibre trichotomy) — THEOREM.** Exhaustively, over all four `(a,c)`:

| `a` | `c` | `g(0)` | `g(1)` | type |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | **identity** |
| 1 | 0 | 1 | 0 | **negation** |
| 0 | 1 | 1 | 1 | **constant 1** |
| 1 | 1 | 0 | 0 | **constant 0** |

> `c_t = 1` makes `g_t` **constant** — the fibre forgets its own value.
> `c_t = 0` makes `g_t` a **bijection** (identity if `a_t = 0`, negation if
> `a_t = 1`).

*Proof.* `c = 1` gives `c OR u = 1` for both `u`, so `g ≡ a XOR 1`. `c = 0`
gives `g(u) = a XOR u`. ∎

This is the same masking phenomenon as Fact F4 of Phase 1 (rule 30 is not
right-permutive) seen in the prefix tower.

## 2. Composition over one base period

Suppose the `(K-1)`-prefix is periodic from `T = T(K-1)` with period
`P = P(K-1)`. Then `(a_t, c_t)` is `P`-periodic for `t >= T`. Let

```
    Phi = g_{T+P-1} o ... o g_{T+1} o g_T .
```

**LEMMA C (composition trichotomy) — THEOREM.** `Phi` is one of:

| condition on the period word `(a_t, c_t)_{t=T}^{T+P-1}` | `Phi` | name |
|---|---|---|
| some `c_t = 1` | constant | **COLLAPSING** |
| all `c_t = 0` and `XOR_t a_t = 0` | identity | **NEUTRAL** |
| all `c_t = 0` and `XOR_t a_t = 1` | negation | **DOUBLING** |

*Proof.* If every `c_t = 0` then every `g_t(u) = u XOR a_t`, so
`Phi(u) = u XOR (XOR_t a_t)`: identity or negation. If some `c_t = 1`, let
`t*` be the **last** such index; `g_{t*}` is constant, and every later `g_t`
is a bijection or a constant, so the composition after `t*` is a fixed function
of a constant — i.e. `Phi` is constant. ∎

*Verification.* `verify_classification(9)`: over 43,690 forcing words of
lengths 1–9, the structural rule (`has c=1` / `XOR a`) predicts the composed
map in **every** case.

## 3. The recursion — **THEOREMS S1, S2, S3**

> **S1 (period trichotomy).** `P(K) ∈ { P(K-1), 2·P(K-1) }`. Consequently
> `P(K) = 2^{m(K)}` with `m` nondecreasing and `m(K) - m(K-1) ∈ {0, 1}`.
>
> **S2 (transient bound).** `T(K) <= T(K-1) + P(K-1)`.
>
> **S3 (monotonicity).** `T(K-1) <= T(K)`.

*Proof of S1.* Write `T = T(K-1)`, `P = P(K-1)`, and `u_{T+jP} = Phi^j(u_T)`.
By Lemma C:

* `Phi` constant `v`: `u_{T+jP} = v` for every `j >= 1`, so coordinate `K` is
  `P`-periodic from `T + P`;
* `Phi` identity: `u_{T+jP} = u_T`, so coordinate `K` is `P`-periodic from `T`;
* `Phi` negation: `u_{T+jP}` alternates, so coordinate `K` is `2P`-periodic
  from `T`.

In every case the whole `K`-prefix (base `P`-periodic from `T`, fibre
`Q`-periodic from `T+P`) is `Q`-periodic from `T+P` with `Q ∈ {P, 2P}`. Hence
`P(K)` divides `Q <= 2P`. By Corollary P1a, `P(K-1) = P` divides `P(K)`.
A multiple of `P` that divides some `Q ∈ {P, 2P}` is `P` or `2P`. ∎

Induction from `P(0) = 1` gives the power-of-two claim. ∎

*Proof of S2.* The three cases above give periodicity from `T + P` at the
latest. ∎

*Proof of S3.* Corollary P1a (projection). ∎

**These are exactly statements E1, E2, E3 of the brief. They are now
theorems, not conjectures** — see `PREFIX_PERIOD_RESULTS.md` §3 for the
computational confirmation (0 counterexamples over `K <= 30000`, as they must
be).

**Corollary S2a.** `T(K) <= T(K0) + sum_{j=K0}^{K-1} P(j)`. With the measured
`P(j) = 16` for `400 <= j <= 30000`, this gives `T(K) <= T(400) + 16(K-400)`,
i.e. `T(K) = O(K)` — the linear growth observed empirically is *bounded above*
by theory, though the theoretical constant (16) is far above the measured
slope (≈ 1.34).

## 4. A lower bound — **THEOREM S4**

> If `w_t(K) = 1` for at least one `t`, then
> ```
>     T(K) + P(K)  >  floor(K/2).
> ```

*Proof.* By the support property `w_t(K) = 0` whenever `2t < K`. Suppose
`T(K) + P(K) <= floor(K/2)`. Every `t ∈ [T(K), T(K)+P(K))` then satisfies
`t <= floor(K/2) - 1 < K/2`, so `w_t(K) = 0` on a full period; periodicity
propagates this to all `t >= T(K)`. For `t < T(K) <= floor(K/2)` we also have
`2t < K`, so `w_t(K) = 0`. Hence `w_t(K) = 0` for every `t`. ∎

**Corollary S4a — the weakest provable form of E4.** For every `K` at which
coordinate `K` is ever 1,

```
    T(K)  >  floor(K/2) - P(K).
```

With `P(K) = 16` for `K >= 400`, this is `T(K) > K/2 - 16`. **The measured
value is `T(K) ≈ 1.34 K`, so a factor ≈ 2.7 separates what is proved from what
is observed.** Closing that gap is open.

**The hypothesis of S4 is not vacuous, and it genuinely fails once.**
`coordinate_ever_one(3000, 8000)` finds exactly one exception:

> **`w_t(7) = 0` for every `t`** — coordinate 7 is permanently zero.

Reason, fully explicit: `T(7) = 2`, `P(7) = 2`, and the 7-prefix cycles between
`11001000` and `11011110` (bits `k = 0..7` left to right); both have bit 7
equal to 0, and `w_0(7) = w_1(7) = 0`. So the 2-cycle certifies it. **THEOREM**,
by exhibition of the cycle. Every other `K <= 3000` takes the value 1.

## 5. Is it "one-bit periodically forced" once the base cycles?

**Yes — THEOREM.** For `t >= T(K-1)` the pair `(a_t, c_t)` is exactly periodic
with period `P(K-1)`, so coordinate `K` obeys a one-bit recurrence driven by a
**purely periodic** forcing word. Lemma C then classifies the outcome
completely, and `FORCED_BIT_CLASSIFICATION.md` works out which class occurs
where, how much extra transient each permits, and whether the initial condition
selects a branch.

## 6. What the skew product does *not* give

* No lower bound on `P(K)`: S1 permits `P(K) = P(K-1)` forever, and indeed the
  measurements show only **four** doublings up to `K = 30000`.
* No lower bound on `T(K)` beyond S4's `K/2`.
* Nothing about the diagonal `w_t(t)`, which is not a coordinate of any fixed
  prefix. That is the subject of `DIAGONAL_BRIDGE_ATTEMPT.md`, and it is where
  the route stalls.
