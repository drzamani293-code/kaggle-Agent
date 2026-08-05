# Coupled left/right towers

**Phase 2I, section 6.** The brief requires explicit answers to four questions;
they are answered in §5, and each answer is either proved or labelled as a
bounded observation.

---

## 1. The two towers, identical indexing

For a seed with support in `[0, M]` (all suite seeds are normalised this way):

```
    left  tower :   w_t(k) := x_t( -t + k )
    right tower :   v_t(k) := x_t( t + M - k )
```

> ### Theorem 6.1 (both towers are autonomous). For rule 30,
> ```
>     w_{t+1}(k) = w_t(k-2) XOR ( w_t(k-1) OR w_t(k) )                (LT)
>     v_{t+1}(k) = v_t(k)   XOR ( v_t(k-1) OR v_t(k-2) )              (RT)
> ```
> with the convention that indices `< 0` read `0`. Both right-hand sides use
> only indices `≤ k`, so every prefix `(·)_t(0..K)` evolves autonomously.

*Proof.* Substitute the coordinate change into (R30). For (LT):
`w_{t+1}(k) = x_{t+1}(-t-1+k) = x_t(-t-2+k) XOR (x_t(-t-1+k) OR x_t(-t+k))`.
For (RT): `v_{t+1}(k) = x_{t+1}(t+1+M-k) = x_t(t+M-k) XOR (x_t(t+1+M-k) OR
x_t(t+2+M-k))`. ∎

Verified against the real orbit: **0 disagreements** in 2520 cells per rule, for
rules 30, 90 and 150.

For rules 90 and 150 the same substitution gives `w_t(k-2) XOR w_t(k)` and
`w_t(k-2) XOR w_t(k-1) XOR w_t(k)` respectively, and the mirrored forms for `v`.

## 2. Every exact coupling identity

The two towers name the same cell exactly when the columns coincide:

```
    w_t(k1) = v_t(k2)   ⟺   -t + k1 = t + M - k2   ⟺   k1 + k2 = 2t + M .
```

For `M = 0` this is `k1 + k2 = 2t`, and the complete list of couplings is the
one-parameter family

> ### Theorem 6.2 (the coupling family). For every `t` and every `r`,
> ```
>       w_t(t - r) = x_t(-r) ,        v_t(t - r) = x_t(+r) ,
> ```
> so `( w_t(t-r), v_t(t-r) )` is exactly the **mirror pair** at distance `r`
> from the centre. The case `r = 0` is the tautology `c_t = w_t(t) = v_t(t)`;
> every `r ≠ 0` is new information.

The brief also names `( w_{t+s}(t-r), v_t(t-r) )`. That pair is
`( x_{t+s}(-r-s), x_t(r) )` — two cells in different rows and different
columns; it carries no coincidence relation and is not used further.

## 3. The mirror defect and its exact driver

> ### Definition 6.3. `δ_t(r) := x_t(-r) + x_t(r) = w_t(t-r) + v_t(t-r)`,
> the **mirror defect** — the natural coupled-tower quantity.

Let `σ` denote the spatial mirror `(σx)(j) = x(-j)`. Rules 90 and 150 are
*symmetric* local rules (`f(l,c,r) = f(r,c,l)`), hence `σ`-equivariant. Rule 30
is not: `f(0,1,1) = 1` but `f(1,1,0) = 0`.

> ### Theorem 6.4 (exact evolution of the mirror defect, rule 30).
> ```
>     δ_{t+1} = L_{150} δ_t + ν_t ,        ν_t(j) := n_t(j) + n_t(-j) ,
> ```
> and for a mirror-symmetric seed `δ_0 = 0`, so by Duhamel
> ```
>     δ_t(j) = Σ_{s<t} Σ_d T(t-1-s, d) ν_s(j-d) .
> ```
> Moreover `δ_t(0) = 0` for every `t`, identically.

*Proof.* `L_{150}` commutes with `σ`; apply `σ` to (R30) and add. For the last
claim, `δ_t(0) = x_t(0)+x_t(0) = 0` directly; consistently, the Duhamel form
gives `Σ_{s}Σ_j T(m,j)(n_s(j) + n_s(-j)) = 0` because `T(m,·)` is symmetric, so
each event is counted twice. ∎

Verified cell by cell: **0 failures** in 1760 cells for each of three seeds.

> ### Corollary 6.5. **The mirror-symmetry breaking of rule 30 is caused
> entirely by the nonlinear field.** `ν ≡ 0` for any linear rule, so a
> symmetric seed under rule 90 or rule 150 gives `δ ≡ 0` forever. Under rule 30,
> even a perfectly symmetric row generates asymmetry, because
> `ν_t(j) = x_t(j)(x_t(j+1) + x_t(j-1))` when `x_t` is symmetric — the product
> `x(j)x(j+1)` is not `σ`-invariant.

## 4. Is any symmetric seed spared?

> ### Bounded observation 6.6. Of **247** mirror-symmetric seeds (all symmetric
> supports within `[-6, 6]`), **0** produced a rule-30 orbit that stayed
> symmetric through `t = 20`.

> ### Failed conjecture 2I-X-03.
> *"Some symmetric rule-30 seed might keep a symmetric orbit."* No such seed
> was found in the scanned family. This is **not** a theorem — 247 seeds is a
> bounded search, and it is reported as "no witness found in the searched
> range", never as UNSAT.

Consequence for the registry: Π3's F1 half holds for every symmetric rule-30
seed tested, so Π3's *seed* discrimination (F3) comes entirely from
**asymmetric** seeds, i.e. from the first conjunct. That is a real limitation
and is recorded as such in `SEED_DISCRIMINATOR_REGISTRY.md` §2.

## 5. The four questions the brief requires answering

### (a) Is the asymmetry a property of the rule alone?

**The tower asymmetry: yes.** Measured image sizes:

| rule | left tower bijective | right tower bijective |
|---|---|---|
| **30** | **NO** (29/64, 101/256, 361/1024, 1325/4096 at `K = 5,7,9,11`) | yes |
| **90** | yes | yes |
| **150** | yes | yes |

`F_K` and `G_K` are defined by the rule; no seed enters. So left-tower
information loss is **RULE-SPECIFIC** and fails filter F3 — a sharpening of
Phase 2H's H3, which had only compared rule 30's two towers. Rule 30 is the
only rule in the control set that loses information in *either* direction.

**The mirror asymmetry: also the rule, given a symmetric seed.** By Theorem 6.4
the driver `ν` is a function of the rule's quadratic term.

### (b) Which parts depend on the seed?

Three, exactly:

1. **Whether the seed is mirror-symmetric at all** — this is the only seed
   input to Π3, and it is decided by the seed alone (Theorem 2.6 shows it is
   equivalent to the linear shadow being constant).
2. **Which collision classes the orbit visits** — `κ_K(t)`, the preimage count
   of the *actual* state. This is seed data, and §7 shows it is degenerate.
3. **The linear shadow `S(t)`** itself, which is a pure function of the seed.

### (c) Can any seed-dependent coupled invariant be formulated?

**Yes — Π3**, and it is the promoted candidate:

> the seed is mirror-symmetric about a cell, **and** the orbit is not.

It couples a seed condition to a rule condition, which §5 of the registry
argues is forced: single properties depend on one input or the other.

**But it does not reach the centre.** By Theorem 6.4, `δ_t(0) = 0` identically.
The invariant is structurally blind to `c_t`. Combining with the exact centre
relations of `NONLINEAR_ACTIVITY_THEORY.md` §5 gives, under EP(T,p),

```
    δ_t(1) = c_{t+1}                      whenever c_t = 0
    δ_t(1) = 1 + c_{t+1} + x_t(1)         whenever c_t = 1
```

— the first is **unconditional** (it is just the rule, no periodicity used), and
the second leaves exactly the residue `x_t(1)` on `{c_t = 1}` that Phase 2H's
H1 already isolated. **No new leverage.** This is why F4 is OPEN with a proved
obstruction rather than merely unproved.

### (d) Does rule 90 make the invariant vanish or become symmetric?

**Vanish, provably.** Rule 90 is `σ`-equivariant, so a symmetric seed gives
`σ x_t = x_t` for all `t` and `δ ≡ 0`. Verified: 0 nonzero cells for rule 90 and
rule 150 on `A_single`, `E_101`, `E_111`, `F_rand05` (recentred), against 864
nonzero cells for rule 30 on `A_single`. This is Π3's F2 pass, and it is a
theorem, not an observation.

---

## 6. Statement inventory

| id | statement | status | filter role |
|---|---|---|---|
| **C-6.1** | both towers autonomous, all three rules | **THEOREM**, 0 disagreements | — |
| **C-6.2** | the coupling family `w_t(t-r) = x_t(-r)`, `v_t(t-r) = x_t(r)` | **THEOREM** | — |
| **C-6.4** | `δ_{t+1} = L δ_t + ν_t`; `δ_t(0) ≡ 0` | **THEOREM**, 5280 cells, 0 failures | the F4 obstruction |
| **C-6.5** | linear rules preserve mirror symmetry | **THEOREM** | Π3's **F2** |
| **BO-6.6** | 0 of 247 symmetric seeds kept a symmetric rule-30 orbit | **BOUNDED OBSERVATION** | limits Π3's F3 |
| **T-6.7** | rule 30's left tower is the only non-bijective tower map in the control set | **THEOREM**, exhaustive to `K = 11` | Π5, rule-specific |
