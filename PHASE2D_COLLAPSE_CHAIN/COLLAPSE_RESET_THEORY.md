# Collapse Time and Reset Value

Exact formulas for a COLLAPSING fibre extension, all verified against the real
orbit on **2995 collapsing levels** (`K <= 3000`) with **zero mismatches** in
every one of the four checks.

Setting (Phase 2C, Theorem P2): at level `K` the fibre obeys

```
    u_{t+1} = a_t XOR ( c_t OR u_t ),   a_t = w_t(K-2), c_t = w_t(K-1), u_t = w_t(K),
```

with the base `(a_t, c_t)` exactly `P = P(K-1)`-periodic for `t >= T = T(K-1)`.
COLLAPSING means `c_t = 1` for at least one phase of the base cycle.

---

## 1. The two distinguished phases

**Definition.** Within one base period `[T, T+P)` let

```
    S = { t in [T, T+P) : c_t = 1 },     tau_K = min S,     sigma_K = max S.
```

COLLAPSING is exactly `S != empty`. `tau_K` is the **first collapse position**,
`sigma_K` the **last**.

## 2. Erasure — **THEOREM R1**

> ```
>     u_{tau_K + 1} = 1 XOR a_{tau_K},
> ```
> independently of `u_{tau_K}`. Hence **all dependence on the fibre's own
> earlier value is erased at time `tau_K + 1`**, and never returns.

*Proof.* `c_{tau_K} = 1` makes `c OR u = 1` for either value of `u`, so
`u_{tau_K+1} = a_{tau_K} XOR 1`. For the "never returns" part: every later
value is a function of `u_{tau_K+1}` and the base, and `u_{tau_K+1}` is already
independent of the past. ∎

*Verification.* `reset_value_mismatches: 0` over 2995 levels.

**Corollary R1a (synchronisation).** Two orbits of the tower that agree on the
`(K-1)`-prefix and differ only in `u_T` **merge at time `tau_K + 1`**. A
collapse is a synchronising event in the sense of automata theory: the fibre's
reset word is `c = 1`.

## 3. The one-period reset map — **THEOREM R2**

> The composition over one full base period is the constant map with value
> ```
>     Phi  =  ( 1 XOR a_{sigma_K} )  XOR  XOR_{t = sigma_K+1}^{T+P-1} a_t .
> ```

*Proof.* After the last collapse position `sigma_K`, `u_{sigma_K+1} = 1 XOR
a_{sigma_K}` by R1's computation. For `t > sigma_K` inside the period, `c_t = 0`
so `g_t(u) = u XOR a_t` (Lemma F). Composing those bijections XORs in
`a_{sigma_K+1}, ..., a_{T+P-1}`. ∎

*Verification.* `phi_constant_mismatches: 0` — the formula reproduces the
measured `w_{T+P}(K)` at all 2995 levels.

Note that `Phi` depends only on the base cycle **after the last collapse**;
everything before `sigma_K` is irrelevant to the cycle value. The prefix of the
period is dead information.

## 4. Periodicity onset — **THEOREM R3**

> Coordinate `K` is `P`-periodic from time `tau_K + 1`, hence
> ```
>     T(K)  <=  max( T(K-1),  tau_K + 1 ).
> ```

*Proof.* Write `tau' = tau_K + P` for the first collapse position in the next
period; by periodicity of the base, `a_{tau'} = a_{tau_K}`, so
`u_{tau'+1} = u_{tau_K+1}` by R1. From those two equal values the driving
sequences are translates of each other, so `u_{t+P} = u_t` for every
`t >= tau_K + 1`. Combining with the base's own periodicity from `T(K-1)` gives
the bound on the prefix preperiod. ∎

*Verification.* `sharp_transient_bound_violations: 0`, and
`erasure_failures: 0` (periodicity from `tau_K+1` checked over four further
base periods at every level).

**R3 is sharper than Phase 2C's S2** (`T(K) <= T(K-1) + P(K-1)`), since
`tau_K + 1 <= T(K-1) + P(K-1)` always, with equality only when the *only*
collapse position is the last phase of the period.

## 5. Summary of the exact formulas

| quantity | formula | verified |
|---|---|---|
| first collapse position | `tau_K = min{ t in [T,T+P) : w_t(K-1) = 1 }` | — (definition) |
| last collapse position | `sigma_K = max{...}` | — |
| erasure time | `tau_K + 1` | **0 failures / 2995** |
| value at erasure | `1 XOR w_{tau_K}(K-2)` | **0 mismatches / 2995** |
| one-period reset map | `Phi ≡ (1 XOR a_{sigma_K}) XOR XOR_{t>sigma_K} a_t` | **0 mismatches / 2995** |
| transient bound | `T(K) <= max(T(K-1), tau_K + 1)` | **0 violations / 2995** |

Reproduce: `python3 -c "import collapse_lab as C; d=C.level_data(3000);
print(C.verify_collapse_formulas(*d))"`.

## 6. What this does not give

* No bound on `tau_K` itself beyond `tau_K < T(K-1) + P(K-1)`. The measured
  transient `T(K) ≈ 1.34 K` is therefore not explained by R3 — R3 bounds
  `T(K)` by the base cycle's structure, and the growth comes from where the
  collapses sit inside the cycle, which is not predicted here.
* Nothing about the diagonal. Collapse is a statement about a fixed coordinate;
  the centre column moves across coordinates.
