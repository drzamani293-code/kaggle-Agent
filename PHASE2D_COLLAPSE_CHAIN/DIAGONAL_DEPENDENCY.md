# Diagonal Dependency

Which cells can influence the centre bit `x_t(0) = w_t(t)`, and how many of
them are already cycle-determined.

---

## 1. The backward cone — **THEOREM DD1**

The update `w_{s+1}(k) = w_s(k-2) XOR ( w_s(k-1) OR w_s(k) )` reaches back from
`(s+1, k)` to `(s, k-2), (s, k-1), (s, k)`: the index can drop by 0, 1 or 2 per
step and **can never rise**.

> The set of ancestors of the diagonal point `(t, t)` at time `s` is exactly the
> index interval
> ```
>     Cone_t(s) = [ max(0, 2s - t) ,  min(t, 2s) ] .
> ```

*Proof.* Going back `t-s` steps from index `t`, the smallest reachable index is
`t - 2(t-s) = 2s - t` and the largest is `t` (indices never rise). Intersect
with the support `[0, 2s]` (Phase 2C: `w_s(k)=0` for `k > 2s`) and with `k >= 0`. ∎

Sanity: `Cone_t(t) = [t,t]`; `Cone_t(t/2) = [0, t]` — the whole row; and
`Cone_t(0) = [0,0]`, the single seed cell. So the centre bit at time `t` depends
on the **entire** row at time `t/2`, and on nothing but the seed at time 0.

## 2. The periodic frontier

Define `K_per(s) = max{ K : T(K) <= s }` — the largest coordinate that has
already entered its cycle by time `s`. From Phase 2C's exact table,
`T(K) ≈ 1.34 K`, so `K_per(s) ≈ 0.746 s`.

A cell `(s, k)` of the cone is **cycle-determined** if `k <= K_per(s)`, and
**transient** otherwise.

## 3. Where the cone leaves the periodic region — **THEOREM DD2 + measurement**

`Cone_t(s)` is entirely transient exactly when `2s - t > K_per(s)`. With
`K_per(s) ≈ 0.746 s` this is `s > t / 1.254 ≈ 0.797 t`. Measured exactly
(using the true `K_per`, not the approximation):

| `t` | first `s` with the cone entirely transient | as a fraction of `t` |
|---|---|---|
| 40 | 37 | 0.925 |
| 120 | 93 | 0.775 |
| 400 | 333 | 0.833 |
| 1200 | 987 | 0.823 |
| 2400 | 1942 | 0.809 |

> **For the last ≈ 19% of its backward history, every ancestor of the diagonal
> point lies strictly in the transient region.** (BOUNDED OBSERVATION for the
> constant; DD1 and the criterion are theorems.)

## 4. Ancestors killed by a collapse — **THEOREM DD3**

The cycle-determined part of the cone carries very little information:

> Within a single COLLAPSING chain of constant period `P`, **the values of
> every cycle-determined cell are functions of just `2P` bits** — the pair of
> cycle words at the bottom of the chain (Corollary CH2).

*Proof.* CH2 gives `C_K = Psi(C_{K-2}, C_{K-1})` throughout the chain, so the
whole family `{C_K}` is a function of one pair; and a cycle-determined cell
`(s,k)` has value `C_k[(s - T*) mod P]`. ∎

So the ancestors that a collapse "kills" are killed in a precise sense: their
own initial fibre values are erased (Corollary R1a), and what remains of them
is `2P = 32` bits for the entire cycle-determined region, however large it is.

**But this does not bound the diagonal bit.** The transient part of the cone
is not covered by DD3, and it is large: at `t = 2400` the cone is entirely
transient for `s > 1942`, and even below that the transient portion is the part
of `[max(0,2s-t), min(t,2s)]` above `K_per(s)`. No bounded-memory description
of `w_t(t)` follows. This is the reason registry item **C1** is not proved.

## 5. Status

| statement | label |
|---|---|
| DD1 cone formula | **THEOREM** |
| the diagonal depends on the whole row at time `t/2` | **THEOREM** (DD1 at `s = t/2`) |
| DD3: cycle-determined cells carry `2P` bits total within a chain | **THEOREM** (via CH2) |
| the cone is entirely transient for `s >~ 0.81 t` | **BOUNDED OBSERVATION**, `t <= 2400` |
| any bounded-memory description of `w_t(t)` | **NOT OBTAINED** |
