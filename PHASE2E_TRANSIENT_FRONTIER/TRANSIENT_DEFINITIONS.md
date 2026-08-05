# Transient Definitions — Phase 2E §1

Exact definitions of the five quantities the brief names, and **proofs of which
of them coincide and which are genuinely distinct**.

Labels: **THEOREM** (proved here, all `K`), **COROLLARY**, **BOUNDED
OBSERVATION** (measured, with its range), **CONJECTURE**.

---

## 0. Setting (imported, already proved)

Edge-aligned coordinates (Phase 2B **Theorem EA1**):

```
    w_t(k) = x_t(-t + k),     w_{t+1}(k) = w_t(k-2) XOR ( w_t(k-1) OR w_t(k) )
    w_0 = (1, 0, 0, ...),     w_t(k) = 0 for k < 0
```

* **EA2** (Phase 2B). The prefix `W_t^K = (w_t(0), …, w_t(K))` evolves
  autonomously, so it is eventually periodic.
* **EA3** (Phase 2B). The centre column is the diagonal, `x_t(0) = w_t(t)`.
* **Support bound** (elementary induction). `w_t(k) = 0` whenever `k > 2t`.
* **S1/S2** (Phase 2C). `P(k) | P(K)` for `k ≤ K`; `P(K) ∈ {P(K-1), 2P(K-1)}`.
* **E3** (Phase 2C). `T` is non-decreasing. Re-proved as Theorem 1.3 below.
* **Fibre trichotomy** (Phase 2C/2D). With `a_t = w_t(K-2)`, `c_t = w_t(K-1)`,
  `u_t = w_t(K)`, the level-`K` fibre is `u_{t+1} = a_t XOR (c_t OR u_t)`;
  the level is COLLAPSING iff `c_t = 1` somewhere on the base cycle.

---

## 1. The definitions

| symbol | definition | type |
|---|---|---|
| `P(K)` | the exact (minimal) period of the orbit of `W^K` | level ↦ int |
| `T(K)` | the exact preperiod: least `t` with `W_t^K = W_{t+P(K)}^K` | level ↦ int |
| `r_q(k)` | least `t` such that `w_s(k) = w_{s+q}(k)` for **every** `s ≥ t` | (lag, coord) ↦ int |
| `R(K)` | `:= r_{P(K)}(K)` — the **coordinate** preperiod of the top coordinate at the level's own lag | level ↦ int |
| `τ(K)` | `:= min{ t ≥ T(K-1) : w_t(K-1) = 1 }` — the **first collapse time** (Phase 2D) | level ↦ int ∪ {undefined} |
| `ρ(K)` | `:= max{ t < T(K) : w_t(K-1) = 1 }` — the **last forcing time before settling** | level ↦ int ∪ {undefined} |
| `A(t,K)` | `:= max(0, T(K) - t)` — the **residual transient age** of level `K` at time `t` | (time, level) ↦ int |
| `A(t)` | `:= A(t,t) = max(0, T(t) - t)` — the **diagonal age deficit** | time ↦ int |
| `K_per(t)` | `:= max{ K : T(K) ≤ t }` (`-1` if none) — the **periodic frontier** | time ↦ int |

`τ(K)` is undefined exactly when level `K` is not COLLAPSING. `ρ(K)` is
undefined exactly when coordinate `K-1` is identically `0` on `[0, T(K))`;
measured, that happens only at `K ∈ {4, 5, 6, 7, 8}` for `K ≤ 30000`.

**Warning about `r_q(k)`.** It is meaningful only when `P(k) | q`. Otherwise the
lag-`q` comparison fails at arbitrarily large `t` and any computed value is an
artefact of where the simulation stopped. Every use below filters on `P(k) | q`,
and `frontier_lab.reliability_horizon` reports the cut-off.

---

## 2. Which are equal, which are distinct

### THEOREM 1.1 (coordinate decomposition of `T`)

> For every `K`,  `T(K) = max_{0 ≤ k ≤ K} r_{P(K)}(k)`.

*Proof.* Write `p = P(K)`. For any `t`, `W_s^K = W_{s+p}^K` for all `s ≥ t` iff
`w_s(k) = w_{s+p}(k)` for all `s ≥ t` and all `k ≤ K`, i.e. iff `t ≥ r_p(k)` for
every `k ≤ K`. The least such `t` is the maximum of the `r_p(k)`.

It remains to note that this least `t` is `T(K)`. `W^K` evolves under a
deterministic map, so its orbit is a "rho" shape: a tail of length `T(K)`
followed by a cycle of length `P(K)`, and `W_s^K = W_{s+p}^K` holds for all
`s ≥ T(K)` and fails at `s = T(K)-1` (else `T(K)` would not be minimal). ∎

Each `r_p(k)` is well defined because `P(k) | p` (Phase 2C S1). ∎

*Verified:* `run_phase2e.py` §1, **30001 levels, 0 mismatches**, `K ≤ 30000`.
`r_p(k)` computed by three independent implementations (backward sweep with an
`unseen` mask; explicit suffix-OR array swept forward; per-coordinate scan
straight from the definition) — all agree.

### COROLLARY 1.2  `R(K) ≤ T(K)` for every `K`.

Immediate from 1.1 (the maximum includes the term `k = K`).
*Verified:* 0 exceptions in 29999 levels. `T(K) = R(K)` at **21139** levels and
`T(K) > R(K)` at **8860** — so **`R` and `T` are genuinely different functions**,
not two names for one object.

### THEOREM 1.3 (monotonicity; = Phase 2C E3, re-proved here)

> `T(K-1) ≤ T(K)` for every `K ≥ 1`.

*Proof.* Let `π` be the projection `W^K ↦ W^{K-1}`. Because the prefix dynamics
is autonomous at both widths, `π` intertwines the two maps. If the `K`-orbit is
periodic from `T(K)`, its image is periodic from `T(K)` too, so the `(K-1)`-orbit
has preperiod at most `T(K)`. ∎

### THEOREM 1.4 (lag monotonicity)

> If `q | q'` then `r_{q'}(k) ≤ r_q(k)`. In particular
> `r_{P(K)}(k) ≤ r_{P(k)}(k) ≤ T(k)` for `k ≤ K`.

*Proof.* `q`-periodicity from `t` implies `q'`-periodicity from `t`. The second
chain uses `P(k) | P(K)` and Theorem 1.1 at level `k`. ∎

This is why Theorem 1.1 is stated at the lag `P(K)` and not levelwise: at a
period-doubling level the earlier coordinates may settle *earlier* when measured
at the longer lag.

### THEOREM 1.5 (one-level decomposition)

> Write `T^{[q]}(J) := max_{k ≤ J} r_q(k)` (the `J`-prefix preperiod at lag `q`).
> Then `T(K) = max( T^{[P(K)]}(K-1), R(K) )`.
> If `P(K) = P(K-1)` this reads `T(K) = max( T(K-1), R(K) )`.

*Proof.* Split the maximum in Theorem 1.1 at `k = K`; use Theorem 1.1 at level
`K-1` for the first term when `P(K) = P(K-1)`. ∎

Theorem 1.5 is what §2 turns into an exact recurrence.

### THEOREM 1.6 (`K_per` and `T` are a Galois pair, not the same object)

> `K_per(t) ≥ K  ⟺  T(K) ≤ t`, and both maps are non-decreasing.
> Consequently `K_per(T(K)) ≥ K` and `T(K_per(t)) ≤ t`, and `K_per` is the
> right inverse of `T` in the order-theoretic sense — but `K_per` is **not**
> injective and `T` is **not** surjective, so neither determines the other
> pointwise without the whole table.

*Proof.* `K_per(t) ≥ K` iff some `K' ≥ K` has `T(K') ≤ t`; by Theorem 1.3 that
happens iff `T(K) ≤ t`. Monotonicity of `K_per` is immediate from the
definition. ∎

*Verified:* `K_per` computed two independent ways (running maximum over levels;
bucket-count of levels by `T`) — identical arrays over `t ≤ 40197`.

### COROLLARY 1.7 (`A` is determined by the frontier)

> `A(t,K) = 0 ⟺ K ≤ K_per(t)`. So `A` carries no information beyond `T`; it is
> a change of variable, not a new quantity.

### THEOREM 1.8 (`τ` and `ρ` coincide exactly on the levels that create transient)

> If `T(K) > T(K-1)` then `τ(K)` is defined and `ρ(K) = τ(K) = T(K) - 1`.

*Proof.* Deferred to §2 (Theorem B), where `T(K) = τ(K)+1` is proved. Given
that, `ρ(K) = max{t < τ(K)+1 : w_t(K-1) = 1} = τ(K)`, since `w_{τ(K)}(K-1) = 1`
by definition of `τ`. ∎

*Verified:* all **16394** such levels, `K ≤ 30000`, 0 exceptions.

On the other levels `ρ` and `τ` are unrelated: `ρ(K)` looks backwards from
`T(K) = T(K-1)` while `τ(K)` looks forwards from `T(K-1)`.

---

## 3. Summary table — the brief's question answered

| pair | relation | status |
|---|---|---|
| `T(K)` vs `R(K)` | `R(K) ≤ T(K)`, with equality at 21139 of 29999 levels | **THEOREM** (inequality) + measurement (equality set) |
| `T(K)` vs `τ(K)+1` | equal **iff** the level is RESETTING (§2); 16394 of 29999 | **THEOREM** |
| `R(K)` vs `τ(K)+1` | `R(K) = τ(K)+1` iff the level is RESETTING, else `R(K) ≤ T(K-1)` | **THEOREM** (§2) |
| `ρ(K)` vs `τ(K)` | equal on every RESETTING level; unrelated elsewhere | **THEOREM** 1.8 |
| `A(t,K)` vs `K_per(t)` | `A(t,K)=0 ⟺ K ≤ K_per(t)`; `A` is a change of variable | **COROLLARY** 1.7 |
| `K_per` vs `T` | Galois pair; mutually determined as functions, neither pointwise-invertible | **THEOREM** 1.6 |

**None of the five is redundant, and none of them is new information about the
centre column.** They are five coordinates on the same transient, and §6–§8
show why re-coordinatising does not by itself reach the diagonal.

---

## 4. The three reset times, and R(K) as the fibre dependence horizon

§§1–3 above use `R(K) = r_{P(K)}(K)` (the coordinate preperiod) and Phase 2D's
`τ(K)` (the *first* collapse at or after `T(K-1)`). The brief asks for two
different objects, and they are genuinely different. Both readings are carried
below, and the relations between all of them are proved.

### 4.1 The brief's `R(K)`: the fibre dependence horizon

**Definition.** Treat the initial fibre bit `w_0(K)` as a free variable, hold
the rest of the orbit fixed, and run the level-`K` fibre recursion
`u_{t+1} = a_t XOR (c_t OR u_t)`. Then

```
    R_dep(K)  :=  max { t : w_t(K) still depends on w_0(K) } .
```

### THEOREM 4.1

> `R_dep(K) = σ(K)`, where `σ(K) := min{ t ≥ 0 : w_t(K-1) = 1 }` is the
> **first reset ever** of level `K` (`R_dep(K) = ∞` if no reset occurs).

*Proof.* Let `u_t, v_t` be the two runs from `u_0 = 0, v_0 = 1`. If `c_t = 0`
then `u_{t+1} XOR v_{t+1} = u_t XOR v_t`; if `c_t = 1` then both equal
`a_t XOR 1`, so the difference is `0` and stays `0` by the first clause. Hence
the runs differ exactly for `t ≤ σ(K)`. ∎

*Verified two independent ways:* by direct perturbation (re-running the fibre
with the bit flipped) and by computing `σ(K)`. Agreement at every sampled
level, `K ≤ 30000`.

### 4.2 The three reset times

```
    σ(K)  = min{ t ≥ 0        : w_t(K-1) = 1 }      first reset ever
    τ(K)  = min{ t ≥ T(K-1)   : w_t(K-1) = 1 }      first reset after the base settles
    ρ(K)  = max{ t < T(K)     : w_t(K-1) = 1 }      last reset before the level settles
```

`σ ≤ τ` always, by definition. The brief's phrase *"the last masking/reset time
relevant to coordinate K"* is `ρ(K)`; Phase 2D's `τ_K` is the first such time
after the base settles. Measured over the first 2000 levels:

| relation | count | verdict |
|---|---|---|
| `σ(K) < τ(K)` | 1988 | **σ and τ are distinct**, and differ almost always |
| `σ(K) = τ(K)` | 8 | only where the first reset already lies past `T(K-1)` |
| `τ(K) = ρ(K)` | 1078 | exactly the RESETTING levels (Theorem 1.8) |
| `ρ(K) < τ(K)` | 914 | the INHERITING levels — there `ρ` looks backwards from `T(K-1)` while `τ` looks forwards |
| `τ(K) > ρ(K)` with `ρ` undefined | 5 | `K = 4…8` |

**So the answer to the brief's question is: all three are distinct.** They
coincide only on the RESETTING levels, where `τ(K) = ρ(K) = T(K) - 1`
(Theorem 1.8), and even there `σ(K)` is generally much smaller.

### 4.3 `R_dep` versus `R = r_{P(K)}(K)`

These are different quantities with different meanings:

* `R_dep(K) = σ(K)` is about **forgetting the initial condition**; it is small
  (the first `1` in coordinate `K-1` appears near `t ≈ K/2`).
* `R(K) = r_{P(K)}(K)` is about **settling into the cycle**; it is large
  (`≈ 1.34 K`).

Both are legitimate readings of "the transient of coordinate `K`", and
conflating them would be an error. Forgetting the seed **does not** mean having
settled: after `σ(K)` the fibre is a function of the base orbit alone, but that
base orbit is itself still transient. The gap `R(K) - R_dep(K)` is exactly the
part of the transient that is inherited from below rather than owned by the
level, and it is what Theorem A decomposes.

---

## 5. Range and controls

* All statements labelled THEOREM are proved for **all `K`**; the computational
  lines say only that the proofs were also checked.
* All numbers quoted are for `K ≤ 30000`, `t ≤ 46000`, the range of
  `run_phase2e.py` with default settings.
* `T(K) ≤ K` holds exactly for `K ∈ {0, …, 17}` (Phase 2C E4). Those levels are
  genuine counterexamples to the unqualified statement "`T(K) > K`" and are
  reported, not suppressed; `T(K) > K` for `18 ≤ K ≤ 30000` is a **BOUNDED
  OBSERVATION**, never an all-`K` theorem.
