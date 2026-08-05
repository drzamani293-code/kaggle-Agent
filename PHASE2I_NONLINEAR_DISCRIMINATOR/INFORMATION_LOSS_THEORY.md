# Information-loss accounting for the left tower

**Phase 2I, section 7.** Everything here is a **finite combinatorial count**.
No probabilistic notion of entropy is used, and the phrase "entropy implies
chaos" — and every informal variant of it — appears nowhere as an argument.

---

## 1. The exact objects

Fix a level `K`. The left-tower map is
`F_K : {0,1}^{K+1} → {0,1}^{K+1}`, `F_K(s)(k) = s(k-2) XOR ( s(k-1) OR s(k) )`
(indices `< 0` read `0`); the right-tower map is
`G_K(s)(k) = s(k) XOR ( s(k-1) OR s(k-2) )`.

> ### Definition 7.1. For a state `u`, the **collision class size** is
> `π_K(u) := |F_K^{-1}(u)|`, an integer in `[0, 2^{K+1}]`. The **image size**
> is `|F_K({0,1}^{K+1})|`, and `Σ_u π_K(u) = 2^{K+1}` exactly (checked by the
> test suite).

> ### Definition 7.2. Along a given orbit, the **realised class sequence** is
> `κ_K(t) := π_K( w_{t} |_{[0,K]} )` for `t ≥ 1` — the collision class the
> **actual** orbit lands in at time `t`.
>
> `π_K` is a property of the **rule**. `κ_K` is a property of the **seed**.
> Section 5's filters F2 and F3 apply to them respectively.

## 2. Measured loss — the rule side

| `K` | states `2^{K+1}` | image size | max `π_K` | histogram of `π_K` over the image |
|---:|---:|---:|---:|---|
| 5 | 64 | 29 | 4 | `{1: 11, 2: 5, 3: 9, 4: 4}` |
| 7 | 256 | 101 | 4 | `{1: 31, 2: 13, 3: 29, 4: 28}` |
| 9 | 1024 | 361 | 4 | `{1: 87, 2: 37, 3: 85, 4: 152}` |
| 11 | 4096 | 1325 | 4 | `{1: 247, 2: 109, 3: 245, 4: 724}` |

The maximum collision class size is **4 for every `K ≥ 4`**. Below that it is
smaller — `2, 3, 3` at `K = 1, 2, 3` — a small-`K` boundary effect, recorded so
that the constant `4` is not read as holding at every level.

The comparison rules are rule 90 and rule 150: for both of them the left map is
a bijection, so every class has size 1 and the table above is entirely a rule 30
phenomenon.

> ### Theorem 7.3 (the control set separates completely).
> Among rules 30, 90, 150 and both towers, **exactly one map is not a
> bijection**: rule 30's left tower.
>
> | rule | `F_K` (left) | `G_K` (right) |
> |---|---|---|
> | **30** | **NOT injective** at every `K ≥ 1` | bijective |
> | **90** | bijective | bijective |
> | **150** | bijective | bijective |

*Proof.* For rule 30's right tower and for both towers of rules 90 and 150, an
explicit inverse exists by induction on `k`: in each case `s(k)` appears in the
image coordinate `k` as an XOR summand whose other arguments have index `< k`,
so `s(k)` is recoverable once `s(0..k-1)` are. For rule 30's left tower this
fails: the image coordinate `k` contains `s(k-2)` as the XOR summand but
`s(k-1), s(k)` inside the OR, i.e. at **higher** indices, which the prefix does
not determine. An explicit collision at `K = 3` is
`F_3(1,1,0,0) = F_3(1,0,1,0) = (1,1,0,1)`. ∎

Verified exhaustively for `K ≤ 11` (rule 30) and `K ≤ 9` (controls).

**Filter verdict.** Π5 = "`F_K` is not injective" passes **F2** (rules 90 and
150 both fail it) and fails **F3** (no seed enters the definition). It is
RULE-SPECIFIC. This sharpens Phase 2H H3, which compared only rule 30's two
towers and could not tell whether the asymmetry was about the rule or about the
direction.

## 3. Measured loss — the seed side, and why it collapses

`κ_K(t)` for rule 30, `t ≤ 400`:

| `K` | `A_single` | `D_two_adjacent` | `E_111` | `F_rand03` | `F_rand04` |
|---|---|---|---|---|---|
| 7 | `{1:1, 4:399}` | `{1:1, 3:1, 4:398}` | `{4:400}` | `{3:2, 4:398}` | `{3:1, 4:399}` |
| 9 | `{1:1, 4:399}` | `{1:1, 4:399}` | `{4:400}` | `{4:400}` | `{4:400}` |
| 11 | `{1:1, 4:399}` | `{1:1, 4:399}` | `{4:400}` | `{4:400}` | `{4:400}` |

and the realised sequence for the single-cell seed at `K = 9` is

```
    kappa_9(t)  =  1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, ...        (t = 1, 2, 3, …)
```

> ### Observation 7.4 (bounded, `t ≤ 400`, `K ∈ {7,9,11}`).
> The realised class is the **maximal** one, `κ_K(t) = 4`, at every `t ≥ 2`,
> for **every seed in the suite**. The only departure is at `t = 1`.

> ### Lemma 7.5 (the `t = 1` exception is real and is a seed effect).
> For the single-cell seed, `w_0 = (1,0,0,…,0)` and `κ_K(1) = 1`: the state
> `w_1 = F_K(w_0)` has a unique preimage. For `E_111` the corresponding state
> already has 4 preimages, so `κ_K(1) = 4`.

*Proof.* Both are finite computations over `2^{K+1}` states, carried out
exhaustively. ∎

> ### Failed candidate 2I-X-02 (repeated here because this is where it dies).
> Π4 = "`κ_K` is non-constant along the orbit" passes F1, F2 and F3 — and is
> **rejected as degenerate**. Its entire discriminating power sits in the single
> step `t = 1`. From `t = 2` on, the realised class is constant and maximal for
> every seed and every rule-30 level tested, so Π4 imposes **no constraint on
> eventual behaviour**, which is the only thing eventual periodicity is about.

**This is the section's main negative result.** The left tower does lose
information — provably, and uniquely in the control set — but along the real
orbit it loses the *same amount at every step after the first*, independently of
the seed. Information-loss accounting therefore separates rules but not seeds.

## 4. The theorem that was sought, and was not found

The brief asks for a theorem of the form

> *eventual periodicity of the shared diagonal forces the actual orbit to
> revisit incompatible collision classes.*

**It was not proved, and Observation 7.4 explains why the obvious route is
blocked.** If the realised class is the constant `4` from `t = 2` onward, then
the class sequence is *already* eventually periodic (constant), for every seed,
whether or not the centre column is periodic. A constant sequence cannot be
incompatible with anything. Any theorem of the stated form must therefore use
a **finer** invariant than the class size — the identity of the class, or the
position of the true preimage within it — and this phase did not find one that
survives filter F3.

Two finer invariants were considered and both fail immediately:

| refinement | why it fails |
|---|---|
| the identity of the visited class (the state itself) | it *is* the tower state; "the state sequence is eventually periodic" at level `K` is already known (Phase 2E), for every seed and every rule — no discrimination |
| which of the `≤ 4` preimages is the true one | the true preimage is `w_{t-1}`, so this is the state sequence again, re-indexed |

## 5. Relation to the nonlinear field

The loss is caused by the same quadratic term as everything else in this phase:

> ### Proposition 7.6. `F_K` fails to be injective exactly because the OR in
> `w_t(k-1) OR w_t(k)` reads indices above `k-2`. Over GF(2) that OR is
> `w(k-1) + w(k) + w(k-1)w(k)`; deleting the quadratic term gives rule 150's
> `F_K`, which is bijective. **The degree-2 monomial is the sole source of the
> information loss.**

*Proof.* Rule 150's left map is `s(k-2) + s(k-1) + s(k)`, triangular with the
identity on the diagonal in the `s(k-2)` variable read upward, hence bijective
(Theorem 7.3). Rule 30's differs from it by exactly `s(k-1)s(k)`. ∎

This is a satisfying rule-level statement and it still fails F3.

## 6. Statement inventory

| id | statement | status | filter role |
|---|---|---|---|
| **I-7.3** | rule 30's left tower is the unique non-bijection in the control set | **THEOREM**, proved; exhaustive `K ≤ 11` | Π5: F2 ✔, F3 ✘ |
| **I-7.5** | the `t = 1` exception, per seed | **THEOREM** (finite computation) | Π4's only content |
| **BO-7.4** | realised class is maximal and constant for `t ≥ 2` | **BOUNDED OBSERVATION**, `t ≤ 400`, `K ≤ 11` | kills Π4 |
| **I-7.6** | the quadratic monomial is the sole source of loss | **THEOREM**, proved | rule-specific |
| **target** | "EP forces incompatible collision classes" | **NOT PROVED**; obvious route blocked by BO-7.4 | — |
