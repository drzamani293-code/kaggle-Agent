# Seed discriminator registry

**Phase 2I, section 5.** This is the phase's central bookkeeping. Every
candidate property `Π` is scored against four filters:

| filter | statement | how established |
|---|---|---|
| **F1** | holds for **rule 30, single-cell seed** | computed / proved |
| **F2** | **fails** for **rule 90, single-cell seed** | computed / proved |
| **F3** | **fails** for at least one **other finite rule-30 seed** | computed / proved |
| **F4** | rigorous path from `Π` + EP(T,p) to a contradiction | **proof obligation** — never computed |

**No candidate is promoted unless it passes F1–F3.** F4 may remain OPEN.
A property passing F2 but not F3 is **RULE-SPECIFIC**; one passing F3 but not
F2 is **SEED-SPECIFIC**; the Phase 2H lesson requires both.

---

## 1. The seed suite (section 5A–5F), all recorded

Seeds are given by their support. All are normalised so `min(support) = 0`, so
that both towers use identical indexing; where mirror symmetry is at issue the
seed is recentred on its support midpoint (recorded explicitly).

| label | support | symmetric? | brief item |
|---|---|---|---|
| `A_single` | `{0}` | yes | **A** (and the rule-90 / rule-150 controls **B**, **C** use the same seed) |
| `D_two_adjacent` | `{0,1}` | no | **D** |
| `E_101` | `{0,2}` | yes (about 1) | **E** |
| `E_111` | `{0,1,2}` | yes (about 1) | **E** |
| `E_1001` | `{0,3}` | no (midpoint not a cell) | **E** |
| `F_rand01_101110101` | `{0,2,3,4,6,8}` | no | **F** |
| `F_rand02_100` | `{0}` | yes | **F** |
| `F_rand03_1111001` | `{0,1,2,3,6}` | no | **F** |
| `F_rand04_11011001` | `{0,1,3,4,7}` | no | **F** |
| `F_rand05_1100011` | `{0,1,5,6}` | yes (about 3) | **F** |
| `F_rand06_100111110` | `{0,3,4,5,6,7}` | no | **F** |

The random seeds come from `random.Random(20250805)` over 9-bit windows,
rejecting the all-zero word; the generator, its seed and the resulting supports
are all stored in `results/phase2i_results.json`. **`F_rand02_100` collided
with `A_single`** — the generator produced the single cell again. It is kept
rather than resampled, because it is a useful internal control: two differently
labelled entries with the same support must score identically on every filter,
and they do.

Rules used: **30** (subject), **90** and **150** (mandatory negative controls).

---

## 2. The candidates

### Π1 — QUADRATIC FORCING
> In `x_{t+1} = L_{150} x_t + F(x_t)`, the ANF of `F` has degree 2.

| | verdict | basis |
|---|---|---|
| F1 | ✔ holds | rule 30: `F = c·r`, degree 2 (exhaustive over 8 triples) |
| F2 | ✔ **fails for rule 90** | rule 90: `F = c`, degree 1 |
| F3 | ✘ holds for every rule-30 seed | `F` does not depend on the seed |

**Verdict: RULE-SPECIFIC. Not promoted.**

### Π2 — CONSTANT LINEAR SHADOW
> `S(t) = Σ_d T(t,d) x_0(-d)` is constant in `t`.

| | verdict | basis |
|---|---|---|
| F1 | ✔ holds | `S(t) = T(t,0) = 1` for all `t` (Theorem 2.4, proved) |
| F2 | ✘ holds for rule 90 too | `S` depends only on the seed, not the rule |
| F3 | ✔ **fails** for `D_two_adjacent`, `E_1001`, `F_rand01/03/04/06` | exhaustive: constant shadow ⟺ mirror-symmetric seed (Theorem 2.6), verified over all 8191 seeds with support in `[-6,6]` |

**Verdict: SEED-SPECIFIC. Not promoted alone.**

### Π3 — MIRROR-BREAKING  ★ promoted
> **The seed is mirror-symmetric about a cell, and the orbit is not.**
> Formally: `x_0(c+d) = x_0(c-d)` for all `d` (some cell `c`), yet
> `δ_t(r) := x_t(c-r) + x_t(c+r) ≢ 0`.

| | verdict | basis |
|---|---|---|
| F1 | ✔ holds | rule 30, `{0}`: first nonzero `δ` at `(t,r) = (2,1)`; 864 nonzero cells for `t ≤ 60`. A single exhibited cell proves it. |
| F2 | ✔ **fails for rule 90** | rule 90 and rule 150 are symmetric local rules (`f(l,c,r) = f(r,c,l)`), so they commute with the mirror; a symmetric seed gives `δ ≡ 0` for all time (proved). Verified: 0 nonzero cells. |
| F3 | ✔ **fails** for `D_two_adjacent`, `E_1001`, `F_rand01/03/04/06` | those seeds are not mirror-symmetric about any cell, so the first conjunct fails |
| F4 | **OPEN**, with a proved obstruction | see below and `DUAL_TOWER_COUPLING.md` §5 |

**Verdict: PASSES F1–F3. Promoted — this is the phase's candidate.**

Both halves are proved, not observed: F1 by exhibiting one asymmetric cell, F2
by the mirror-equivariance argument. Π3 is **strictly weaker than Problem 1**
(it is decided by a finite computation) so it is not rejected under the
"equivalent to Problem 1" rule.

**The obstruction to F4, stated up front so Π3 is not oversold.** The mirror
defect is *structurally blind to the centre*: `δ_t(0) = x_t(0) + x_t(0) = 0`
identically, and the Duhamel form of `δ` reproduces this
(`DUAL_TOWER_COUPLING.md` Theorem 6.4). So Π3 cannot constrain `c_t` through
`δ` alone. F4 would need a coupling that this phase did not find.

### Π4 — MIXED COLLISION SPECTRUM  ✗ degenerate
> Along the actual orbit, the left-tower preimage count `κ_K(t)` is
> non-constant.

| | verdict | basis |
|---|---|---|
| F1 | ✔ holds | rule 30, `{0}`, `K = 9`: histogram `{1: 1, 4: 399}` for `t ≤ 400` |
| F2 | ✔ fails for rule 90 | rule 90's tower maps are bijections, so `κ ≡ 1` |
| F3 | ✔ fails for `E_111`, `F_rand03`, `F_rand04` | histogram `{4: 400}` — constant |

Passes F1–F3 **and is nevertheless rejected**, because it is degenerate:

> ### Failed candidate 2I-X-02.
> The `κ ≠ 4` event occurs **only at `t = 1`**, at every `K ∈ {7,9,11}` tested.
> From `t = 2` onward the orbit sits at the maximal preimage count `κ = 4` for
> **every seed and every level**. So Π4 distinguishes seeds only through a
> single transient step and says nothing about eventual behaviour — which is
> the only thing periodicity is about. Recorded, not promoted.

At `K = 5` the picture is different again (`κ ∈ {1,2,3,4}`), which is a
finite-`K` boundary effect; the `K = 7,9,11` behaviour is the stable one within
the measured range.

### Π5 — LEFT-TOWER NON-INJECTIVITY
> The level-`K` left-tower map `F_K` is not injective.

| | verdict | basis |
|---|---|---|
| F1 | ✔ | rule 30 left tower: image `29/64`, `101/256`, `361/1024`, `1325/4096` at `K = 5,7,9,11` |
| F2 | ✔ | rule 90 **and** rule 150 are bijective in **both** towers |
| F3 | ✘ | `F_K` is a property of the rule, independent of the seed |

**Verdict: RULE-SPECIFIC. Not promoted.** (Rule 30's *right* tower is
bijective; the left/right asymmetry is rule-30-only, which is a sharpening of
Phase 2H H3 — it is not merely that rule 30's left tower loses information, it
is that no other rule in the control set loses any.)

### Π6 — THE `n_t(-1)` IDENTITY
> `n_t(-1) = c_t(1 + c_{t+1})`.

| | verdict | basis |
|---|---|---|
| F1 | ✔ | 0 failures over 800 rows |
| F2 | ✔ | fails for rule 90 and rule 150 |
| F3 | ✘ | holds for every rule-30 seed tested (0 failures, all seeds) |

**Verdict: RULE-SPECIFIC. Not promoted.**

### Π7 — PARITY REPRESENTATION `c_t = S(t) + |N(t)| mod 2`
**Rejected before scoring:** it is an identity, and the statement "`|N(t)| mod 2`
is aperiodic" is *equivalent* to Problem 1. The brief requires rejecting any
candidate equivalent to Problem 1.

---

## 3. Scoreboard

| candidate | F1 | F2 | F3 | F4 | classification |
|---|:--:|:--:|:--:|:--:|---|
| **Π1** quadratic forcing | ✔ | ✔ | ✘ | — | rule-specific |
| **Π2** constant shadow | ✔ | ✘ | ✔ | — | seed-specific |
| **Π3** mirror-breaking | ✔ | ✔ | ✔ | **OPEN** | **PROMOTED** |
| **Π4** mixed collision spectrum | ✔ | ✔ | ✔ | — | passes but **degenerate**, rejected |
| **Π5** left-tower non-injectivity | ✔ | ✔ | ✘ | — | rule-specific |
| **Π6** `n_t(-1)` identity | ✔ | ✔ | ✘ | — | rule-specific |
| **Π7** parity representation | — | — | — | — | rejected: equivalent to Problem 1 |

**Exactly one candidate is promoted, and its F4 carries a proved obstruction.**

## 4. The pattern, and why it is the honest headline

Five of the six scored candidates fail exactly one filter, and they fail it in
a systematic way: **properties of the local rule pass F2 and fail F3;
properties of the initial condition pass F3 and fail F2.** Π3 passes both only
because it is explicitly a *conjunction of a seed condition and a rule
condition* — the seed must be symmetric (F3) and the rule must fail to commute
with the mirror (F2).

That is not an accident of the search. It is the Phase 2H obstruction seen from
the other side: **the rule and the seed are independent inputs, and any single
property depends on one or the other.** A discriminator must therefore couple
them, and Π3 is the only natural coupling this phase found. Whether a coupled
property can also reach the centre column — F4 — is precisely what remains
open, and §6 shows the obvious route is closed.

## 5. Preserved failures

| id | content | where |
|---|---|---|
| **2I-C-01** | "constant shadow ⟺ single-cell seed" — false, it characterises mirror-symmetric seeds | `NONLINEAR_DUHAMEL_FORMULA.md` §3 |
| **2I-C-02** | Stern criterion claimed for all `d` — false for `d > 0` | `MOD2_KERNEL_GEOMETRY.md` §2 |
| **2I-X-01** | difference kernel has smaller support — false, ~24 % larger | `PERIODICITY_NONLINEAR_CONSTRAINT.md` §3 |
| **2I-X-02** | Π4 mixed collision spectrum — passes F1–F3 but is a `t = 1` transient | this file, §2 |
| **2I-X-03** | "a symmetric rule-30 seed might keep a symmetric orbit" — false: 247 symmetric seeds tested, **0** stayed symmetric | `DUAL_TOWER_COUPLING.md` §4 |
| **2I-X-04** | brute-force and SAT searches disagreed at `p = 1` — a window mismatch (`W = 4` vs `W = 5`) in the driver, not a mathematical discrepancy; both methods agree once the windows match | `NONLINEAR_SYMBOLIC_RESULTS.md` §4 |
