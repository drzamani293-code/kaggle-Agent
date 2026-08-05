# Phase 2I — Nonlinear Seed Discriminator

Rule 30 Prize Problem 1 research corpus.

> ## Problem 1 is not solved. No partial claim on it is made here.
> **Decision gate: B** — a precise candidate passed filters 1–3; filter 4
> remains open, with a proved obstruction. See `DECISION_GATE.md`.

---

## The constraint this phase works under

Phase 2H established: any argument using only radius 1, left-permutivity,
finite support, the light cone and the frozen edge **cannot** prove
centre-column aperiodicity, because rule 90 has all of those and an eventually
periodic single-seed centre column.

So every candidate here must use a property that separates rule 30 from rule 90.
The separating property is made explicit and exact: writing
`x_{t+1} = L₁₅₀ x_t + F(x_t)`, the forcing `F` has ANF **degree 2** for rule 30,
**degree 1** for rule 90, and **degree 0** for rule 150.

## The promoted candidate

> **Π3 / NLB4.** The seed is mirror-symmetric about a cell, and the orbit is not.

* **F1** ✔ holds for rule 30, single-cell seed (proved by exhibiting one cell).
* **F2** ✔ fails for rule 90 — proved: symmetric rules commute with the mirror.
* **F3** ✔ fails for non-symmetric rule-30 seeds.
* **F4** **OPEN**, and the obvious route is *provably blocked*: the mirror
  defect satisfies `δ_t(0) = 0` identically, so it is structurally blind to the
  centre column.

## Contents

| file | brief § | what it contains |
|---|---|---|
| `NONLINEAR_ACTIVITY_THEORY.md` | 1 | `rule30 = rule150 + n`; forcing degrees; `n_t(-1) = c_t(1+c_{t+1})`; defect equations |
| `NONLINEAR_DUHAMEL_FORMULA.md` | 2 | the forced-linear representation, the mod-2 trinomial Green function, `T(t,0) = 1`, constant shadow ⟺ symmetric seed |
| `PERIODICITY_NONLINEAR_CONSTRAINT.md` | 3 | `Q_p = SEED + OLD + SLAB`; the self-similar `p = 2^a` family; the refuted support-reduction conjecture |
| `MOD2_KERNEL_GEOMETRY.md` | 4 | exact digit identities: three-slit, three disjoint copies, scaled self-similarity, support recurrences |
| `SEED_DISCRIMINATOR_REGISTRY.md` | 5 | the seed suite A–F and the four-filter scoreboard for Π1–Π7 |
| `DUAL_TOWER_COUPLING.md` | 6 | both towers, the mirror pair, `δ_{t+1} = Lδ_t + ν_t`, and the four required answers |
| `INFORMATION_LOSS_THEORY.md` | 7 | preimage counts as finite combinatorics; why they separate rules but not seeds |
| `NONLINEAR_SKELETON.md` | 8 | `Ker(t)` vs `N(t)`, kept strictly apart; kernel-size recurrence |
| `NONLINEAR_BRIDGE_REGISTRY.md` | 9 | NLB1–NLB5, each with all eight required fields |
| `NONLINEAR_SYMBOLIC_RESULTS.md` | 10 | ANFs, two independent derivations, brute-force vs SAT |
| `DECISION_GATE.md` | 11 | the classification, argued against all four options |

### Code

| file | role |
|---|---|
| `nonlinear_lab.py` | GF(2) form, `n` field, trinomial kernel `T`, Duhamel, `Q` |
| `kernel_geometry.py` | section 4 digit identities and support recurrences |
| `discriminator_lab.py` | seed suite A–F, candidate properties, filter data |
| `tower_coupling_lab.py` | both towers, mirror defect, preimage counts |
| `symbolic_lab.py` | ANF arithmetic, brute-force and SAT searches |
| `run_phase2i.py` | driver → `results/phase2i_results.json` |
| `run_phase2i_tests.py` | validation suite (groups A–D) |

## Reproducing

```bash
python3 run_phase2i.py            # ~2.5 min
python3 run_phase2i.py --quick    # ~40 s
python3 run_phase2i_tests.py      # validation
python3 run_phase2i_tests.py --fast
```

`pysat` is required for the SAT cross-check; if absent, that check reports
itself unavailable rather than failing.

## Headline results

**Proved.**

* Forcing degrees 2 / 1 / 0 for rules 30 / 90 / 150 — the exact discriminating
  property, exhaustive over all 8 local inputs.
* `x_t(j) = Σ_d T(t,d) x_0(j-d) + Σ_{s<t} Σ_d T(t-1-s,d) n_s(j-d)` — verified on
  4073 cells across three seeds with **0 mismatches**, and confirmed as an
  algebraic identity by matching ANFs at every `t ≤ 7`.
* `T(t,0) = 1` for every `t` (central trinomial coefficient always odd), so for
  the single-cell seed `c_t = 1 + |N(t)| mod 2`.
* The linear shadow is constant **iff** the seed is mirror-symmetric —
  exhaustive over all 8191 seeds with support in `[-6,6]`.
* `n_t(-1) = c_t(1 + c_{t+1})`: the nonlinear field beside the centre is a
  function of the centre column alone. Fails for rules 90 and 150.
* Mod-2 kernel geometry: `T(2^a,·)` is a three-slit; `T(2^a+r,·)` is three
  disjoint copies (254/254); `T(q·2^a+r,·)` is exactly self-similar
  (1240/1240); the difference-kernel convolution (419/419).
* `δ_{t+1} = L₁₅₀ δ_t + ν_t` with `δ_t(0) ≡ 0` — mirror breaking is caused
  entirely by the quadratic term.
* Rule 30's left tower is the **only** non-bijective map among
  `{30,90,150} × {left,right}`, and the degree-2 monomial is its sole source.

**Refuted or rejected, and preserved.**

* 2I-C-01 constant shadow ⟺ single-cell seed — false (it is symmetry).
* 2I-C-02 the Stern criterion for all `d` — false for `d > 0`.
* 2I-X-01 the difference kernel is sparser — false, ~24 % **larger**.
* 2I-X-02 Π4 mixed collision spectrum — passes F1–F3 but is a `t = 1` transient.
* 2I-X-03 a symmetric seed keeping a symmetric rule-30 orbit — none found in 247.
* 2I-X-04 a brute-force/SAT window mismatch in the driver.
* NLB1 and NLB5's operative clause — restatements of Problem 1, rejected.

## Standing rules observed

Finite computation is never presented as proving non-periodicity. Theorem,
lemma, corollary, bounded observation and failed conjecture are labelled
separately. Bounded searches are reported as "no witness found in the searched
range", never as UNSAT. **Rule 90 is a mandatory negative control and rule 150
is used as a second one**; at least one other rule-30 finite seed is used
everywhere a seed property is claimed. Rule-specific and seed-specific
properties are separated explicitly in every registry. No statistical or
randomness test was used. No theorem is inferred from powers-of-two data alone.
No novelty is claimed: no primary source has been retrievable from this
environment in any phase, re-confirmed at the start and end of this one.
