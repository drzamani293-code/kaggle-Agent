# Decision gate

**Phase 2I, section 11.** One classification, chosen from the four the brief
offers, with no optimistic inflation.

---

## Classification: **B**

> **B. A precise candidate passed filters 1–3 but filter 4 remains open.**

The candidate is **Π3 / NLB4**:

> The seed is mirror-symmetric about a cell, and the orbit is not.
> Formally: `x_0(c+d) = x_0(c-d)` for all `d`, yet
> `δ_t(r) := x_t(c-r) + x_t(c+r) ≢ 0`.

| filter | verdict | how established |
|---|---|---|
| **F1** holds for rule 30, single seed | ✔ | first nonzero `δ` at `(t,r) = (2,1)`; 864 nonzero cells for `t ≤ 60`. One exhibited cell proves it. |
| **F2** fails for rule 90, single seed | ✔ | **proved**: rules 90 and 150 are symmetric local rules, commute with the mirror, so a symmetric seed gives `δ ≡ 0` forever. Verified: 0 nonzero cells. |
| **F3** fails for another finite rule-30 seed | ✔ | `D_two_adjacent`, `E_1001`, `F_rand01/03/04/06` are not mirror-symmetric about any cell |
| **F4** `Π3` + EP ⟹ contradiction | **OPEN** | see the obstruction below |

Π3 is **strictly weaker than Problem 1** (it is decided by a finite
computation), so it is not rejected under the brief's "no candidate equivalent
to Problem 1" rule. Both of its filter passes are theorems, not observations.

---

## Why not A

No bridge lemma was proved. Filter 4 is untouched for Π3, and the natural route
is **provably blocked**: by Theorem 6.4 the mirror defect satisfies
`δ_t(0) = 0` identically — the invariant is structurally blind to the centre
column. Coupling it to `c` through the exact centre relations yields
`δ_t(1) = c_{t+1}` on `{c_t = 0}`, which is **unconditional** (it uses the rule,
not periodicity), and on `{c_t = 1}` leaves precisely the one-bit-per-row
residue `x_t(1)` that Phase 2H had already isolated. **No new leverage was
obtained anywhere in this phase.**

## Why not C

C would say every discriminator failed F2 or F3. Two did not: Π3, and Π4 (the
mixed collision spectrum). C is therefore factually wrong. But C is *nearly*
right, and the near-miss is the phase's real finding — see below.

## Why not D — and where D does apply

D would say the nonlinear decomposition produced only a restatement of centre
periodicity. **D is correct about sections 2 and 3 specifically**, and this is
stated in those documents rather than hidden:

* `c_t = S(t) + |N(t)| mod 2` is an identity; "`|N(t)| mod 2` is aperiodic" is
  *equivalent* to Problem 1, and is rejected as a bridge candidate (Π7).
* `Q_p(t) = 0` is *equivalent* to `c_{t+p} = c_t`. Theorem 3.2 rewrites it and
  Theorem 3.4 rewrites it again; neither weakens it. **NLB1 is rejected on
  exactly this ground.**
* NLB5's operative clause is likewise a restatement, and is rejected.

D is not the classification only because the phase also produced Π3, which is
not a restatement, and a body of proved structure (sections 1, 4, 6, 7) that is
independent of the periodicity question.

---

## The finding worth stating plainly

Five of the six scored candidates fail exactly one filter, and they fail
systematically:

> **properties of the local rule pass F2 and fail F3;
> properties of the initial condition pass F3 and fail F2.**

| candidate | nature | F2 | F3 |
|---|---|:--:|:--:|
| Π1 quadratic forcing | rule | ✔ | ✘ |
| Π5 left-tower non-injectivity | rule | ✔ | ✘ |
| Π6 `n_t(-1) = c_t(1+c_{t+1})` | rule | ✔ | ✘ |
| Π2 constant linear shadow | seed | ✘ | ✔ |
| Π3 mirror-breaking | **rule × seed** | ✔ | ✔ |
| Π4 mixed collision spectrum | rule × seed | ✔ | ✔ (degenerately) |

This is not an artefact of which properties were tried. The rule and the seed
are independent inputs to the orbit, so any property that mentions only one of
them is blind to the other. **A discriminator must couple them**, and Π3 is the
only non-degenerate coupling this phase found: the seed must be symmetric (F3)
and the rule must fail to commute with the mirror (F2).

Π4 shows how easily such a coupling can be vacuous. It passes F1–F3 and is
rejected anyway, because its entire discriminating power sits at `t = 1`: from
`t = 2` onward the orbit sits in the maximal collision class `κ = 4` for every
seed and every level tested. A coupled property can be seed-sensitive only
transiently, and then it says nothing about eventual periodicity.

---

## What Phase 2I added to the corpus

**Proved, and independent of the periodicity question:**

* the forcing-degree separation (2 / 1 / 0 for rules 30 / 90 / 150) — the exact
  form of the property Phase 2H's lesson demands;
* `n_t(-1) = c_t(1 + c_{t+1})` — the nonlinear field adjacent to the centre is a
  function of the centre column alone, by a GF(2) cancellation;
* the Duhamel representation with its mod-2 trinomial Green function, verified
  cell-by-cell on three seeds and confirmed as an **algebraic identity** by
  matching ANFs;
* `T(t,0) = 1` for every `t` — the central trinomial coefficient is always odd;
* constant linear shadow ⟺ **mirror-symmetric seed** (exhaustive to `W = 6`);
* the mod-2 kernel geometry: three-slit at `2^a`, three disjoint copies at
  `2^a + r`, exact scaled self-similarity at `q·2^a + r`, the difference-kernel
  convolution, and the support recurrences `A(2m) = A(m)`,
  `A(2m+1) = 3A(m) − 2C(m)`, `B(2t) = 4B(t) − 2ΣC`;
* `δ_{t+1} = L₁₅₀ δ_t + ν_t` — mirror-symmetry breaking is caused **entirely**
  by the quadratic term;
* rule 30's left tower is the **only** non-bijective tower map in the control
  set `{30, 90, 150} × {left, right}`, and the degree-2 monomial is its sole
  source.

**Negative results, recorded so the next phase does not repeat them:**

* the difference kernel is ~24 % **larger** in support than the plain kernel —
  the "smallest support" goal of section 3 is refuted (2I-X-01);
* information-loss accounting separates rules but **not** seeds: the realised
  collision class is maximal and constant from `t = 2` for every seed (2I-X-02);
* the `2^a` structure lives in the **kernel**, which is shared with rules 90 and
  150 and is seed-independent — so NLB3's every proved ingredient fails both
  filters;
* NLB1 and NLB5's operative clause are restatements of Problem 1 and are
  rejected as bridges;
* NLB2's proved part transfers to rule 90, hence cannot settle Problem 1.

**Corrections preserved:** 2I-C-01 (constant shadow characterises symmetric,
not single-cell, seeds), 2I-C-02 (the Stern criterion holds only for `d ≤ 0`),
2I-X-03 (no symmetric rule-30 seed kept a symmetric orbit — a bounded search,
not a theorem), 2I-X-04 (a brute-force/SAT window mismatch in the driver).

---

## Standing statement

**Rule 30 Prize Problem 1 is not solved, and Phase 2I makes no partial claim on
it.** No finite computation here is evidence of non-periodicity. Every bounded
search is reported as "no witness found in the searched range", never as UNSAT.
No statistical or randomness test was used. No theorem is inferred from
powers-of-two data. No novelty is claimed for any result: no primary source has
been readable from this environment in any phase (`RETRIEVAL_NOTE.md` in
Phase 2H records the verbatim 403s, re-confirmed at the start of this phase).
