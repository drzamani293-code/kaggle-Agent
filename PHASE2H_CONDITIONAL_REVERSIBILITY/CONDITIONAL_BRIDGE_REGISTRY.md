# Conditional bridge registry

**Phase 2H, section 10.** The brief: *"Prioritize one candidate as the next
main target."* Section 4 does that, and the recommendation is shaped by the
phase's most useful result, which is a negative one.

---

## 1. The Bridge, unchanged

> **`BR-01`.** From *"the centre column of the rule-30 single-cell orbit is
> eventually periodic"*, derive *"some second column is eventually periodic"*.
> With **T-03**, that would settle Rule 30 Prize Problem 1.

**`BR-01` is open.** It was open after Phase 1 and it is open after Phase 2H.
No phase has produced a second periodic column, and none has produced a
contradiction from the periodicity hypothesis.

## 2. Everything Phase 2H added, and what it is worth

| id | statement | status | closes `BR-01`? |
|---|---|---|---|
| **R-1.1 … R-1.4** | conditional backward determinism; the cone `[b-i, m+1]`; count `(h+1)(m-b+2) + h(h+1)/2` | **PROVED**, verified on 12 296 cells | no — one-step algebra |
| **T-2.A** | one column determines nothing | **PROVED** + exhaustive `h ≤ 12` | no |
| **T-2.B** | two columns determine `h(h-1)/2` cells; envelope sharp | **PROVED**; sharpness exhaustive `h ≤ 10` | no |
| **T-2.C′** | the diagonal recursion `E_{k+2}(s) = E_k(s+1) XOR (E_{k+1}(s) OR E_k(s))` | **PROVED**, 0 failures at `T ≤ 300` | no — no compression |
| **T-3.5 / H1** | EP ⟹ the reduced boundary word `β` is not eventually `p`-periodic | **PROVED** | no — H5 shows why |
| **T-6.3 / H3** | the right-aligned tower map `G_K` is a **bijection**; preperiod 0 at every level; `F_K` is not injective | **PROVED**, exhaustive `K ≤ 12`, inverse checked on 65 534 states | no |
| **T-8.3** | `L(h, n) = n + h + 1` | **PROVED** | no — `L(h) → ∞` |
| **T-8.7/8.8 / H4** | `p_{01}(h) ≤ p_0(h+1)·N(h)`; `T+p ≥ p_{01}(h)/N(h)` | **PROVED**; best value 995, vs `CT-01`'s 998 140 | no — strictly weaker |
| **T-6.5 / H5** | rule 90: centre column EP, column 1 aperiodic, T-01/02/03 all apply | **PROVED** | **no, and it proves H1/H2 cannot** |
| **BO-4.3–4.5** | `W(t)` records at powers of two; the white triangle below `2^n` | **BOUNDED OBSERVATION**, `t ≤ 1200` | no |
| **C-5.4** | the restart events impose no constraint on the centre | **NEGATIVE RESULT**, with inequalities I–IV | no |

## 3. What Phase 2H actually changed

Three things, none of them progress toward a proof, all of them progress toward
knowing where not to look.

1. **The missing data shrank by about half.** The Bridge needs "a second
   column". H1 shows it needs only the bits of column 1 at the rows where the
   centre column is white — measured 9880 of 20 000. The width is still 2
   (T-2.A forbids 1); the bit count is roughly halved.

2. **Phase 2G's "we have no handle on the right" is now closed, and the right
   is no better.** The right-aligned tower is *strictly better behaved* than the
   left one — its map is a bijection, preperiods are exactly 0, periods are
   powers of two — and the centre column is its moving diagonal just as it is
   the left tower's. The Phase 2E obstruction was never about choosing the
   wrong side.

3. **The T-02/T-03 machinery is now provably incapable of settling Problem 1.**
   Rule 90 satisfies every hypothesis those theorems use — left-permutivity,
   light cone, frozen edge, radius 1, finite seed — and rule 90's centre column
   **is** eventually periodic while its column 1 is provably not. Any argument
   built only from that machinery would prove a false statement about rule 90.
   This converts Phase 1's informal seed-blindness worry into a theorem with an
   explicit witness.

Point 3 is the most useful output of the phase and it retires a family of
approaches, including H1 and H2 themselves.

## 4. Priority for the next phase — one candidate

> ### Recommended next main target: **`BR-02` — find a property of the
> single-cell seed that rule 90 does not share, and that survives the
> reduced-boundary reduction.**

**Why this and not something else.** H5 is a filter. It says: *any* candidate
bridge whose proof uses only left-permutivity, the light cone, the frozen edge,
radius 1, and finite support is dead on arrival, because rule 90 is a
counterexample. Every bridge candidate this corpus has produced across Phases
1–2H — H1, H2, T-02, T-03, the wall identities, the strip automaton — is of
exactly that kind. The next phase should therefore not propose another
candidate of the same type; it should first identify a **discriminating
property**.

**Concrete form of `BR-02`.** Find a predicate `Π` such that:

1. `Π` holds for rule 30 with the single-cell seed;
2. `Π` fails for rule 90 with the single-cell seed;
3. `Π` fails for rule 30 with at least one *other* finite seed (otherwise the
   seed-blindness obstruction of Phase 1 §10.3 recurs);
4. `Π` together with EP(T,p) is contradictory.

Conditions (2) and (3) are the two filters the corpus has proved are necessary.
Condition (4) is the work. **No candidate `Π` is proposed here**, because
proposing one without checking (2) and (3) is precisely the error the phase
learned to avoid.

**Candidate raw material for `Π`**, listed without endorsement:

* rule 30's **failure of right-permutivity** (Lemma 1.0′). It is the one
  structural property in this phase that rule 90 does not share, and it is what
  makes the `OR` blind (Lemma 3.2) and the left tower non-injective (T-6.3b).
  Whether it can be made to bear weight is unknown.
* the **asymmetry between the two towers** — `G_K` bijective, `F_K` not. Rule 90
  makes both towers bijective (its rule is XOR in both outer arguments), so this
  asymmetry passes filter (2). Whether it passes (3) — whether it is a property
  of the *seed* rather than the *rule* — is **not** known, and on the face of it
  it is a property of the rule alone, which would fail (3).

That second bullet is the honest state of the recommendation: the most
promising discriminator found in this phase probably fails filter (3), and
checking that is the first task of the next phase.

## 5. Demoted and closed

| id | was | now |
|---|---|---|
| **restart-event route** (§4/§5) | a candidate bridge mechanism | **CLOSED**, C-5.4: arrival at `t ≥ 2^{n+1} - W_n`, indistinguishable from the bulk |
| **width-2 factor counting** (H4) | a hoped-for improvement on `CT-01` | **CLOSED**, X-2H-06: weaker by a factor `N(h)` |
| **"the right side is the missing handle"** (Phase 2G §3) | the corpus's stated main gap | **CLOSED**, H3: the right side has the same obstruction |
| **H1, H2** | weak bridge targets | **PROVED, but shown incapable** by H5 |

## 6. Failed conjectures and corrections preserved this phase

| id | content | where |
|---|---|---|
| **X-2H-01** | white-strip seed at `m = 40` for `t_bot = 60`: 413/882 mismatches | `MINIMAL_BOUNDARY_DATA.md` §6 |
| **X-2H-02** | "the centre word pins the neighbour" — refuted at `h = 1` | §3.1, `SMALL_TRIANGLE_ANALYSIS.md` |
| **X-2H-03** | "two diagonals compress the boundary" — refuted by the light cone | `MINIMAL_BOUNDARY_DATA.md` §6 |
| **X-2H-04** | narrowing `Z` by `x_{s+1}(0)` — refuted | `PERIODIC_CENTER_TRIANGLES.md` §6 |
| **X-2H-05** | iterating OR-blindness — true but saves nothing | `PERIODIC_CENTER_TRIANGLES.md` §6 |
| **X-2H-06** | width-2 counting beats centre counting — refuted, 995 vs 998 140 | `TRIANGLE_TO_TRACE_TRANSFER.md` §4 |
| **C-2H-01** | vacuous right-edge/apex test — retracted, result preserved | `MINIMAL_BOUNDARY_DATA.md` §4.1 |

## 7. Standing statement

**Rule 30 Prize Problem 1 is not solved, and Phase 2H makes no partial claim on
it.** No finite computation in this phase constitutes evidence of
non-periodicity. Every bounded search reported here is described as "no witness
found in the searched range", never as UNSAT. No novelty is claimed for any
result, because no primary source was readable (`RETRIEVAL_NOTE.md`).
