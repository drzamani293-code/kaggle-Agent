# The Zero-Wall Automaton

A finite-state representation of the local data around column 0, built to test
whether a bi-infinite zero wall is even *locally* possible — and, if so, how
constrained it is.

---

## 1. Construction

**State.** For radius `r`, a state is the pair of windows

```
    S = ( X, Y ),   X = x_t(-r..r),   Y = x_{t+p}(-r..r),
```

each `2r+1` bits, so `2(2r+1)` bits in total. The defect window is `X XOR Y`;
the **wall condition** is that its centre bit vanishes, `X[0] = Y[0]`, i.e.
`d_t(0) = 0`.

**Transition.** Computing `x_{t+1}(-r..r)` needs `x_t(-r-1..r+1)`: two bits
beyond the window, one on each side. The same for `Y`. Those four bits are
**not** determined by the state, so the transition relation is
**nondeterministic with exactly 16 successors per state**, one for each choice
of the incoming bits. This is the standard light-cone-honest construction: the
automaton **over-approximates** the true dynamics.

**Why over-approximation is the right direction.** Any real orbit carrying an
infinite wall projects to an infinite path through wall states. So:

* **empty invariant set ⟹ no infinite wall exists** (a genuine theorem);
* **non-empty invariant set ⟹ nothing** about the real orbit.

We are looking for emptiness. Everything below reports that we did not find it.

**Maximal invariant set.** From the wall-restricted graph, iteratively delete
every state with no surviving successor or no surviving predecessor, to a fixed
point. What remains is exactly the set of states lying on some **bi-infinite**
wall path — the object whose emptiness would settle the question.

---

## 2. Exhaustive results, `r = 1..5`

| `r` | window bits | wall states `2^(4r+1)` | maximal invariant | fraction | empty? | time |
|---|---|---|---|---|---|---|
| 1 | 3 | 32 | 16 | 0.5000 | no | 0.0 s |
| 2 | 5 | 512 | 128 | 0.2500 | no | 0.0 s |
| 3 | 7 | 8 192 | 992 | 0.1211 | no | 0.6 s |
| 4 | 9 | 131 072 | 7 616 | 0.0581 | no | 11.8 s |
| 5 | 11 | 2 097 152 | 59 136 | 0.0282 | no | 245.3 s |

**RESULT: the maximal invariant set is non-empty at every radius computed.**
Therefore **no bounded impossibility is obtained.** A bi-infinite zero wall is
locally admissible at radius ≤ 5, and the automaton route yields no theorem.

**COMPUTATIONAL OBSERVATION (growth).** The invariant sets have sizes
`16, 128, 992, 7616, 59136`, with successive ratios
`8.00, 7.75, 7.68, 7.77` — consistent with `|Inv(r)| ≈ C · 7.7^r`, against a
total of `2 · 16^r`. The surviving fraction therefore decays like `0.48^r`: the
wall constraint removes a constant proportion of the state space per unit of
radius, but never all of it. Extrapolating, `r = 8` would leave `≈ 2.7 × 10^7`
of `3.4 × 10^{10}` states — still non-empty, and the extrapolation is
**CONJECTURE**, not measurement.

### 2.1 Radii 6, 7, 8 were NOT computed

The brief asks for `r = 1..8`. Exhaustive enumeration needs `2^(4r+1)` states
and 16 transitions each:

| `r` | wall states | edges | feasible here? |
|---|---|---|---|
| 6 | 33 554 432 | 5.4 × 10⁸ | no (memory) |
| 7 | 536 870 912 | 8.6 × 10⁹ | no |
| 8 | 8 589 934 592 | 1.4 × 10¹¹ | no |

`r = 5` already took 245 s and ~1 GB in this Python implementation. **`r ≥ 6`
is recorded as NOT COMPUTED**, not as "no result found". Doing it would need a
bitset/BDD or symbolic (SAT/BMC) formulation rather than explicit enumeration —
noted as future work, not attempted.

---

## 3. Locally admissible versus observed in the real orbit

Wall states occurring in the actual single-cell orbit (lags 1, 2, 3, 7, 16, 27;
`t ≤ 20000`), compared with the two locally defined sets:

| `r` | admissible wall states | maximal invariant | observed in real orbit | observed ∩ invariant | invariant, not observed |
|---|---|---|---|---|---|
| 1 | 32 | 16 | 32 | 16 | 0 |
| 2 | 512 | 128 | 512 | 128 | 0 |
| 3 | 8 192 | 992 | 8 025 | 977 | 15 |
| 4 | 131 072 | 7 616 | 31 080 | 1 763 | 5 853 |

**How to read this table — the labelling matters.**

* *"observed"* means the state occurred at some `t` with `d_t(0) = 0` in the
  sampled orbit. Such a `t` lies on a **finite** wall (the real orbit has no
  infinite one, since the longest observed wall is 18 —
  `DEFECT_GEOMETRY_RESULTS.md` §4).
* Therefore *observed but not invariant* (29 317 states at `r = 4`) is exactly
  what one should expect: those states can begin or end a finite wall but
  cannot sit on a bi-infinite one. No tension.
* *"invariant, not observed"* (5 853 states at `r = 4`) is the interesting
  direction, and it must be labelled carefully:

> **These states are NOT shown to be globally unreachable.** They were not seen
> in the sampled range (6 lags, `t ≤ 20000`). Proving that a locally admissible
> state never occurs in the single-cell orbit is a global reachability
> question — the same difficulty as the main problem. The correct label is
> **NOT OBSERVED IN THE SAMPLED RANGE**, and that is how they are recorded in
> `results/phase2b_results.json`.

**COMPUTATIONAL OBSERVATION.** The observed set shrinks relative to the
admissible set as `r` grows (100 %, 100 %, 98 %, 24 %), while the invariant set
shrinks faster. At `r = 4` only 1 763 of 7 616 invariant states (23 %) were
observed. Whether that gap is real structure or sampling is undetermined —
larger `t` and more lags would move the number, and no extrapolation is offered.

---

## 4. Strongly connected structure

The maximal invariant set is by construction a union of states each having a
bi-infinite path; it therefore contains at least one cycle at every radius
computed. Two things are worth recording:

* The **all-zero state** `X = Y = 0` is a fixed point (all-zero windows with
  all-zero incoming bits map to themselves, and `d(0) = 0`). It is a trivial
  member of the invariant set at every radius. It is **not** a state of the real
  orbit near column 0 at large `t`, where the pattern is not all-zero.
* Removing the all-zero state and its basin does not empty the invariant set at
  any radius computed: the counts above are dominated by non-trivial states.

**No SCC decomposition beyond this was computed.** A full SCC census at `r = 5`
(2 × 10⁶ states) is feasible but was not run; it is recorded as not done rather
than estimated.

---

## 5. Verdict

| Question from the brief | Answer |
|---|---|
| Enumerate reachable wall states in the actual orbit | done, `r ≤ 4`, table §3 |
| Enumerate all locally admissible wall states without the seed | done, `r ≤ 5`, table §2 |
| Identify states locally admissible but globally unreachable | **not achievable**; reported as *not observed in the sampled range* (§3) |
| Search for SCCs that could support `d_t(0) = 0` forever | found at every radius `r ≤ 5`; §2 |
| If a closed SCC exists, do not infer it occurs in the single-cell orbit | **no such inference is made** |
| If none exists, record bounded impossibility | **not applicable — the invariant set is non-empty at every radius computed** |

**Bottom line: the automaton route gives no impossibility result, and the
growth pattern suggests it will not give one at any radius.** The reason is
structural and worth stating: a fixed-radius window around column 0 cannot see
the seed, and Phase 1 §10.3 already proved that seed-blind arguments cannot
settle Problem 1. The automaton is a seed-blind argument by construction. Its
value here is a quantitative one — it measures *how much* of the local state
space a wall forbids (a factor `≈ 0.48` per unit radius) — and a negative one:
it closes off a route.
