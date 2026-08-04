# Doubling Point Analysis

The four doubling points `K = 3, 8, 29, 400`, their exact cycle data, and the
structural condition that produces them.

---

## 1. Exact data

At a doubling point the base cycle word of coordinate `K-1` (the `c`-word) must
be **all zeros** (Theorem N1) and the parity of the coordinate `K-2` cycle word
(the `a`-word) must be **odd** (Phase 2C Lemma C).

| `K` | `T_base` | `P_base` | `a`-word (coordinate `K-2`) | `c`-word (coordinate `K-1`) | parity | `P` after |
|---|---|---|---|---|---|---|
| 3 | 2 | 1 | `1` | `0` | 1 | 2 |
| 8 | 2 | 2 | `01` | `00` | 1 | 4 |
| 29 | 8 | 4 | `1011` | `0000` | 1 | 8 |
| 400 | 29 | 8 | `01100001` | `00000000` | 1 | 16 |

Every `c`-word is all zeros, as N1 requires; every parity is odd, which is what
makes them DOUBLING rather than NEUTRAL.

The corresponding **eventually-zero coordinates** are `K-1 = 2, 7, 28, 399`.
Only `7` is permanently zero; `2, 28, 399` are nonzero somewhere in their
transient.

## 2. The structural condition

Putting N1, N2 and Lemma C together, a doubling point at `K` is exactly:

```
    (i)  coordinate K-1 is zero throughout its cycle;
    (ii) the cycle word of coordinate K-2 has odd weight.
```

and by **N2** condition (i) forces coordinate `K-2` and coordinate `K-3` to
agree on that cycle, so (ii) may equivalently be read at level `K-3`. Both
readings were checked at all four points and agree (parity 1 in both).

Condition (i) is the scarce one: it happened four times in 30000 levels.
Condition (ii) then happened to hold all four times.

## 3. No formula is fitted

The doubling points are

```
    3, 8, 29, 400
```

with gaps `5, 21, 371` and ratios `2.67, 3.63, 13.79`. **Per the brief, no
formula is fitted to four data points, and none is proposed.** Observations
that can be stated without fitting:

* the points are strictly increasing and the gaps grow;
* `P` doubles at each, so `P(K) = 16` for all `400 <= K <= 30000` (Phase 2C);
* the fifth doubling point, if it exists, is beyond `K = 30000`.

Whether the sequence is infinite — equivalently whether `P(K) -> infinity` — is
**open**. Both "finitely many doubling points" (so `P` eventually constant) and
"infinitely many" are consistent with everything measured. This is registry
item **C4**.

## 4. Recursive relation: searched, not found

We looked for a relation expressing each doubling point in terms of its
predecessors, using:

* the cycle words above (they grow in length `1, 2, 4, 8` — one doubling each
  time, so each `a`-word is *twice* as long as the previous, and no obvious
  substitution relates consecutive `a`-words `1`, `01`, `1011`, `01100001`);
* the transient values `T_base = 2, 2, 8, 29` — note `T_base` at one doubling
  point equals the *previous doubling point* for the last two entries
  (`8 = K_2`, `29 = K_3`), but not the first (`2 != K_1 = 3`). With one
  exception in three, this is **not** treated as a pattern;
* gap ratios — no structure from three gaps.

**Nothing was found, and nothing is asserted.** Recorded as a negative result.

## 5. Why doubling points matter here

Each doubling point ends a COLLAPSING chain (`COLLAPSE_CHAIN_THEOREMS.md` §1)
and doubles the memory of the cycle-word transducer from `2P` to `4P` bits. If
doubling points were finite in number, the transducer memory would be
eventually constant and Theorem CH3 (pigeonhole) would eventually apply — with
an astronomically large threshold `2^{2P}`, but applying. If they are infinite,
`P` grows without bound and CH3 never bites.

Neither branch is decided, and neither branch is currently known to help with
the diagonal.
