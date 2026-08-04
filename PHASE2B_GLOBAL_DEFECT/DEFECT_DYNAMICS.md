# The Exact Temporal Defect Equations for Rule 30

Every identity below is **exhaustively verified over all Boolean assignments**
— no sampling — by `defect_lab.verify_defect_identities()` and
`verify_zero_wall_reduction()`, and separately re-checked cell-by-cell against
the real single-cell orbit by `verify_defect_dynamics_on_orbit()`.

---

## 0. Setting and notation

Rule 30 on `Z`, single-cell seed, Phase 1 conventions unchanged:

```
    x_{t+1}(j) = x_t(j-1) XOR ( x_t(j) OR x_t(j+1) ),
    x_0(0) = 1,  x_0(j) = 0 for j != 0.
```

Fix a lag `p >= 1` and write `y_t(j) = x_{t+p}(j)`. The **temporal defect
field** is

```
    d_t(j) = x_{t+p}(j) XOR x_t(j) = y_t(j) XOR x_t(j).
```

`d` is a field on the same space-time lattice: `d_t(j) = 0` means rows `t` and
`t+p` agree at site `j`.

**Hypothesis under study (H-WALL).** If the centre column is eventually
`p`-periodic after time `T`, then

```
    d_t(0) = 0    for every t >= T                       (the "zero wall").
```

Phase 2B asks whether the real defect field can be compatible with an infinite
zero wall. **Nothing in this package settles that.**

Throughout, abbreviate the four quantities the update depends on:

```
    u = x_t(j),   v = x_t(j+1),   e = d_t(j),   g = d_t(j+1),
    c = d_t(j-1).
```

---

## 1. Derivation

Apply the rule to both rows and XOR:

```
    d_{t+1}(j) = y_{t+1}(j) XOR x_{t+1}(j)
               = [ y_t(j-1) XOR ( y_t(j) OR y_t(j+1) ) ]
                 XOR
                 [ x_t(j-1) XOR ( x_t(j) OR x_t(j+1) ) ].
```

XOR is associative and commutative, so the two left-neighbour terms combine
into `y_t(j-1) XOR x_t(j-1) = d_t(j-1) = c`, leaving

```
    d_{t+1}(j) = c  XOR  [ ( y_t(j) OR y_t(j+1) ) XOR ( x_t(j) OR x_t(j+1) ) ].
```

Substituting `y_t(j) = u XOR e` and `y_t(j+1) = v XOR g` defines the
**OR-difference**

```
    Delta(u, v, e, g)  =  ( (u XOR e) OR (v XOR g) )  XOR  ( u OR v ).
```

### 1.1 XOR/OR form  — **THEOREM D1**

```
    d_{t+1}(j) = d_t(j-1)
                 XOR [ ( (x_t(j)   XOR d_t(j)  )
                     OR  ( x_t(j+1) XOR d_t(j+1) ) )
                     XOR ( x_t(j) OR x_t(j+1) ) ].
```

*Verification.* `verify_defect_identities()` evaluates both sides for all
`2^6 = 64` assignments of `(x_t(j-1), x_t(j), x_t(j+1), y_t(j-1), y_t(j),
y_t(j+1))`, computing the left side from the *definition* (apply rule 30 to
each row, then XOR). **64/64 exact.**

### 1.2 Algebraic normal form over GF(2)  — **THEOREM D2**

Using `a OR b = a + b + ab` over GF(2):

```
    Delta = (u+e) + (v+g) + (u+e)(v+g)  +  u + v + uv
          = e + g + [ uv + ug + ev + eg + uv ]
          = e + g + ug + ev + eg.
```

Hence

```
    d_{t+1}(j) = d_t(j-1) + d_t(j) + d_t(j+1)
                 + x_t(j)·d_t(j+1) + x_t(j+1)·d_t(j) + d_t(j)·d_t(j+1)   (mod 2).
```

*Verification.* Same 64 assignments, plus all 16 assignments of `(u,v,e,g)` for
`Delta` alone. **Exact.**

Three structural readings of D2, each immediate from the ANF:

* **Linear part** `c + e + g` — the defect field would obey rule 90 (the XOR
  rule) if the quadratic terms vanished.
* **Quadratic part** `ug + ve + eg` — all three terms contain a defect
  variable, so `d ≡ 0` is invariant: **no defect can be created from nothing**.
* **Left-permutivity is inherited.** `c = d_t(j-1)` appears linearly and alone,
  so `d_t(j-1)` is recoverable from `d_{t+1}(j)` and the data at `(j, j+1)`:
  ```
      d_t(j-1) = d_{t+1}(j) + d_t(j) + d_t(j+1)
                 + x_t(j)·d_t(j+1) + x_t(j+1)·d_t(j) + d_t(j)·d_t(j+1).
  ```
  This is the defect-field analogue of Phase 1's Lemma 1.

### 1.3 Case table  — **THEOREM D3**

`Delta` conditioned on `(x_t(j), x_t(j+1), d_t(j), d_t(j+1))`. All 16 rows:

| `x_t(j)` | `x_t(j+1)` | `d_t(j)` | `d_t(j+1)` | `Delta` | comment |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | **0** | no defect in, none out |
| 0 | 0 | 0 | 1 | **1** | |
| 0 | 0 | 1 | 0 | **1** | |
| 0 | 0 | 1 | 1 | **1** | |
| 0 | 1 | 0 | 0 | **0** | |
| 0 | 1 | 0 | 1 | **1** | |
| 0 | 1 | 1 | 0 | **0** | defect absorbed |
| 0 | 1 | 1 | 1 | **0** | defect absorbed |
| 1 | 0 | 0 | 0 | **0** | |
| 1 | 0 | 0 | 1 | **0** | **right defect invisible** |
| 1 | 0 | 1 | 0 | **1** | |
| 1 | 0 | 1 | 1 | **0** | |
| 1 | 1 | 0 | 0 | **0** | |
| 1 | 1 | 0 | 1 | **0** | **right defect invisible** |
| 1 | 1 | 1 | 0 | **0** | |
| 1 | 1 | 1 | 1 | **1** | |

and then `d_{t+1}(j) = d_t(j-1) XOR Delta`.

**The rows that matter.** Whenever `x_t(j) = 1` and `d_t(j) = 0`, `Delta = 0`
*regardless of `d_t(j+1)`* (rows 9–10 and 13–14 above). A black cell with no
defect on it **masks** the defect immediately to its right. This is the defect
form of Fact F4 (rule 30 is not right-permutive) and is the single mechanism
behind every zero-wall lemma in `ZERO_WALL_LEMMAS.md`.

### 1.4 Two collapse identities  — **THEOREM D4**

```
    Delta(u, v, 0, g) = g · (1 XOR u)          "no defect here"
    Delta(u, v, e, 0) = e · (1 XOR v)          "no defect to the right"
```

*Verification.* Exhaustive over the 8 remaining assignments each. **Exact.**

---

## 2. Immediate consequences

**Corollary D5 (defect light cone).** `d_{t+1}(j)` depends only on
`d_t(j-1), d_t(j), d_t(j+1)` (and on `x_t(j), x_t(j+1)`). Defects therefore
propagate at speed at most 1 in each direction, and if `d_t ≡ 0` on an
interval `[a, b]` with the neighbours also zero, then `d_{t+1} ≡ 0` on
`[a+1, b-1]`.

**Corollary D6 (quiet transport).** If `d_t(j) = d_t(j+1) = 0` then
`d_{t+1}(j) = d_t(j-1)`: in the absence of local defect the field is
**transported one site to the right per time step**. Note the direction: the
defect field drifts right, while rule 30's information flow (Phase 1 Lemma 2)
runs left. The two are consistent — one is a statement about differences, the
other about reconstruction.

**Corollary D7 (support-edge defects).** For every `t >= 0` and every `p >= 1`:

```
    d_t( +(t+p) ) = 1 ,    d_t( -(t+p) ) = 1 ,
    d_t(j) = 0  for |j| > t + p.
```

*Proof.* By the Phase 1 light-cone fact, `x_t(j) = 0` for `|j| > t`, and by the
frozen-edge fact `x_{t+p}(±(t+p)) = 1`. Since `t + p > t`, `x_t(±(t+p)) = 0`,
so the XOR is 1. Beyond `t+p` both rows vanish. ∎

**This is the reason the defect field is never empty.** For every lag and every
time there are defects at exactly `±(t+p)`, and the defect support is exactly
the interval `[-(t+p), t+p]`. Verified for every lag in the Phase 2B sweep
(`support_edge_theorem_holds_for_all_lags`).

Note what D7 does *not* say: the edge defects travel **outward** at speed 1,
away from column 0. They do not by themselves threaten a wall at the centre.
The relevance of D7 to a global argument is registered as **G2** in
`GLOBAL_LEMMA_REGISTRY.md`, where its current status is *insufficient*.

---

## 3. Verification summary

| Identity | Method | Assignments | Result |
|---|---|---|---|
| D1 XOR/OR form | exhaustive | 64 | exact |
| D2 ANF over GF(2) | exhaustive | 64 | exact |
| D3 `Delta` case table | exhaustive | 16 | exact |
| D4 collapse identities | exhaustive | 8 + 8 | exact |
| D1/D2 on the real orbit | cell-by-cell, lags 1,3,7,16,27 | 24,300 cells per lag | 0 mismatches |
| D7 support edges | every lag in the sweep | all sampled rows | holds |

Reproduce: `python3 -c "import defect_lab as D, json;
print(json.dumps(D.verify_defect_identities(), indent=1))"`.
