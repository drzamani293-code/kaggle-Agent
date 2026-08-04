# Proof Dependency Graph

Complete logical dependencies behind the four load-bearing results. Every node
is either a **definition**, a **proved statement** (with the file and section
where its proof is written out), or a **finite computation** (with the script
that performs it and the artifact holding its output).

Legend:

* `[DEF]` definition — nothing to verify beyond internal consistency
* `[FACT]` / `[LEM]` / `[THM]` proved here; proof location given
* `[COMP]` finite computation; script and artifact given
* `[TEST]` mechanical check of a proved statement's computational content —
  **supports** confidence in the implementation, is **not** part of the proof
* `[EXT]` external input

Arrows point **from prerequisite to consequence**.

---

## 0. Shared foundation

```mermaid
graph TD
  D1["[DEF] f(l,c,r) = l XOR (c OR r)<br/>FORMAL_PROOF §0, Def 0.1"]
  D2["[DEF] orbit a(t,i) by recursion on t<br/>seed a(0,i) = delta_0<br/>FORMAL_PROOF §0, Def 0.2"]
  D3["[DEF] col_i(t) = a(t,i)<br/>Def 0.3"]
  D4["[DEF] EP(x): exists T,p, all t>=T, x(t+p)=x(t)<br/>Def 0.4"]
  T0["[TEST] verify_local_rule<br/>8/8 neighbourhoods vs rule number 30"]
  F1["[FACT] F1 light cone: |i|>t implies a(t,i)=0<br/>FORMAL_PROOF §1, induction on t"]
  F2["[FACT] F2 left edge: a(t,-t)=1<br/>FORMAL_PROOF §1, induction on t"]
  F3["[FACT] F3 left-permutive<br/>FORMAL_PROOF §1, exhaustion"]

  D1 --> D2 --> D3 --> D4
  D1 -.-> T0
  D2 --> F1
  F1 --> F2
  D2 --> F2
  D1 --> F3
```

`F1` and `F2` are the **only** properties of the initial condition used
anywhere in §1 below. Both hold for any finitely supported seed — this is why
Theorem W2′ cannot by itself settle Prize Problem 1 (`CLAIM_LEDGER.md` E4).

---

## 1. Theorem W2′ — "at most one eventually periodic column"

```mermaid
graph TD
  D1["[DEF] local rule f"]
  F1["[FACT] F1 light cone"]
  F2["[FACT] F2 a(t,-t)=1"]
  F3["[FACT] F3 left-permutive"]

  L1["[LEM] Lemma 1 local inversion<br/>a(t,i-1) = a(t+1,i) XOR (a(t,i) OR a(t,i+1))<br/>FORMAL_PROOF §2"]
  L2["[LEM] Lemma 2 normalisation<br/>T = max(T1,T2), p = lcm(p1,p2)<br/>FORMAL_PROOF §3"]
  L3["[LEM] Lemma 3 leftward propagation<br/>UNIFORM T and p for all col_(i-k)<br/>FORMAL_PROOF §4"]
  W2["[THM] W2: no two ADJACENT columns<br/>both eventually periodic<br/>FORMAL_PROOF §4"]
  L4["[LEM] Lemma 4 boundary forcing<br/>v(t+1) = G(v(t), u(t))<br/>FORMAL_PROOF §5"]
  PH["[FACT] pigeonhole on Sigma^m<br/>2^m + 1 iterates in a set of size 2^m<br/>FORMAL_PROOF §5, Claim 5.2"]
  L5["[LEM] Lemma 5 interior inherits EP<br/>period <= 2^(j-i-1) * p<br/>FORMAL_PROOF §5"]
  W2P["[THM] W2': at most ONE eventually<br/>periodic column in the diagram<br/>FORMAL_PROOF §6"]

  T1["[TEST] verify_lemma_left_reconstruction<br/>202,000 (t,i) pairs"]
  T2["[TEST] verify_lemma_leftward_determinism<br/>columns -1..-200 rebuilt from 0,1"]
  T3["[TEST] verify_lemma_strip_determinism<br/>interior -5..5 from boundary -6,6"]

  F3 --> L1
  D1 --> L1
  L1 --> L3
  L2 --> L3
  L3 --> W2
  F1 --> W2
  F2 --> W2
  D1 --> L4
  L4 --> L5
  L2 --> L5
  PH --> L5
  W2 --> W2P
  L5 --> W2P

  L1 -.-> T1
  L3 -.-> T2
  L4 -.-> T3
```

### Dependency table

| Node | Depends on | Proof | Mechanical check |
|---|---|---|---|
| Lemma 1 | Def 0.1, F3 | `FORMAL_PROOF §2` (one line: XOR both sides) | `verify_lemma_left_reconstruction` — PASS |
| Lemma 2 | — | `FORMAL_PROOF §3` (induction on `s ≤ k` where `p = k·p₁`) | none (pure arithmetic) |
| Lemma 3 | Lemma 1, Lemma 2 | `FORMAL_PROOF §4` (induction on `k`, paired hypothesis `P(k)`) | `verify_lemma_leftward_determinism` — PASS |
| **Theorem W2** | Lemma 3, F1, F2 | `FORMAL_PROOF §4` (choose `j = min(i, −(T+p))`; three steps) | **none possible** |
| Lemma 4 | Def 0.1 (radius 1) | `FORMAL_PROOF §5` | `verify_lemma_strip_determinism` — PASS |
| Lemma 5 | Lemma 2, Lemma 4, pigeonhole | `FORMAL_PROOF §5` (Claims 5.1–5.3) | none |
| **Theorem W2′** | Theorem W2, Lemma 5 | `FORMAL_PROOF §6` (two cases: `j = i+1`, `j ≥ i+2`) | **none possible** |

### Critical steps a reviewer should attack first

1. **Uniformity of `T` in Lemma 3.** If the preperiod grew with `k`, Step 2 of
   Theorem W2 would fail. Proof at `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md:184`;
   the point is that only indices `t` and `t+1` are consulted, both `≥ T`.
2. **The choice `j = min(i, −(T+p))`** and the inequality `T + p ≤ n = −j`,
   which is what makes the interval `[T, T+p)` lie entirely below the light
   cone's arrival time at site `j`.
3. **Claim 5.1**, `v(T+kp) = Hᵏ(v(T))`: it needs `u(T+kp+s) = u(T+s)` for all
   `0 ≤ s < p`, which is `(∗)` applied `k` times.

### What is *not* used

No compactness, no König's lemma, no axiom of choice, no reversibility of the
global map (only the local rule is inverted, in its first argument only), and
no bi-infinite time. Argued in full at
`FORMAL_PROOF_AT_MOST_ONE_COLUMN.md §7`.

---

## 2. The bound `T + p ≥ 998140`

```mermaid
graph TD
  D1["[DEF] rule 30 local rule"]
  T0["[TEST] verify_local_rule (8/8)"]
  E1["[COMP] engine C big-integer bit-parallel<br/>rule30_lab.py:196"]
  E2["[COMP] mirror engine, rule 86, reversed shifts<br/>verify_sequence_independent.py:46"]
  E3["[COMP] numpy engine B (prefix only)<br/>rule30_lab.py:148"]
  SEQ["[COMP] centre column, N = 1000001 bits<br/>sha256 0bb02e4e...<br/>phase1_results/center_column_1000001.txt"]
  CA["[COMP] method A rolling-int hash set"]
  CB["[COMP] method B numpy bit-pack + sort-unique"]
  CC["[COMP] method C string slicing set"]
  K["[COMP] K = 998140 distinct length-28 factors<br/>ceiling N-n+1 = 999974"]
  LA["[LEM] Lemma A: EP with (T,p) implies<br/>at most T+p factors of each length<br/>FACTOR_COMPLEXITY_BOUND §2"]
  CAP["[LEM] Corollary A': prefix factors are<br/>genuine factors, so T+p >= K"]
  B["[COMP+LEM] T + p >= 998140"]

  D1 -.-> T0
  D1 --> E1 --> SEQ
  E2 --> SEQ
  E3 --> SEQ
  SEQ --> CA --> K
  SEQ --> CB --> K
  SEQ --> CC --> K
  LA --> CAP --> B
  K --> B
```

### Dependency table

| Node | Value / location | Verification |
|---|---|---|
| Sequence | `center_column_1000001.txt`, sha256 `0bb02e4e…` | regenerated bit-for-bit by the **mirror engine** (225 s) and prefix-matched by **numpy**: `phase1_results/verify_seq_1M.log` |
| Word length | `n = 28` | largest count among `n ∈ {20,24,26,28}` |
| Count | `K = 998140` | **three algorithms agree** — hashing (A, C) and sorting (B): `phase1_results/verify_factor_1M.log` |
| Lemma A | proof `FACTOR_COMPLEXITY_BOUND.md §2` | proof only; no computation |
| Result | `T + p ≥ 998140` | `verify_factor_bound.py` |

### Assumptions and limits

* Depends on the sequence being correct — hence the mirror-engine
  regeneration, which is the only check that catches a shift-direction error.
* Ceiling `N − n + 1 = 999974`: the count is at 99.82% of the maximum
  possible, so this `N` is nearly exhausted. Bigger bounds need bigger `N`,
  linearly.
* **One-sided.** Gives no information about whether such `(T,p)` exists.

---

## 3. The bound `T > 979998` for periods `p ≤ 20000`

```mermaid
graph TD
  SEQ["[COMP] centre column, N = 1000001<br/>sha256 0bb02e4e..."]
  DEF["[DEF] EP with period p and preperiod T"]
  MM["[COMP] for each p <= 20000:<br/>last_mismatch(p) = max{t : c(t) != c(t+p)}<br/>period_scan_fast, rule30_lab.py:595"]
  XCHK["[COMP] cross-check vs naive O(Np) scan<br/>for p <= 64 on first 20000 bits<br/>run_phase1.py:150"]
  ARG["[LEM] if c is (T,p)-periodic then<br/>T > last_mismatch(p)"]
  MIN["[COMP] min over p <= 20000 of last_mismatch(p)<br/>= 979998, attained at p = 19999"]
  RES["[COMP+LEM] any eventual period p <= 20000<br/>forces T > 979998"]

  SEQ --> MM --> MIN --> RES
  DEF --> ARG --> RES
  MM -.-> XCHK
```

### The elementary lemma

If `c(t+p) = c(t)` for every `t ≥ T`, then no `t ≥ T` can have
`c(t) ≠ c(t+p)`. So every mismatch index is `< T`, and in particular
`T > last_mismatch(p)`. Applied to the `p` minimising `last_mismatch(p)`, this
yields a bound valid **for every** `p ≤ 20000` simultaneously.

### Dependency table

| Node | Value | Location |
|---|---|---|
| Scan implementation | big-integer `S ^ (S >> p)` | `rule30_lab.py:595` (`period_scan_fast`) |
| Cross-check | naive `O(Np)` scan, `p ≤ 64`, first 20000 bits — agrees exactly | `run_phase1.py:150`; `phase1_results.json: period_scan_crosscheck_naive_vs_fast = true` |
| Result at `N = 10⁶` | `min last_mismatch = 979998` at `p = 19999`; no `p ≤ 20000` survives the window | `anomaly_followup.json: period_scan` |
| Result at `N = 2·10⁵` | `min last_mismatch = 189998` at `p = 10000` | `phase1_results.json: period_scan` |

### Limits

Covers only `p ≤ 20000`. Says nothing about larger periods — for those, D1
(factor bound) is the only constraint, and it constrains only the **sum**
`T + p`.

---

## 4. The MB1 equivalence (Proposition 6)

```mermaid
graph TD
  L1["[LEM] Lemma 1 local inversion at i = 0<br/>a(t,-1) = a(t+1,0) XOR (a(t,0) OR a(t,1))"]
  HYP["[HYP] col_0 is (T,p)-periodic<br/>CONJECTURALLY FALSE - see vacuity note"]
  XOR["[LEM] XOR the identity at t and t+p:<br/>a(t,-1) XOR a(t+p,-1)<br/>= (a(t,0) OR a(t,1)) XOR (a(t,0) OR a(t+p,1))"]
  C1["[LEM] case a(t,0)=1: both sides 1, difference 0<br/>(this is Fact F4, non-right-permutivity)"]
  C0["[LEM] case a(t,0)=0: difference = a(t,1) XOR a(t+p,1)"]
  P6["[THM] Prop 6: col_(-1) is p-periodic from T<br/>IFF col_1 agrees at lag p on Z = {t>=T : a(t,0)=0}<br/>WIDTH2 §11"]
  MB1["[CONJ] MB1: the right-hand side holds<br/>REGISTRY C3 - open, and untestable"]
  W2["[THM] Theorem W2"]
  PP1["[CONJ] Prize Problem 1"]

  L1 --> XOR
  HYP --> XOR
  XOR --> C1
  XOR --> C0
  C1 --> P6
  C0 --> P6
  P6 --> MB1
  MB1 --> PP1
  W2 --> PP1
```

### Derivation, step by step

Assume `col_0` is `(T,p)`-periodic. For `t ≥ T`, Lemma 1 at `i = 0` gives both

```
a(t,-1)   = a(t+1,0)   XOR ( a(t,0)   OR a(t,1)   )
a(t+p,-1) = a(t+1+p,0) XOR ( a(t+p,0) OR a(t+p,1) )
```

XOR them. Since `t+1 ≥ T`, periodicity kills `a(t+1,0) XOR a(t+1+p,0) = 0`,
and `a(t+p,0) = a(t,0)`. Hence

```
a(t,-1) XOR a(t+p,-1) = ( a(t,0) OR a(t,1) ) XOR ( a(t,0) OR a(t+p,1) ).
```

* If `a(t,0) = 1`: both parentheses are `1`, so the difference is `0`.
  **This case is exactly Fact F4** (rule 30 is not right-permutive): when the
  centre cell is black, the right neighbour is invisible.
* If `a(t,0) = 0`: the difference is `a(t,1) XOR a(t+p,1)`.

So `col_{−1}` is `p`-periodic from `T` **iff** `col_1` agrees at lag `p` at
every `t ∈ Z = { t ≥ T : a(t,0) = 0 }`. ∎

### Consequence chain

```
MB1  ==>  col_(-1) eventually periodic          [Prop 6]
     ==>  col_(-1) and col_0 adjacent, both EP
     ==>  contradiction                          [Theorem W2]
     ==>  col_0 is not eventually periodic       [Prize Problem 1]
```

### Vacuity warning (structural, not incidental)

The hypothesis `HYP` is conjecturally false. If Prize Problem 1 has the
expected answer, **MB1 is vacuously true and no computation can bear on it**.
What was tested instead is the finitary statement MB1-loc(`a`), a *different*
proposition, refuted for `a = 0,4,…,28` (`CLAIM_LEDGER.md` D5). Refuting
MB1-loc does **not** refute MB1.

### Dependency table

| Node | Type | Location | Verification |
|---|---|---|---|
| Lemma 1 at `i = 0` | LEM | `FORMAL_PROOF §2` | `verify_lemma_left_reconstruction` — PASS |
| Fact F4 (case `a(t,0)=1`) | FACT | `WIDTH2 §3` | exhaustive over 8 neighbourhoods |
| Proposition 6 | THM | `WIDTH2 §11`, restated above | proof only |
| MB1 | CONJ | `CONJECTURE_REGISTRY.md` C3 | **untestable by construction** |
| MB1-loc(`a`) | REF for `a ≤ 28` | `find_mb1loc_witnesses.py` | 8 witnesses, each re-simulated twice |

---

## 5. Global summary

```mermaid
graph LR
  FOUND["Definitions + F1 + F2 + F3"]
  W2["THM W2<br/>no adjacent pair"]
  W2P["THM W2'<br/>at most one column"]
  P6["THM Prop 6<br/>MB1 equivalence"]
  GAP["OPEN: the Bridge<br/>Registry C2 / C3"]
  PP1["OPEN: Prize Problem 1<br/>Registry C1"]
  BND["COMP bounds<br/>T+p >= 998140<br/>T > 979998 for p <= 20000"]

  FOUND --> W2 --> W2P
  FOUND --> P6
  W2P --> GAP
  P6 --> GAP
  GAP -.->|"missing"| PP1
  BND -.->|"constrains only"| PP1
```

The dashed edge into Prize Problem 1 is the entire remaining problem. Nothing
in this package crosses it, and the computational bounds constrain a
hypothetical `(T, p)` without excluding one.
