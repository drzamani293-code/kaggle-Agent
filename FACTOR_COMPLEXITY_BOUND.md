# The Factor-Complexity Lower Bound `T + p ≥ 998140`

Everything needed to check this claim independently: the combinatorial lemma
and its proof, the exact parameters, the code, the checksum of the sequence the
count was taken from, and three mutually independent recomputations.

---

## 1. The claim

> **Claim B1.** Let `c(t) = a_t(0)` be the centre column of rule 30 from the
> single-cell seed. If `c` is eventually periodic with preperiod `T ≥ 0` and
> period `p ≥ 1` — i.e. `c(t+p) = c(t)` for all `t ≥ T` — then
>
> ```
> T + p ≥ 998140.
> ```

Type: **computation + proved lemma**. The lemma is Lemma A below (proved); the
number 998140 is a measured factor count (verified three ways).

**This is a lower bound and nothing more.** It does not show that no `(T, p)`
exists. See §7.

---

## 2. The combinatorial lemma

**Definition.** For an infinite sequence `x : N → Σ` over a finite alphabet `Σ`
and `n ≥ 1`, the *factor of length `n` at position `i`* is the block

```
w_i = ( x(i), x(i+1), ..., x(i+n-1) ) ∈ Σⁿ,        i ∈ N.
```

The *factor complexity* is `P_x(n) = | { w_i : i ∈ N } |`, the number of
distinct length-`n` factors occurring anywhere in `x`.

> **Lemma A (factor counting).** Let `x : N → Σ` be `(T, p)`-periodic, i.e.
> `x(t+p) = x(t)` for every `t ≥ T`, with `T ≥ 0` and `p ≥ 1`. Then
>
> ```
> ∀ n ≥ 1 :   P_x(n) ≤ T + p.
> ```

**Proof.** Fix `n ≥ 1`. Partition the set of starting positions `N` into
`A = { i : i < T }` and `B = { i : i ≥ T }`.

*Positions in `A`.* There are exactly `T` of them, so they contribute at most
`T` distinct factors.

*Positions in `B`.* Let `i ≥ T`. Write `i - T = q·p + r` by division with
remainder, so `q ≥ 0` and `0 ≤ r < p`; then `i = T + r + q·p`. We claim

```
w_i = w_{T+r} .
```

It suffices to show `x(i + j) = x(T + r + j)` for every `j` with
`0 ≤ j ≤ n-1`. Fix such a `j` and set `s = T + r + j`. Note `s ≥ T`. We prove
by induction on `k ∈ {0, 1, ..., q}` that `x(s + k·p) = x(s)`:

* `k = 0`: trivial.
* `k → k+1`: the index `s + k·p` satisfies `s + k·p ≥ s ≥ T`, so
  `(T,p)`-periodicity applies at it and gives
  `x(s + (k+1)·p) = x( (s + k·p) + p ) = x(s + k·p)`, which equals `x(s)` by
  the inductive hypothesis.

Taking `k = q` gives `x(s + q·p) = x(s)`, i.e. `x(i + j) = x(T + r + j)` since
`i + j = T + r + j + q·p = s + q·p`. As `j` was arbitrary, `w_i = w_{T+r}`.

Hence every factor starting in `B` equals `w_{T+r}` for one of the `p` values
`r ∈ {0, ..., p-1}`, contributing at most `p` distinct factors.

Total: `P_x(n) ≤ T + p`. **∎**

**Corollary A′ (the form actually used).** Let `x` be any infinite sequence and
let `K` be the number of distinct length-`n` blocks occurring in the finite
prefix `x(0), ..., x(N-1)`. Every such block is a factor of `x`, so
`K ≤ P_x(n)`. Therefore, if `x` is `(T,p)`-periodic then

```
T + p ≥ P_x(n) ≥ K.
```

**∎**

*Remark (why the bound is one-sided).* Corollary A′ converts *observed
richness* into a lower bound on `T + p`. There is no converse: a sequence can
have `P_x(n) ≤ T + p` for every `n` without being eventually periodic at all
(e.g. any Sturmian sequence has `P_x(n) = n+1`, and is aperiodic). So no
factor count can ever establish aperiodicity.

*Remark (ceiling).* A prefix of length `N` contains exactly `N - n + 1` blocks
of length `n`, so `K ≤ N - n + 1`. The method can therefore never produce a
bound exceeding `N - n + 1 ≈ N`. Doubling the computation at best doubles the
bound.

---

## 3. Exact parameters

| Quantity | Value |
|---|---|
| Sequence | centre column `c(t) = a_t(0)`, `t = 0 .. 1000000` |
| Length `N` | **1000001** bits |
| Ones / zeros | 500768 / 499233 |
| **Word length `n` used** | **28** |
| **Distinct length-28 factors `K`** | **998140** |
| Blocks available (`N - n + 1`) | 999974 |
| Fraction of the ceiling attained | 998140 / 999974 = 0.998166 |
| **Resulting bound** | **`T + p ≥ 998140`** |

Counts at the other lengths measured on the same file:

| `n` | distinct factors `K` | ceiling `N-n+1` |
|---|---|---|
| 20 | 644259 | 999982 |
| 24 | 970865 | 999978 |
| 26 | 992634 | 999976 |
| **28** | **998140** | 999974 |

`n = 28` gives the largest count and is therefore the witness length. (Larger
`n` cannot help much: `K` is already within 0.19% of the ceiling.)

---

## 4. The sequence the count was taken from

```
file    : phase1_results/center_column_1000001.txt
format  : one line of 1000001 ASCII characters '0'/'1', then a newline
bits    : c(0) c(1) ... c(1000000),  c(0) = 1
sha256  : 0bb02e4ed6c3d80bd832eed0b6991cc99f6de01a5f32fa795281d3efb2e9104e
first 32: 11011100110001011001001110101110
last 32 : 11111100110010000011011011111110
```

The 200001-bit file used elsewhere in the audit:

```
file    : phase1_results/center_column_200001.txt
sha256  : cf018b09db932412d5f42c3d03076181a34d806173fbee749e30c871fe5d994a
```

Both checksums are recorded in `phase1_results/SHA256SUMS.txt` and re-checked
by `run_all_tests.py` (step 2).

Regenerate with:

```
python3 gen_center_column.py --steps 1000000     # ~334 s
```

### 4.1 The sequence itself, independently regenerated

A count is only as good as the sequence it is taken from. The file was produced
by engine C (big-integer bit-parallel, `rule30_lab.center_column_bitwise`,
`rule30_lab.py:196`), whose update is

```
new = (row << 1) ^ (row | (row >> 1))
```

A reversed-shift bug here would yield a self-consistent but wrong sequence.
`verify_sequence_independent.py` regenerates it two other ways:

1. **Mirror engine (rule 86).** Rule 86 is the left–right mirror of rule 30
   (checked exhaustively in `verify_local_rule`, test `mirror_is_86`), and its
   bit-parallel update has the shifts reversed:
   `new = (row >> 1) ^ (row | (row << 1))`. Since the single-cell seed is
   mirror-symmetric, the rule-86 centre column must be *bit-for-bit identical*
   to the rule-30 one. A shift-direction error cannot survive this test.
2. **numpy engine (engine B).** Table-lookup on a `uint8` array, sharing no
   code with either big-integer engine; run over a prefix (it is `O(N²)` in
   cells, so only a prefix is affordable).

Results are in `phase1_results/verify_seq_200001.log` and
`verify_seq_1M.log`.

---

## 5. The code that computed the count

`verify_factor_bound.py` implements three independent counting algorithms:

| Method | Function | Line | Principle | Data structure |
|---|---|---|---|---|
| A | `count_A_rolling_int` | `verify_factor_bound.py:80` | rolling `n`-bit integer window | Python `set` (hashing) |
| B | `count_B_numpy_unique` | `verify_factor_bound.py:92` | bit-pack all windows, then `np.unique` | numpy array (**sorting**) |
| C | `count_C_string_set` | `verify_factor_bound.py:105` | string slicing | `set` of `str` (hashing, different key type) |

A and B differ in *algorithmic principle* — hashing versus sorting — so their
agreement is a genuine cross-check of the number, not a rerun of the same code
path. C additionally varies the key representation.

The original Phase-1 count came from a fourth code path,
`run_anomaly_followup.factor_count` (`run_anomaly_followup.py:38`), and the
per-length table in `PHASE1_AUDIT.md` §5.3 came from a fifth,
`rule30_lab.subword_complexity` (`rule30_lab.py:491`). All five agree.

### 5.1 Recomputation output (verbatim)

```
sequence file      : phase1_results/center_column_1000001.txt
sha256(file)       : 0bb02e4ed6c3d80bd832eed0b6991cc99f6de01a5f32fa795281d3efb2e9104e
length N           : 1000001 bits (t = 0..1000000)
ones / zeros       : 500768 / 499233

   n  A rolling-int B numpy-unique   C string-set      agree  ceiling N-n+1
  20         644259         644259         644259        YES         999982
  24         970865         970865         970865        YES         999978
  26         992634         992634         992634        YES         999976
  28         998140         998140         998140        YES         999974

Lemma A  =>  T + p >= 998140   (witness length n = 28)
ceiling of this method at N = 1000001 is 999974

RESULT: all methods agree
```

Reproduce:

```
python3 verify_factor_bound.py --file phase1_results/center_column_1000001.txt \
        --lengths 20 24 26 28
```

The same script on the 200001-bit file gives `T + p ≥ 199900` at `n = 28`
(ceiling 199974), again with all three methods agreeing.

---

## 6. Chain of dependency

```
Definition of rule 30  (rule30_lab.py:83, verified verify_local_rule)
        |
        v
centre column c(0..10^6)  (engine C; cross-checked by mirror engine + numpy)
        |                  sha256 0bb02e4e...
        v
count of distinct length-28 factors = 998140   (methods A, B, C agree)
        |
        +---- Lemma A (proved in §2, no computation)
        v
   T + p >= 998140
```

No step uses any unproved assumption. The only inputs are the definition of
rule 30, the definition of eventual periodicity, and a finite computation.

---

## 7. What this does **not** show

1. **It does not show the centre column is aperiodic.** It bounds `T + p` from
   below *conditionally on* eventual periodicity. Both the hypothesis and its
   negation remain open.
2. **It cannot be pushed past `N`.** The ceiling `N - n + 1` means a bound of
   `10^9` needs a sequence of `10^9` bits. The current generator is `O(N²/64)`
   (334 s at `N = 10^6`, so ≈ 9 hours at `N = 10^7`); a light-cone-restricted
   generator would be needed to go further.
3. **Corollary A′ requires only that the observed blocks be genuine factors** —
   which holds for any prefix of the true sequence. So the bound is robust to
   *where* in the sequence the blocks come from, but it is **not** robust to
   the sequence being wrong. That risk is addressed in §4.1, not by the count.
4. **The comparison with `2ⁿ` is not part of the argument.** `PHASE1_AUDIT.md`
   §5.3 notes that all `2^14` words of length 14 occur; that is a descriptive
   observation, and the bound here uses only the raw count at `n = 28`.
