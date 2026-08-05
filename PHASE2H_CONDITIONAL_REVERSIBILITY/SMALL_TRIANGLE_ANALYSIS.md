# Exhaustive small-height triangle enumeration, `h ≤ 12`

**Phase 2H, section 9.** The brief: *"Use this only to discover and refute
conjectures."* That is what it was used for. Three conjectures were tested;
one was refuted, one was confirmed and then **proved independently**, one was
found to be vacuous and retracted.

**No claim below is extended beyond the enumerated range.** `h ≤ 12` is twelve
data points, and the standing rule against inferring all-`h` statements from
bounded enumeration is in force throughout.

---

## 1. What was enumerated

`triangle_enumerator.py` evolves **every** top row of `2h+1` cells for `h`
steps — all `2^{2h+1}` of them, so the enumeration is exhaustive, not a sample.
At `h = 12` that is 33 554 432 triangles.

```
   row 0 : columns 0 .. 2h                (free, 2h+1 bits)
   row s : columns s .. 2h-s
   apex  : (h, h)

   centre word    C  = ( x_s(h) )_{s=0..h}                    h+1 bits
   width-2 word   W2 = ( x_s(h), x_s(h+1) )_{s=0..h-1}        2h  bits
   right edge     Rt = ( x_s(2h-s) )_{s=0..h}                 h+1 bits
```

These triangles are **locally valid**, not orbit segments: they are constrained
only by the rule, not by any seed. Every window of the real orbit is one of
them, which is what makes §4's inequality usable in one direction only.

## 2. Results — rule 30

| `h` | triangles | distinct `C` / possible | triangles per `C` | `N(h)` = max `W2` per `C` | mean `W2` per `C` | max `Rt` per `C` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 4 / 4 | 2 | 2 | 2.000 | 2 |
| 2 | 32 | 8 / 8 | 4 | 3 | 3.000 | 3 |
| 3 | 128 | 16 / 16 | 8 | 4 | 4.000 | 5 |
| 4 | 512 | 32 / 32 | 16 | 5 | 5.000 | 9 |
| 5 | 2 048 | 64 / 64 | 32 | 7 | 6.250 | 16 |
| 6 | 8 192 | 128 / 128 | 64 | 9 | 7.750 | 27 |
| 7 | 32 768 | 256 / 256 | 128 | 12 | 9.438 | 48 |
| 8 | 131 072 | 512 / 512 | 256 | 15 | 11.391 | 79 |
| 9 | 524 288 | 1 024 / 1 024 | 512 | 19 | 13.602 | 141 |
| 10 | 2 097 152 | 2 048 / 2 048 | 1 024 | 24 | 16.090 | 247 |
| 11 | 8 388 608 | 4 096 / 4 096 | 2 048 | 30 | 18.855 | 417 |
| 12 | 33 554 432 | 8 192 / 8 192 | 4 096 | 36 | 21.935 | 759 |

Two exact regularities hold at every `h ≤ 12`:

* **every** one of the `2^{h+1}` centre words occurs; and
* **exactly `2^h`** triangles carry each centre word (max = min).

The second is provable and is not merely observed:

> ### Lemma 9.0 (iterated left-permutivity).
> For every `t ≥ 0` and `j`, there is a function `G` such that
> ```
>      x_t(j)  =  x_0(j-t)  XOR  G( x_0(j-t+1), …, x_0(j+t) ) .
> ```
> In particular, flipping `x_0(j-t)` alone flips `x_t(j)`.

*Proof.* Induction on `t`. `t = 0` is trivial. For `t ≥ 1`,
`x_t(j) = x_{t-1}(j-1) XOR ( x_{t-1}(j) OR x_{t-1}(j+1) )`. By induction
`x_{t-1}(j-1) = x_0(j-t) XOR G'(x_0(j-t+1 … j+t-2))`. The cells `x_{t-1}(j)`
and `x_{t-1}(j+1)` depend on `x_0` over `[j-t+1, j+t-1]` and `[j-t+2, j+t]`
respectively, neither of which contains `j-t`. Collect everything except
`x_0(j-t)` into `G`. ∎

> ### Proposition 9.1 (the centre map is balanced).
> For every `h ≥ 0`, the map `M_h :` (top row, `2h+1` bits) `↦` (centre word,
> `h+1` bits) is surjective and exactly `2^h`-to-one.

*Proof.* Induction on `h`. For `h = 0` the top row is one bit and the centre
word is that bit: `M_0` is a bijection, `2^0 = 1`.

For `h ≥ 1`, split the centre word into its first `h` letters
`(x_s(h))_{s=0..h-1}` and its apex `x_h(h)`.

*The first `h` letters.* Cell `(s, h)` depends only on top-row columns
`h-s … h+s`, so for `s ≤ h-1` it depends only on columns `1 … 2h-1`. Those
`2h-1` bits are the top row of a height-`(h-1)` triangle whose centre is again
column `h`, and the first `h` letters are exactly its centre word. By the
inductive hypothesis that map is surjective and `2^{h-1}`-to-one.

*The remaining two bits of freedom.* Top-row column `2h` is not read by any of
the first `h` letters, so it is free: a factor of 2. And by Lemma 9.0 with
`t = j = h`, `x_h(h) = x_0(0) XOR G(x_0(1), …, x_0(2h))`, so once columns
`1 … 2h` are fixed, `x_0(0)` is determined by the apex value — and both apex
values are attained.

Hence `M_h` is surjective, and each fibre has size
`2^{h-1} (inductive) × 2 (column 2h) = 2^h`, with `x_0(0)` pinned. ∎

The enumeration agrees: max fibre = min fibre = `2^h` and all `2^{h+1}` centre
words occur, at every `h ≤ 12`.

## 3. Conjectures tested

### 3.1 **REFUTED** — "the centre word determines the neighbouring column"

`centre_determines_width2` is **false at every `h` from 1 to 12**. The minimal
counterexample is at `h = 1`: top rows `010` and `011` both give centre word
`11` and differ at column 2. Recorded as **X-2H-02**.

The quantitative form is `N(h)`, tabulated above:

```
      N(h)  =  2, 3, 4, 5, 7, 9, 12, 15, 19, 24, 30, 36        (h = 1 … 12)
   first differences  1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6
```

**No formula is fitted and no growth rate is claimed.** Twelve terms admit many
continuations, and the standing rule forbids choosing one. What is used
downstream (`TRIANGLE_TO_TRACE_TRANSFER.md` Theorem 8.7) is only the individual
measured values, inside an inequality that holds separately for each `h`.

### 3.2 **CONFIRMED, then PROVED** — "two adjacent columns determine a triangle of exactly `h(h-1)/2` cells"

`boundary_lab.candidate_B_region` confirms, for `2 ≤ h ≤ 10`, that the `W2`
word determines exactly the envelope `{ (s, h-k) : 1 ≤ k ≤ h-1, 0 ≤ s ≤ h-1-k }`
— **0 violations**, and every probe cell one row below the envelope takes both
values on some fibre. Counts match `h(h-1)/2` at every `h`.

The enumeration only *suggested* this. It is **proved** in
`MINIMAL_BOUNDARY_DATA.md` Theorem 2.B, by induction on the backward sweep, for
all `h`. The bounded enumeration is therefore not carrying the claim.

### 3.3 **VACUOUS, RETRACTED** — "the right edge determines the apex"

Reported `True` for `h ≤ 10`, then withdrawn: the apex `(h,h)` **is** the last
letter of the right-edge word, so the test asked whether a word determines its
own letter. Recorded as **C-2H-01**; the retracted result is preserved in
`MINIMAL_BOUNDARY_DATA.md` §4.1 rather than deleted.

Re-run against the honest target (the centre word with the shared apex
removed), the right edge determines **nothing**: false at every `h ≤ 10`, with
fibres of size `2^h`. Two rightmost *diagonals*, by contrast, determine the
centre word with fibres of size exactly **2** — and that is proved for all `h`
by the diagonal recursion (Theorem 2.C′), not inferred from the enumeration.

## 4. Rule 90 control

| `h` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| rule 90 `N(h)` | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| rule 30 `N(h)` | 2 | 3 | 4 | 5 | 7 | 9 | 12 | 15 | 19 |

For rule 90, `N(h) = 2^h` exactly over the whole range: the centre word
constrains the neighbouring column **not at all**, which is what one expects
from a rule that is permutive on both sides. Rule 30's `N(h)` is much smaller
over the same range.

**The control's job here is to stop an over-reading.** A small `N(h)` for rule
30 is *not* evidence that rule 30's centre column is aperiodic — rule 90 has the
largest possible `N(h)` and its centre column **is** eventually periodic
(`WEAK_BRIDGE_THEOREMS.md` H5). The two quantities are unrelated.

## 5. Cross-check between two independent implementations

Standing rule 3 requires two implementations. `triangle_enumerator.py` (numpy,
grouping by a packed `(C, W2)` key) and `boundary_lab.py` (argsort +
`reduceat` over fibres, different word layout) were written separately. Their
counts agree up to the known offset: `triangle_enumerator`'s `(C, W2)` pairs
include the apex bit, `boundary_lab`'s `W2` word does not, so the former should
be exactly twice the latter.

| `h` | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| `(C,W2)` pairs | 24 | 64 | 160 | 400 | 992 | 2416 | 5832 | 13928 | 32952 |
| `W2` words | 12 | 32 | 80 | 200 | 496 | 1208 | 2916 | 6964 | 16476 |
| ratio | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

Exactly 2 at every `h`, as Proposition 9.1 predicts (the apex is free given
`W2`). The implementations agree.

## 6. What the enumeration did **not** establish

1. **No all-`h` claim.** Every table above stops at `h ≤ 12` (or `h ≤ 10`), and
   nothing is extrapolated. Where an all-`h` statement is needed downstream it
   is proved separately.
2. **No growth rate for `N(h)`.** The observed values are sub-exponential over
   the range. That is a description of twelve numbers, not an asymptotic.
3. **No conclusion about the orbit.** These are locally valid triangles. The
   single-cell orbit uses a measure-zero subset of them, and the enumeration
   says nothing about which.
4. **No progress on Problem 1.** The one inequality that reaches the orbit
   (Theorem 8.7) yields a bound weaker than `CT-01` by three orders of
   magnitude (X-2H-06).
