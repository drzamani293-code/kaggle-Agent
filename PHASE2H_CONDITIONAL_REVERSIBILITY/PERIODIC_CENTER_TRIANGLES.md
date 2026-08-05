# The periodic-centre hypothesis, and the smallest missing boundary word

**Phase 2H, section 3.** The brief: *"The main objective is to isolate the
smallest missing boundary word `B(t,h,p)`."* This section does that, and the
answer is smaller than "a second column".

**Nothing here proves or disproves anything about rule 30's centre column.**
Every statement is of the form *if the centre column is eventually periodic,
then …*. That is a conditional, and it stays a conditional.

---

## 1. The hypothesis, stated exactly

> ### Hypothesis EP(T, p).
> There exist `T ≥ 0` and `p ≥ 1` such that
> ```
>       x_{s+p}(0) = x_s(0)      for every  s ≥ T .
> ```

Problem 1 asks whether EP(T,p) holds for some `(T,p)` for the single-cell seed.
`CT-01` proves that **if** it holds, then `T + p ≥ 998140`. Nothing else about
`(T,p)` is known.

## 2. What is missing, in the crudest form

By Theorem 2.B, the pair of columns `0` and `1` determines every column to the
left. So:

> ### Proposition 3.1 (the crude gap).
> Assume EP(T,p). If **in addition** column 1 is eventually `p`-periodic, then
> every column `j ≤ 1` is eventually `p`-periodic, contradicting **T-02** (no
> two adjacent eventually periodic columns, for any finite seed). Hence
> **column 1 is not eventually `p`-periodic.**

*Proof.* If columns 0 and 1 are both eventually `p`-periodic from some row `T'`,
apply (BW) at `j = 0`: `x_s(-1) = x_{s+1}(0) XOR ( x_s(0) OR x_s(1) )`. Every
term on the right is `p`-periodic for `s ≥ T'`, so `x_{s+p}(-1) = x_s(-1)` for
`s ≥ T'`. Induction leftwards gives column `-k` eventually `p`-periodic for
every `k ≥ 0`. Columns `0` and `-1` are adjacent and both eventually periodic,
contradicting T-02. ∎

So the "missing boundary word" is at most **one bit per row**: the values
`x_s(1)`. That is `h+1` bits on a window of height `h+1`. **Section 3's
contribution is that it is strictly fewer.**

## 3. OR-blindness: the bits that do not matter

> ### Lemma 3.2 (OR-blindness).
> For every `s`,
> ```
>       x_s(0) = 1   ⟹   x_s(-1) = x_{s+1}(0) XOR 1 ,
> ```
> independently of `x_s(1)`.

*Proof.* If `x_s(0) = 1` then `x_s(0) OR x_s(1) = 1` whatever `x_s(1)` is;
substitute into (BW) at `j = 0`. ∎

Trivial, and it is the whole point: **column `-1` depends on column 1 only at
the rows where column 0 is white.**

*Verification.* `boundary_lab.reduced_boundary_lemma(20000)`: over 20 000 rows
of the real orbit, the identity (BW) at `j=0` held at every row (**0
failures**), and flipping `x_s(1)` at every row with `x_s(0)=1` changed the
reconstructed `x_s(-1)` at **0** of those rows.

## 4. The reduced boundary word `B(t, h, p)`

> ### Definition 3.3.
> For a window `S = [t, t+h]` put
> ```
>       Z(S)  :=  { s ∈ S : x_s(0) = 0 } ,
>       B(t,h) :=  ( x_s(1) )_{s ∈ Z(S)}          — the reduced boundary word.
> ```
> Its length is `|Z(S)|`, not `h+1`.

> ### Theorem 3.4 (reduced sufficiency).
> The centre column on `[t, t+h]` together with `B(t,h)` determines
> `x_s(-1)` for every `s ∈ [t, t+h-1]`. No other bit of column 1 is used.

*Proof.* Fix `s ∈ [t, t+h-1]`. If `x_s(0) = 1`, Lemma 3.2 gives `x_s(-1)` from
`x_{s+1}(0)` alone, and `s+1 ∈ S`. If `x_s(0) = 0` then `s ∈ Z(S)` and
`x_s(1) = B(t,h)_s` is supplied; (BW) at `j=0` gives `x_s(-1)`. ∎

> ### Theorem 3.5 (the periodic transfer — the exact conditional).
> Assume EP(T, p). Then `Z := { s ≥ T : x_s(0) = 0 }` satisfies `Z + p = Z`.
> Define `β : Z → {0,1}`, `β(s) := x_s(1)`. If
> ```
>       β(s + p) = β(s)      for all  s ∈ Z  with  s ≥ T′,           (†)
> ```
> for some `T′ ≥ T`, then column `-1` is eventually `p`-periodic, and hence by
> Prop. 3.1 **T-02 is contradicted**.
>
> Equivalently, and this is the usable form:
>
> > **If the centre column is eventually periodic with period `p`, then the
> > reduced boundary word `β` is not eventually `p`-periodic along `Z`.**

*Proof.* `Z + p = Z` because `x_{s+p}(0) = x_s(0)` for `s ≥ T`. Take
`s ≥ max(T, T′)`. **Case `x_s(0) = 1`:** then `x_{s+p}(0) = 1` too, so by
Lemma 3.2 both `x_s(-1)` and `x_{s+p}(-1)` equal `1 XOR` their respective
`x_{·+1}(0)`, which agree by EP(T,p). **Case `x_s(0) = 0`:** then `s, s+p ∈ Z`,
so `x_s(1) = β(s) = β(s+p) = x_{s+p}(1)` by (†), and the remaining terms of
(BW) agree by EP(T,p). Either way `x_{s+p}(-1) = x_s(-1)`. Column `-1` is thus
eventually `p`-periodic, and Prop. 3.1's induction applies. ∎

## 5. How much smaller is `B`?

`|Z(S)| / (h+1)` is the density of white cells in the centre column. Over
20 000 rows of the real orbit:

```
   rows tested                20000
   |Z| = #{ s : x_s(0) = 0 }   9880
   |Z| / t_max               0.4940
   bits saved                10120
```

**This is a bounded observation and nothing more.** No limit is claimed, no
constant is fitted, and the standing rule against inferring an asymptotic from
a finite run applies. If the density does tend to `1/2`, the reduced boundary
word is asymptotically **half** the naive one; if it does not, Theorem 3.5 is
still exactly true, with whatever `|Z(S)|` turns out to be.

## 6. Can it be reduced further? Two attempts, both failed

| id | attempted reduction | outcome |
|---|---|---|
| **X-2H-04** | *"only the rows where `x_s(0)=0` **and** `x_{s+1}(0)=1` matter"* | **false.** (BW) at `j=0` uses `x_{s+1}(0)` only as an XOR summand; it never gates the OR. Both values of `x_{s+1}(0)` leave `x_s(1)` relevant. |
| **X-2H-05** | *"iterate the blindness: to get column `-2` we may drop the rows where `x_s(-1)=1`"* | **true but useless.** The OR at `j=-1` is `x_s(-1) OR x_s(0)`, and column `-1` has just been reconstructed in full, so nothing is saved — the saving would have to be in the *given* data, and column 1 is the only given. |

So the reduction of §4 is, as far as this phase could push it, the end of the
line: **one bit at each row where the centre column is white**.

## 7. What this section does **not** establish

1. It does **not** show the centre column is aperiodic. Theorem 3.5 is a
   conditional with an un-checkable hypothesis: to refute (†) one must know
   `β`, which requires knowing column 1, which is exactly what is unavailable
   under the hypothesis.
2. It does **not** improve `CT-01`'s bound `T + p ≥ 998140`.
3. It gives **no** bound on `p`.
4. The rule-90 control (section 11, `WEAK_BRIDGE_THEOREMS.md` §5) shows the
   whole family of arguments in this section is compatible with an eventually
   periodic centre column. Rule 90's centre column *is* eventually periodic and
   its column 1 *is* aperiodic — precisely the configuration Theorem 3.5
   permits. **Theorem 3.5 cannot decide Problem 1, and the control proves it
   cannot.**

## 8. Statement inventory

| id | statement | status |
|---|---|---|
| **P-3.1** | EP + column 1 periodic ⟹ contradiction with T-02 | **THEOREM**, proved, domain FIN |
| **L-3.2** | OR-blindness | **THEOREM**, proved, domain ANY; verified 20 000 rows |
| **T-3.4** | reduced sufficiency of `B(t,h)` | **THEOREM**, proved, domain ANY |
| **T-3.5** | EP ⟹ `β` not eventually `p`-periodic along `Z` | **THEOREM**, proved, domain FIN |
| **BO-3.6** | `\|Z\|/t = 0.4940` at `t = 20000` | **BOUNDED OBSERVATION**; no limit claimed |
| **X-2H-04/05** | two failed further reductions | **REFUTED**, preserved |
