"""rule30_lab.py -- Rule 30 laboratory (Phase 1).

Purpose
-------
Independent, cross-checked implementations of Wolfram's elementary cellular
automaton Rule 30 on the two-sided infinite lattice Z, started from the single
black cell configuration, together with the diagnostic statistics used in the
Phase 1 audit (PHASE1_AUDIT.md).

Conventions fixed once and for all (see PHASE1_AUDIT.md Sec. 1)
--------------------------------------------------------------
* State at time t is a map a_t : Z -> {0,1}.
* Initial condition: a_0(0) = 1 and a_0(i) = 0 for i != 0.
* Local rule (Wolfram numbering, neighbourhood index 4*l + 2*c + r):

      a_{t+1}(i) = f(a_t(i-1), a_t(i), a_t(i+1)),
      f(l, c, r) = (30 >> (4*l + 2*c + r)) & 1.

  Equivalently (proved exhaustively in `verify_local_rule`):

      a_{t+1}(i) = a_t(i-1) XOR (a_t(i) OR a_t(i+1)).

  Note the asymmetry: the rule is *permutive in the left argument* l and is
  NOT permutive in the right argument r.  This asymmetry is the whole content
  of the width-2 theorem (see WIDTH2_PROOF_RECONSTRUCTION.md).
* "Left" = decreasing i, drawn to the left; time increases downward.
* Center column: c(t) := a_t(0) for t = 0, 1, 2, ...  so c(0) = 1.
  (OEIS A051023 uses offset 1; dropping/adding finitely many initial terms
  does not change whether a sequence is eventually periodic, so the choice of
  origin is immaterial to Prize Problem 1 -- but it matters for any claim
  about specific indices such as t = 2^k.)

Three independent implementations are provided:

  A. `step_naive`     -- pure Python list, explicit 8-entry lookup table
                         derived from the rule *number* by bit extraction.
  B. `step_numpy`     -- vectorised table lookup on a numpy uint8 array.
  C. `step_bitwise`   -- big-integer bit-parallel evaluation of the boolean
                         formula  new = (row << 1) ^ (row | (row >> 1)).

A and B share the table-lookup idea but not the code path; C shares nothing
with A/B except the mathematics.  A fourth, structurally different check is
the mirror check: rule 86 is the left-right mirror of rule 30, so evolving
rule 86 from a single cell and reading its center column must reproduce the
same sequence (`verify_mirror`).

Boundary handling
-----------------
All simulations use a finite array of radius R with zero padding.  Because the
light cone of the single-cell seed grows at speed exactly 1 (a_t(i) = 0
whenever |i| > t), a finite array of radius R reproduces the infinite-lattice
evolution *exactly* for all cells with |i| <= t as long as t <= R - 1.  Every
public routine here enforces R >= T + 2 for T steps, so no reported value is
contaminated by the artificial boundary.  `verify_boundary_independence`
checks this empirically by re-running with a larger radius.

Nothing in this file proves anything about non-periodicity.  Finite
computation can only ever *lower-bound* a hypothetical (preperiod, period);
see `periodicity_lower_bounds`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is required for impl. B only
    np = None

RULE30 = 30
RULE86 = 86  # left-right mirror of rule 30 (verified in verify_mirror)


# ---------------------------------------------------------------------------
# 1. Local rule
# ---------------------------------------------------------------------------


def rule_table(rule_number: int) -> Tuple[int, ...]:
    """Wolfram's elementary-CA table: entry k is the output for neighbourhood
    (l, c, r) with k = 4*l + 2*c + r."""
    if not 0 <= rule_number <= 255:
        raise ValueError("elementary CA rule number must be in [0, 255]")
    return tuple((rule_number >> k) & 1 for k in range(8))


def rule30_formula(l: int, c: int, r: int) -> int:
    """The standard closed form for rule 30: l XOR (c OR r)."""
    return l ^ (c | r)


def mirror_rule_number(rule_number: int) -> int:
    """Rule number of the left-right mirror image automaton."""
    tab = rule_table(rule_number)
    out = 0
    for l in (0, 1):
        for c in (0, 1):
            for r in (0, 1):
                if tab[4 * r + 2 * c + l]:  # mirrored neighbourhood
                    out |= 1 << (4 * l + 2 * c + r)
    return out


# ---------------------------------------------------------------------------
# 2. Implementation A: pure Python list + lookup table
# ---------------------------------------------------------------------------


def step_naive(row: Sequence[int], table: Sequence[int]) -> List[int]:
    """One step on a finite array with zero padding outside."""
    n = len(row)
    out = [0] * n
    for i in range(n):
        l = row[i - 1] if i - 1 >= 0 else 0
        c = row[i]
        r = row[i + 1] if i + 1 < n else 0
        out[i] = table[4 * l + 2 * c + r]
    return out


def evolve_naive(steps: int, rule_number: int = RULE30) -> List[List[int]]:
    """Rows 0..steps as lists of length 2R+1, position i stored at index i+R."""
    table = rule_table(rule_number)
    R = steps + 2
    row = [0] * (2 * R + 1)
    row[R] = 1
    rows = [row]
    for _ in range(steps):
        row = step_naive(row, table)
        rows.append(row)
    return rows


def center_column_naive(steps: int, rule_number: int = RULE30) -> List[int]:
    R = steps + 2
    return [row[R] for row in evolve_naive(steps, rule_number)]


# ---------------------------------------------------------------------------
# 3. Implementation B: numpy vectorised
# ---------------------------------------------------------------------------


def step_numpy(row, table):
    if np is None:
        raise RuntimeError("numpy not available")
    left = np.empty_like(row)
    left[0] = 0
    left[1:] = row[:-1]
    right = np.empty_like(row)
    right[-1] = 0
    right[:-1] = row[1:]
    idx = (left << 2) | (row << 1) | right
    return table[idx]


def evolve_numpy(steps: int, rule_number: int = RULE30, keep_rows: bool = False):
    """Returns (center_column, rows_or_None).  rows only if keep_rows."""
    if np is None:
        raise RuntimeError("numpy not available")
    table = np.array(rule_table(rule_number), dtype=np.uint8)
    R = steps + 2
    row = np.zeros(2 * R + 1, dtype=np.uint8)
    row[R] = 1
    center = [int(row[R])]
    rows = [row.copy()] if keep_rows else None
    for _ in range(steps):
        row = step_numpy(row, table)
        center.append(int(row[R]))
        if keep_rows:
            rows.append(row.copy())
    return center, rows


def center_column_numpy(steps: int, rule_number: int = RULE30) -> List[int]:
    return evolve_numpy(steps, rule_number)[0]


# ---------------------------------------------------------------------------
# 4. Implementation C: big-integer bit-parallel (rule 30 only, formula based)
# ---------------------------------------------------------------------------
#
# Encoding: bit k of the integer holds the cell at lattice position k - R.
# Then  a(i-1)  reads off  (row << 1),  and  a(i+1)  reads off  (row >> 1),
# so the rule  a' = a(i-1) XOR (a(i) OR a(i+1))  becomes
#
#       new = (row << 1) ^ (row | (row >> 1))
#
# masked to the array width.  This shares no code with A or B.


def center_column_bitwise(steps: int) -> List[int]:
    R = steps + 2
    width = 2 * R + 1
    mask = (1 << width) - 1
    row = 1 << R
    center = [1]
    for _ in range(steps):
        row = ((row << 1) ^ (row | (row >> 1))) & mask
        center.append((row >> R) & 1)
    return center


def evolve_bitwise_rows(steps: int) -> Tuple[List[int], int]:
    """Full rows as integers plus the center bit index R (memory heavy)."""
    R = steps + 2
    width = 2 * R + 1
    mask = (1 << width) - 1
    row = 1 << R
    rows = [row]
    for _ in range(steps):
        row = ((row << 1) ^ (row | (row >> 1))) & mask
        rows.append(row)
    return rows, R


# ---------------------------------------------------------------------------
# 5. Verification suite
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def verify_local_rule() -> List[Check]:
    checks = []
    tab = rule_table(RULE30)
    expected = (0, 1, 1, 1, 1, 0, 0, 0)  # k = 0..7, i.e. binary 00011110 = 30
    checks.append(
        Check(
            "rule_number_bits",
            tab == expected,
            f"table(30) by index 4l+2c+r = {tab}; expected {expected}",
        )
    )
    ok = True
    rows = []
    for l in (0, 1):
        for c in (0, 1):
            for r in (0, 1):
                a = tab[4 * l + 2 * c + r]
                b = rule30_formula(l, c, r)
                ok &= a == b
                rows.append(f"{l}{c}{r}->{a}")
    checks.append(
        Check(
            "table_equals_xor_or_formula",
            ok,
            "exhaustive over all 8 neighbourhoods: " + ", ".join(rows),
        )
    )
    # permutivity structure
    left_perm = all(
        len({rule30_formula(l, c, r) for l in (0, 1)}) == 2
        for c in (0, 1)
        for r in (0, 1)
    )
    right_perm = all(
        len({rule30_formula(l, c, r) for r in (0, 1)}) == 2
        for l in (0, 1)
        for c in (0, 1)
    )
    checks.append(
        Check(
            "left_permutive",
            left_perm,
            "for every (c,r), l -> f(l,c,r) is a bijection of {0,1}",
        )
    )
    checks.append(
        Check(
            "not_right_permutive",
            not right_perm,
            "f(l,1,0) = f(l,1,1) = l XOR 1, so r -> f(l,c,r) is NOT injective "
            "when c = 1; rule 30 is one-sided permutive only",
        )
    )
    checks.append(
        Check(
            "mirror_is_86",
            mirror_rule_number(RULE30) == RULE86,
            f"mirror(30) = {mirror_rule_number(RULE30)}",
        )
    )
    return checks


HAND_ROWS = {
    # rows computed by hand in PHASE1_AUDIT.md Sec. 2.3, listed as the window
    # of positions -3..+3 (index 0 <-> position -3)
    0: [0, 0, 0, 1, 0, 0, 0],
    1: [0, 0, 1, 1, 1, 0, 0],
    2: [0, 1, 1, 0, 0, 1, 0],
    3: [1, 1, 0, 1, 1, 1, 1],
}


def verify_hand_rows(steps: int = 3) -> List[Check]:
    rows = evolve_naive(max(steps, 3))
    R = max(steps, 3) + 2
    checks = []
    for t, expected in HAND_ROWS.items():
        got = rows[t][R - 3 : R + 4]
        checks.append(
            Check(
                f"hand_row_t{t}",
                got == expected,
                f"positions -3..3 at t={t}: got {got}, hand-computed {expected}",
            )
        )
    return checks


def verify_implementations(steps_full: int = 400, steps_center: int = 5000) -> List[Check]:
    checks = []
    # full-row agreement between A and B
    rows_a = evolve_naive(steps_full)
    _, rows_b = evolve_numpy(steps_full, keep_rows=True)
    same = all(list(map(int, rb)) == ra for ra, rb in zip(rows_a, rows_b))
    checks.append(
        Check(
            "A_vs_B_full_rows",
            same,
            f"all {steps_full + 1} complete rows (width {2 * (steps_full + 2) + 1}) "
            "identical between list-lookup and numpy-lookup implementations",
        )
    )
    # full-row agreement between A and C (bitwise), comparing as bit strings
    rows_c, Rc = evolve_bitwise_rows(steps_full)
    Ra = steps_full + 2
    ok = True
    for t, (ra, rc) in enumerate(zip(rows_a, rows_c)):
        as_int = 0
        for k, v in enumerate(ra):
            if v:
                as_int |= 1 << k
        ok &= as_int == rc
        if not ok:
            break
    checks.append(
        Check(
            "A_vs_C_full_rows",
            ok and Ra == Rc,
            f"all {steps_full + 1} complete rows identical between list-lookup "
            "and big-integer bit-parallel implementations",
        )
    )
    # center column agreement over a longer range, B vs C
    cb = center_column_numpy(steps_center)
    cc = center_column_bitwise(steps_center)
    checks.append(
        Check(
            "B_vs_C_center_column",
            cb == cc,
            f"center columns agree on all {steps_center + 1} entries "
            f"t = 0..{steps_center}",
        )
    )
    return checks


def verify_mirror(steps: int = 2000) -> List[Check]:
    """Rule 86 is the mirror of rule 30; from a symmetric (single cell) seed the
    center column must therefore be identical, and rows must be reversals."""
    c30 = center_column_bitwise(steps)
    c86, rows86 = evolve_numpy(steps, RULE86, keep_rows=True)
    _, rows30 = evolve_numpy(steps, RULE30, keep_rows=True)
    rows_ok = all(
        list(map(int, r30)) == list(map(int, r86))[::-1]
        for r30, r86 in zip(rows30, rows86)
    )
    return [
        Check(
            "mirror86_center_column",
            c30 == c86,
            f"rule 86 (mirror of 30) reproduces the same center column for "
            f"t = 0..{steps}",
        ),
        Check(
            "mirror86_rows_reversed",
            rows_ok,
            "every rule-30 row equals the reversal of the corresponding rule-86 row",
        ),
    ]


def verify_structural_invariants(steps: int = 3000) -> List[Check]:
    """Three facts provable in two lines from the local rule; a correct
    implementation must satisfy them (they test orientation, not randomness).

      (i)  a_t(-t) = 1 for all t          [left light-cone edge is frozen at 1]
      (ii) a_t(+t) = 1 for all t          [right edge likewise]
      (iii) a_t(t-1) alternates: a_{t+1}(t) = a_t(t-1) XOR 1  for t >= 1
    """
    _, rows = evolve_numpy(steps, keep_rows=True)
    R = steps + 2
    left_edge = all(int(rows[t][R - t]) == 1 for t in range(steps + 1))
    right_edge = all(int(rows[t][R + t]) == 1 for t in range(steps + 1))
    alt = all(
        int(rows[t + 1][R + t]) == (int(rows[t][R + t - 1]) ^ 1)
        for t in range(1, steps)
    )
    return [
        Check("invariant_left_edge_all_ones", left_edge,
              "a_t(-t) = 1 verified for t = 0..%d (proof: a_{t+1}(-(t+1)) = "
              "0 XOR (0 OR a_t(-t)) = a_t(-t))" % steps),
        Check("invariant_right_edge_all_ones", right_edge,
              "a_t(+t) = 1 verified for t = 0..%d" % steps),
        Check("invariant_subdiagonal_alternates", alt,
              "a_{t+1}(t) = a_t(t-1) XOR 1 verified for t = 1..%d" % (steps - 1)),
    ]


def verify_boundary_independence(steps: int = 2000, extra: int = 500) -> List[Check]:
    """Re-run with a much larger radius; the center column must not move."""
    base = center_column_bitwise(steps)
    R = steps + 2 + extra
    width = 2 * R + 1
    mask = (1 << width) - 1
    row = 1 << R
    wide = [1]
    for _ in range(steps):
        row = ((row << 1) ^ (row | (row >> 1))) & mask
        wide.append((row >> R) & 1)
    return [
        Check(
            "boundary_independence",
            base == wide,
            f"center column for t = 0..{steps} unchanged when the padded array "
            f"radius is increased from {steps + 2} to {R}",
        )
    ]


def run_all_verifications(fast: bool = False) -> List[Check]:
    checks: List[Check] = []
    checks += verify_local_rule()
    checks += verify_hand_rows()
    if fast:
        checks += verify_implementations(200, 1000)
        checks += verify_mirror(500)
        checks += verify_structural_invariants(500)
        checks += verify_boundary_independence(500, 100)
        checks += verify_lemma_left_reconstruction(500, 20)
        checks += verify_lemma_leftward_determinism(600, 50)
        checks += verify_lemma_strip_determinism(600)
    else:
        checks += verify_implementations(400, 5000)
        checks += verify_mirror(2000)
        checks += verify_structural_invariants(3000)
        checks += verify_boundary_independence(2000, 500)
        checks += verify_lemma_left_reconstruction(2000, 50)
        checks += verify_lemma_leftward_determinism(3000, 200)
        checks += verify_lemma_strip_determinism(3000)
    return checks


# ---------------------------------------------------------------------------
# 6. Statistics
# ---------------------------------------------------------------------------


def bit_frequencies(bits: Sequence[int], prefixes: Iterable[int]) -> List[Dict]:
    out = []
    running = 0
    prefixes = sorted(p for p in prefixes if p <= len(bits))
    idx = 0
    for n in prefixes:
        while idx < n:
            running += bits[idx]
            idx += 1
        ones = running
        zeros = n - ones
        # z-score under the (heuristic!) fair-coin null hypothesis
        z = (ones - n / 2) / ((n ** 0.5) / 2) if n else 0.0
        out.append(
            {"n": n, "ones": ones, "zeros": zeros, "freq_one": ones / n,
             "excess_ones": ones - zeros, "z_fair_coin": z}
        )
    return out


def subword_complexity(bits: Sequence[int], max_len: int, prefix: int = None) -> List[Dict]:
    """p(n) = number of distinct length-n factors occurring in the observed
    prefix.  Every such factor is a genuine factor of the infinite word, so
    p_obs(n) is a valid lower bound for the true factor complexity."""
    s = "".join(map(str, bits[:prefix] if prefix else bits))
    N = len(s)
    out = []
    for n in range(1, max_len + 1):
        seen = {s[i : i + n] for i in range(N - n + 1)}
        rec = {
            "n": n,
            "p_obs": len(seen),
            "max_possible_2^n": 2 ** n if n <= 30 else None,
            "saturation_bound_N-n+1": N - n + 1,
            "sturmian_bound_n+1": n + 1,
        }
        # If almost every binary word of length n occurs, name the absentees:
        # a word that stays absent as N grows would be a genuine structural
        # fact (a forbidden factor), not a sampling artefact.
        if n <= 22 and 2 ** n - len(seen) <= 8:
            rec["missing_words"] = sorted(
                format(v, "0%db" % n) for v in range(2 ** n)
                if format(v, "0%db" % n) not in seen
            )
        out.append(rec)
    return out


def berlekamp_massey_gf2(bits: Sequence[int]) -> Tuple[int, List[int]]:
    """Linear complexity over GF(2) of a finite bit sequence.

    Bit-packed variant: C and B are polynomials stored as integers (bit i =
    coefficient of x^i); the discrepancy is the parity of popcount(C & R),
    where R holds the reversed prefix.  Mathematically identical to the
    textbook O(n^2) algorithm, but the inner loop runs at word speed.
    """
    C = 1  # connection polynomial, C(x) = 1
    B = 1
    L = 0
    m = 1
    R = 0  # bit i = s_{N-i}
    for N, s in enumerate(bits):
        R = (R << 1) | int(s)
        d = bin(C & R).count("1") & 1
        if d == 0:
            m += 1
        elif 2 * L <= N:
            T = C
            C ^= B << m
            L = N + 1 - L
            B = T
            m = 1
        else:
            C ^= B << m
            m += 1
    poly = [(C >> i) & 1 for i in range(L + 1)]
    return L, poly


def linear_complexity_profile(bits: Sequence[int], checkpoints: Iterable[int]) -> List[Dict]:
    checkpoints = sorted(c for c in checkpoints if c <= len(bits))
    out = []
    for n in checkpoints:
        L, _ = berlekamp_massey_gf2(bits[:n])
        out.append({"N": n, "L": L, "L_over_N": L / n if n else 0.0})
    return out


def period_scan(bits: Sequence[int], max_period: int) -> Dict:
    """For each candidate period p, report

      first_mismatch(p) = min{ t >= 0 : c(t) != c(t+p) }        (t + p < N)
      last_mismatch(p)  = max{ t     : c(t) != c(t+p) }         (t + p < N)

    The *last* mismatch is the meaningful one: if the center column really had
    eventual period p, its preperiod T would have to satisfy T > last_mismatch(p).
    A large min over p of last_mismatch(p) is a rigorous lower bound on the
    preperiod of ANY eventual period p <= max_period -- and nothing more.
    """
    N = len(bits)
    firsts, lasts = {}, {}
    for p in range(1, max_period + 1):
        first = None
        last = None
        for t in range(N - p):
            if bits[t] != bits[t + p]:
                if first is None:
                    first = t
                last = t
        firsts[p] = first
        lasts[p] = last
    finite_last = {p: v for p, v in lasts.items() if v is not None}
    worst_p = min(finite_last, key=lambda p: finite_last[p]) if finite_last else None
    return {
        "N": N,
        "max_period": max_period,
        "first_mismatch": firsts,
        "last_mismatch": lasts,
        "periods_with_no_mismatch_in_window": [p for p, v in lasts.items() if v is None],
        "min_last_mismatch_over_p": finite_last[worst_p] if worst_p else None,
        "argmin_p": worst_p,
    }


def period_scan_fast(bits: Sequence[int], max_period: int) -> Dict:
    """Same as period_scan but using big-integer XOR; O(max_period * N / 64)."""
    N = len(bits)
    S = 0
    for i, b in enumerate(bits):
        if b:
            S |= 1 << i
    firsts, lasts = {}, {}
    for p in range(1, max_period + 1):
        window = (1 << (N - p)) - 1
        diff = (S ^ (S >> p)) & window  # bit t set iff c(t) != c(t+p)
        if diff == 0:
            firsts[p] = None
            lasts[p] = None
        else:
            firsts[p] = (diff & -diff).bit_length() - 1
            lasts[p] = diff.bit_length() - 1
    finite_last = {p: v for p, v in lasts.items() if v is not None}
    worst_p = min(finite_last, key=lambda p: finite_last[p]) if finite_last else None
    return {
        "N": N,
        "max_period": max_period,
        "first_mismatch": firsts,
        "last_mismatch": lasts,
        "periods_with_no_mismatch_in_window": [p for p, v in lasts.items() if v is None],
        "min_last_mismatch_over_p": finite_last[worst_p] if worst_p else None,
        "argmin_p": worst_p,
    }


def powers_of_two_behaviour(bits: Sequence[int], kmax: int = None) -> List[Dict]:
    N = len(bits)
    out = []
    k = 0
    while True:
        base = 2 ** k
        if base + 1 >= N or (kmax is not None and k > kmax):
            break
        out.append(
            {
                "k": k,
                "2^k-1": base - 1,
                "c(2^k-1)": bits[base - 1],
                "2^k": base,
                "c(2^k)": bits[base],
                "2^k+1": base + 1,
                "c(2^k+1)": bits[base + 1],
            }
        )
        k += 1
    return out


def run_length_stats(bits: Sequence[int], max_report: int = 12) -> Dict:
    runs = {0: {}, 1: {}}
    cur = bits[0]
    length = 1
    longest = {0: 0, 1: 0}
    for b in bits[1:]:
        if b == cur:
            length += 1
        else:
            runs[cur][length] = runs[cur].get(length, 0) + 1
            longest[cur] = max(longest[cur], length)
            cur, length = b, 1
    runs[cur][length] = runs[cur].get(length, 0) + 1
    longest[cur] = max(longest[cur], length)
    return {
        "longest_run_of_0": longest[0],
        "longest_run_of_1": longest[1],
        "run_counts_0": {k: runs[0].get(k, 0) for k in range(1, max_report + 1)},
        "run_counts_1": {k: runs[1].get(k, 0) for k in range(1, max_report + 1)},
    }


def block_entropy(bits: Sequence[int], max_len: int = 16) -> List[Dict]:
    s = "".join(map(str, bits))
    N = len(s)
    import math

    out = []
    for n in range(1, max_len + 1):
        counts = {}
        for i in range(N - n + 1):
            w = s[i : i + n]
            counts[w] = counts.get(w, 0) + 1
        tot = N - n + 1
        H = -sum((v / tot) * math.log2(v / tot) for v in counts.values())
        out.append({"n": n, "H_n_bits": H, "H_n_over_n": H / n, "distinct": len(counts)})
    return out


def periodicity_lower_bounds(bits: Sequence[int], subword: List[Dict], L: int) -> Dict:
    """The only *rigorous* things finite computation yields here.

    Lemma A (factor counting).  If x is eventually periodic with preperiod T
    and period p, then x has at most T + p distinct factors of each length.
    Hence  T + p >= max_n p_obs(n).

    Lemma B (linear recurrence).  If x is eventually periodic with preperiod T
    and period p, then x satisfies the GF(2) linear recurrence with
    characteristic polynomial x^{T+p} + x^{T}, so the linear complexity of
    every prefix is at most T + p.  Hence  T + p >= L(N).

    Both are lower bounds on T + p.  Neither can ever establish that no
    (T, p) exists.
    """
    max_pobs = max(d["p_obs"] for d in subword) if subword else 0
    arg = max(subword, key=lambda d: d["p_obs"])["n"] if subword else None
    return {
        "N": len(bits),
        "factor_bound_T_plus_p_at_least": max_pobs,
        "factor_bound_witness_n": arg,
        "linear_complexity_bound_T_plus_p_at_least": L,
        "combined_T_plus_p_at_least": max(max_pobs, L),
    }


# ---------------------------------------------------------------------------
# 7. Left-region geometry (relevant to the width-2 theorem, see Sec. 6 of
#    WIDTH2_PROOF_RECONSTRUCTION.md)
# ---------------------------------------------------------------------------


def _edge_match_width(rows, R, t, tau, sigma):
    """Largest w with a_t(-t+k) == a_{t+tau}(-(t+tau)+sigma+k) for all k < w,
    i.e. the width (measured from the left light-cone edge) of the band on
    which the pattern is invariant under the space-time shift
    (t, i) -> (t + tau, i - tau + sigma)."""
    a, b = rows[t], rows[t + tau]
    w = 0
    while w <= t:
        ia = R - t + w
        ib = R - (t + tau) + sigma + w
        if ib < 0 or ia > R + t:
            break
        if a[ia] != b[ib]:
            break
        w += 1
    return w


def left_region_shift_symmetry(
    steps: int = 6000,
    taus=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64),
    sigmas=(-2, -1, 0, 1, 2),
    n_samples: int = 40,
):
    """Search for the space-time shift symmetries of the regular left region.

    For each candidate shift, measure the band width W(t) from the left
    light-cone edge and fit W(t) = slope * t + intercept by least squares.

      * slope ~ 0 with W bounded  -> the symmetry holds only in a finite band
        hugging the light-cone edge;
      * slope > 0                 -> the symmetric region is a wedge whose
        boundary sits at lattice position ~ -(1 - slope) * t, i.e. the
        boundary between the regular left region and the chaotic bulk drifts
        leftwards at speed 1 - slope.

    A positive drift speed is required by the width-2 theorem: every fixed
    column must eventually be swallowed by the non-periodic bulk, since two
    adjacent columns inside a shift-periodic region would both be eventually
    periodic.  This routine measures that speed; it does not prove it.
    """
    _, rows = evolve_numpy(steps, keep_rows=True)
    R = steps + 2
    lo, hi = steps // 2, steps - max(taus) - 1
    ts = [lo + (hi - lo) * j // (n_samples - 1) for j in range(n_samples)]
    results = []
    for tau in taus:
        for sigma in sigmas:
            widths = [(t, _edge_match_width(rows, R, t, tau, sigma)) for t in ts]
            n = len(widths)
            sx = sum(t for t, _ in widths)
            sy = sum(w for _, w in widths)
            sxx = sum(t * t for t, _ in widths)
            sxy = sum(t * w for t, w in widths)
            den = n * sxx - sx * sx
            slope = (n * sxy - sx * sy) / den if den else 0.0
            results.append(
                {
                    "tau": tau,
                    "sigma": sigma,
                    "slope": slope,
                    "intercept": (sy - slope * sx) / n,
                    "min_width": min(w for _, w in widths),
                    "max_width": max(w for _, w in widths),
                    "widths_sample": widths[:: max(1, n // 5)],
                }
            )
    best = max(results, key=lambda r: r["slope"])
    # the minimal tau achieving essentially the maximal wedge
    minimal = min(
        (r for r in results if r["slope"] > 0.5 * best["slope"]),
        key=lambda r: (r["tau"], abs(r["sigma"])),
        default=best,
    )
    return {
        "steps": steps,
        "best_by_slope": best,
        "minimal_tau_with_wedge": minimal,
        "implied_boundary_speed_leftward": 1.0 - best["slope"],
        "all": [r for r in results if r["max_width"] > 8],
    }


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rule 30 laboratory")
    ap.add_argument("command", choices=["verify", "center", "stats"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.command == "verify":
        checks = run_all_verifications(fast=args.fast)
        for c in checks:
            print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        print(f"\n{sum(c.passed for c in checks)}/{len(checks)} checks passed")
        return 0 if all(c.passed for c in checks) else 1

    bits = center_column_bitwise(args.steps)
    if args.command == "center":
        print("".join(map(str, bits)))
        return 0

    stats = {
        "frequencies": bit_frequencies(bits, [10, 100, 1000, 10000, len(bits)]),
        "subword_complexity": subword_complexity(bits, 16),
        "linear_complexity": linear_complexity_profile(bits, [len(bits)]),
        "powers_of_two": powers_of_two_behaviour(bits),
    }
    print(json.dumps(stats, indent=2) if args.json else stats)
    return 0



# ---------------------------------------------------------------------------
# 9. Columns near the center, and the locality probe for the bridging lemma
#    (see WIDTH2_PROOF_RECONSTRUCTION.md Sec. 7 and CONJECTURE_REGISTRY.md C5)
# ---------------------------------------------------------------------------


def columns(steps: int, positions: Sequence[int]) -> Dict[int, List[int]]:
    """Time series of several columns at once, via the bit-parallel engine."""
    R = steps + 2
    width = 2 * R + 1
    mask = (1 << width) - 1
    row = 1 << R
    out = {i: [(row >> (R + i)) & 1] for i in positions}
    for _ in range(steps):
        row = ((row << 1) ^ (row | (row >> 1))) & mask
        for i in positions:
            out[i].append((row >> (R + i)) & 1)
    return out


def agreement_mask(col: Sequence[int], p: int) -> int:
    """Integer whose bit t is 1 iff col(t) == col(t+p)."""
    S = 0
    for i, b in enumerate(col):
        if b:
            S |= 1 << i
    N = len(col)
    window = (1 << (N - p)) - 1
    return (~(S ^ (S >> p))) & window


def _runs_of_ones(x: int, N: int):
    """Yield (start, length) for every maximal run of 1-bits in x below bit N.

    Scans a bit *string* rather than shifting the integer once per bit: each
    `x >> t` on a 200 kbit integer copies the whole number, which turned this
    routine into the dominant cost of `locality_probe`."""
    bits = bin(x)[2:][::-1]          # LSB first
    if len(bits) < N:
        bits += "0" * (N - len(bits))
    t = 0
    while t < N:
        if bits[t] != "1":
            t += 1
            continue
        start = t
        while t < N and bits[t] == "1":
            t += 1
        yield start, t - start


def _to_int(bits: Sequence[int]) -> int:
    v = 0
    for i, b in enumerate(bits):
        if b:
            v |= 1 << i
    return v


def locality_probe(steps: int = 200000, periods=range(1, 257), min_window: int = 8):
    """Does *local* periodicity of column 0 force *local* periodicity of column -1?

    The exact identity   a_t(-1) = a_{t+1}(0) XOR (a_t(0) OR a_t(1))   gives,
    for any lag p:

        [a_t(-1) == a_{t+p}(-1)]  <=  [a_t(0)==a_{t+p}(0)]
                                   &  [a_{t+1}(0)==a_{t+1+p}(0)]
                                   &  ( a_t(0) == 1  OR  a_t(1) == a_{t+p}(1) )

    i.e. once column 0 agrees at lag p at times t and t+1, column -1 agrees at
    lag p at time t automatically whenever a_t(0) = 1, and *only* when column 1
    also agrees at lag p whenever a_t(0) = 0.

    A *finitary* bridging lemma would say: "column 0 p-periodic on a window of
    length W  =>  column -1 p-periodic on a window of length W - O(1)".  This
    routine tests exactly that implication:

      * `full_transfer_fraction` -- fraction of maximal windows on which
        column -1 is p-periodic throughout;
      * `cond_agree_col1_on_zeroset` -- the empirical value of
        P[ a_t(1) == a_{t+p}(1) | column 0 agrees at t, t+1 and a_t(0) = 0 ],
        the quantity a finitary bridge would have to control;
      * `survival_per_step` -- the empirical per-step survival rate of the
        column -1 agreement inside a column-0 window, to be compared with the
        heuristic prediction 1/2 * 1 + 1/2 * 1/2 = 3/4.
    """
    cols = columns(steps, (-1, 0, 1))
    N = steps + 1
    S = {i: _to_int(cols[i]) for i in (-1, 0, 1)}
    tot_windows = 0
    full_transfer = 0
    win_lens = []
    best_runs = []
    zero_trials = 0
    zero_agree = 0
    steps_taken = 0
    steps_survived = 0
    max_window = {"p": None, "len": 0, "minus1_run": 0}
    for p in periods:
        window = (1 << (N - p)) - 1
        agree0 = (~(S[0] ^ (S[0] >> p))) & window
        agree1 = (~(S[1] ^ (S[1] >> p))) & window
        agreem1 = (~(S[-1] ^ (S[-1] >> p))) & window
        # column 0 agrees at t AND t+1
        agree0_pair = agree0 & (agree0 >> 1)
        for start, length in _runs_of_ones(agree0_pair, N - p - 1):
            if length < min_window:
                continue
            tot_windows += 1
            win_lens.append(length)
            seg = (agreem1 >> start) & ((1 << length) - 1)
            if seg == (1 << length) - 1:
                full_transfer += 1
            # longest run of column -1 agreement inside the window
            best = cur = 0
            for k in range(length):
                if (seg >> k) & 1:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
                steps_taken += 1
                steps_survived += (seg >> k) & 1
                t = start + k
                if not ((S[0] >> t) & 1):          # a_t(0) == 0
                    zero_trials += 1
                    zero_agree += (agree1 >> t) & 1
            best_runs.append(best)
            if length > max_window["len"]:
                max_window = {"p": p, "len": length, "minus1_run": best}
    if not tot_windows:
        return {"windows": 0}
    return {
        "steps": steps,
        "periods_scanned": [min(periods), max(periods)],
        "min_window": min_window,
        "windows": tot_windows,
        "max_col0_window": max(win_lens),
        "mean_col0_window": sum(win_lens) / len(win_lens),
        "mean_colminus1_run": sum(best_runs) / len(best_runs),
        "full_transfer_windows": full_transfer,
        "full_transfer_fraction": full_transfer / tot_windows,
        "cond_agree_col1_on_zeroset": zero_agree / zero_trials if zero_trials else None,
        "zeroset_trials": zero_trials,
        "survival_per_step": steps_survived / steps_taken,
        "heuristic_survival_prediction": 0.75,
        "largest_window": max_window,
        "colminus1_run_histogram": {
            k: sum(1 for b in best_runs if b == k) for k in range(0, 16)
        },
    }


def transfer_profile(steps: int = 200000, periods=range(1, 257), kmax: int = 16):
    """How much does lag-p agreement in column 0 predict lag-p agreement in
    column 1?

    For each lag p and each k, let A_k(t) = 1 iff column 0 agrees at lag p at
    all of t, t+1, ..., t+k.  We report

        P[ col1 agrees at lag p at t | A_k(t) ]                and
        P[ col1 agrees at lag p at t | A_k(t) and a_t(0) = 0 ]

    aggregated over all p in `periods`, together with the unconditional
    baseline P[ col1 agrees at lag p ].  Under the naive "columns are
    independent fair coins" heuristic every one of these is 1/2.  Deviation
    measures how strongly local repetition of the center column drags its
    right neighbour along -- which is exactly the content of the bridging
    lemma MB1 (WIDTH2_PROOF_RECONSTRUCTION.md Sec. 7).
    """
    cols = columns(steps, (0, 1))
    N = steps + 1
    S0, S1 = _to_int(cols[0]), _to_int(cols[1])
    pc = lambda x: bin(x).count("1")
    num = [0] * (kmax + 1)
    den = [0] * (kmax + 1)
    znum = [0] * (kmax + 1)
    zden = [0] * (kmax + 1)
    base_num = base_den = 0
    for p in periods:
        window = (1 << (N - p - kmax - 1)) - 1
        agree0 = (~(S0 ^ (S0 >> p))) & window
        agree1 = (~(S1 ^ (S1 >> p))) & window
        zero0 = (~S0) & window
        base_num += pc(agree1)
        base_den += N - p - kmax - 1
        A = window
        for k in range(kmax + 1):
            A &= agree0 >> k
            num[k] += pc(A & agree1)
            den[k] += pc(A)
            znum[k] += pc(A & agree1 & zero0)
            zden[k] += pc(A & zero0)
    return {
        "steps": steps,
        "periods_scanned": [min(periods), max(periods)],
        "baseline_P_col1_agrees": base_num / base_den,
        "profile": [
            {
                "k": k,
                "window_len": k + 1,
                "n": den[k],
                "P_col1_agrees_given_col0_window": (num[k] / den[k]) if den[k] else None,
                "n_zeroset": zden[k],
                "P_col1_agrees_given_col0_window_and_col0_zero":
                    (znum[k] / zden[k]) if zden[k] else None,
            }
            for k in range(kmax + 1)
        ],
    }


def backward_transfer(steps: int = 200000, max_period: int = 4096, avals=(0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16)):
    """P[ a_t(1) == a_{t+p}(1) | column 0 agrees at lag p throughout
        [t-a, t] and a_t(0) = 0 ], as a function of the look-back depth a.

    Phase 1 finding: the *forward* analogue (conditioning on [t, t+b]) is flat
    at 1/2, while this backward version rises with a.  The light cone explains
    the asymmetry: a_t(1) is determined by the cells at time t-k in positions
    [1-k, 1+k], which contain a_{t-1}(0), a_{t-2}(0), ...; conditioning on
    those matching at lag p matches part of a_t(1)'s own dependency cone.
    Times after t lie outside that cone and carry no information.

    Whether this probability tends to 1 as a -> infinity is the quantitative
    form of the bridging lemma MB1 (CONJECTURE_REGISTRY.md C5).
    """
    cols = columns(steps, (0, 1))
    N = steps + 1
    S0, S1 = _to_int(cols[0]), _to_int(cols[1])
    pc = lambda x: bin(x).count("1")
    num = {a: 0 for a in avals}
    den = {a: 0 for a in avals}
    amax = max(avals)
    for p in range(1, max_period + 1):
        W = (1 << (N - p - 1)) - 1
        agree0 = (~(S0 ^ (S0 >> p))) & W
        agree1 = (~(S1 ^ (S1 >> p))) & W
        zero0 = (~S0) & W
        A = W
        prev = 0
        for a in sorted(avals):
            for j in range(prev, a + 1):
                A &= (agree0 << j) if j else agree0
            A &= W
            prev = a + 1
            d = A & zero0 & (W >> amax) << 0
            den[a] += pc(d)
            num[a] += pc(d & agree1)
    return {
        "steps": steps,
        "max_period": max_period,
        "profile": [
            {"lookback_a": a, "n": den[a],
             "P_col1_agrees": (num[a] / den[a]) if den[a] else None}
            for a in sorted(avals)
        ],
    }


# ---------------------------------------------------------------------------
# 10. Mechanical verification of the lemmas used in the width-2 proof
# ---------------------------------------------------------------------------


def verify_lemma_left_reconstruction(steps: int = 2000, span: int = 50) -> List[Check]:
    """Lemma 1:  a_t(i-1) = a_{t+1}(i) XOR (a_t(i) OR a_t(i+1))."""
    _, rows = evolve_numpy(steps, keep_rows=True)
    R = steps + 2
    ok = True
    bad = None
    for t in range(steps):
        for i in range(-span, span + 1):
            lhs = int(rows[t][R + i - 1])
            rhs = int(rows[t + 1][R + i]) ^ (int(rows[t][R + i]) | int(rows[t][R + i + 1]))
            if lhs != rhs:
                ok = False
                bad = (t, i)
                break
        if not ok:
            break
    return [Check("lemma1_left_reconstruction", ok,
                  "identity checked at %d (t, i) pairs, t < %d, |i| <= %d%s"
                  % (steps * (2 * span + 1), steps, span,
                     "" if ok else "; FAILS at %s" % (bad,)))]


def verify_lemma_leftward_determinism(steps: int = 3000, depth: int = 200) -> List[Check]:
    """Lemma 2: columns i and i+1 determine every column to the left.

    Reconstructs columns -1, -2, ..., -depth from columns 0 and 1 alone and
    compares against the true evolution."""
    _, rows = evolve_numpy(steps, keep_rows=True)
    R = steps + 2
    truth = {i: [int(rows[t][R + i]) for t in range(steps + 1)] for i in range(-depth, 2)}
    right = truth[1][:]      # column i+1
    mid = truth[0][:]        # column i
    ok = True
    first_bad = None
    for j in range(-1, -depth - 1, -1):
        horizon = len(mid) - 1
        rec = [mid[t + 1] ^ (mid[t] | right[t]) for t in range(horizon)]
        if rec[: len(truth[j]) if len(truth[j]) < horizon else horizon] != truth[j][:horizon]:
            ok = False
            first_bad = j
            break
        right, mid = mid[:horizon], rec
    return [Check("lemma2_leftward_determinism", ok,
                  "columns -1..-%d reconstructed from columns 0 and 1 only "
                  "(time horizon shrinks by 1 per column); %s"
                  % (depth, "all match" if ok else "MISMATCH at column %s" % first_bad))]


def verify_lemma_strip_determinism(steps: int = 3000, i0: int = -6, j0: int = 6) -> List[Check]:
    """Lemma 3 (gap version): the two boundary columns i, j together with the
    initial row segment determine every column strictly between them.

    This is the mechanical content of Theorem W2' ("at most one eventually
    periodic column"): the strip is a finite-state machine driven by the two
    boundary columns."""
    _, rows = evolve_numpy(steps, keep_rows=True)
    R = steps + 2
    tab = rule_table(RULE30)
    left_col = [int(rows[t][R + i0]) for t in range(steps + 1)]
    right_col = [int(rows[t][R + j0]) for t in range(steps + 1)]
    state = [int(rows[0][R + k]) for k in range(i0, j0 + 1)]
    ok = True
    bad = None
    for t in range(steps):
        nxt = [0] * len(state)
        for k in range(1, len(state) - 1):
            nxt[k] = tab[4 * state[k - 1] + 2 * state[k] + state[k + 1]]
        nxt[0] = left_col[t + 1]     # boundary columns supplied, not computed
        nxt[-1] = right_col[t + 1]
        state = nxt
        true_row = [int(rows[t + 1][R + k]) for k in range(i0, j0 + 1)]
        if state != true_row:
            ok = False
            bad = t + 1
            break
    return [Check("lemma3_strip_determinism", ok,
                  "interior columns %d..%d driven by boundary columns %d, %d "
                  "for %d steps; %s" % (i0 + 1, j0 - 1, i0, j0, steps,
                                        "all match" if ok else "MISMATCH at t=%d" % bad))]

if __name__ == "__main__":
    raise SystemExit(main())
