"""boundary_lab.py -- Phase 2H sections 2, 3, 5, 6, 8, 11.

Everything here is derived from the local rule.  No paper was readable
(RETRIEVAL_NOTE.md), so nothing is imported from the literature.

Coordinates (original, absolute columns):

    forward   x_{t+1}(j) = x_t(j-1) XOR ( x_t(j) OR x_t(j+1) )          (FW)
    backward  x_s(j-1)   = x_{s+1}(j) XOR ( x_s(j) OR x_s(j+1) )        (BW)

(BW) is (FW) solved for the left input; it is legitimate because XOR is a
group operation in the first argument (left-permutivity, T-01).
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------- rules ---
def f30(l, c, r):
    return l ^ (c | r)


def f90(l, c, r):
    return l ^ r


RULES = {30: f30, 90: f90}


def orbit(t_max: int, rule: int = 30):
    """Single-cell seed.  Returns (grid, C) with grid[t][j + C] = x_t(j)."""
    f = RULES[rule]
    C = t_max + 2
    grid = [[0] * (2 * C + 1) for _ in range(t_max + 1)]
    grid[0][C] = 1
    for t in range(t_max):
        row, nxt = grid[t], grid[t + 1]
        for j in range(1, 2 * C):
            nxt[j] = f(row[j - 1], row[j], row[j + 1])
    return grid, C


def columns_bitparallel(t_max: int, cols, rule: int = 30):
    """Bit-parallel single-cell orbit; returns {j: [x_t(j)]_{t=0..t_max}}.

    Row t is one Python integer R_t with x_t(j) at bit position j + C,
    C = t_max + 2 fixed for every row.  Then

        R_{t+1} = ( (R_t << 1) XOR ( R_t OR (R_t >> 1) ) ) & mask

    reproduces (FW) bit for bit, because bit p of (R<<1) is bit p-1 of R.
    Only the requested columns are stored, so memory is O(t_max), not
    O(t_max^2).  Cross-checked against the explicit grid in run_phase2h_tests.
    """
    C = t_max + 2
    w = 2 * C + 1
    mask = (1 << w) - 1
    R = 1 << C
    out = {j: [] for j in cols}
    for _ in range(t_max + 1):
        for j in cols:
            out[j].append((R >> (j + C)) & 1)
        if rule == 30:
            R = ((R << 1) ^ (R | (R >> 1))) & mask
        else:
            R = ((R << 1) ^ (R >> 1)) & mask
    return out


# ================================================== SECTION 2: candidates ===
# Triangle geometry, identical to triangle_enumerator.py:
#   top row = columns 0 .. 2h ; row s = columns s .. 2h-s ; apex = (h, h).
#
# candidate A  centre column       C  = (x_s(h))_{s=0..h}          h+1 bits
# candidate B  width-2 word        W2 = (x_s(h), x_s(h+1))_{s<h}   2h  bits
# candidate C  right light-cone edge Rt = (x_s(2h-s))_{s=0..h}     h+1 bits
# candidate D  bottom row alone (one row of the orbit)             -- see D_*
# candidate E  bottom row + white right half-line                  -- see D_*

def _evolve_all_tops(h: int, rule: int = 30):
    """All 2^(2h+1) top rows evolved h steps; returns list of numpy arrays,
    rows_by_time[s] = the row-s state of every triangle, packed in the SAME
    2h+1 bit positions (bits outside columns s..2h-s are garbage and must not
    be read)."""
    w = 2 * h + 1
    mask = np.uint64((1 << w) - 1)
    cur = np.arange(1 << w, dtype=np.uint64)
    out = [cur]
    for _ in range(h):
        if rule == 30:
            cur = (cur << np.uint64(1)) ^ (cur | (cur >> np.uint64(1)))
        else:
            cur = (cur << np.uint64(1)) ^ (cur >> np.uint64(1))
        cur = cur & mask
        out.append(cur)
    return out


def _bit(arr, k):
    return (arr >> np.uint64(k)) & np.uint64(1)


def candidate_B_region(h: int, rule: int = 30) -> Dict[str, object]:
    """Exhaustive test of the exact region determined by the width-2 word.

    Claim (Theorem 2.B):  W2 = (x_s(h), x_s(h+1))_{s=0..h-1} determines
        x_s(h-k)  for  1 <= k <= h-1  and  0 <= s <= h-1-k,
    and determines nothing beyond that column-by-column envelope: for every
    k the cell (h-k, h-k) -- one row past the envelope -- is NOT determined.
    """
    rows = _evolve_all_tops(h, rule)
    # width-2 key
    key = np.zeros(1 << (2 * h + 1), dtype=np.uint64)
    for s in range(h):
        key |= _bit(rows[s], h) << np.uint64(2 * s)
        key |= _bit(rows[s], h + 1) << np.uint64(2 * s + 1)

    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    groups = np.add.reduceat(np.ones(ks.size, dtype=np.int64), starts)

    def constant_within_groups(vals: np.ndarray) -> bool:
        v = vals[order]
        mn = np.minimum.reduceat(v, starts)
        mx = np.maximum.reduceat(v, starts)
        return bool(np.array_equal(mn, mx))

    determined, not_determined, violations = [], [], []
    for k in range(1, h):
        for s in range(0, h - k):                      # inside the envelope
            if not constant_within_groups(_bit(rows[s], h - k)):
                violations.append([s, h - k])
            else:
                determined.append([s, h - k])
        s = h - k                                       # one row past it
        if s <= h and h - k >= s:
            if constant_within_groups(_bit(rows[s], h - k)):
                not_determined.append(["UNEXPECTEDLY DETERMINED", s, h - k])
            else:
                not_determined.append([s, h - k])
    return {
        "h": h, "rule": rule,
        "distinct_width2_words": int(starts.size),
        "max_tops_per_width2": int(groups.max()),
        "min_tops_per_width2": int(groups.min()),
        "cells_claimed_determined": len(determined),
        "predicted_count_h_h_minus_1_over_2": h * (h - 1) // 2,
        "determination_violations": violations,
        "envelope_is_sharp": all(isinstance(x[0], int) for x in not_determined),
        "sharpness_probes": len(not_determined),
    }


def forward_collisions(n: int, rule: int = 30, limit: int = 3):
    """Distinct rows of length n with the SAME image on the interior columns
    1..n-2.  Their existence is why backward reconstruction needs a seed: the
    one-step map is not injective, so 'run the rule backwards' is meaningless
    without boundary data.  (Rule 90 has the same non-injectivity, but is
    right-permutive, so it can be seeded from either side.)"""
    f = RULES[rule]
    seen: Dict[Tuple[int, ...], int] = {}
    coll = []
    for r in range(1 << n):
        row = [(r >> i) & 1 for i in range(n)]
        img = tuple(f(row[i - 1], row[i], row[i + 1]) for i in range(1, n - 1))
        if img in seen:
            coll.append({"row_a": [(seen[img] >> i) & 1 for i in range(n)],
                         "row_b": row, "common_image": list(img)})
            if len(coll) >= limit:
                break
        else:
            seen[img] = r
    return {"n": n, "rule": rule, "collisions_found": len(coll),
            "map_is_injective": len(coll) == 0, "examples": coll}


def crosscheck_implementations(t_max: int = 400, rule: int = 30):
    """Rule 3 of the standing scientific rules: two independent
    implementations must agree.  `orbit` is an explicit cell-by-cell grid;
    `columns_bitparallel` is big-integer arithmetic."""
    grid, C = orbit(t_max, rule)
    col = columns_bitparallel(t_max, (-3, -1, 0, 1, 5), rule)
    bad = 0
    for j in (-3, -1, 0, 1, 5):
        for t in range(t_max + 1):
            if grid[t][j + C] != col[j][t]:
                bad += 1
    return {"t_max": t_max, "rule": rule, "columns_compared": 5,
            "cells_compared": 5 * (t_max + 1), "disagreements": bad}


def candidate_C_right_edge(h: int, rule: int = 30, diagonals: int = 1
                           ) -> Dict[str, object]:
    """Candidate C  (diagonals=1): the right light-cone edge
           Rt = ( x_s(2h-s) )_{s=0..h}.
       Candidate C' (diagonals=2): the two rightmost diagonals
           ( x_s(2h-s), x_s(2h-s-1) )_{s=0..h-1}.

    The target must NOT be a sub-word of the boundary, or the test is vacuous.
    The apex (h,h) IS the last letter of Rt, so 'does Rt determine the apex'
    is trivially yes and says nothing; an earlier version of this file asked
    exactly that and is corrected here (see MINIMAL_BOUNDARY_DATA.md, C-2H-01).
    The honest target is the CENTRE WORD ( x_s(h) )_{s=0..h-1}, i.e. the
    centre column with the shared apex removed.
    """
    rows = _evolve_all_tops(h, rule)
    key = np.zeros(1 << (2 * h + 1), dtype=np.uint64)
    bit = 0
    for s in range(h + 1):
        if diagonals == 1:
            key |= _bit(rows[s], 2 * h - s) << np.uint64(bit); bit += 1
        elif s < h:
            key |= _bit(rows[s], 2 * h - s) << np.uint64(bit); bit += 1
            key |= _bit(rows[s], 2 * h - s - 1) << np.uint64(bit); bit += 1
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    groups = np.add.reduceat(np.ones(ks.size, dtype=np.int64), starts)

    # target: the centre word MINUS the apex (which the boundary may contain)
    tgt = np.zeros(1 << (2 * h + 1), dtype=np.uint64)
    for s in range(h):
        tgt |= _bit(rows[s], h) << np.uint64(s)
    tv = tgt[order]
    mn = np.minimum.reduceat(tv, starts)
    mx = np.maximum.reduceat(tv, starts)
    return {"h": h, "rule": rule, "diagonals": diagonals,
            "distinct_boundary_words": int(starts.size),
            "max_tops_per_boundary_word": int(groups.max()),
            "determines_centre_word_minus_apex": bool(np.array_equal(mn, mx))}


def diagonal_recursion(T: int, rule: int = 30) -> Dict[str, object]:
    """Theorem 2.C' on the real single-cell orbit.

    For a target cell (T, 0), index its backward light cone by DIAGONALS
    measured in from the right edge:

        E_k(s) := x_s( T - s - k ),      0 <= k,   0 <= s <= T - k.

    E_0 and E_1 are the two rightmost diagonals of the cone.  Claim:

        E_{k+2}(s) = E_k(s+1) XOR ( E_{k+1}(s) OR E_k(s) )            (DIAG)

    so E_0 and E_1 on rows 0..T-1 determine every E_k on rows 0..T-k, and in
    particular the centre column x_s(0) = E_{T-s}(s) for s = 0..T-1.

    Note E_0(s) = x_s(T-s) = 0 for every s < T/2 by the light cone, so the
    first half of the boundary data is identically zero and carries nothing.
    """
    grid, C = orbit(T, rule)
    v = lambda t, j: grid[t][j + C]
    E = [[0] * (T + 1) for _ in range(T + 1)]
    for s in range(T):
        E[0][s] = v(s, T - s)
        E[1][s] = v(s, T - s - 1)
    bad_rec, bad_centre = 0, 0
    for k in range(0, T - 1):
        for s in range(0, T - k - 1):
            E[k + 2][s] = E[k][s + 1] ^ (E[k + 1][s] | E[k][s])
            if E[k + 2][s] != v(s, T - s - k - 2):
                bad_rec += 1
    for s in range(T):
        if E[T - s][s] != v(s, 0):
            bad_centre += 1
    zero_prefix = sum(1 for s in range(T) if E[0][s] == 0 and s < T / 2)
    return {"T": T, "rule": rule,
            "recursion_cells_checked": sum(max(0, T - k - 1)
                                           for k in range(0, T - 1)),
            "recursion_failures": bad_rec,
            "centre_rows_reconstructed": T,
            "centre_failures": bad_centre,
            "E0_zero_on_first_half": zero_prefix == (T + 1) // 2,
            "boundary_bits_total": 2 * T,
            "boundary_bits_nonzero_possible": 2 * T - 2 * ((T + 1) // 2)}


def rule90_column1_ones(t_max: int):
    """Rule 90, single-cell seed: column 1 is 1 exactly at t = 2^n - 1.

    Proof (Kummer/Lucas): x_t(j) = C(t, (t+j)/2) mod 2 when t+j is even and 0
    otherwise.  For j = 1 and t = 2m+1 the entry is C(2m+1, m+1), which is odd
    iff (m+1) AND m = 0, i.e. iff m = 2^a - 1, i.e. t = 2^(a+1) - 1.
    Hence the gaps between successive ones are unbounded and column 1 is NOT
    eventually periodic.  This routine checks the closed form."""
    col = columns_bitparallel(t_max, (0, 1), 90)
    ones = [t for t in range(t_max + 1) if col[1][t]]
    pred = [2 ** a - 1 for a in range(1, t_max.bit_length() + 2)
            if 2 ** a - 1 <= t_max]
    return {"t_max": t_max, "ones_at": ones, "closed_form_2n_minus_1": pred,
            "closed_form_matches": ones == pred,
            "largest_gap": max((ones[i + 1] - ones[i]
                                for i in range(len(ones) - 1)), default=0)}


# =========================================== SECTION 3: reduced boundary ===
def reduced_boundary_lemma(t_max: int, rule: int = 30) -> Dict[str, object]:
    """Lemma 3.1 verified on the real orbit.

        x_s(-1) = x_{s+1}(0) XOR ( x_s(0) OR x_s(1) )
        and when x_s(0) = 1 the value of x_s(1) is IRRELEVANT.

    So the only bits of column 1 that column -1 depends on are those at rows
    where column 0 is white.  Z = { s : x_s(0) = 0 }.
    """
    col = columns_bitparallel(t_max, (-1, 0, 1), rule)
    cm1, c0, c1 = col[-1], col[0], col[1]
    bad_identity, bad_irrelevance = 0, 0
    Z = 0
    for s in range(t_max):
        if cm1[s] != (c0[s + 1] ^ (c0[s] | c1[s])):
            bad_identity += 1
        if c0[s] == 1:
            # flipping x_s(1) must not change the reconstructed x_s(-1)
            if (c0[s + 1] ^ (c0[s] | (1 - c1[s]))) != cm1[s]:
                bad_irrelevance += 1
        else:
            Z += 1
    return {"t_max": t_max, "rule": rule,
            "identity_failures": bad_identity,
            "irrelevance_failures": bad_irrelevance,
            "rows_tested": t_max,
            "|Z|": Z, "|Z|/t_max": round(Z / t_max, 6),
            "bits_saved": t_max - Z}


def widest_white_pair_run(t_max: int, cols=(0, 1), rule: int = 30):
    """Longest run of consecutive rows on which BOTH columns are white.
    This is the length of boundary data that is free (all zero)."""
    col = columns_bitparallel(t_max, tuple(cols), rule)
    best, run, best_t = 0, 0, -1
    for t in range(t_max + 1):
        if all(col[c][t] == 0 for c in cols):
            run += 1
            if run > best:
                best, best_t = run, t - run + 1
        else:
            run = 0
    return {"cols": list(cols), "t_max": t_max, "rule": rule,
            "longest_white_run": best, "starts_at": best_t}


# ================================== SECTION 6/8: factor-complexity bridge ===
def column_factor_counts(t_max: int, h_list, rule: int = 30):
    """p_0(n)   = # distinct length-n factors of the centre column
       p_01(h)  = # distinct length-h factors of the width-2 column pair
    Both are counted inside a FINITE run, so both are LOWER bounds on the
    true factor complexities.  They are never upper bounds."""
    col = columns_bitparallel(t_max, (0, 1), rule)
    c0, c1 = col[0], col[1]
    out = []
    for h in h_list:
        s0 = set()
        for t in range(t_max + 1 - (h + 1)):
            s0.add(tuple(c0[t:t + h + 1]))
        s01 = set()
        for t in range(t_max + 1 - (h + 1)):
            s01.add(tuple(c0[t:t + h]) + tuple(c1[t:t + h]))
        out.append({"h": h, "p0(h+1)": len(s0), "p01(h)": len(s01),
                    "ratio": round(len(s01) / max(1, len(s0)), 4)})
    return out


# ================================ SECTION 11: rule 90 as negative control ===
def rule90_centre_column(t_max: int) -> Dict[str, object]:
    """Rule 90 from a single cell: the CENTRE column is eventually periodic
    (it is 1,0,0,0,...).  Rule 90 is left-permutive, so T-01/T-02/T-03 all
    apply to it.  Hence T-03 ('at most one eventually periodic column') is
    ATTAINED, and no theorem of that family can settle Problem 1."""
    col = columns_bitparallel(t_max, (0, 1, 2, 3, 4), 90)
    col0 = col[0]
    nonzero = [t for t in range(t_max + 1) if col0[t]]
    # neighbouring columns: count distinct length-n factors as evidence of
    # NON-periodicity (a lower bound only -- never a proof)
    cols = {}
    for j in (1, 2, 3, 4):
        cj = col[j]
        n = 24
        fac = {tuple(cj[t:t + n]) for t in range(t_max + 1 - n)}
        cols[j] = {"distinct_factors_len24": len(fac),
                   "ones": sum(cj)}
    return {"t_max": t_max,
            "centre_column_nonzero_times": nonzero,
            "centre_column_eventually_zero": nonzero == [0],
            "neighbour_columns": cols}


def main():
    import json, os
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    out = {}

    print("=== 0. cross-check of the two implementations ===")
    out["crosscheck"] = [crosscheck_implementations(400, 30),
                         crosscheck_implementations(400, 90)]
    for r in out["crosscheck"]:
        print("  rule %d: %d cells compared, %d disagreements"
              % (r["rule"], r["cells_compared"], r["disagreements"]))
    out["forward_collisions"] = [forward_collisions(8, 30),
                                 forward_collisions(8, 90)]
    for r in out["forward_collisions"]:
        print("  rule %d one-step map on 8-cell rows injective: %s"
              % (r["rule"], r["map_is_injective"]))

    print("\n=== 2. minimal backward-determining boundary ===")
    out["candidate_B"] = [candidate_B_region(h) for h in range(2, 11)]
    for r in out["candidate_B"]:
        print("  h=%2d  distinct W2 words=%7d  tops/W2 in [%d,%d]  "
              "cells determined=%3d (predicted %3d)  violations=%d  sharp=%s"
              % (r["h"], r["distinct_width2_words"], r["min_tops_per_width2"],
                 r["max_tops_per_width2"], r["cells_claimed_determined"],
                 r["predicted_count_h_h_minus_1_over_2"],
                 len(r["determination_violations"]), r["envelope_is_sharp"]))
    out["candidate_B_rule90"] = [candidate_B_region(h, 90) for h in range(2, 9)]
    out["candidate_C_prime_rule90"] = [candidate_C_right_edge(h, 90, 2)
                                       for h in range(2, 9)]
    out["diagonal_recursion"] = [diagonal_recursion(T)
                                 for T in (40, 120, 300)]
    for r in out["diagonal_recursion"]:
        print("  DIAG T=%3d: %d cells checked, %d recursion failures, "
              "centre column %d rows, %d failures; E0 zero on first half: %s"
              % (r["T"], r["recursion_cells_checked"], r["recursion_failures"],
                 r["centre_rows_reconstructed"], r["centre_failures"],
                 r["E0_zero_on_first_half"]))
    out["candidate_C"] = [candidate_C_right_edge(h, 30, 1) for h in range(2, 11)]
    out["candidate_C_prime"] = [candidate_C_right_edge(h, 30, 2)
                                for h in range(2, 11)]
    for lbl, key in (("C  (1 diagonal )", "candidate_C"),
                     ("C' (2 diagonals)", "candidate_C_prime")):
        for r in out[key]:
            print("  %s h=%2d  boundary words=%6d  tops/word <= %6d  "
                  "determines centre word: %s"
                  % (lbl, r["h"], r["distinct_boundary_words"],
                     r["max_tops_per_boundary_word"],
                     r["determines_centre_word_minus_apex"]))

    print("\n=== 3. reduced boundary word ===")
    out["reduced_boundary"] = reduced_boundary_lemma(20000)
    print("  ", out["reduced_boundary"])
    out["white_pair_runs"] = [widest_white_pair_run(20000, (0, 1)),
                              widest_white_pair_run(20000, (-1, 0)),
                              widest_white_pair_run(20000, (0, 1), 90)]
    for r in out["white_pair_runs"]:
        print("  ", r)

    print("\n=== 6/8. factor counts in a finite run (lower bounds only) ===")
    out["factor_counts"] = column_factor_counts(60000, [4, 8, 12, 16, 20, 24])
    for r in out["factor_counts"]:
        print("  ", r)

    print("\n=== 11. rule 90 negative control ===")
    out["rule90_control"] = rule90_centre_column(4000)
    out["rule90_column1"] = rule90_column1_ones(4000)
    r90 = out["rule90_column1"]
    print("   rule 90 column 1 ones at %s ... closed form 2^n-1 matches: %s ;"
          " largest gap %d"
          % (r90["ones_at"][:6], r90["closed_form_matches"], r90["largest_gap"]))
    r = out["rule90_control"]
    print("   centre column nonzero at t in %s ; eventually zero: %s"
          % (r["centre_column_nonzero_times"][:5],
             r["centre_column_eventually_zero"]))
    for j, d in r["neighbour_columns"].items():
        print("   column %s: %d distinct length-24 factors, %d ones"
              % (j, d["distinct_factors_len24"], d["ones"]))

    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open("results/phase2h_boundary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/phase2h_boundary.json (%.1f s)" % out["elapsed_sec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
