"""defect_lab.py -- Phase 2B-GLOBAL: the temporal defect field of rule 30.

Setting.  x_t(j) is the rule-30 orbit of the single-cell seed.  For a fixed
lag p >= 1 define the *temporal defect field*

    d_t(j) = x_{t+p}(j) XOR x_t(j).

If the centre column were eventually p-periodic after time T then d_t(0) = 0
for every t >= T: an infinite "zero wall" at column 0.  Phase 2B asks whether
the real defect field can be compatible with such a wall.

Nothing here proves or disproves that.  Every routine is labelled with what it
does and does not establish; see PHASE2B_METHOD-style notes in the .md files.

Conventions are inherited unchanged from Phase 1 (rule30_lab.py):
    x_{t+1}(j) = x_t(j-1) XOR ( x_t(j) OR x_t(j+1) ),
    x_0(0) = 1, x_0(j) = 0 otherwise, sites increase to the right.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import rule30_lab as L

TABLE30 = L.rule_table(30)


def f30(l: int, c: int, r: int) -> int:
    return TABLE30[4 * l + 2 * c + r]


# ==========================================================================
# 1. The exact defect equations
# ==========================================================================


def delta(u: int, v: int, e: int, g: int) -> int:
    """The OR-difference term

        Delta(u,v,e,g) = ((u XOR e) OR (v XOR g)) XOR (u OR v)

    where u = x_t(j), v = x_t(j+1), e = d_t(j), g = d_t(j+1)."""
    return ((u ^ e) | (v ^ g)) ^ (u | v)


def delta_anf(u: int, v: int, e: int, g: int) -> int:
    """Algebraic normal form of Delta over GF(2):

        Delta = e + g + u*g + v*e + e*g          (all sums mod 2)"""
    return e ^ g ^ (u & g) ^ (v & e) ^ (e & g)


def defect_step_xor_or(dm: int, u: int, v: int, e: int, g: int) -> int:
    """d_{t+1}(j) = d_t(j-1) XOR [ ((x_t(j) XOR d_t(j)) OR (x_t(j+1) XOR d_t(j+1)))
                                   XOR (x_t(j) OR x_t(j+1)) ]"""
    return dm ^ delta(u, v, e, g)


def defect_step_anf(dm: int, u: int, v: int, e: int, g: int) -> int:
    """d_{t+1}(j) = d_t(j-1) + d_t(j) + d_t(j+1)
                    + x_t(j)*d_t(j+1) + x_t(j+1)*d_t(j) + d_t(j)*d_t(j+1)"""
    return dm ^ e ^ g ^ (u & g) ^ (v & e) ^ (e & g)


def verify_defect_identities() -> Dict[str, object]:
    """Exhaustive verification over ALL 2^6 assignments of the six underlying
    cells (x and y = x_{t+p} at sites j-1, j, j+1).  No sampling."""
    ok_xor = ok_anf = ok_delta = True
    rows = []
    for bits in range(64):
        xm, x0, xp, ym, y0, yp = [(bits >> i) & 1 for i in range(6)]
        dm, e, g = xm ^ ym, x0 ^ y0, xp ^ yp
        lhs = f30(ym, y0, yp) ^ f30(xm, x0, xp)          # d_{t+1}(j), by definition
        ok_xor &= lhs == defect_step_xor_or(dm, x0, xp, e, g)
        ok_anf &= lhs == defect_step_anf(dm, x0, xp, e, g)
    for u, v, e, g in itertools.product((0, 1), repeat=4):
        ok_delta &= delta(u, v, e, g) == delta_anf(u, v, e, g)
        rows.append({"x_t(j)": u, "x_t(j+1)": v, "d_t(j)": e, "d_t(j+1)": g,
                     "Delta": delta(u, v, e, g)})
    return {
        "assignments_checked_for_step": 64,
        "assignments_checked_for_delta": 16,
        "xor_or_form_exact": bool(ok_xor),
        "anf_form_exact": bool(ok_anf),
        "delta_anf_matches": bool(ok_delta),
        "delta_table": rows,
    }


def verify_zero_wall_reduction() -> Dict[str, object]:
    """With d_t(j) = 0 the OR-difference collapses:

        Delta(u, v, 0, g) = g * (1 XOR u).

    Exhaustive over the 8 remaining assignments."""
    ok = all(delta(u, v, 0, g) == (g & (1 ^ u))
             for u in (0, 1) for v in (0, 1) for g in (0, 1))
    ok2 = all(delta(u, v, e, 0) == (e & (1 ^ v))
              for u in (0, 1) for v in (0, 1) for e in (0, 1))
    return {"delta_with_e0_equals_g_and_not_u": bool(ok),
            "delta_with_g0_equals_e_and_not_v": bool(ok2),
            "assignments_checked": 8}


# ==========================================================================
# 2. The real orbit and its defect field
# ==========================================================================


def orbit_rows(t_max: int) -> Tuple[List[int], int]:
    """Rows 0..t_max of the single-cell orbit as integers; bit (R+j) is site j."""
    R = t_max + 2
    width = 2 * R + 1
    mask = (1 << width) - 1
    row = 1 << R
    rows = [row]
    for _ in range(t_max):
        row = ((row << 1) ^ (row | (row >> 1))) & mask
        rows.append(row)
    return rows, R


def defect_rows(p: int, t_max: int) -> Tuple[List[int], int]:
    """d_t = x_{t+p} XOR x_t for t = 0..t_max, as integers."""
    rows, R = orbit_rows(t_max + p)
    return [rows[t + p] ^ rows[t] for t in range(t_max + 1)], R


def bit(row: int, R: int, j: int) -> int:
    return (row >> (R + j)) & 1


def verify_defect_dynamics_on_orbit(p: int, t_max: int, span: int) -> Dict[str, object]:
    """Check the derived step equation cell by cell against the real orbit."""
    rows, R = orbit_rows(t_max + p + 1)
    d = [rows[t + p] ^ rows[t] for t in range(t_max + 2)]
    n = bad = 0
    for t in range(t_max):
        for j in range(-span, span + 1):
            got = bit(d[t + 1], R, j)
            want = defect_step_anf(bit(d[t], R, j - 1),
                                   bit(rows[t], R, j), bit(rows[t], R, j + 1),
                                   bit(d[t], R, j), bit(d[t], R, j + 1))
            n += 1
            bad += (got != want)
    return {"p": p, "cells_checked": n, "mismatches": bad, "exact": bad == 0}


# ==========================================================================
# 3. Zero-wall lemmas, checked where their hypotheses actually hold
# ==========================================================================


def check_wall_relation(p: int, t_max: int) -> Dict[str, object]:
    """THEOREM W1:  if d_t(0) = 0 and d_{t+1}(0) = 0 then
           d_t(-1) = d_t(1) * (1 XOR x_t(0)).
    Checked at every t in the real orbit where the hypothesis holds."""
    rows, R = orbit_rows(t_max + p + 1)
    d = [rows[t + p] ^ rows[t] for t in range(t_max + 2)]
    hits = viol = 0
    for t in range(t_max):
        if bit(d[t], R, 0) or bit(d[t + 1], R, 0):
            continue
        hits += 1
        lhs = bit(d[t], R, -1)
        rhs = bit(d[t], R, 1) & (1 ^ bit(rows[t], R, 0))
        viol += (lhs != rhs)
    return {"p": p, "t_max": t_max, "hypothesis_holds_at": hits,
            "violations": viol, "exact": viol == 0}


def check_ones_run_triangle(p: int, t_max: int, min_run: int = 3) -> Dict[str, object]:
    """THEOREM W3 (triangle):  if the wall holds on [t0, t1+1] and
    x_s(0) = 1 for all s in [t0, t1], then  d_s(-m) = 0 for every
    m = 1..(t1-t0+1) and every s in [t0, t1-(m-1)].

    Checked wherever the hypothesis occurs in the real orbit."""
    rows, R = orbit_rows(t_max + p + 1)
    d = [rows[t + p] ^ rows[t] for t in range(t_max + 2)]
    windows = viol = cells = 0
    longest = 0
    t = 0
    while t < t_max:
        if bit(d[t], R, 0) or bit(rows[t], R, 0) == 0:
            t += 1
            continue
        t0 = t
        while (t + 1 <= t_max and bit(d[t + 1], R, 0) == 0
               and bit(rows[t + 1], R, 0) == 1):
            t += 1
        t1 = t
        # need the wall one step beyond the run
        if t1 + 1 <= t_max and bit(d[t1 + 1], R, 0) == 0 and (t1 - t0 + 1) >= min_run:
            windows += 1
            longest = max(longest, t1 - t0 + 1)
            for m in range(1, t1 - t0 + 2):
                for s in range(t0, t1 - (m - 1) + 1):
                    cells += 1
                    viol += bit(d[s], R, -m)
        t += 1
    return {"p": p, "windows_found": windows, "longest_run": longest,
            "cells_checked": cells, "violations": viol, "exact": viol == 0}


def check_left_slaving(p: int, t_max: int) -> Dict[str, object]:
    """THEOREM W4: if the wall holds at t, t+1 and t+2 then
           d_t(-2) = d_{t+1}(1)*(1 XOR x_{t+1}(0)) XOR d_t(1)*(1 XOR x_t(0)).
    i.e. the second left column is determined by right-hand defects."""
    rows, R = orbit_rows(t_max + p + 2)
    d = [rows[t + p] ^ rows[t] for t in range(t_max + 3)]
    hits = viol = 0
    for t in range(t_max):
        if bit(d[t], R, 0) or bit(d[t + 1], R, 0) or bit(d[t + 2], R, 0):
            continue
        hits += 1
        lhs = bit(d[t], R, -2)
        rhs = (bit(d[t + 1], R, 1) & (1 ^ bit(rows[t + 1], R, 0))) ^ \
              (bit(d[t], R, 1) & (1 ^ bit(rows[t], R, 0)))
        viol += (lhs != rhs)
    return {"p": p, "hypothesis_holds_at": hits, "violations": viol,
            "exact": viol == 0}


# ==========================================================================
# 4. Defect geometry in the real orbit
# ==========================================================================


def row_runs(row: int, R: int, lo: int, hi: int) -> List[Tuple[int, int]]:
    """Maximal intervals of consecutive 1-bits of `row` within sites [lo, hi]."""
    out = []
    j = lo
    while j <= hi:
        if not bit(row, R, j):
            j += 1
            continue
        s = j
        while j <= hi and bit(row, R, j):
            j += 1
        out.append((s, j - 1))
    return out


def defect_geometry(p: int, t_max: int, sample_every: int = 1) -> Dict[str, object]:
    """Per-row geometry of the defect field, plus the support-edge facts."""
    rows, R = orbit_rows(t_max + p)
    edge_ok = True
    stats = []
    wall_zero = 0
    crossing_runs = 0
    total_runs = 0
    density_sum = 0.0
    n_rows = 0
    for t in range(0, t_max + 1, sample_every):
        d = rows[t + p] ^ rows[t]
        # THEOREM E1: the leftmost and rightmost defects sit exactly at -+(t+p)
        edge_ok &= bit(d, R, t + p) == 1 and bit(d, R, -(t + p)) == 1
        edge_ok &= bit(d, R, t + p + 1) == 0 and bit(d, R, -(t + p) - 1) == 0
        runs = row_runs(d, R, -(t + p), t + p)
        ndef = bin(d).count("1")
        crossing = [r for r in runs if r[0] <= 0 <= r[1]]
        total_runs += len(runs)
        crossing_runs += len(crossing)
        density_sum += ndef / (2 * (t + p) + 1)
        n_rows += 1
        if bit(d, R, 0) == 0:
            wall_zero += 1
        stats.append({
            "t": t,
            "leftmost": -(t + p), "rightmost": t + p,
            "n_defects": ndef,
            "n_runs": len(runs),
            "longest_run": max((b - a + 1 for a, b in runs), default=0),
            "d0": bit(d, R, 0), "dm1": bit(d, R, -1), "d1": bit(d, R, 1),
            "run_covering_0": crossing[0] if crossing else None,
        })
    return {
        "p": p, "t_max": t_max, "rows_sampled": n_rows,
        "support_edge_defect_theorem_holds": bool(edge_ok),
        "mean_defect_density": density_sum / n_rows,
        "rows_with_zero_wall": wall_zero,
        "zero_wall_fraction": wall_zero / n_rows,
        "total_runs": total_runs,
        "runs_covering_column_0": crossing_runs,
        "per_row": stats,
    }


def wall_run_lengths(p: int, t_max: int) -> Dict[str, object]:
    """Maximal runs of consecutive t with d_t(0) = 0 -- the finite walls that
    actually occur in the real orbit."""
    rows, R = orbit_rows(t_max + p)
    runs = []
    t = 0
    while t <= t_max:
        if bit(rows[t + p] ^ rows[t], R, 0):
            t += 1
            continue
        s = t
        while t <= t_max and bit(rows[t + p] ^ rows[t], R, 0) == 0:
            t += 1
        runs.append((s, t - s))
    lens = [n for _, n in runs]
    hist = {}
    for n in lens:
        hist[n] = hist.get(n, 0) + 1
    return {"p": p, "t_max": t_max, "n_walls": len(runs),
            "longest_wall": max(lens) if lens else 0,
            "longest_wall_start": max(runs, key=lambda r: r[1])[0] if runs else None,
            "mean_wall": sum(lens) / len(lens) if lens else 0.0,
            "length_histogram": {k: hist[k] for k in sorted(hist)}}


def crossing_events(p: int, t_max: int, k: int = 6) -> Dict[str, object]:
    """Relation between the centre word x_t(0..k-1 in time) and whether the
    defect run covering column 0 exists (a 'crossing').  Descriptive only."""
    rows, R = orbit_rows(t_max + p + k)
    counts = {}
    for t in range(t_max + 1):
        d = rows[t + p] ^ rows[t]
        word = "".join(str(bit(rows[t + s], R, 0)) for s in range(k))
        cross = 1 if bit(d, R, 0) else 0
        c = counts.setdefault(word, [0, 0])
        c[cross] += 1
    return {"p": p, "k": k,
            "words": {w: {"wall": c[0], "defect": c[1],
                          "defect_fraction": c[1] / (c[0] + c[1])}
                      for w, c in sorted(counts.items())}}


# ==========================================================================
# 5. The zero-wall automaton
# ==========================================================================


def wall_automaton(r: int) -> Dict[str, object]:
    """Nondeterministic automaton on states (X, Y) = (x_t(-r..r), x_{t+p}(-r..r)).

    A step consumes two unknown incoming bits on each side (one for x, one for
    y), so each state has 16 successors.  We keep only *wall states*, those with
    X[0] == Y[0] (i.e. d(0) = 0), and compute the maximal invariant subset --
    the states that have both an infinite forward and an infinite backward path
    inside the wall set.  That set is exactly the set of states that can occur
    on a bi-infinite wall in the RELAXED model (no seed).

    LABEL: this is a relaxation.  A nonempty invariant set does NOT imply that
    a wall occurs in the single-cell orbit.  An EMPTY invariant set, however,
    would prove that no wall of any length beyond the transient exists at all --
    that is the outcome worth looking for.
    """
    w = 2 * r + 1
    mask = (1 << w) - 1
    centre = 1 << r          # bit of site 0 inside a window

    def step(win: int, left_in: int, right_in: int) -> int:
        """One rule-30 step on a window with two supplied incoming bits.
        Bit i of `win` is site (i - r)."""
        ext = (win << 1) | left_in            # now w+1 bits, sites -r-1..r
        ext |= right_in << (w + 1)            # sites -r-1..r+1
        out = 0
        for i in range(w):                    # output site (i - r)
            l = (ext >> i) & 1                # site i-r-1
            c = (ext >> (i + 1)) & 1          # site i-r
            rr = (ext >> (i + 2)) & 1         # site i-r+1
            out |= f30(l, c, rr) << i
        return out

    def is_wall(X: int, Y: int) -> bool:
        return ((X ^ Y) & centre) == 0

    states = [(X, Y) for X in range(1 << w) for Y in range(1 << w)
              if is_wall(X, Y)]
    index = {s: i for i, s in enumerate(states)}
    succ = [[] for _ in states]
    pred = [[] for _ in states]
    for i, (X, Y) in enumerate(states):
        for lx, rx, ly, ry in itertools.product((0, 1), repeat=4):
            X2, Y2 = step(X, lx, rx), step(Y, ly, ry)
            if is_wall(X2, Y2):
                j = index[(X2, Y2)]
                succ[i].append(j)
                pred[j].append(i)
    # maximal invariant set: iteratively drop states with no successor or no
    # predecessor inside the surviving set
    alive = [True] * len(states)
    changed = True
    while changed:
        changed = False
        for i in range(len(states)):
            if not alive[i]:
                continue
            if not any(alive[j] for j in succ[i]) or not any(alive[j] for j in pred[i]):
                alive[i] = False
                changed = True
    inv = [i for i in range(len(states)) if alive[i]]
    return {
        "radius": r,
        "window_bits": w,
        "wall_states_total": len(states),
        "maximal_invariant_states": len(inv),
        "invariant_is_empty": len(inv) == 0,
        "index": index,
        "states": states,
        "alive": alive,
        "succ": succ,
    }


def observed_wall_states(r: int, p_list: Sequence[int], t_max: int) -> Dict[int, set]:
    """Wall states (X, Y) that actually occur in the single-cell orbit."""
    out = {}
    w = 2 * r + 1
    for p in p_list:
        rows, R = orbit_rows(t_max + p)
        seen = set()
        for t in range(t_max + 1):
            X = (rows[t] >> (R - r)) & ((1 << w) - 1)
            Y = (rows[t + p] >> (R - r)) & ((1 << w) - 1)
            if ((X ^ Y) >> r) & 1:
                continue                       # not a wall state at this t
            seen.add((X, Y))
        out[p] = seen
    return out


# ==========================================================================
# 6. Edge-aligned coordinates
# ==========================================================================


def edge_word(rows: List[int], R: int, t: int, k_max: int = None) -> List[int]:
    """w_t(k) = x_t(-t + k) for 0 <= k <= 2t (0 beyond the support)."""
    n = 2 * t if k_max is None else min(2 * t, k_max)
    return [bit(rows[t], R, -t + k) for k in range(n + 1)]


def verify_edge_update(t_max: int, k_max: int) -> Dict[str, object]:
    """THEOREM EA1:  w_{t+1}(k) = w_t(k-2) XOR ( w_t(k-1) OR w_t(k) ),
    with w_t(m) = 0 for m < 0.  Note the rule became ONE-SIDED: the new value
    at k depends only on k-2, k-1, k -- never on k+1."""
    rows, R = orbit_rows(t_max + 1)
    n = bad = 0
    for t in range(1, t_max):
        w = edge_word(rows, R, t, k_max + 2)
        w2 = edge_word(rows, R, t + 1, k_max)

        def g(m):
            return w[m] if 0 <= m < len(w) else 0

        for k in range(min(k_max, len(w2) - 1) + 1):
            n += 1
            bad += (w2[k] != (g(k - 2) ^ (g(k - 1) | g(k))))
    return {"cells_checked": n, "mismatches": bad, "exact": bad == 0,
            "one_sided": True}


def prefix_period(K: int, t_max: int) -> Dict[str, object]:
    """The edge-aligned prefix W_t = (w_t(0..K)) evolves autonomously (EA1), so
    it is eventually periodic in t.  Returns its preperiod and period."""
    rows, R = orbit_rows(t_max + 1)
    seen = {}
    for t in range(t_max + 1):
        if 2 * t < K:
            continue
        w = tuple(edge_word(rows, R, t, K))
        if len(w) < K + 1:
            continue
        if w in seen:
            return {"K": K, "preperiod": seen[w], "period": t - seen[w],
                    "found": True}
        seen[w] = t
    return {"K": K, "preperiod": None, "period": None, "found": False,
            "t_max": t_max}


def centre_is_diagonal(t_max: int) -> Dict[str, object]:
    """THEOREM EA2:  x_t(0) = w_t(t).  The centre column is the DIAGONAL of the
    edge-aligned array, and the p-periodicity condition becomes
        w_{t+p}(t+p) = w_t(t)."""
    rows, R = orbit_rows(t_max + 1)
    ok = all(bit(rows[t], R, 0) == edge_word(rows, R, t, t)[t]
             for t in range(1, t_max + 1))
    return {"t_max": t_max, "centre_equals_diagonal": bool(ok)}
