"""reversibility_lab.py -- Phase 2H: conditional backward determinism.

Coordinates (original):  x_{t+1}(j) = x_t(j-1) XOR ( x_t(j) OR x_t(j+1) ).

Backward step (left-permutivity, Phase 2F T-01):

    x_s(j-1) = x_{s+1}(j) XOR ( x_s(j) OR x_s(j+1) ).                    (BW)

(BW) determines x_s(j-1) from ONE cell of the later row s+1 and TWO cells of
row s itself, both strictly to the right.  So within a row the recursion runs
RIGHT TO LEFT and must be seeded on the right.

Nothing here is taken from any paper; every statement is derived from the local
rule.  Rowland's texts could not be retrieved (see RETRIEVAL_NOTE.md).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

UNKNOWN = None


def f30(l, c, r):
    return l ^ (c | r)


def f90(l, c, r):
    return l ^ r


RULES = {30: f30, 90: f90}


# ---------------------------------------------------------------- forward ---
def evolve_rows(top: Sequence[int], h: int, rule=30) -> List[List[Optional[int]]]:
    """Forward light cone: row s has width len(top) - 2s, indices shrink by one
    on each side.  Returns rows as lists (row s occupies absolute columns
    [s, len(top)-1-s])."""
    f = RULES[rule]
    rows = [list(top)]
    cur = list(top)
    for _ in range(h):
        nxt = [f(cur[i - 1], cur[i], cur[i + 1]) for i in range(1, len(cur) - 1)]
        rows.append(nxt)
        cur = nxt
    return rows


# --------------------------------------------------------------- backward ---
def or_determined(a, b):
    """Is (a OR b) determined by what we know?  a,b may be None (unknown)."""
    if a == 1 or b == 1:
        return 1
    if a == 0 and b == 0:
        return 0
    return UNKNOWN


def backward_row(later: Dict[int, int], seed: Dict[int, int],
                 j_hi: int, j_lo: int) -> Dict[int, int]:
    """One backward step.

    `later`  : known cells of row s+1, keyed by column.
    `seed`   : already-known cells of row s (the right-hand seed), keyed by col.
    Fills row s from column j_hi-1 down to j_lo using (BW), stopping as soon as
    a required input is missing.  Returns the cells of row s that got
    determined (including the seed cells passed in)."""
    out = dict(seed)
    j = j_hi
    while j >= j_lo:
        o = or_determined(out.get(j), out.get(j + 1))
        if o is UNKNOWN or (j not in later):
            break
        out[j - 1] = later[j] ^ o
        j -= 1
    return out


def backward_cone(bottom: Dict[int, int], white_cols: Sequence[int],
                  t_bot: int, h: int) -> Dict[Tuple[int, int], int]:
    """Reconstruct backward from row t_bot up to row t_bot-h.

    `bottom`     : known cells of row t_bot, keyed by column.
    `white_cols` : columns assumed 0 on EVERY row in [t_bot-h, t_bot].
    Returns {(row, col): value} for everything determined."""
    m = min(white_cols)
    known: Dict[Tuple[int, int], int] = {}
    for s in range(t_bot - h, t_bot + 1):
        for c in white_cols:
            known[(s, c)] = 0
    for c, v in bottom.items():
        known[(t_bot, c)] = v
    for s in range(t_bot - 1, t_bot - h - 1, -1):
        later = {c: known[(s + 1, c)] for (r, c) in known if r == s + 1}
        seed = {c: 0 for c in white_cols}
        lo = min(c for (r, c) in known if r == s + 1) - 1
        row = backward_row(later, seed, m, lo)
        for c, v in row.items():
            known[(s, c)] = v
    return known


def verify_backward_on_orbit(t_bot: int, h: int, m: int, width: int,
                             rule=30) -> Dict[str, object]:
    """Run the backward reconstruction inside the real single-cell orbit, where
    the white tail is supplied by the light cone (x_t(j)=0 for j>t)."""
    f = RULES[rule]
    C = t_bot + 4
    grid = [[0] * (2 * C + 1) for _ in range(t_bot + 1)]
    grid[0][C] = 1
    for t in range(t_bot):
        for j in range(1, 2 * C):
            grid[t + 1][j] = f(grid[t][j - 1], grid[t][j], grid[t][j + 1])
    val = lambda t, j: grid[t][j + C]
    # white columns m, m+1 must really be white on every row in the range
    ok_white = all(val(s, c) == 0
                   for s in range(t_bot - h, t_bot + 1) for c in (m, m + 1))
    bottom = {c: val(t_bot, c) for c in range(m - width, m + 2)}
    known = backward_cone(bottom, [m, m + 1], t_bot, h)
    bad = [(s, c) for (s, c), v in known.items() if val(s, c) != v]
    per_row = {}
    for (s, c) in known:
        per_row[s] = per_row.get(s, 0) + 1
    return {"t_bot": t_bot, "h": h, "m": m, "width": width,
            "white_strip_really_white": ok_white,
            "cells_reconstructed": len(known), "mismatches": len(bad),
            "exact": not bad, "cells_per_row": dict(sorted(per_row.items()))}


# --------------------------------------------------- white regions in orbit --
def orbit_grid(t_max: int, rule=30):
    f = RULES[rule]
    C = t_max + 2
    grid = [[0] * (2 * C + 1) for _ in range(t_max + 1)]
    grid[0][C] = 1
    for t in range(t_max):
        row, nxt = grid[t], grid[t + 1]
        for j in range(1, 2 * C):
            nxt[j] = f(row[j - 1], row[j], row[j + 1])
    return grid, C


def white_strips(t_max: int, min_h: int, w: int, rule=30):
    """All maximal (time-interval, column-pair) rectangles of zeros of width w
    and height >= min_h, inside the light cone.  These are the seeds that make
    backward reconstruction possible."""
    grid, C = orbit_grid(t_max, rule)
    found = []
    for m in range(-t_max, t_max - w + 1):
        run = 0
        for t in range(t_max + 1):
            if abs(m) > t + 1 or abs(m + w - 1) > t + 1:
                run = 0
                continue
            if all(grid[t][m + i + C] == 0 for i in range(w)):
                run += 1
            else:
                if run >= min_h:
                    found.append({"col": m, "width": w,
                                  "t_start": t - run, "t_end": t - 1,
                                  "height": run})
                run = 0
        if run >= min_h:
            found.append({"col": m, "width": w, "t_start": t_max - run + 1,
                          "t_end": t_max, "height": run})
    return found
