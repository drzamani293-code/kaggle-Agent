"""tower_coupling_lab.py -- Phase 2I sections 6 and 7.

Left tower  :  w_t(k) = x_t(-t+k)        (seeds normalised to min support 0)
Right tower :  v_t(k) = x_t(t+M-k)       (M = max support)

Coupling.  Both towers name the same cell when the columns agree:
        w_t(k1) = v_t(k2)   <==>   -t + k1 = t + M - k2   <==>  k1 + k2 = 2t + M.
For M = 0 this is k1 + k2 = 2t, and in particular

        w_t(t-r) = x_t(-r),      v_t(t-r) = x_t(+r),      c_t = w_t(t) = v_t(t).

So the natural coupled object is the MIRROR PAIR ( w_t(t-r), v_t(t-r) ) and its
defect

        delta_t(r) := x_t(-r) + x_t(r) = w_t(t-r) + v_t(t-r).

Rules 90 and 150 are symmetric local rules (f(l,c,r) = f(r,c,l)), so they
commute with the mirror and a symmetric seed gives delta == 0 for all time.
Rule 30 is not symmetric, and delta is driven exactly by the mirror defect of
the nonlinear field -- see DUAL_TOWER_COUPLING.md.

Section 7 measures INFORMATION LOSS as a finite combinatorial count only:
preimage counts of the left tower map along the ACTUAL orbit.  No probabilistic
notion of entropy is used anywhere.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from nonlinear_lab import RULE_TABLES, orbit, make_xfield


# ------------------------------------------------------------- tower maps ---
def F_left(s: Tuple[int, ...], rule: int = 30) -> Tuple[int, ...]:
    """w_{t+1}(k) from w_t, derived from the local rule in left-aligned
    coordinates.  Rule 30:  w_t(k-2) XOR ( w_t(k-1) OR w_t(k) ).
    Rule 90 :  w_t(k-2) XOR w_t(k).      Rule 150:  w_t(k-2) XOR w_t(k-1) XOR w_t(k)."""
    f = RULE_TABLES[rule]
    K = len(s) - 1
    return tuple(f(s[k - 2] if k >= 2 else 0,
                   s[k - 1] if k >= 1 else 0,
                   s[k]) for k in range(K + 1))


def G_right(s: Tuple[int, ...], rule: int = 30) -> Tuple[int, ...]:
    """v_{t+1}(k) from v_t.  Rule 30: v_t(k) XOR ( v_t(k-1) OR v_t(k-2) )."""
    f = RULE_TABLES[rule]
    K = len(s) - 1
    return tuple(f(s[k],
                   s[k - 1] if k >= 1 else 0,
                   s[k - 2] if k >= 2 else 0) for k in range(K + 1))


def tower_agreement(seed: Dict[int, int], K: int, t_max: int, rule: int
                    ) -> Dict[str, object]:
    """Both tower recurrences really are the rule, on the real orbit."""
    g, C = orbit(seed, t_max, rule)
    xf = make_xfield(g, C)
    M = max(seed) if seed else 0
    wl = tuple(1 if xf(0, k) else 0 for k in range(K + 1))       # w_0(k)=x_0(k)
    vr = tuple(1 if xf(0, M - k) else 0 for k in range(K + 1))   # v_0(k)=x_0(M-k)
    badw = badv = 0
    for t in range(t_max):
        for k in range(K + 1):
            if wl[k] != xf(t, -t + k):
                badw += 1
            if vr[k] != xf(t, t + M - k):
                badv += 1
        wl = F_left(wl, rule)
        vr = G_right(vr, rule)
    return {"rule": rule, "K": K, "t_max": t_max,
            "left_disagreements": badw, "right_disagreements": badv,
            "cells": t_max * (K + 1)}


# ------------------------------------------------- section 7: information loss ---
def preimage_table(K: int, rule: int, which: str = "left") -> Dict[Tuple[int, ...], int]:
    """Exact preimage COUNT of every state under the level-K tower map.
    This is a finite combinatorial quantity: |F_K^{-1}(u)|.  Summing to 2^(K+1)
    is an invariant checked by the test suite."""
    step = F_left if which == "left" else G_right
    cnt: Dict[Tuple[int, ...], int] = {}
    for m in range(1 << (K + 1)):
        s = tuple((m >> i) & 1 for i in range(K + 1))
        u = step(s, rule)
        cnt[u] = cnt.get(u, 0) + 1
    return cnt


def image_size(K: int, rule: int, which: str = "left") -> int:
    return len(preimage_table(K, rule, which))


def kappa_along_orbit(seed: Dict[int, int], K: int, t_max: int, rule: int,
                      which: str = "left") -> Dict[str, object]:
    """kappa(t) := | (tower map)^{-1} ( state at time t+1 ) |  along the ACTUAL
    single-seed orbit.  This is the seed-dependent part of the information-loss
    accounting: the tower map is a property of the rule, but WHICH collision
    classes the real orbit visits is a property of the seed."""
    tbl = preimage_table(K, rule, which)
    g, C = orbit(seed, t_max, rule)
    xf = make_xfield(g, C)
    M = max(seed) if seed else 0
    ks = []
    for t in range(1, t_max + 1):
        if which == "left":
            st = tuple(xf(t, -t + k) for k in range(K + 1))
        else:
            st = tuple(xf(t, t + M - k) for k in range(K + 1))
        ks.append(tbl.get(st, 0))
    hist: Dict[int, int] = {}
    for v in ks:
        hist[v] = hist.get(v, 0) + 1
    return {"rule": rule, "K": K, "which": which, "t_max": t_max,
            "kappa_values": sorted(hist),
            "histogram": {str(k): v for k, v in sorted(hist.items())},
            "distinct_classes_visited": len(hist),
            "always_one": set(hist) == {1},
            "first_20": ks[:20]}


# ------------------------------------------------- section 6: mirror coupling ---
def mirror_defect(seed: Dict[int, int], t_max: int, rule: int
                  ) -> Dict[str, object]:
    """delta_t(r) = x_t(-r) + x_t(r) = w_t(t-r) + v_t(t-r)."""
    g, C = orbit(seed, t_max, rule)
    xf = make_xfield(g, C)
    rows = []
    nz = 0
    for t in range(t_max + 1):
        d = [xf(t, -r) ^ xf(t, r) for r in range(0, t + 2)]
        nz += sum(d)
        rows.append(d)
    return {"rule": rule, "nonzero": nz, "identically_zero": nz == 0,
            "delta_t_0_always_zero": all(row[0] == 0 for row in rows)}


def seed_is_symmetric(seed: Dict[int, int]) -> bool:
    """Symmetric about column 0 after recentring on the support midpoint."""
    cols = sorted(seed)
    if not cols:
        return True
    lo, hi = cols[0], cols[-1]
    if (lo + hi) % 2 != 0:
        return False
    c = (lo + hi) // 2
    return all(seed.get(c + d, 0) == seed.get(c - d, 0)
               for d in range(0, hi - c + 1))


def recentre(seed: Dict[int, int]) -> Dict[int, int]:
    cols = sorted(seed)
    if not cols:
        return {}
    c = (cols[0] + cols[-1]) // 2
    return {j - c: 1 for j in cols}


def mirror_driver_identity(seed: Dict[int, int], t_max: int
                           ) -> Dict[str, object]:
    """The exact evolution of the mirror defect for rule 30:

        delta_{t+1} = L150 delta_t + nu_t ,     nu_t(j) := n_t(j) + n_t(-j),

    with delta_0 = 0 for a mirror-symmetric seed.  Checked cell by cell."""
    g, C = orbit(seed, t_max, rule=30)
    xf = make_xfield(g, C)
    bad = 0
    tot = 0
    for t in range(t_max):
        for j in range(-(t + 2), t + 3):
            tot += 1
            d = lambda tt, jj: xf(tt, -jj) ^ xf(tt, jj)
            n = lambda tt, jj: xf(tt, jj) & xf(tt, jj + 1)
            lhs = d(t + 1, j)
            rhs = d(t, j - 1) ^ d(t, j) ^ d(t, j + 1) ^ n(t, j) ^ n(t, -j)
            if lhs != rhs:
                bad += 1
    return {"cells": tot, "failures": bad, "holds": bad == 0}
