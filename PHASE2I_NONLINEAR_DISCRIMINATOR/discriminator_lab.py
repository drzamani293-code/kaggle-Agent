"""discriminator_lab.py -- Phase 2I sections 5, 6, 7.

The mandatory lesson from Phase 2H:

    any argument using only radius 1, left-permutivity, finite support, the
    light cone and the frozen edge cannot prove centre-column aperiodicity,
    because rule 90 has all of those and an eventually periodic single-seed
    centre column.

So every candidate property Pi here is scored against FOUR filters:

    F1  holds for rule 30, single-cell seed
    F2  FAILS for rule 90, single-cell seed          (rule discrimination)
    F3  FAILS for at least one other finite rule-30 seed  (seed discrimination)
    F4  rigorous path from Pi + eventual centre periodicity to a contradiction

F1-F3 are decidable inside a bounded range and are computed here.  F4 is never
computed; it is a proof obligation and is reported as OPEN unless discharged in
the documents.  A property passing only F2 is RULE-SPECIFIC; one passing only
F3 is SEED-SPECIFIC; the brief demands both.

Nothing in this file is a probabilistic or statistical test.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from nonlinear_lab import RULE_TABLES, T, T_row, orbit, make_nfield, make_xfield

# ------------------------------------------------------------- the seed suite ---
# All seeds are normalised to have min(support) = 0, so the left light-cone edge
# sits at column -t for every one of them and the towers use identical indexing.
BASE_SEEDS = {
    "A_single":      {0: 1},
    "D_two_adjacent": {0: 1, 1: 1},
    "E_101":         {0: 1, 2: 1},
    "E_111":         {0: 1, 1: 1, 2: 1},
    "E_1001":        {0: 1, 3: 1},
}


def random_seeds(n: int, width: int, rng_seed: int) -> Dict[str, Dict[int, int]]:
    """Section 5F: several randomly chosen finite seeds, ALL RECORDED.
    The RNG seed is fixed and stored so the suite is reproducible; no
    statistical claim is attached to these seeds -- they are extra witnesses."""
    rng = random.Random(rng_seed)
    out = {}
    i = 0
    while len(out) < n:
        i += 1
        bits = [rng.randint(0, 1) for _ in range(width)]
        if not any(bits):
            continue
        first = bits.index(1)
        seed = {k - first: b for k, b in enumerate(bits) if b}
        key = "F_rand%02d_%s" % (len(out) + 1,
                                 "".join(str(bits[k]) for k in
                                         range(first, width)))
        if key in out:
            continue
        out[key] = seed
    return out


def seed_suite(rng_seed: int = 20250805, n_random: int = 6, width: int = 9):
    s = dict(BASE_SEEDS)
    s.update(random_seeds(n_random, width, rng_seed))
    return s


# ------------------------------------------------------- candidate properties ---
def linear_shadow(seed: Dict[int, int], t_max: int) -> List[int]:
    """The rule-150 shadow of the centre column:  sum_d T(t,d) x_0(-d).
    This is the FIRST term of the Duhamel formula and depends on the seed only."""
    out = []
    for t in range(t_max + 1):
        acc = 0
        for d, v in T_row(t).items():
            if v:
                acc ^= seed.get(-d, 0)
        out.append(acc)
    return out


def P2_constant_shadow(seed: Dict[int, int], t_max: int) -> Dict[str, object]:
    sh = linear_shadow(seed, t_max)
    return {"constant": len(set(sh)) == 1, "value": sh[0],
            "first_deviation": next((t for t in range(len(sh))
                                     if sh[t] != sh[0]), None)}


def P1_forcing_degree(rule: int) -> int:
    """Algebraic degree over GF(2) of F in  x_{t+1} = L150 x_t + F(x_t).
        rule 30 : F(x)(j) = x(j)x(j+1)          degree 2
        rule 90 : F(x)(j) = x(j)                degree 1
        rule 150: F  = 0                        degree 0
    Computed, not asserted: the ANF of (rule(l,c,r) + l + c + r)."""
    f = RULE_TABLES[rule]
    # Moebius transform over the 3 variables of  g = f + l + c + r
    g = {}
    for l in (0, 1):
        for c in (0, 1):
            for r in (0, 1):
                g[(l, c, r)] = f(l, c, r) ^ l ^ c ^ r
    anf = dict(g)
    for bit in range(3):
        for k in list(anf):
            if k[bit] == 1:
                lo = list(k)
                lo[bit] = 0
                anf[k] ^= anf[tuple(lo)]
    deg = max([sum(k) for k, v in anf.items() if v], default=-1)
    monomials = sorted(k for k, v in anf.items() if v)
    return {"rule": rule, "degree": deg, "monomials": monomials}


def P4_mirror_defect(seed: Dict[int, int], t_max: int, rule: int
                     ) -> Dict[str, object]:
    """mu_t(r) := x_t(-r) + x_t(r), the coupled-tower mirror pair
    ( w_t(t-r), v_t(t-r) ) of section 6.  Rule 90 and rule 150 are symmetric
    local rules, so a symmetric seed gives mu == 0 identically."""
    g, C = orbit(seed, t_max, rule)
    xf = make_xfield(g, C)
    nz = 0
    first = None
    for t in range(t_max + 1):
        for r in range(1, t + 2):
            if xf(t, -r) ^ xf(t, r):
                nz += 1
                if first is None:
                    first = (t, r)
    return {"nonzero_cells": nz, "identically_zero": nz == 0,
            "first_nonzero": first}


def P8_n_minus1_identity(seed: Dict[int, int], t_max: int, rule: int
                         ) -> Dict[str, object]:
    """n_t(-1) = c_t * (1 + c_{t+1})  -- the nonlinear field one step left of
    the centre is a function of the CENTRE COLUMN ALONE.  Proved for rule 30
    in NONLINEAR_ACTIVITY_THEORY.md; checked here for every rule/seed."""
    g, C = orbit(seed, t_max, rule)
    xf = make_xfield(g, C)
    bad = 0
    for t in range(t_max):
        n = xf(t, -1) & xf(t, 0)
        pred = xf(t, 0) & (1 ^ xf(t + 1, 0))
        if n != pred:
            bad += 1
    return {"failures": bad, "holds": bad == 0, "rows": t_max}


def centre_column(seed: Dict[int, int], t_max: int, rule: int) -> List[int]:
    g, C = orbit(seed, t_max, rule)
    return [g[t][C] for t in range(t_max + 1)]


def n_event_count(seed: Dict[int, int], t_max: int, rule: int) -> List[int]:
    """|N(t)| : number of ACTIVE nonlinear events in the centre kernel.
    Section 8's skeleton size.  For a linear rule the field n is still defined
    (n = x(j)x(j+1)) but it does not drive the evolution; the count is reported
    for all rules so the comparison is like-for-like."""
    g, C = orbit(seed, t_max, rule)
    nf = make_nfield(g, C)
    out = []
    for t in range(t_max + 1):
        cnt = 0
        for s in range(t):
            for j, v in T_row(t - 1 - s).items():
                if v and nf(s, j):
                    cnt += 1
        out.append(cnt)
    return out
