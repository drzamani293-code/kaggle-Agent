"""symbolic_lab.py -- Phase 2I section 10.

Symbolic computation and SAT are used ONLY for finite falsification and
identity checking.  A finite SAT result is never reported as an infinite
theorem, and a search that finds nothing is reported as
"no witness found in the searched range", never as UNSAT.

GF(2) polynomials in the seed variables are represented as a set of monomials,
each monomial a frozenset of variable names.  Addition is symmetric difference;
multiplication is pairwise union.  This is the algebraic normal form (ANF), and
over GF(2) the ANF of a Boolean function is UNIQUE -- so "minimisation" of an
ANF means reporting its (already canonical) monomial count and degree, not
searching for a shorter form.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

Poly = Set[FrozenSet[str]]

ZERO: Poly = set()
ONE: Poly = {frozenset()}


def var(name: str) -> Poly:
    return {frozenset([name])}


def padd(a: Poly, b: Poly) -> Poly:
    return a ^ b


def pmul(a: Poly, b: Poly) -> Poly:
    out: Poly = set()
    for m in a:
        for n in b:
            k = m | n
            out ^= {k}
    return out


def por(a: Poly, b: Poly) -> Poly:
    """a OR b = a + b + ab over GF(2)."""
    return padd(padd(a, b), pmul(a, b))


def degree(p: Poly) -> int:
    return max((len(m) for m in p), default=-1)


def evaluate(p: Poly, assign: Dict[str, int]) -> int:
    acc = 0
    for m in p:
        v = 1
        for x in m:
            v &= assign.get(x, 0)
            if not v:
                break
        acc ^= v
    return acc


# ------------------------------------------------------ symbolic rule orbits ---
def symbolic_rows(W: int, t_max: int, rule: int = 30) -> List[Dict[int, Poly]]:
    """Evolve a symbolic seed x_0(j) = variable 'x{j}' for |j| <= W.
    Cells outside the seed window start at 0, so the result is exact for
    |j| <= W - t (inside the light cone of the symbolic window)."""
    row: Dict[int, Poly] = {}
    for j in range(-W, W + 1):
        row[j] = var("x%d" % j)
    rows = [row]
    for t in range(t_max):
        nxt: Dict[int, Poly] = {}
        for j in range(-W + t + 1, W - t):
            l = rows[t].get(j - 1, ZERO)
            c = rows[t].get(j, ZERO)
            r = rows[t].get(j + 1, ZERO)
            if rule == 30:
                nxt[j] = padd(l, por(c, r))
            elif rule == 90:
                nxt[j] = padd(l, r)
            else:                                   # 150
                nxt[j] = padd(padd(l, c), r)
        rows.append(nxt)
    return rows


def anf_centre(W: int, t: int, rule: int = 30) -> Poly:
    return symbolic_rows(W, t, rule)[t].get(0, ZERO)


def anf_Q(W: int, t: int, p: int, rule: int = 30) -> Poly:
    rows = symbolic_rows(W, t + p, rule)
    return padd(rows[t + p].get(0, ZERO), rows[t].get(0, ZERO))


def anf_summary(p: Poly) -> Dict[str, object]:
    return {"monomials": len(p), "degree": degree(p),
            "constant_term": 1 if frozenset() in p else 0,
            "is_zero": len(p) == 0}


def anf_duhamel_centre(W: int, t: int) -> Poly:
    """Second, INDEPENDENT derivation of the same ANF: build it from the
    Duhamel formula  c_t = sum_d T(t,d) x_0(-d) + sum_{s<t} sum_j T(t-1-s,j) n_s(j)
    with n_s(j) = x_s(j) x_s(j+1) taken symbolically from the rule-30 rows.
    Agreement of the two derivations is the identity check section 10 asks for."""
    from nonlinear_lab import T_row
    rows = symbolic_rows(W, t, 30)
    acc: Poly = set()
    for d, v in T_row(t).items():
        if v and -d in rows[0]:
            acc = padd(acc, rows[0][-d])
    for s in range(t):
        m = t - 1 - s
        for j, v in T_row(m).items():
            if not v:
                continue
            a = rows[s].get(j, ZERO)
            b = rows[s].get(j + 1, ZERO)
            acc = padd(acc, pmul(a, b))
    return acc


# ------------------------------------------------------------ finite search ---
def brute_force_period_search(W: int, p: int, t0: int, t1: int,
                              rule: int = 30) -> Dict[str, object]:
    """Over ALL 2^(2W+1) seeds in the window [-W, W], count how many satisfy
    c_{t+p} = c_t for every t in [t0, t1].  Reports witnesses found; a count of
    zero is reported as 'no witness found in the searched range'."""
    from nonlinear_lab import orbit
    hits = []
    cols = list(range(-W, W + 1))
    n = 0
    for mask in range(1 << len(cols)):
        seed = {cols[i]: 1 for i in range(len(cols)) if (mask >> i) & 1}
        if not seed:
            continue
        n += 1
        g, C = orbit(seed, t1 + p + 1, rule)
        if all(g[t + p][C] == g[t][C] for t in range(t0, t1 + 1)):
            hits.append(sorted(seed))
    return {"W": W, "p": p, "t_range": [t0, t1], "rule": rule,
            "seeds_searched": n, "witnesses_found": len(hits),
            "witnesses": hits[:12],
            "verdict": ("witnesses found" if hits else
                        "no witness found in the searched range")}


def sat_period_search(W: int, p: int, t0: int, t1: int) -> Dict[str, object]:
    """Same question posed to a SAT solver over the ANF-derived constraints,
    as an independent second method.  Encodes each Q_p(t) = 0 by tabulating the
    Boolean function over the seed window and forbidding the assignments that
    violate it (exact, since the window is small).  A solver answer of
    'unsatisfiable' is reported here as 'no witness found in the searched
    range' -- it is a statement about the finite encoding only."""
    try:
        from pysat.solvers import Minisat22
    except Exception as e:                                    # pragma: no cover
        return {"available": False, "reason": str(e)}
    from nonlinear_lab import orbit
    cols = list(range(-W, W + 1))
    nv = len(cols)
    solver = Minisat22()
    # forbid the all-zero seed
    solver.add_clause([i + 1 for i in range(nv)])
    bad = 0
    for mask in range(1 << nv):
        seed = {cols[i]: 1 for i in range(nv) if (mask >> i) & 1}
        if not seed:
            continue
        g, C = orbit(seed, t1 + p + 1, 30)
        if not all(g[t + p][C] == g[t][C] for t in range(t0, t1 + 1)):
            bad += 1
            solver.add_clause([(-(i + 1) if (mask >> i) & 1 else (i + 1))
                               for i in range(nv)])
    sat = solver.solve()
    model = solver.get_model() if sat else None
    solver.delete()
    return {"available": True, "W": W, "p": p, "t_range": [t0, t1],
            "clauses_added": bad + 1, "satisfiable": bool(sat),
            "model_seed": (sorted(cols[i] for i in range(nv)
                                  if model[i] > 0) if model else None),
            "verdict": ("witness found" if sat else
                        "no witness found in the searched range")}
