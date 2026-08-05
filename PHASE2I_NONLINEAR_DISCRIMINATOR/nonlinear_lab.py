"""nonlinear_lab.py -- Phase 2I sections 1, 2, 3.

Rule 30 over GF(2).  Writing  a OR b = a + b + ab  (GF(2) arithmetic),

    x_{t+1}(j) = x_t(j-1) XOR ( x_t(j) OR x_t(j+1) )
               = x_t(j-1) + x_t(j) + x_t(j+1) + x_t(j) x_t(j+1)
               = (L x_t)(j)  +  n_t(j)                                  (R30)

where  L  is the RULE 150 operator  (Lx)(j) = x(j-1) + x(j) + x(j+1)
and    n_t(j) = x_t(j) x_t(j+1)                     is the NONLINEAR FIELD.

Rule 90 is  (L90 x)(j) = x(j-1) + x(j+1);  it and rule 150 are linear, so for
them n is identically absent.  That is the discriminating property this whole
phase is required to use.

Laurent-polynomial picture.  Encode a row as  X(z) = sum_j x(j) z^j.  Then L is
multiplication by  (z + 1 + z^{-1}) = z^{-1}(1 + z + z^2), so

    L^m  <->  z^{-m} (1 + z + z^2)^m ,
    (L^m x)(j) = sum_d T(m,d) x(j-d) ,     T(m,d) := [z^{d+m}] (1+z+z^2)^m.

T(m,.) is the mod-2 TRINOMIAL kernel, supported on |d| <= m, and symmetric.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# ------------------------------------------------------------------ rules ---
RULE_TABLES = {
    30:  lambda l, c, r: l ^ (c | r),
    90:  lambda l, c, r: l ^ r,
    150: lambda l, c, r: l ^ c ^ r,
}


def f30_gf2(l, c, r):
    """The GF(2) form -- must agree with the boolean form everywhere."""
    return (l + c + r + c * r) % 2


def nonlinear_bit(c, r):
    return c * r


# ------------------------------------------------------- orbits (explicit) ---
def orbit(seed: Dict[int, int], t_max: int, rule: int = 30):
    """seed maps column -> bit.  Returns (grid, C) with grid[t][j+C] = x_t(j)."""
    f = RULE_TABLES[rule]
    span = max(abs(j) for j in seed) if seed else 0
    C = t_max + span + 3
    grid = [[0] * (2 * C + 1) for _ in range(t_max + 1)]
    for j, b in seed.items():
        grid[0][j + C] = b
    for t in range(t_max):
        row, nxt = grid[t], grid[t + 1]
        for j in range(1, 2 * C):
            nxt[j] = f(row[j - 1], row[j], row[j + 1])
    return grid, C


def n_field(grid, C, t_max):
    """n_t(j) = x_t(j) x_t(j+1), same indexing as grid."""
    out = []
    for t in range(t_max + 1):
        row = grid[t]
        out.append([row[i] & row[i + 1] for i in range(len(row) - 1)] + [0])
    return out


# ------------------------------------------------- the mod-2 trinomial kernel ---
def trinomial_row(m: int) -> List[int]:
    """Coefficients of (1+z+z^2)^m mod 2, index e = 0 .. 2m.  Built by the
    Frobenius product over the binary digits of m:

        (1+z+z^2)^m = prod_{i : bit i of m = 1} (1 + z^{2^i} + z^{2^{i+1}}).

    Verified against direct polynomial powering in run_phase2i_tests.py."""
    poly = [1]
    i = 0
    mm = m
    while mm:
        if mm & 1:
            step = 1 << i
            new = [0] * (len(poly) + 2 * step)
            for e, cv in enumerate(poly):
                if cv:
                    new[e] ^= 1
                    new[e + step] ^= 1
                    new[e + 2 * step] ^= 1
            poly = new
        mm >>= 1
        i += 1
    return poly


def T(m: int, d: int) -> int:
    """T(m,d) = [z^{d+m}] (1+z+z^2)^m mod 2 ; the L^m Green function."""
    if m < 0 or abs(d) > m:
        return 0
    row = trinomial_row(m)
    e = d + m
    return row[e] if 0 <= e < len(row) else 0


def T_row(m: int) -> Dict[int, int]:
    """{ d : T(m,d) } over the full support -m .. m."""
    row = trinomial_row(m)
    return {e - m: row[e] for e in range(len(row))}


def T_support(m: int) -> List[int]:
    return [d for d, v in T_row(m).items() if v]


def hyperbinary_count(e: int, bits: Sequence[int]) -> int:
    """Number of ways to write e = sum_i e_i 2^{a_i} with e_i in {0,1,2},
    a_i ranging over `bits`.  T(m, d) is this count mod 2 with e = d+m and
    `bits` = the set bits of m.  Used as an INDEPENDENT second derivation."""
    ways = {0: 1}
    for a in bits:
        nxt: Dict[int, int] = {}
        step = 1 << a
        for v, c in ways.items():
            for k in (0, 1, 2):
                nxt[v + k * step] = nxt.get(v + k * step, 0) + c
        ways = nxt
    return ways.get(e, 0)


# ---------------------------------------------------- the Duhamel formula ---
def duhamel_cell(seed: Dict[int, int], nfield, t: int, j: int) -> int:
    """x_t(j) = sum_d T(t,d) x_0(j-d)  +  sum_{s<t} sum_d T(t-1-s,d) n_s(j-d).

    `nfield` is a callable (s, j) -> n_s(j).  Returns the GF(2) value."""
    acc = 0
    for d, v in T_row(t).items():
        if v:
            acc ^= seed.get(j - d, 0)
    for s in range(t):
        m = t - 1 - s
        for d, v in T_row(m).items():
            if v:
                acc ^= nfield(s, j - d)
    return acc


def duhamel_centre(seed: Dict[int, int], nfield, t: int) -> int:
    """c_t = x_t(0) = sum_d T(t,d) x_0(-d) + sum_{s<t} sum_j T(t-1-s,j) n_s(j),
    using the symmetry T(m,-j) = T(m,j)."""
    acc = 0
    for d, v in T_row(t).items():
        if v:
            acc ^= seed.get(-d, 0)
    for s in range(t):
        m = t - 1 - s
        for j, v in T_row(m).items():
            if v:
                acc ^= nfield(s, j)
    return acc


def active_events(seed: Dict[int, int], nfield, t: int) -> List[Tuple[int, int]]:
    """N(t) = { (s,j) : T(t-1-s, j) = 1 and n_s(j) = 1 }  -- the NONLINEAR
    SKELETON of the centre bit c_t (section 8)."""
    out = []
    for s in range(t):
        m = t - 1 - s
        for j, v in T_row(m).items():
            if v and nfield(s, j):
                out.append((s, j))
    return out


def kernel_support_centre(t: int) -> List[Tuple[int, int]]:
    """The kernel support of c_t BEFORE looking at the orbit:
    { (s,j) : T(t-1-s, j) = 1 }.  Section 8 asks for this separately."""
    out = []
    for s in range(t):
        for j in T_support(t - 1 - s):
            out.append((s, j))
    return out


# ------------------------------------------------------- helper for orbits ---
def make_nfield(grid, C):
    def nf(s, j):
        i = j + C
        if s < 0 or s >= len(grid) or i < 0 or i + 1 >= len(grid[0]):
            return 0
        return grid[s][i] & grid[s][i + 1]
    return nf


def make_xfield(grid, C):
    def xf(t, j):
        i = j + C
        if t < 0 or t >= len(grid) or i < 0 or i >= len(grid[0]):
            return 0
        return grid[t][i]
    return xf


# =============================== section 3: the periodicity constraint Q ===
def Q(seed: Dict[int, int], nfield, p: int, t: int) -> int:
    """Q_p(t) := c_{t+p} + c_t  (GF(2)).  Eventual p-periodicity of the centre
    is exactly  Q_p(t) = 0 for all large t.  The point of section 3 is the
    CANONICAL FORM below, not this definition."""
    return duhamel_centre(seed, nfield, t + p) ^ duhamel_centre(seed, nfield, t)


def kernel_difference(m: int, p: int) -> Dict[int, int]:
    """K_p(m, .) := T(m+p, .) + T(m, .)  -- the coefficient with which an event
    at time s = t-1-m enters Q_p(t)."""
    a, b = T_row(m + p), T_row(m)
    out = {}
    for d in set(a) | set(b):
        v = a.get(d, 0) ^ b.get(d, 0)
        if v:
            out[d] = 1
    return out


def Q_canonical(seed: Dict[int, int], nfield, p: int, t: int) -> Dict[str, int]:
    """The canonical decomposition of Q_p(t) into three exactly-defined parts:

        Q_p(t) = SEED_p(t)  +  OLD_p(t)  +  SLAB_p(t)

        SEED_p(t) = sum_d [T(t+p,d) + T(t,d)] x_0(-d)
        OLD_p(t)  = sum_{s<t} sum_j K_p(t-1-s, j) n_s(j)      (difference kernel)
        SLAB_p(t) = sum_{s=t}^{t+p-1} sum_j T(t+p-1-s, j) n_s(j)   (the new slab)

    SLAB_p(t) is supported on the p rows between t and t+p -- exactly the
    'spacetime slab between times t and t+p' the brief asks for."""
    seed_part = 0
    a, b = T_row(t + p), T_row(t)
    for d in set(a) | set(b):
        if a.get(d, 0) ^ b.get(d, 0):
            seed_part ^= seed.get(-d, 0)
    old = 0
    for s in range(t):
        for j in kernel_difference(t - 1 - s, p):
            old ^= nfield(s, j)
    slab = 0
    for s in range(t, t + p):
        for j, v in T_row(t + p - 1 - s).items():
            if v:
                slab ^= nfield(s, j)
    return {"seed": seed_part, "old": old, "slab": slab,
            "Q": seed_part ^ old ^ slab}
