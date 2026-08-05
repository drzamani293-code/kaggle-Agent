"""kernel_geometry.py -- Phase 2I section 4.

Exact binary-digit structure of the mod-2 Green function

    T(m, d) = [z^{d+m}] (1 + z + z^2)^m   over GF(2).

Everything here is a digit identity.  No asymptotic, statistical or
"randomness" statement is made or used.

The one tool is the Frobenius factorisation over GF(2):

    (1+z+z^2)^{2^i} = 1 + z^{2^i} + z^{2^{i+1}} ,

hence, writing  Q_i(z) := 1 + z^{2^i} + z^{2^{i+1}} ,

    P_m(z) := (1+z+z^2)^m = prod_{i in bits(m)} Q_i(z) .              (FROB)

Immediate consequence, used throughout:

    bits(a) and bits(b) disjoint   ==>   P_{a+b} = P_a * P_b .        (DISJ)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from nonlinear_lab import T, T_row, T_support, trinomial_row


def bits(m: int) -> List[int]:
    return [i for i in range(m.bit_length()) if (m >> i) & 1]


def popcount(m: int) -> int:
    return bin(m).count("1")


def conv(a: Dict[int, int], b: Dict[int, int]) -> Dict[int, int]:
    """GF(2) convolution of two sparse kernels keyed by displacement."""
    out: Dict[int, int] = {}
    for i, u in a.items():
        if not u:
            continue
        for j, v in b.items():
            if v:
                out[i + j] = out.get(i + j, 0) ^ 1
    return {k: v for k, v in out.items() if v}


def delta(k: int = 0) -> Dict[int, int]:
    return {k: 1}


def sparse(m: int) -> Dict[int, int]:
    return {d: 1 for d in T_support(m)}


# ------------------------------------------------------- special-time shapes ---
def shape_power_of_two(a: int) -> Dict[str, object]:
    """T(2^a, .) is supported on exactly { -2^a, 0, 2^a }."""
    A = 1 << a
    return {"m": A, "support": sorted(T_support(A)),
            "predicted": [-A, 0, A],
            "matches": sorted(T_support(A)) == [-A, 0, A]}


def shape_power_minus_one(a: int) -> Dict[str, object]:
    """m = 2^a - 1 (all low bits set).  T(m,d) counts hyperbinary
    representations of e = d+m using digits 0..a-1, mod 2.  That count is
    Stern's diatomic s(e+1), and s(n) is even iff 3 | n, so

        T(2^a - 1, d) = 1   <==>   3 does not divide (d + 2^a).
    """
    m = (1 << a) - 1
    row = T_row(m)
    pred = {d: (0 if (d + (1 << a)) % 3 == 0 else 1) for d in row}
    return {"m": m, "support_size": sum(row.values()),
            "rule": "T=1 iff 3 does not divide d + 2^a",
            "matches": all(row[d] == pred[d] for d in row)}


def shape_power_plus_r(a: int, r: int) -> Dict[str, object]:
    """m = 2^a + r with 0 <= r < 2^a.  By (DISJ),

        T(2^a + r, d) = T(r, d + 2^a) + T(r, d) + T(r, d - 2^a),

    i.e. THREE copies of the kernel T(r,.) translated to -2^a, 0, +2^a.  The
    copies are pairwise DISJOINT exactly when r < 2^{a-1} (support of T(r,.)
    is contained in [-r, r], so the copies separate iff 2*r < 2^a)."""
    A = 1 << a
    m = A + r
    lhs = T_row(m)
    rhs: Dict[int, int] = {}
    for d, v in T_row(r).items():
        if v:
            for sh in (-A, 0, A):
                rhs[d + sh] = rhs.get(d + sh, 0) ^ 1
    ok = all(lhs.get(d, 0) == rhs.get(d, 0)
             for d in set(lhs) | set(rhs))
    disjoint = 2 * r < A
    return {"a": a, "r": r, "m": m, "identity_holds": ok,
            "copies_disjoint": disjoint,
            "support_size": sum(lhs.values()),
            "3x_support_of_r": 3 * sum(T_row(r).values()),
            "disjointness_confirmed": (not disjoint) or
                                      (sum(lhs.values())
                                       == 3 * sum(T_row(r).values()))}


def shape_scaled(q: int, a: int, r: int) -> Dict[str, object]:
    """m = q * 2^a + r  with  0 <= r < 2^a.  By (FROB), P_{q 2^a}(z) = P_q(z^{2^a}),
    and bits(q*2^a) and bits(r) are disjoint, so

        T(q*2^a + r, d) = sum_e T(q, e) * T(r, d - e * 2^a).

    The kernel is a 2^a-SCALED COPY of T(q,.) with every atom replaced by a
    translated copy of T(r,.).  This is the exact self-similar decomposition
    section 4 asks for; the copies are disjoint iff 2*r < 2^a."""
    A = 1 << a
    m = q * A + r
    lhs = T_row(m)
    rhs: Dict[int, int] = {}
    for e, u in T_row(q).items():
        if not u:
            continue
        for d, v in T_row(r).items():
            if v:
                k = d + e * A
                rhs[k] = rhs.get(k, 0) ^ 1
    ok = all(lhs.get(d, 0) == rhs.get(d, 0) for d in set(lhs) | set(rhs))
    return {"q": q, "a": a, "r": r, "m": m, "identity_holds": ok,
            "support_size": sum(lhs.values()),
            "product_of_supports": sum(T_row(q).values())
                                   * sum(T_row(r).values()),
            "copies_disjoint": 2 * r < A}


def disjoint_bit_convolution(m1: int, m2: int) -> bool:
    """(DISJ) itself: T(m1+m2, .) = T(m1,.) * T(m2,.) when bits are disjoint."""
    if m1 & m2:
        return True                      # hypothesis fails, nothing asserted
    lhs = sparse(m1 + m2)
    rhs = conv(sparse(m1), sparse(m2))
    return lhs == rhs


# ---------------------------------------------- support size and recurrence ---
def A_support(m: int) -> int:
    """A(m) := number of odd coefficients of (1+z+z^2)^m."""
    return sum(trinomial_row(m))


def support_recurrence_check(m_max: int) -> Dict[str, object]:
    """P_{2m}(z) = P_m(z^2) gives A(2m) = A(m) exactly.  The odd case is tested
    against the candidate A(2m+1) = A(m) + 2*A_odd(m) forms; whatever holds is
    reported, and nothing is extrapolated."""
    A = [A_support(m) for m in range(m_max + 1)]
    even_ok = all(A[2 * m] == A[m] for m in range(m_max // 2 + 1))
    # A(2m+1) in terms of A(m), A(m+1)?  test the two natural candidates
    cand1 = all(A[2 * m + 1] == A[m] + A[m + 1]
                for m in range((m_max - 1) // 2))
    cand2 = all(A[2 * m + 1] == 2 * A[m] + 1
                for m in range((m_max - 1) // 2))
    return {"m_max": m_max, "A_2m_equals_A_m": even_ok,
            "A_2m1_equals_A_m_plus_A_m1": cand1,
            "A_2m1_equals_2A_m_plus_1": cand2,
            "A_first_40": A[:40],
            "A_at_powers_of_two": {str(1 << a): A[1 << a]
                                   for a in range(m_max.bit_length())
                                   if (1 << a) <= m_max},
            "A_at_powers_minus_one": {str((1 << a) - 1): A[(1 << a) - 1]
                                      for a in range(1, m_max.bit_length())
                                      if (1 << a) - 1 <= m_max}}


# ------------------------------------------ difference kernel (section 3 tie) ---
def difference_kernel_theorem(m: int, p: int) -> Dict[str, object]:
    """K_p(m,.) := T(m+p,.) + T(m,.).

    THEOREM (verified here, proved in MOD2_KERNEL_GEOMETRY.md):
        bits(m) and bits(p) disjoint  ==>  K_p(m,.) = ( T(p,.) + delta_0 ) * T(m,.).

    For p = 2^a this reads  K_p(m,d) = T(m, d-2^a) + T(m, d+2^a):  the SAME
    kernel evaluated on a 2^a-shifted difference -- a self-similar event family.
    """
    lhs: Dict[int, int] = {}
    a_, b_ = T_row(m + p), T_row(m)
    for d in set(a_) | set(b_):
        if a_.get(d, 0) ^ b_.get(d, 0):
            lhs[d] = 1
    factor = dict(sparse(p))
    factor[0] = factor.get(0, 0) ^ 1
    factor = {k: v for k, v in factor.items() if v}
    rhs = conv(factor, sparse(m))
    return {"m": m, "p": p, "disjoint_bits": (m & p) == 0,
            "identity_holds": lhs == rhs,
            "supp_T_m": len(sparse(m)), "supp_K": len(lhs)}
