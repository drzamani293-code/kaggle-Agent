"""prefix_lab.py -- Phase 2C: the edge-aligned prefix dynamics of rule 30.

Object.  In edge-aligned coordinates w_t(k) = x_t(-t+k) the rule-30 orbit of
the single-cell seed obeys the ONE-SIDED rule (Phase 2B Theorem EA1)

    w_{t+1}(k) = w_t(k-2) XOR ( w_t(k-1) OR w_t(k) ),      w_t(m) = 0 for m < 0,

with w_0 = (1, 0, 0, ...).  For each K the prefix W_t^K = (w_t(0..K)) evolves
autonomously under a map F_K on {0,1}^{K+1}, so its orbit has an exact tail
T(K) (preperiod) and cycle length P(K) (period).

Everything here is exact: T(K) and P(K) are the true rho/lambda of a
deterministic orbit, never estimates.  Three independent computations are
provided (reference hashing, suffix-minimum scan, incremental skew-product) and
cross-checked on overlapping ranges.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import rule30_lab as L

INF = float("inf")


# ==========================================================================
# 1. The map F_K
# ==========================================================================


def F_coord(z: Sequence[int], k: int) -> int:
    """Coordinate k of F(z):  z[k-2] XOR (z[k-1] OR z[k]), z[m]=0 for m<0."""
    a = z[k - 2] if k - 2 >= 0 else 0
    b = z[k - 1] if k - 1 >= 0 else 0
    c = z[k]
    return a ^ (b | c)


def F_K(z: Sequence[int]) -> List[int]:
    """The autonomous prefix map on {0,1}^{K+1}, K = len(z)-1."""
    return [F_coord(z, k) for k in range(len(z))]


def F_K_int(W: int, mask: int) -> int:
    """Bit-parallel form: bit k of W is w(k).

    w(k-2) sits at bit k of (W << 2); w(k-1) at bit k of (W << 1)."""
    return ((W << 2) ^ ((W << 1) | W)) & mask


def verify_F_forms(K_max: int = 12, samples: int = 4096) -> Dict[str, object]:
    """The list form and the bit-parallel form agree, exhaustively for small K."""
    ok = True
    n = 0
    for K in range(0, K_max + 1):
        mask = (1 << (K + 1)) - 1
        total = 1 << (K + 1)
        step = 1 if total <= samples else total // samples
        for code in range(0, total, step):
            z = [(code >> i) & 1 for i in range(K + 1)]
            a = F_K(z)
            b = F_K_int(code, mask)
            ok &= all(((b >> i) & 1) == a[i] for i in range(K + 1))
            n += 1
    return {"K_max": K_max, "assignments_checked": n, "forms_agree": bool(ok)}


def verify_projection(K_max: int = 10, samples: int = 4096) -> Dict[str, object]:
    """THEOREM P1:  pi_K( F_{K+1}(z) ) = F_K( pi_K(z) ).

    pi_K drops the last coordinate.  Verified exhaustively for K+1 <= 12."""
    ok = True
    n = 0
    for K in range(0, K_max + 1):
        total = 1 << (K + 2)
        step = 1 if total <= samples else total // samples
        for code in range(0, total, step):
            z = [(code >> i) & 1 for i in range(K + 2)]
            lhs = F_K(z)[: K + 1]
            rhs = F_K(z[: K + 1])
            ok &= lhs == rhs
            n += 1
    return {"K_max": K_max, "assignments_checked": n, "projection_commutes": bool(ok)}


def verify_new_coordinate_drive(K_max: int = 10, samples: int = 4096):
    """THEOREM P2 (skew product): the new coordinate obeys

        (F_{K+1} z)_{K+1} = z_{K-1} XOR ( z_K OR z_{K+1} ),

    i.e. it is driven by the OLD prefix coordinates K-1, K and by its own
    value only -- a one-bit fibre over the K-prefix base."""
    ok = True
    n = 0
    for K in range(1, K_max + 1):
        total = 1 << (K + 2)
        step = 1 if total <= samples else total // samples
        for code in range(0, total, step):
            z = [(code >> i) & 1 for i in range(K + 2)]
            got = F_K(z)[K + 1]
            want = z[K - 1] ^ (z[K] | z[K + 1])
            ok &= got == want
            n += 1
    return {"K_max": K_max, "assignments_checked": n, "skew_form_exact": bool(ok)}


# ==========================================================================
# 2. The fibre map and its classification
# ==========================================================================


def fibre_map(a: int, c: int):
    """g(u) = a XOR (c OR u).  Returns ('const', v) | ('id',) | ('neg',)."""
    g0, g1 = a ^ (c | 0), a ^ (c | 1)
    if g0 == g1:
        return ("const", g0)
    return ("id",) if g0 == 0 else ("neg",)


def fibre_table() -> List[Dict[str, object]]:
    out = []
    for a, c in itertools.product((0, 1), repeat=2):
        out.append({"a": a, "c": c, "g(0)": a ^ (c | 0), "g(1)": a ^ (c | 1),
                    "type": fibre_map(a, c)[0]})
    return out


def classify_forcing_word(word: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    """Compose one base period of fibre maps and classify the result.

    word = [(a_t, c_t)] over one period of the base (K-1)-prefix cycle.

      * some c_t = 1                -> COLLAPSING: composition is constant;
                                       fibre period = base period; the initial
                                       condition is forgotten; extra transient
                                       up to one base period.
      * all c_t = 0, XOR a_t = 0    -> NEUTRAL: composition is the identity;
                                       fibre period = base period; the initial
                                       value is preserved (two invariant fibres).
      * all c_t = 0, XOR a_t = 1    -> DOUBLING: composition is negation;
                                       fibre period = 2 x base period.
    """
    u0 = 0
    u1 = 1
    for a, c in word:
        u0 = a ^ (c | u0)
        u1 = a ^ (c | u1)
    any_c = any(c for _, c in word)
    xor_a = 0
    for a, c in word:
        xor_a ^= a
    if u0 == u1:
        kind = "COLLAPSING"
    elif u0 == 0:
        kind = "NEUTRAL"
    else:
        kind = "DOUBLING"
    return {"kind": kind, "phi(0)": u0, "phi(1)": u1,
            "has_c_equal_1": bool(any_c), "xor_of_a": xor_a,
            "period_multiplier": 2 if kind == "DOUBLING" else 1}


def verify_classification(max_len: int = 10, samples: int = 20000):
    """The three-way classification is complete and matches the composition."""
    ok = True
    n = 0
    counts = {"COLLAPSING": 0, "NEUTRAL": 0, "DOUBLING": 0}
    for m in range(1, max_len + 1):
        total = 4 ** m
        step = 1 if total <= samples else max(1, total // samples)
        for code in range(0, total, step):
            word = []
            c2 = code
            for _ in range(m):
                word.append((c2 & 1, (c2 >> 1) & 1))
                c2 >>= 2
            r = classify_forcing_word(word)
            counts[r["kind"]] += 1
            # predicted kind from the structural rule
            if r["has_c_equal_1"]:
                pred = "COLLAPSING"
            elif r["xor_of_a"] == 0:
                pred = "NEUTRAL"
            else:
                pred = "DOUBLING"
            ok &= (pred == r["kind"])
            n += 1
    return {"words_checked": n, "structural_rule_exact": bool(ok),
            "counts": counts}


# ==========================================================================
# 3. Orbit generation and exact (T, P)
# ==========================================================================


def edge_rows(K_max: int, T_max: int) -> List[int]:
    """W_t for t = 0..T_max, bit k = w_t(k), truncated to k <= K_max.

    Truncation is exact: F_K only reads coordinates <= k."""
    mask = (1 << (K_max + 1)) - 1
    W = 1                      # w_0 = (1, 0, 0, ...)
    rows = [W]
    for _ in range(T_max):
        W = F_K_int(W, mask)
        rows.append(W)
    return rows


def verify_against_rule30(K_max: int = 200, T_max: int = 300) -> Dict[str, object]:
    """The edge-aligned rows really are x_t(-t+k)."""
    rows = edge_rows(K_max, T_max)
    orb, R = _orbit_rows(T_max)
    bad = n = 0
    for t in range(1, T_max + 1):
        for k in range(0, min(2 * t, K_max) + 1):
            got = (rows[t] >> k) & 1
            want = (orb[t] >> (R + (-t + k))) & 1
            n += 1
            bad += got != want
    return {"cells_checked": n, "mismatches": bad, "exact": bad == 0}


def _orbit_rows(t_max: int):
    R = t_max + 2
    width = 2 * R + 1
    mask = (1 << width) - 1
    row = 1 << R
    rows = [row]
    for _ in range(t_max):
        row = ((row << 1) ^ (row | (row >> 1))) & mask
        rows.append(row)
    return rows, R


def lowbit_index(d: int) -> int:
    return (d & -d).bit_length() - 1


def exact_TP_first_repeat(K_max: int, T_max: int, q_max_exp: int = 14):
    """EXACT T(K), P(K) for every K <= K_max.

    Determinism makes a SINGLE coincidence sufficient: if W_T^K = W_{T+Q}^K
    then the orbit is periodic from T with period Q forever.  Hence

        P(K) = min { Q : exists t with W_t^K = W_{t+Q}^K },
        T(K) = min { t : W_t^K = W_{t+P(K)}^K },

    and both are the true lambda and rho of the orbit -- not bounds, and not
    dependent on how far the simulation is carried (only on it being far
    enough for the repeat to occur).

    Implementation: let L_t^Q = index of the lowest set bit of
    W_{t+Q} XOR W_t (or K_max+1 if the rows agree).  Then W_t^K = W_{t+Q}^K
    iff L_t^Q > K.  For fixed Q, min{t : L_t^Q > K} is nondecreasing in K, so
    one two-pointer sweep settles every K.  Q runs over powers of two, which
    Theorem S1 proves is exhaustive.
    """
    rows = edge_rows(K_max, T_max)
    n = len(rows)
    best_T = [None] * (K_max + 1)
    best_P = [None] * (K_max + 1)
    for e in range(q_max_exp + 1):
        Q = 1 << e
        if Q >= n:
            break
        m = n - Q                                  # t ranges over 0..m-1
        Lq = [0] * m
        for t in range(m):
            d = rows[t + Q] ^ rows[t]
            Lq[t] = (K_max + 1) if d == 0 else lowbit_index(d)
        T = 0
        for K in range(K_max + 1):
            while T < m and Lq[T] <= K:
                T += 1
            if T >= m:
                break                              # this Q fails for all larger K
            if best_T[K] is None:
                best_T[K] = T
                best_P[K] = Q
    return best_T, best_P


def exact_TP_reference(K_max: int, T_max: int):
    """Independent reference: hash the whole prefix and find the first repeat.

    For a deterministic map the first repeated state gives the exact tail and
    cycle length.  O(K * T) memory-heavy, used only for small K."""
    rows = edge_rows(K_max, T_max)
    out = {}
    for K in range(K_max + 1):
        mask = (1 << (K + 1)) - 1
        seen = {}
        for t, W in enumerate(rows):
            s = W & mask
            if s in seen:
                out[K] = (seen[s], t - seen[s])
                break
            seen[s] = t
        else:
            out[K] = (None, None)
    return out


def exact_TP_incremental(K_max: int, T_max: int):
    """Third computation, using the skew-product theorem directly.

    Given (T(K-1), P(K-1)), Theorem S1 says P(K) is P(K-1) or 2P(K-1) and
    Theorem S2 says coordinate K is periodic from at most T(K-1)+P(K-1).  So
    only a window of size O(P) needs to be examined per K."""
    rows = edge_rows(K_max, T_max)
    n = len(rows)
    T = [0]
    P = [1]
    for K in range(1, K_max + 1):
        Tm, Pm = T[K - 1], P[K - 1]
        chosen = None
        for Q in (Pm, 2 * Pm):
            hi = min(n - Q, Tm + Pm + 2 * Q + 4)
            last_bad = -1
            for t in range(Tm, hi):
                if (((rows[t + Q] ^ rows[t]) >> K) & 1):
                    last_bad = t
            if last_bad + 1 <= Tm + Pm:
                chosen = (max(Tm, last_bad + 1), Q)
                break
        if chosen is None:                      # theorem violated -> report
            return T, P, {"failed_at_K": K}
        T.append(chosen[0])
        P.append(chosen[1])
    return T, P, {"failed_at_K": None}


def full_verify_TP(K: int, T: int, P: int, rows: List[int]) -> Dict[str, bool]:
    """Independent full-range check that (T,P) really is the exact rho/lambda:
      (a) W_{t+P}^K = W_t^K for every t in [T, T_max-P];
      (b) minimality of T: T = 0, or W_{T-1+P}^K != W_{T-1}^K;
      (c) minimality of P: no proper divisor of P works from T."""
    mask = (1 << (K + 1)) - 1
    n = len(rows)
    per = all(((rows[t + P] ^ rows[t]) & mask) == 0 for t in range(T, n - P))
    minT = (T == 0) or (((rows[T - 1 + P] ^ rows[T - 1]) & mask) != 0)
    minP = True
    d = 1
    while d < P:
        if P % d == 0:
            if all(((rows[t + d] ^ rows[t]) & mask) == 0 for t in range(T, n - d)):
                minP = False
                break
        d *= 2
    return {"periodic_from_T": per, "T_minimal": minT, "P_minimal": minP,
            "all": bool(per and minT and minP)}


# ==========================================================================
# 4. Structural facts about the real orbit
# ==========================================================================


def coordinate_ever_one(K_max: int, T_max: int) -> Dict[str, object]:
    """Is w_t(K) = 1 for some t?  Needed by the support lower bound (S4)."""
    rows = edge_rows(K_max, T_max)
    never = []
    for K in range(K_max + 1):
        if not any((W >> K) & 1 for W in rows):
            never.append(K)
    return {"K_max": K_max, "coordinates_never_one": never,
            "all_coordinates_take_value_one": len(never) == 0}


def forcing_word_census(T: List[int], P: List[int], K_max: int, T_max: int):
    """Which of the three fibre classes actually occurs at each K in the real
    orbit, read off the base cycle."""
    rows = edge_rows(K_max, T_max)
    counts = {"COLLAPSING": 0, "NEUTRAL": 0, "DOUBLING": 0}
    doubling_K = []
    per_K = []
    for K in range(2, K_max + 1):
        Tm, Pm = T[K - 1], P[K - 1]
        if Tm + Pm >= len(rows):
            break
        word = [(((rows[t] >> (K - 2)) & 1), ((rows[t] >> (K - 1)) & 1))
                for t in range(Tm, Tm + Pm)]
        r = classify_forcing_word(word)
        counts[r["kind"]] += 1
        per_K.append({"K": K, "kind": r["kind"], "P_prev": Pm, "P": P[K]})
        if r["kind"] == "DOUBLING":
            doubling_K.append(K)
        # consistency: DOUBLING iff the period actually doubled
        if (P[K] == 2 * Pm) != (r["kind"] == "DOUBLING"):
            per_K[-1]["INCONSISTENT"] = True
    bad = [x for x in per_K if x.get("INCONSISTENT")]
    return {"counts": counts, "doubling_K": doubling_K,
            "classification_matches_measured_period": len(bad) == 0,
            "inconsistencies": bad[:20], "per_K_sample": per_K[:5]}


def diagonal_relation_range(T: List[int], P: List[int], K_max: int):
    """THEOREM D1 gives  x_{t+P(K)}(0) = x_t(P(K))  for T(K) <= t <= K - P(K).
    Report for how many K that range is non-empty."""
    nonempty = [K for K in range(K_max + 1)
                if T[K] is not None and P[K] is not None
                and T[K] <= K - P[K]]
    return {"K_max": K_max, "K_with_nonempty_range": nonempty,
            "count": len(nonempty),
            "note": "empty for every K means Theorem D1 is vacuous on the real orbit"}
