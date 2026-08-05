"""frontier_lab.py -- Phase 2E: the transient frontier of the edge-aligned tower.

Coordinates (Phase 2B Theorem EA1, Phase 2C):

    w_t(k) = x_t(-t + k),   w_{t+1}(k) = w_t(k-2) XOR ( w_t(k-1) OR w_t(k) )

with w_0 = (1, 0, 0, ...) and w_t(k) = 0 for k < 0.  The K-prefix
W_t^K = (w_t(0), ..., w_t(K)) evolves autonomously, hence is eventually
periodic: exact preperiod T(K), exact period P(K).

The centre column is the diagonal, x_t(0) = w_t(t) (Theorem EA3).

This module studies the TRANSIENT: where T(K) comes from, which levels create
it, and what the diagonal reads before any level settles.

Nothing in this file proves anything about the centre column.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import prefix_lab as PL


# ==========================================================================
# 1. Coordinate preperiods  (the atoms of T(K))
# ==========================================================================


def coord_preperiods_backward(rows: List[int], Q: int, K_max: int) -> List[int]:
    """r_Q(k) = min{ t : w_s(k) = w_{s+Q}(k) for every s >= t }, all k <= K_max.

    Equivalently 1 + (the last time index at which the lag-Q comparison fails),
    or 0 if it never fails.  Backward sweep with an `unseen` mask: the first
    failure met going down in t is the last one going up.

    NOTE.  The answer is only meaningful for coordinates whose minimal period
    divides Q.  For the others the comparison fails at arbitrarily large t and
    the value returned is an artefact of where the simulation stopped; callers
    must filter on P(k) | Q.  `reliability_horizon` below flags them.
    """
    n = len(rows)
    unseen = (1 << (K_max + 1)) - 1
    R = [0] * (K_max + 1)
    for t in range(n - Q - 1, -1, -1):
        d = (rows[t + Q] ^ rows[t]) & unseen
        if d:
            m = d
            while m:
                b = m & -m
                R[b.bit_length() - 1] = t + 1
                m ^= b
            unseen ^= d                       # d is a subset of unseen
            if not unseen:
                break
    return R


def coord_preperiods_suffix_or(rows: List[int], Q: int, K_max: int) -> List[int]:
    """Independent implementation of `coord_preperiods_backward`.

    Builds the explicit suffix-OR array S_t = OR_{t' >= t} (W_{t'+Q} XOR W_{t'})
    and then sweeps FORWARD: bit k of S_t is monotone non-increasing in t, and
    r_Q(k) is the index at which it switches off.  Different data structure,
    different sweep direction, no early exit."""
    n = len(rows)
    m = n - Q
    mask = (1 << (K_max + 1)) - 1
    S = [0] * (m + 1)
    for t in range(m - 1, -1, -1):
        S[t] = S[t + 1] | ((rows[t + Q] ^ rows[t]) & mask)
    R = [0] * (K_max + 1)
    for t in range(m):
        diff = S[t] ^ S[t + 1]
        while diff:
            b = diff & -diff
            R[b.bit_length() - 1] = t + 1
            diff ^= b
    return R


def coord_preperiod_direct(rows: List[int], Q: int, k: int) -> int:
    """Third implementation, one coordinate at a time, from the definition:
    scan the whole lag-Q comparison of coordinate k and return 1 + the last
    index at which it fails."""
    last = -1
    for t in range(len(rows) - Q):
        if ((rows[t + Q] >> k) ^ (rows[t] >> k)) & 1:
            last = t
    return last + 1


def reliability_horizon(rows: List[int], Q: int) -> int:
    """Largest t index used by a lag-Q comparison; r_Q(k) at or above this is
    not a settled value but a statement that no repeat was seen."""
    return len(rows) - Q


def prefix_max(a: Sequence[int]) -> List[int]:
    out = []
    m = -1
    for v in a:
        if v > m:
            m = v
        out.append(m)
    return out


# ==========================================================================
# 2. The level table:  class, tau, the defect bit, and the exact recurrence
# ==========================================================================


def bit(rows: List[int], t: int, k: int) -> int:
    if k < 0:
        return 0
    return (rows[t] >> k) & 1


def first_one_at_or_after(rows: List[int], k: int, t0: int,
                          cap: Optional[int] = None) -> Optional[int]:
    """min{ t >= t0 : w_t(k) = 1 }, or None if not found within `cap` steps."""
    n = len(rows)
    hi = n if cap is None else min(n, t0 + cap)
    for t in range(t0, hi):
        if (rows[t] >> k) & 1:
            return t
    return None


def last_one_strictly_before(rows: List[int], k: int, t1: int,
                             cap: Optional[int] = None) -> Optional[int]:
    """max{ t < t1 : w_t(k) = 1 }, or None."""
    lo = 0 if cap is None else max(0, t1 - cap)
    for t in range(t1 - 1, lo - 1, -1):
        if (rows[t] >> k) & 1:
            return t
    return None


def level_window(rows: List[int], K: int, t_lo: int, t_hi: int) -> Dict[int, int]:
    """{ t : (w_t(K-2), w_t(K-1), w_t(K)) packed in bits 0,1,2 } for t in the
    window.  One big-integer shift per time step instead of three."""
    sh = K - 2
    n = len(rows)
    return {t: (rows[t] >> sh) & 7
            for t in range(max(0, t_lo), min(n, t_hi + 1))}


def level_table(K_max: int, T_max: Optional[int] = None, rho_cap: int = 256):
    """Everything Phase 2E needs, level by level.

    Returns (records, T, P, rows) where `records[i]` describes level K = i + 2.

    Fields
      K, P_prev, P_K, fibre_kind          Phase 2C/2D classification
      T_prev, T_K                         exact preperiods
      tau_K      = min{ t >= T(K-1) : w_t(K-1) = 1 }        (first collapse)
      D_bit      = w_{T(K-1)}(K) XOR w_{T(K-1)+P(K)}(K)     (the defect bit)
      cls        RESETTING (T(K) > T(K-1)) or INHERITING (T(K) = T(K-1))
      T_pred     the value Theorem B predicts from (T_prev, tau_K, D_bit)
      R_K        = r_{P(K)}(K), the coordinate-K preperiod at the level's lag
      rho_K      = max{ t < T(K) : w_t(K-1) = 1 }
    """
    T_max = T_max or int(1.5 * K_max) + 1000
    T, P = PL.exact_TP_first_repeat(K_max, T_max)
    rows = PL.edge_rows(K_max, T_max)

    lags = sorted({p for p in P if p is not None})
    Rq = {Q: coord_preperiods_backward(rows, Q, K_max) for Q in lags}

    recs = []
    for K in range(2, K_max + 1):
        if T[K] is None or T[K - 1] is None:
            break
        Tp, Pp, Pk = T[K - 1], P[K - 1], P[K]
        if Tp + Pk + 1 >= len(rows):
            break
        win = level_window(rows, K, min(Tp, T[K]) - rho_cap,
                           max(Tp + 2 * Pp + 2, T[K] + Pk + 1))
        b0 = lambda t: (win[t] >> 0) & 1       # w_t(K-2)
        b1 = lambda t: (win[t] >> 1) & 1       # w_t(K-1)
        b2 = lambda t: (win[t] >> 2) & 1       # w_t(K)
        word = [(b0(t), b1(t)) for t in range(Tp, Tp + Pp)]
        kind = PL.classify_forcing_word(word)["kind"]
        tau = None
        for t in range(Tp, min(Tp + 2 * Pp + 2, len(rows))):
            if b1(t):
                tau = t
                break
        D = b2(Tp) ^ b2(Tp + Pk)
        if kind == "COLLAPSING" and Pk == Pp:
            T_pred = Tp if D == 0 else (tau + 1 if tau is not None else None)
        else:
            T_pred = Tp                        # NEUTRAL / DOUBLING: Theorem B2
        rho = None
        for t in range(T[K] - 1, max(-1, T[K] - 1 - rho_cap), -1):
            if t in win and b1(t):
                rho = t
                break
        recs.append({
            "K": K, "P_prev": Pp, "P_K": Pk, "fibre_kind": kind,
            "T_prev": Tp, "T_K": T[K], "tau_K": tau, "D_bit": D,
            "cls": "RESETTING" if T[K] > Tp else "INHERITING",
            "T_pred": T_pred, "R_K": Rq[Pk][K], "rho_K": rho,
        })
    return recs, T, P, rows, Rq


# ==========================================================================
# 3. Verification of the structure theorems
# ==========================================================================


def verify_theorem_A(recs, T, P, Rq, K_max: int) -> Dict[str, object]:
    """THEOREM A.  T(K) = max_{k <= K} r_{P(K)}(k).

    Checked for every K for which the whole prefix has period exactly P(K),
    i.e. for every K in range."""
    pm = {Q: prefix_max(R) for Q, R in Rq.items()}
    bad = []
    n = 0
    for K in range(K_max + 1):
        if T[K] is None:
            break
        n += 1
        if pm[P[K]][K] != T[K]:
            bad.append({"K": K, "T": T[K], "prefix_max": pm[P[K]][K]})
    return {"levels_checked": n, "mismatches": len(bad),
            "first_mismatches": bad[:10], "exact": not bad}


def verify_theorem_B(recs) -> Dict[str, object]:
    """THEOREM B (exact recurrence).  For P(K) = P(K-1):
         D = 0  =>  T(K) = T(K-1)          (INHERITING)
         D = 1  =>  T(K) = tau_K + 1       (RESETTING)
       For P(K) = 2 P(K-1) (DOUBLING) and for NEUTRAL levels: T(K) = T(K-1).
    """
    bad, n = [], 0
    per_class = {"RESETTING": 0, "INHERITING": 0}
    for r in recs:
        if r["T_pred"] is None:
            bad.append({"K": r["K"], "reason": "tau not found in 2 periods"})
            continue
        n += 1
        per_class[r["cls"]] += 1
        if r["T_pred"] != r["T_K"]:
            bad.append({"K": r["K"], "pred": r["T_pred"], "actual": r["T_K"],
                        "D": r["D_bit"], "kind": r["fibre_kind"]})
    return {"levels_checked": n, "mismatches": len(bad),
            "first_mismatches": bad[:10], "exact": not bad,
            "class_counts": per_class}


def verify_dichotomy(recs) -> Dict[str, object]:
    """The RESETTING/INHERITING split is exactly the defect bit D, and
    RESETTING happens only at COLLAPSING levels."""
    bad_d, bad_kind, bad_reset = [], [], []
    for r in recs:
        if r["P_K"] == r["P_prev"] and r["fibre_kind"] == "COLLAPSING":
            if (r["D_bit"] == 1) != (r["cls"] == "RESETTING"):
                bad_d.append(r["K"])
        if r["cls"] == "RESETTING" and r["fibre_kind"] != "COLLAPSING":
            bad_kind.append(r["K"])
        if r["cls"] == "RESETTING" and r["tau_K"] is not None:
            if r["T_K"] != r["tau_K"] + 1:
                bad_reset.append(r["K"])
    return {"D_bit_predicts_class": not bad_d, "D_violations": bad_d[:10],
            "resetting_implies_collapsing": not bad_kind,
            "kind_violations": bad_kind[:10],
            "resetting_T_equals_tau_plus_1": not bad_reset,
            "reset_violations": bad_reset[:10]}


def definition_comparison(recs) -> Dict[str, object]:
    """How often do T(K), R(K) and tau(K)+1 coincide?  The brief asks which of
    the definitions are equal and which are genuinely distinct; this counts the
    coincidences that the theorems predict and the ones they do not."""
    n = len(recs)
    c = {"T_eq_R": 0, "T_gt_R": 0, "T_lt_R": 0,
         "R_eq_tau_plus_1": 0, "R_le_T_prev": 0, "R_other": 0,
         "T_eq_tau_plus_1": 0}
    for r in recs:
        T_, R_, tp = r["T_K"], r["R_K"], r["T_prev"]
        tau = r["tau_K"]
        c["T_eq_R"] += (T_ == R_)
        c["T_gt_R"] += (T_ > R_)
        c["T_lt_R"] += (T_ < R_)
        if tau is not None and R_ == tau + 1:
            c["R_eq_tau_plus_1"] += 1
        elif R_ <= tp:
            c["R_le_T_prev"] += 1
        else:
            c["R_other"] += 1
        if tau is not None and T_ == tau + 1:
            c["T_eq_tau_plus_1"] += 1
    return {"levels": n, "counts": c,
            "T_never_below_R": c["T_lt_R"] == 0,
            "R_always_classified": c["R_other"] == 0}


def verify_R_two_ways(rows, Rq, K_max: int, K_cross: int = 4000,
                      spot: Sequence[int] = ()) -> Dict[str, object]:
    """Three implementations of the coordinate preperiods must agree.

    The suffix-OR implementation is run on the sub-range k <= K_cross (it
    materialises an O(T_max) array of K_max-bit integers, so it is not run at
    full width); `spot` coordinates are additionally checked one at a time by
    the direct definition, at full width."""
    K_cross = min(K_cross, K_max)
    out = {}
    for Q, Rb in Rq.items():
        Rf = coord_preperiods_suffix_or(rows, Q, K_cross)
        diff = [k for k in range(K_cross + 1) if Rb[k] != Rf[k]]
        out[str(Q)] = {"mismatches": len(diff), "first": diff[:5]}
    spots = []
    for Q, k in spot:
        d = coord_preperiod_direct(rows, Q, k)
        spots.append({"Q": Q, "k": k, "direct": d, "sweep": Rq[Q][k],
                      "agree": d == Rq[Q][k]})
    return {"K_cross": K_cross, "per_lag": out, "spot_checks": spots,
            "all_agree": all(v["mismatches"] == 0 for v in out.values())
                         and all(s["agree"] for s in spots)}


def verify_tau_two_ways(recs, rows, T, P) -> Dict[str, object]:
    """tau_K by direct minimum-scan (level_table) vs. by the Phase 2D phase
    index on the base cycle word."""
    bad = []
    for r in recs:
        K, Tp, Pp = r["K"], r["T_prev"], r["P_prev"]
        cpos = [j for j in range(Pp) if bit(rows, Tp + j, K - 1)]
        alt = Tp + cpos[0] if cpos else None
        if alt != r["tau_K"]:
            bad.append({"K": K, "scan": r["tau_K"], "phase": alt})
    return {"levels_checked": len(recs), "mismatches": len(bad),
            "first": bad[:10], "agree": not bad}


# ==========================================================================
# 4. Reset schedule
# ==========================================================================


def reset_schedule(recs) -> Dict[str, object]:
    """rho(K) = max{ t < T(K) : w_t(K-1) = 1 }.  Is T(K) = rho(K) + 1?"""
    tot = eq = miss = 0
    by_cls = {"RESETTING": [0, 0], "INHERITING": [0, 0]}
    gaps = {}
    for r in recs:
        tot += 1
        if r["rho_K"] is None:
            miss += 1
            continue
        hit = (r["T_K"] == r["rho_K"] + 1)
        by_cls[r["cls"]][0] += 1
        by_cls[r["cls"]][1] += int(hit)
        eq += int(hit)
        g = r["T_K"] - r["rho_K"] - 1
        gaps[g] = gaps.get(g, 0) + 1
    return {"levels": tot, "rho_not_found": miss,
            "T_equals_rho_plus_1": eq,
            "fraction": (eq / tot) if tot else None,
            "by_class": {k: {"levels": v[0], "hits": v[1],
                             "fraction": (v[1] / v[0]) if v[0] else None}
                         for k, v in by_cls.items()},
            "gap_histogram": dict(sorted(gaps.items())[:20])}


def reset_increments(recs) -> Dict[str, object]:
    """At a RESETTING level the jump is T(K) - T(K-1) = tau_K + 1 - T(K-1),
    i.e. one more than the wait for a 1 in coordinate K-1 after T(K-1)."""
    inc = {}
    waits = {}
    n_reset = 0
    for r in recs:
        if r["cls"] != "RESETTING":
            continue
        n_reset += 1
        d = r["T_K"] - r["T_prev"]
        inc[d] = inc.get(d, 0) + 1
        if r["tau_K"] is not None:
            w = r["tau_K"] - r["T_prev"]
            waits[w] = waits.get(w, 0) + 1
    tot_jump = sum(k * v for k, v in inc.items())
    return {"resetting_levels": n_reset,
            "increment_histogram": dict(sorted(inc.items())),
            "wait_histogram": dict(sorted(waits.items())),
            "total_transient_created": tot_jump,
            "mean_increment": (tot_jump / n_reset) if n_reset else None}


def resetting_gaps(recs) -> Dict[str, object]:
    """Spacing in K between consecutive RESETTING levels."""
    ks = [r["K"] for r in recs if r["cls"] == "RESETTING"]
    gaps = [b - a for a, b in zip(ks, ks[1:])]
    h = {}
    for g in gaps:
        h[g] = h.get(g, 0) + 1
    return {"n_resetting": len(ks), "first": ks[:20], "last": ks[-5:],
            "gap_histogram": dict(sorted(h.items())),
            "max_gap": max(gaps) if gaps else None,
            "mean_gap": (sum(gaps) / len(gaps)) if gaps else None}


# ==========================================================================
# 5. Frontier dynamics
# ==========================================================================


def frontier(T: List[int], K_max: int, t_max: int) -> List[int]:
    """K_per(t) = max{ K : T(K) <= t }, or -1 if none."""
    out = [-1] * (t_max + 1)
    K = -1
    for t in range(t_max + 1):
        while K + 1 <= K_max and T[K + 1] is not None and T[K + 1] <= t:
            K += 1
        out[t] = K
    return out


def frontier_from_counts(T: List[int], K_max: int, t_max: int) -> List[int]:
    """Independent implementation of `frontier`: bucket the levels by T and
    take a running count."""
    cnt = [0] * (t_max + 2)
    for K in range(K_max + 1):
        if T[K] is not None and T[K] <= t_max:
            cnt[T[K]] += 1
    out = []
    run = -1
    for t in range(t_max + 1):
        run += cnt[t]
        out.append(run)
    return out


def frontier_stats(T, P, K_max: int, t_max: int) -> Dict[str, object]:
    kp = frontier(T, K_max, t_max)
    kp2 = frontier_from_counts(T, K_max, t_max)
    steps = {}
    for a, b in zip(kp, kp[1:]):
        d = b - a
        steps[d] = steps.get(d, 0) + 1
    dT = {}
    viol_mono = []
    viol_P = []
    for K in range(K_max):
        if T[K] is None or T[K + 1] is None:
            break
        d = T[K + 1] - T[K]
        dT[d] = dT.get(d, 0) + 1
        if d < 0:
            viol_mono.append(K)
        if d > P[K]:
            viol_P.append({"K": K, "dT": d, "P": P[K]})
    return {"two_implementations_agree": kp == kp2,
            "K_per_step_histogram": dict(sorted(steps.items())),
            "max_K_per_step": max(steps) if steps else None,
            "dT_histogram": dict(sorted(dT.items())),
            "monotonicity_violations": viol_mono[:10],
            "bound_dT_le_P_violations": viol_P[:10],
            "n_bound_violations": len(viol_P)}


def age_profile(T, K_list) -> List[Dict[str, object]]:
    """A(t, K) = max(0, T(K) - t); the diagonal age deficit is A(t, t)."""
    out = []
    for K in K_list:
        if K < len(T) and T[K] is not None:
            out.append({"K": K, "T": T[K], "A_diag": T[K] - K,
                        "T_over_K": (T[K] / K) if K else None})
    return out


# ==========================================================================
# 6. The co-moving transient strip
# ==========================================================================


def strip_identity(t_max: int, R: int) -> Dict[str, object]:
    """THEOREM.  q_t(r) := w_t(t + r) = x_t(r): the co-moving strip IS the
    original column set.  Verified cell by cell against an independent
    original-coordinate simulation."""
    K_max = 2 * t_max + R + 2
    wrows = PL.edge_rows(K_max, t_max)
    orb, C = PL._orbit_rows(t_max)
    bad = n = 0
    for t in range(t_max + 1):
        for r in range(-R, R + 1):
            k = t + r
            if k < 0:
                continue
            n += 1
            if ((wrows[t] >> k) & 1) != ((orb[t] >> (C + r)) & 1):
                bad += 1
    return {"cells_checked": n, "mismatches": bad, "exact": bad == 0,
            "t_max": t_max, "radius": R}


def strip_words(t_max: int, R: int, base: Optional[List[int]] = None) -> List[int]:
    """The strip state at time t: bits r = 0..R of q_t, packed low-to-high.

    `base` may be a wider strip-word list computed once; it is masked down."""
    mask = (1 << (R + 1)) - 1
    if base is not None:
        return [b & mask for b in base]
    orb, C = PL._orbit_rows(t_max)
    return [(row >> C) & mask for row in orb]


def strip_not_autonomous(ws: List[int], R: int) -> Dict[str, object]:
    """The strip is NOT a finite-state system: exhibit two times with the same
    width-(R+1) strip word and different successors.

    This is a finite disproof, not a heuristic: it is a counterexample to the
    statement 'q_{t+1}(0..R) is a function of q_t(0..R)'."""
    seen = {}
    for t in range(len(ws) - 1):
        s = ws[t]
        if s in seen:
            t0 = seen[s]
            if ws[t0 + 1] != ws[t + 1]:
                return {"radius": R, "autonomous": False,
                        "witness": {"t1": t0, "t2": t, "state": s,
                                    "succ1": ws[t0 + 1], "succ2": ws[t + 1]}}
        else:
            seen[s] = t
    return {"radius": R, "autonomous": None,
            "note": "no witness within t_max; NOT a proof of autonomy"}


def strip_census(t_max: int, R_max: int) -> List[Dict[str, object]]:
    """For each radius: distinct strip words observed, first repeat time, and
    whether a counterexample to autonomy was found.  The orbit is generated
    once and masked down to each radius."""
    wide = strip_words(t_max, R_max)
    out = []
    for R in range(1, R_max + 1):
        ws = strip_words(t_max, R, base=wide)
        seen = {}
        first_rep = None
        for t, s in enumerate(ws):
            if s in seen and first_rep is None:
                first_rep = (seen[s], t)
            seen.setdefault(s, t)
        na = strip_not_autonomous(ws, R)
        out.append({"R": R, "width": R + 1, "distinct_words": len(seen),
                    "possible_words": 1 << (R + 1),
                    "all_words_seen": len(seen) == (1 << (R + 1)),
                    "first_repeat": first_rep,
                    "autonomy_counterexample": na.get("witness"),
                    "is_autonomous": na["autonomous"]})
    return out


def strip_dependency_growth(t_max: int, R: int) -> Dict[str, object]:
    """Backward dependency cone of the strip q_t(0..R).

    One step back from column r reaches r-1, r, r+1, so the cone at time t-s is
    [-(t-s), R+(t-s)]: it opens by one column on EACH side per step.  The right
    edge is the one that matters -- rule 30 is left-permutive and not
    right-permutive (Phase 1 Fact F4), so the rightward dependence cannot be
    eliminated.  Tracked directly, and the genuineness of the right dependence
    is tested by flipping q_t(R+1) and re-evaluating the local rule."""
    lo, hi = 0, R
    widths = []
    for _ in range(t_max):
        lo -= 1
        hi += 1
        widths.append(hi - lo + 1)
    # genuineness: q_{t+1}(R) reads q_t(R+1) whenever q_t(R) = 0
    sensitive = [(l, m, rr) for l in (0, 1) for m in (0, 1) for rr in (0, 1)
                 if (l ^ (m | 0)) != (l ^ (m | 1))]
    return {"radius": R, "steps": t_max,
            "final_cone": [lo, hi], "final_cone_width": hi - lo + 1,
            "growth_per_step_each_side": 1,
            "right_dependence_is_genuine": len(sensitive) > 0,
            "local_patterns_sensitive_to_right_neighbour":
                sorted({(l, m) for l, m, _ in sensitive}),
            "cone_widths_first_10": widths[:10]}


# ==========================================================================
# 7. Diagonal reset ancestry
# ==========================================================================


def cone_interval(t: int, s: int) -> Tuple[int, int]:
    """Phase 2D Theorem DD1: ancestors of (t, t) at time s."""
    return max(0, 2 * s - t), min(t, 2 * s)


def reset_events_in_cone(recs, t: int) -> Dict[str, object]:
    """Which reset events (tau_K, K) lie inside the backward cone of (t, t)?"""
    inside = []
    for r in recs:
        if r["cls"] != "RESETTING" or r["tau_K"] is None:
            continue
        s, K = r["tau_K"], r["K"]
        if s > t:
            continue
        lo, hi = cone_interval(t, s)
        if lo <= K <= hi:
            inside.append(K)
    return {"t": t, "reset_events_in_cone": len(inside),
            "max_level_in_cone": max(inside) if inside else None,
            "max_level_over_t": (max(inside) / t) if inside and t else None}


def diagonal_never_settled(T, K_max: int) -> Dict[str, object]:
    """The diagonal reads level K at time K.  Where T(K) > K it reads that
    level strictly before the level settles.

    Phase 2C's E4 records that T(K) <= K holds exactly for K <= 17; those small
    levels are genuine counterexamples to the unqualified statement and are
    reported here, not suppressed."""
    bad = [K for K in range(0, K_max + 1)
           if T[K] is not None and T[K] <= K]
    return {"K_max": K_max, "levels_with_T_le_K": bad,
            "count": len(bad),
            "first_K_from_which_T_gt_K": (max(bad) + 1) if bad else 0,
            "diagonal_strictly_transient_above_that": True}


def ancestry_skeleton(recs, T, t: int) -> Dict[str, object]:
    """For the diagonal point (t, t): the cone, the periodic frontier inside
    it, and the reset events it contains."""
    kp = frontier(T, len(T) - 1, t)
    rows = []
    first_all_transient = None
    for s in range(t + 1):
        lo, hi = cone_interval(t, s)
        inper = max(0, min(hi, kp[s]) - lo + 1)
        rows.append({"s": s, "lo": lo, "hi": hi, "width": hi - lo + 1,
                     "in_periodic": inper})
        if s > 0 and inper == 0 and first_all_transient is None:
            first_all_transient = s
    ev = reset_events_in_cone(recs, t)
    return {"t": t,
            "first_s_with_cone_entirely_transient": first_all_transient,
            "crossover_fraction": (first_all_transient / t) if (first_all_transient and t) else None,
            "reset_events": ev,
            "cone_cells": sum(r["width"] for r in rows),
            "cone_cells_in_periodic": sum(r["in_periodic"] for r in rows)}


# ==========================================================================
# 8. The proof DAG for the diagonal, with the periodicity rewrite
# ==========================================================================


def canon(s: int, k: int, T, P):
    """Canonical representative of the cell (s, k) under the SOUND rewrite

        w_s(k) = w_{s - P(k)}(k)   whenever  s >= T(k) + P(k)

    (Theorem EA2 plus the exact (T, P) table.)  Returns either
    ('c', v) for a folded constant or ('n', s', k)."""
    if k < 0:
        return ("c", 0)
    if k > 2 * s:
        return ("c", 0)                        # support bound: w_s(k) = 0
    if s == 0:
        return ("c", 1 if k == 0 else 0)
    tk, pk = T[k], P[k]
    if tk is not None and pk and s >= tk + pk:
        s = tk + ((s - tk) % pk)
    return ("n", s, k)


def diagonal_proof_dag(t: int, T, P, node_cap: int = 4_000_000):
    """Hash-consed DAG for w_t(t) under the periodicity rewrite.

    Nodes are canonical cells; each has children canon(s-1, k-2),
    canon(s-1, k-1), canon(s-1, k).  Returns node/edge counts and the size of
    the un-rewritten cone for comparison."""
    root = canon(t, t, T, P)
    if root[0] == "c":
        return {"t": t, "nodes": 0, "constant": root[1]}
    seen = {root}
    stack = [root]
    edges = 0
    capped = False
    while stack:
        _, s, k = stack.pop()
        for kk in (k - 2, k - 1, k):
            ch = canon(s - 1, kk, T, P)
            edges += 1
            if ch[0] == "n" and ch not in seen:
                seen.add(ch)
                if len(seen) > node_cap:
                    capped = True
                    stack = []
                    break
                stack.append(ch)
    cone = sum(min(t, 2 * s) - max(0, 2 * s - t) + 1 for s in range(t + 1))
    irreducible = 0
    for s in range(t + 1):
        lo, hi = cone_interval(t, s)
        for k in range(lo, hi + 1):
            if T[k] is not None and s < T[k] + P[k]:
                irreducible += 1
    return {"t": t, "nodes": len(seen), "edges": edges, "capped": capped,
            "naive_cone_cells": cone,
            "irreducible_cone_cells": irreducible,
            "reduction_factor": (cone / len(seen)) if seen else None,
            "nodes_over_t_squared": len(seen) / (t * t) if t else None}


def dag_growth(ts: Sequence[int], T, P) -> List[Dict[str, object]]:
    return [diagonal_proof_dag(t, T, P) for t in ts]


# ==========================================================================
# 9. Consequences of the periodicity hypothesis (all conditional)
# ==========================================================================


def hypothesis_consequences(T, P, K_max: int, p_list: Sequence[int]):
    """(H) : the centre column is eventually p-periodic after T*.

    Nothing here is a proof of anything about (H).  These are the statements
    that follow from (H) together with theorems already proved in Phases 1-2D,
    reported with the range on which their non-trivial content is testable."""
    out = []
    for p in p_list:
        # x_{t+p}(0) = x_t(0)  <=>  w_{t+p}(t+p) = w_t(t)
        # If T(K) <= K - p for some K then Theorem D1 (Phase 2C) has content
        # at level K.  Report where that happens.
        content = [K for K in range(K_max + 1)
                   if T[K] is not None and T[K] <= K - p]
        out.append({"p": p, "levels_where_D1_has_content": len(content),
                    "first": content[:5]})
    return {"per_p": out,
            "note": "empty on the whole computed range: Theorem D1 is vacuous"}


def diagonal_lines(t_max: int, r_list: Sequence[int],
                   q_max: int = 64) -> Dict[str, object]:
    """The lines t -> w_t(t + r) are exactly the original columns x_t(r).

    Under (H) together with Phase 1 Theorem W2', AT MOST ONE of these lines can
    be eventually periodic.  This routine only extracts the lines and reports
    finite period scans -- a finite scan can refute periodicity for a given
    (period, preperiod) budget, never establish it."""
    orb, C = PL._orbit_rows(t_max)
    out = {}
    for r in r_list:
        seq = [(row >> (C + r)) & 1 for row in orb]
        n = len(seq)
        best = None
        per_q = {}
        for q in range(1, q_max + 1):
            last = -1
            for i in range(n - q):
                if seq[i] != seq[i + q]:
                    last = i
            pre = last + 1                      # minimal preperiod for this lag
            per_q[q] = pre
            if best is None or pre + q < best[0]:
                best = (pre + q, q, pre)
        out[str(r)] = {"length": n, "min_T_plus_p": best[0],
                       "argmin_p": best[1], "preperiod_there": best[2],
                       "smallest_preperiod_over_q": min(per_q.values())}
    return {"lines": out, "t_max": t_max, "q_max": q_max,
            "statement": "for every line and every lag q <= q_max, the lag-q "
                         "comparison still fails at the index reported, so any "
                         "eventual q-periodicity must start later than that",
            "control": "this REFUTES bounded (preperiod, period) budgets; it "
                       "never establishes periodicity and never proves any "
                       "line aperiodic"}
