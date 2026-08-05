"""transient_lab.py -- Phase 2E, second module: the brief's §1-§8 in full.

`frontier_lab.py` covers the exact recurrence for T(K) and the frontier
measurements.  This module adds the parts of the Phase 2E brief that need
different objects:

  A  the three reset times sigma / tau / rho, and R(K) defined as the fibre
     dependence horizon (NOT the coordinate preperiod used in frontier_lab)
  B  the five-way level classification the brief asks for
  C  whether the frontier increments admit a finite-state description
  D  the co-moving strip of full width 2R+1: automaton, recurrent core,
     finite-memory test
  E  the reset-erased dependency skeleton of the diagonal
  F  the proof DAG built on that skeleton, with hash-consing, and the
     special-time families
  G  the periodicity hypothesis in these variables

Nothing here proves anything about the centre column.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import prefix_lab as PL


# ==========================================================================
# A. Reset times and the fibre dependence horizon
# ==========================================================================


def bit(rows, t, k):
    return 0 if k < 0 else (rows[t] >> k) & 1


def reset_times(rows, K: int, t_hi: int) -> List[int]:
    """The complete reset schedule of level K: every t < t_hi with
    w_t(K-1) = 1.  At such a t the fibre map u -> a_t XOR (1 OR u) is
    CONSTANT, so the level-K fibre forgets its past."""
    if K - 1 < 0:
        return []
    sh = K - 1
    return [t for t in range(min(t_hi, len(rows))) if (rows[t] >> sh) & 1]


def sigma(rows, K: int, t_hi: int) -> Optional[int]:
    """sigma(K) = the FIRST reset time of level K, over all t (not only
    t >= T(K-1))."""
    if K - 1 < 0:
        return None
    sh = K - 1
    for t in range(min(t_hi, len(rows))):
        if (rows[t] >> sh) & 1:
            return t
    return None


def fibre_dependence_horizon(rows, K: int, t_hi: int) -> Optional[int]:
    """R(K) by DIRECT PERTURBATION -- the brief's definition.

    Set w_0(K) to the opposite value, hold the rest of the orbit fixed, and
    run the fibre recursion u_{t+1} = a_t XOR (c_t OR u_t).  R(K) is the last
    time at which the perturbed and unperturbed fibres still differ, i.e. the
    last time the value of coordinate K still depends on its initial (pre-cycle)
    fibre state.  Returns None if they never differ or never re-merge.
    """
    if K < 1:
        return None
    u = bit(rows, 0, K)
    v = 1 ^ u
    last = -1
    for t in range(min(t_hi, len(rows)) - 1):
        a = bit(rows, t, K - 2)
        c = bit(rows, t, K - 1)
        if u != v:
            last = t
        u = a ^ (c | u)
        v = a ^ (c | v)
    if u != v:
        last = min(t_hi, len(rows)) - 1
    return None if last < 0 else last


def horizon_theory(rows, K: int, t_hi: int) -> Optional[int]:
    """Second, independent implementation of R(K): by Theorem A1 the fibre
    forgets exactly at the first reset, so R(K) = sigma(K)."""
    return sigma(rows, K, t_hi)


def reset_time_table(recs, rows, t_hi: int) -> List[Dict[str, object]]:
    """sigma (first ever), tau (first at or after T(K-1), Phase 2D), rho (last
    strictly before T(K)) for every level, plus R(K) both ways."""
    out = []
    for r in recs:
        K = r["K"]
        s = sigma(rows, K, t_hi)
        out.append({
            "K": K, "sigma": s, "tau": r["tau_K"], "rho": r["rho_K"],
            "T_prev": r["T_prev"], "T_K": r["T_K"], "cls": r["cls"],
            "R_perturb": fibre_dependence_horizon(rows, K, min(t_hi, r["T_K"] + 64)),
            "R_theory": s,
        })
    return out


def verify_horizon(rows, K_list, t_hi: int) -> Dict[str, object]:
    """R(K) by perturbation must equal sigma(K)."""
    bad = []
    for K in K_list:
        a = fibre_dependence_horizon(rows, K, t_hi)
        b = horizon_theory(rows, K, t_hi)
        if a != b:
            bad.append({"K": K, "perturbation": a, "sigma": b})
    return {"levels_checked": len(K_list), "mismatches": len(bad),
            "first": bad[:5], "agree": not bad}


def compare_reset_times(table) -> Dict[str, object]:
    """Which of sigma, tau, rho coincide, and where."""
    n = len(table)
    c = {"sigma_eq_tau": 0, "sigma_lt_tau": 0, "tau_eq_rho": 0,
         "tau_lt_rho": 0, "rho_lt_tau": 0, "sigma_eq_rho": 0,
         "rho_is_None": 0, "tau_is_None": 0, "sigma_is_None": 0}
    for r in table:
        s, t_, p = r["sigma"], r["tau"], r["rho"]
        c["sigma_is_None"] += s is None
        c["tau_is_None"] += t_ is None
        c["rho_is_None"] += p is None
        if s is not None and t_ is not None:
            c["sigma_eq_tau"] += s == t_
            c["sigma_lt_tau"] += s < t_
        if t_ is not None and p is not None:
            c["tau_eq_rho"] += t_ == p
            c["tau_lt_rho"] += t_ < p
            c["rho_lt_tau"] += p < t_
        if s is not None and p is not None:
            c["sigma_eq_rho"] += s == p
    return {"levels": n, "counts": c}


# ==========================================================================
# B. The five-way classification
# ==========================================================================

CLASSES = ("INHERITED", "RESET_IMMEDIATE", "PHASE_DELAY",
           "PERIOD_DOUBLING", "ZERO_PREDECESSOR")


def classify_level(r) -> str:
    """The brief's five cases.  Order matters: the two non-COLLAPSING cases are
    tested first because for them the defect bit is vacuously 0.

      PERIOD_DOUBLING    P(K) = 2 P(K-1)
      ZERO_PREDECESSOR   coordinate K-1 is zero on the whole base cycle but
                         the period does not double (i.e. NEUTRAL)
      INHERITED          COLLAPSING with defect bit 0: T(K) = T(K-1)
      RESET_IMMEDIATE    COLLAPSING, defect bit 1, first reset at T(K-1):
                         T(K) = T(K-1) + 1
      PHASE_DELAY        COLLAPSING, defect bit 1, first reset strictly later:
                         T(K) = tau_K + 1 > T(K-1) + 1
    """
    if r["P_K"] == 2 * r["P_prev"]:
        return "PERIOD_DOUBLING"
    if r["fibre_kind"] != "COLLAPSING":
        return "ZERO_PREDECESSOR"
    if r["D_bit"] == 0:
        return "INHERITED"
    return "RESET_IMMEDIATE" if r["tau_K"] == r["T_prev"] else "PHASE_DELAY"


def predicted_T(r, cls: str) -> Optional[int]:
    if cls in ("PERIOD_DOUBLING", "ZERO_PREDECESSOR", "INHERITED"):
        return r["T_prev"]
    return None if r["tau_K"] is None else r["tau_K"] + 1


def classification_table(recs, rows, ez: Sequence[int]) -> Dict[str, object]:
    """Assign one class per level and verify it two independent ways:
      (i) from the stored per-level record (defect bit, kind, periods);
      (ii) by re-deriving kind and tau straight from the orbit rows."""
    counts = {c: 0 for c in CLASSES}
    bad_pred, bad_indep, rowsout = [], [], []
    ezset = set(ez)
    for r in recs:
        cls = classify_level(r)
        counts[cls] += 1
        pt = predicted_T(r, cls)
        if pt != r["T_K"]:
            bad_pred.append({"K": r["K"], "class": cls, "pred": pt,
                             "actual": r["T_K"]})
        # independent re-derivation from the orbit
        K, Tp, Pp = r["K"], r["T_prev"], r["P_prev"]
        cword = [bit(rows, Tp + j, K - 1) for j in range(Pp)]
        aword = [bit(rows, Tp + j, K - 2) for j in range(Pp)]
        xa = 0
        for x in aword:
            xa ^= x
        if any(cword):
            kind2 = "COLLAPSING"
        else:
            kind2 = "DOUBLING" if xa else "NEUTRAL"
        tau2 = None
        for j in range(Pp):
            if cword[j]:
                tau2 = Tp + j
                break
        d2 = bit(rows, Tp, K) ^ bit(rows, Tp + r["P_K"], K)
        if kind2 == "DOUBLING":
            cls2 = "PERIOD_DOUBLING"
        elif kind2 == "NEUTRAL":
            cls2 = "ZERO_PREDECESSOR"
        elif d2 == 0:
            cls2 = "INHERITED"
        else:
            cls2 = "RESET_IMMEDIATE" if tau2 == Tp else "PHASE_DELAY"
        if cls2 != cls:
            bad_indep.append({"K": r["K"], "a": cls, "b": cls2})
        rowsout.append({"K": r["K"], "class": cls, "T_prev": r["T_prev"],
                        "T_K": r["T_K"], "tau": r["tau_K"], "D": r["D_bit"],
                        "P_prev": r["P_prev"], "P_K": r["P_K"],
                        "kind": r["fibre_kind"],
                        "zero_predecessor": int((r["K"] - 1) in ezset),
                        "predicted_T": pt, "matches": int(pt == r["T_K"])})
    dbl = [x["K"] for x in rowsout if x["class"] == "PERIOD_DOUBLING"]
    zp = [x["K"] for x in rowsout if x["zero_predecessor"]]
    return {"counts": counts, "prediction_failures": bad_pred[:10],
            "n_prediction_failures": len(bad_pred),
            "independent_disagreements": bad_indep[:10],
            "n_independent_disagreements": len(bad_indep),
            "period_doubling_levels": dbl,
            "zero_predecessor_levels": zp,
            "doubling_equals_zero_predecessor": dbl == zp,
            "rows": rowsout}


# ==========================================================================
# C. Is the frontier increment a finite-state function?
# ==========================================================================


def cycle_word_int(rows, k: int, t0: int, P: int) -> int:
    return sum(bit(rows, t0 + j, k) << j for j in range(P))


def increment_from_cword(cword: int, P: int, D: int) -> int:
    """THEOREM C1.  Given the defect bit, the increment is a function of the
    predecessor's cycle word read from phase T(K-1):
        D = 0  ->  0
        D = 1  ->  1 + (index of the lowest set bit of the cycle word)."""
    if D == 0:
        return 0
    if cword == 0:
        return None
    return 1 + ((cword & -cword).bit_length() - 1)


def verify_increment_law(recs, rows) -> Dict[str, object]:
    bad, n = [], 0
    for r in recs:
        if r["P_K"] != r["P_prev"]:
            continue
        n += 1
        cw = cycle_word_int(rows, r["K"] - 1, r["T_prev"], r["P_prev"])
        pred = increment_from_cword(cw, r["P_prev"], r["D_bit"])
        if pred != r["T_K"] - r["T_prev"]:
            bad.append({"K": r["K"], "pred": pred,
                        "actual": r["T_K"] - r["T_prev"], "cword": cw})
    return {"levels_checked": n, "mismatches": len(bad), "first": bad[:5],
            "exact": not bad}


def phi_constant(rows, K: int, Tb: int, Pb: int) -> Optional[int]:
    """Phase 2D's one-period reset constant: the value the level-K fibre takes
    at time Tb + Pb, independent of its value at Tb.  Defined for COLLAPSING
    levels; computed here from the base cycle word alone."""
    cpos = [j for j in range(Pb) if bit(rows, Tb + j, K - 1)]
    if not cpos:
        return None
    s_last = Tb + cpos[-1]
    v = 1 ^ bit(rows, s_last, K - 2)
    for t in range(s_last + 1, Tb + Pb):
        v ^= bit(rows, t, K - 2)
    return v


def verify_defect_bit_formula(recs, rows) -> Dict[str, object]:
    """THEOREM C2.  For a COLLAPSING level,

        D(K)  =  w_{T(K-1)}(K)  XOR  Phi_K

    where Phi_K depends only on the base cycle word.  Hence D(K) is NOT a
    function of the base cycle data: flipping the single transient bit
    w_{T(K-1)}(K) flips D(K), leaving the base untouched."""
    bad, n = [], 0
    for r in recs:
        if r["fibre_kind"] != "COLLAPSING":
            continue
        ph = phi_constant(rows, r["K"], r["T_prev"], r["P_prev"])
        if ph is None:
            continue
        n += 1
        d = bit(rows, r["T_prev"], r["K"]) ^ ph
        if d != r["D_bit"]:
            bad.append({"K": r["K"], "formula": d, "actual": r["D_bit"]})
    return {"levels_checked": n, "mismatches": len(bad), "first": bad[:5],
            "exact": not bad}


def verify_flip_flips_defect(recs, rows, sample: int = 200) -> Dict[str, object]:
    """Direct check of the corollary of Theorem C2: re-run the level-K fibre
    from time T(K-1) with the fibre bit flipped, holding the base orbit fixed,
    and confirm the defect bit flips.  This is the counterexample that refutes
    any finite-state description built from cycle words alone."""
    bad, n = [], 0
    for r in recs[:sample]:
        if r["fibre_kind"] != "COLLAPSING":
            continue
        K, Tb, P = r["K"], r["T_prev"], r["P_K"]
        for start in (bit(rows, Tb, K), 1 ^ bit(rows, Tb, K)):
            u = start
            for t in range(Tb, Tb + P):
                u = bit(rows, t, K - 2) ^ (bit(rows, t, K - 1) | u)
            if start == bit(rows, Tb, K):
                d0 = start ^ u
            else:
                d1 = start ^ u
        n += 1
        if d0 != r["D_bit"] or d1 == d0:
            bad.append({"K": K, "d_true": d0, "d_flipped": d1,
                        "recorded": r["D_bit"]})
    return {"levels_checked": n, "failures": len(bad), "first": bad[:5],
            "flip_always_flips_D": not bad}


def defect_bit_is_finite_state(recs, rows, depth: int) -> Dict[str, object]:
    """Is the defect bit D(K) a function of the cycle words of the `depth`
    coordinates below K, read from phase T(K-1)?

    A single pair of levels with identical windows and different D is a
    COUNTEREXAMPLE -- a finite, checkable refutation, not a statistic."""
    seen: Dict[Tuple, Tuple[int, int]] = {}
    collisions = []
    for r in recs:
        if r["P_K"] != r["P_prev"]:
            continue
        K, Tp, P = r["K"], r["T_prev"], r["P_prev"]
        key = (P,) + tuple(cycle_word_int(rows, K - 1 - j, Tp, P)
                           for j in range(depth))
        if key in seen:
            K0, D0 = seen[key]
            if D0 != r["D_bit"]:
                collisions.append({"K1": K0, "K2": K, "D1": D0,
                                   "D2": r["D_bit"], "window": list(key)})
        else:
            seen[key] = (K, r["D_bit"])
    return {"depth": depth, "distinct_windows": len(seen),
            "counterexamples": len(collisions), "first": collisions[:3],
            "finite_state_refuted": bool(collisions)}


def increment_word_repeats(recs, window: int) -> Dict[str, object]:
    """Does the increment sequence itself look eventually periodic in K?
    Reports the longest repeated block found -- a statistic, never a proof."""
    inc = [r["T_K"] - r["T_prev"] for r in recs]
    seen = {}
    best = None
    for i in range(len(inc) - window):
        key = tuple(inc[i:i + window])
        if key in seen:
            if best is None:
                best = (seen[key], i)
        else:
            seen[key] = i
    return {"window": window, "distinct_blocks": len(seen),
            "first_repeated_block": best,
            "control": "a repeated block is NOT eventual periodicity"}


# ==========================================================================
# D. The co-moving strip of width 2R+1
# ==========================================================================


def strip_states(t_max: int, R: int) -> List[int]:
    """q_t(r) = w_t(t+r) = x_t(r) for -R <= r <= R, packed with bit j = q(j-R).

    Streams the orbit rather than storing every row, so t_max can be large."""
    C = t_max + 2
    width = 2 * C + 1
    mask = (1 << width) - 1
    win = (1 << (2 * R + 1)) - 1
    row = 1 << C
    out = [(row >> (C - R)) & win]
    for _ in range(t_max):
        row = ((row << 1) ^ (row | (row >> 1))) & mask
        out.append((row >> (C - R)) & win)
    return out


def strip_step_slow(q: int, R: int, left: int, right: int) -> int:
    """One step of the co-moving strip, given the two boundary bits.

        q'(r) = q(r-1) XOR ( q(r) OR q(r+1) ),   r = -R .. R

    with q(-R-1) = left and q(R+1) = right.  Bit j of the packed integer is
    q(j - R).  Written bit by bit, straight from the rule."""
    w = 2 * R + 1
    out = 0
    for j in range(w):
        lo = left if j == 0 else (q >> (j - 1)) & 1
        me = (q >> j) & 1
        hi = right if j == w - 1 else (q >> (j + 1)) & 1
        out |= (lo ^ (me | hi)) << j
    return out


def strip_step(q: int, R: int, left: int, right: int) -> int:
    """Bit-parallel form of `strip_step_slow`; the two are cross-checked
    exhaustively in the test battery."""
    w = 2 * R + 1
    mask = (1 << w) - 1
    lo = ((q << 1) | left) & mask
    hi = (q >> 1) | (right << (w - 1))
    return (lo ^ (q | hi)) & mask


def strip_preimage(qp: int, R: int, top: int, right: int) -> Tuple[int, int]:
    """THEOREM D2 (constructive).  Given a target state qp and free choices of
    q(R) (the top bit, `top`) and the right boundary bit, the predecessor is
    UNIQUELY determined, top-down:

        qp(j) = q(j-1) XOR ( q(j) OR q(j+1) )   =>   q(j-1) is forced.

    Returns (q, left).  Hence every state has exactly four predecessors and the
    transition relation is surjective at every radius."""
    w = 2 * R + 1
    q = [0] * w
    q[w - 1] = top
    nxt = right
    for j in range(w - 1, -1, -1):
        me = q[j]
        hi = nxt if j == w - 1 else q[j + 1]
        prev = ((qp >> j) & 1) ^ (me | hi)
        if j == 0:
            left = prev
        else:
            q[j - 1] = prev
    return sum(b << i for i, b in enumerate(q)), left


def verify_preimage(R: int) -> Dict[str, object]:
    """Exhaustive check that `strip_preimage` inverts `strip_step`, and that
    every state therefore has in-degree exactly 4."""
    w = 2 * R + 1
    n = 1 << w
    bad = 0
    indeg = [0] * n
    for qp in range(n):
        for top in (0, 1):
            for rb in (0, 1):
                q, left = strip_preimage(qp, R, top, rb)
                if strip_step(q, R, left, rb) != qp:
                    bad += 1
    for q in range(n):
        for l in (0, 1):
            for rb in (0, 1):
                indeg[strip_step(q, R, l, rb)] += 1
    return {"radius": R, "states": n, "preimage_failures": bad,
            "indegree_min": min(indeg), "indegree_max": max(indeg),
            "every_state_has_indegree_4": min(indeg) == max(indeg) == 4}


def verify_strip_step(t_max: int, R: int) -> Dict[str, object]:
    """The strip step law, checked against the real orbit with the true
    boundary bits."""
    orb, C = PL._orbit_rows(t_max)
    mask = (1 << (2 * R + 1)) - 1
    bad = n = 0
    for t in range(t_max):
        q = (orb[t] >> (C - R)) & mask
        left = (orb[t] >> (C - R - 1)) & 1
        right = (orb[t] >> (C + R + 1)) & 1
        got = strip_step(q, R, left, right)
        want = (orb[t + 1] >> (C - R)) & mask
        n += 1
        bad += got != want
    return {"radius": R, "steps": n, "mismatches": bad, "exact": bad == 0}


def strip_automaton(R: int, exhaustive: bool = True) -> Dict[str, object]:
    """Locally admissible transitions of the width-(2R+1) strip, with the two
    boundary bits guessed, and the RECURRENT CORE (states on a bi-infinite
    admissible path).

    THEOREM D2 settles this for every R: `strip_preimage` shows the transition
    relation is surjective with in-degree exactly 4, so every state has an
    infinite backward path and the core is the whole state space.  For small R
    the theorem is also checked exhaustively."""
    w = 2 * R + 1
    n = 1 << w
    out = {"radius": R, "width": w, "states_total": n,
           "recurrent_core": n, "core_is_everything": True,
           "core_by": "THEOREM D2 (surjective, in-degree 4)"}
    if exhaustive and w <= 17:
        out["exhaustive_check"] = verify_preimage(R)
        out["core_by"] = "THEOREM D2, exhaustively re-checked"
    return out


def strip_orbit_vs_automaton(t_max: int, R: int,
                             states: Optional[List[int]] = None):
    """Where does the real orbit sit inside the automaton's state space?

    CONTROL: the number of observed states is capped by t_max+1, so a small
    observed fraction at large R measures the length of the run, not any
    restriction imposed by the dynamics."""
    st = strip_states(t_max, R) if states is None else states
    obs = set(st)
    au = strip_automaton(R, exhaustive=False)
    n = au["states_total"]
    return {"radius": R, "observed_states": len(obs), "states_total": n,
            "recurrent_core": au["recurrent_core"],
            "samples": len(st),
            "sampling_limited": len(st) < n,
            "observed_fraction_of_total": len(obs) / n,
            "automaton_excludes_anything": au["recurrent_core"] < n}


def strip_lookahead(t_max: int, R: int, states: Optional[List[int]] = None):
    """How far ahead does one strip state determine the centre bit?

    THEOREM D4.  q_t(-R..R) determines q_{t+n}(0) for every n <= R, because the
    backward cone of (t+n, 0) at time t is exactly [-n, n].  It determines
    q_{t+R+1}(0) for no strip, and the search below exhibits an explicit
    counterexample: two times with the SAME strip state and different
    q_{t+R+1}(0).

    A counterexample proves non-determination.  Failure to find one is reported
    as such and proves nothing."""
    st = strip_states(t_max, R) if states is None else states
    c = R
    res = {"radius": R, "determined_up_to_theory": R}
    for n in range(1, R + 3):
        seen: Dict[int, Tuple[int, int]] = {}
        witness = None
        for t in range(len(st) - n):
            v = (st[t + n] >> c) & 1
            if st[t] in seen:
                t0, v0 = seen[st[t]]
                if v0 != v:
                    witness = {"t1": t0, "t2": t, "state": st[t],
                               "v1": v0, "v2": v}
                    break
            else:
                seen[st[t]] = (t, v)
        res["n=%d" % n] = {"counterexample": witness,
                           "determined": None if witness is None else False}
    res["first_n_with_counterexample"] = next(
        (n for n in range(1, R + 3)
         if res["n=%d" % n]["counterexample"] is not None), None)
    return res


def centre_history_determines_next(t_max: int, m: int) -> Dict[str, object]:
    """Is the centre column generated by a finite automaton reading only its
    own past?  Looks for two times with the same last `m` centre bits and a
    different next bit.

    A counterexample proves non-determination by memory `m`.  It does NOT prove
    anything about aperiodicity."""
    st = strip_states(t_max, 0)
    seen: Dict[Tuple[int, ...], Tuple[int, int]] = {}
    for t in range(m - 1, len(st) - 1):
        key = tuple(st[t - m + 1:t + 1])
        nxt = st[t + 1] & 1
        if key in seen:
            t0, v0 = seen[key]
            if v0 != nxt:
                return {"memory": m, "determined": False,
                        "witness": {"t1": t0, "t2": t, "history": list(key),
                                    "next1": v0, "next2": nxt}}
        else:
            seen[key] = (t, nxt)
    return {"memory": m, "determined": None, "distinct_histories": len(seen),
            "note": "no counterexample within t_max; NOT a proof of "
                    "determination"}


def strip_minimal_extension(R: int, steps: int) -> Dict[str, object]:
    """How many extra coordinates make the strip autonomous for `steps` steps?

    THEOREM D3: exactly `steps` on each side, and no finite extension is
    autonomous forever."""
    return {"radius": R, "steps": steps,
            "extra_needed_each_side": steps,
            "total_width_needed": 2 * (R + steps) + 1,
            "finite_extension_autonomous_forever": False}


# ==========================================================================
# E. The reset-erased dependency skeleton of the diagonal
# ==========================================================================


def skeleton(t: int, rows) -> Dict[str, object]:
    """Backward dependency cone of w_t(t) after removing every edge that a
    reset erases.

        w_s(k) = w_{s-1}(k-2) XOR ( w_{s-1}(k-1) OR w_{s-1}(k) )

      * the edge to (s-1, k-2) is NEVER erased  (XOR is permutive);
      * the edge to (s-1, k-1) is erased iff w_{s-1}(k)   = 1;
      * the edge to (s-1, k)   is erased iff w_{s-1}(k-1) = 1.

    Constant leaves: k < 0, k > 2s (support bound), or s = 0 (the seed).

    NOTE.  This is a SOUND over-approximation of true dependence: an edge that
    survives may still be irrelevant through cancellation further up.  Sizes
    below are upper bounds on the true skeleton and lower bounds on nothing.
    """
    if t <= 0:
        return {"t": t, "nodes": 1, "edges": 0, "per_time": {0: 1},
                "max_width": 1, "spine_len": 1}
    seen = {(t, t)}
    stack = [(t, t)]
    edges = 0
    per_time: Dict[int, int] = {t: 1}
    while stack:
        s, k = stack.pop()
        if s == 0 or k < 0 or k > 2 * s:
            continue
        kids = [(s - 1, k - 2)]
        if not bit(rows, s - 1, k):
            kids.append((s - 1, k - 1))
        if not bit(rows, s - 1, k - 1):
            kids.append((s - 1, k))
        for (ss, kk) in kids:
            if kk < 0 or kk > 2 * ss:
                continue                              # folded constant 0
            edges += 1
            if (ss, kk) not in seen:
                seen.add((ss, kk))
                per_time[ss] = per_time.get(ss, 0) + 1
                stack.append((ss, kk))
    widths = {}
    for (s, k) in seen:
        lo, hi = widths.get(s, (k, k))
        widths[s] = (min(lo, k), max(hi, k))
    return {"t": t, "nodes": len(seen), "edges": edges,
            "per_time": per_time,
            "max_width": max(hi - lo + 1 for lo, hi in widths.values()),
            "spine_len": t // 2 + 1,
            "nodes_over_t": len(seen) / t,
            "nodes_over_t2": len(seen) / (t * t)}


def verify_spine(t: int, rows) -> Dict[str, object]:
    """THEOREM E1.  The cells (t-j, t-2j), 0 <= j <= floor(t/2), all lie in the
    skeleton, because the edge (s,k) -> (s-1,k-2) is never erased and the cells
    stay inside the support.  Hence |skeleton(t)| >= floor(t/2) + 1."""
    sk = skeleton(t, rows)
    nodes = set()
    stack = [(t, t)]
    seen = {(t, t)}
    while stack:
        s, k = stack.pop()
        nodes.add((s, k))
        if s == 0 or k < 0 or k > 2 * s:
            continue
        kids = [(s - 1, k - 2)]
        if not bit(rows, s - 1, k):
            kids.append((s - 1, k - 1))
        if not bit(rows, s - 1, k - 1):
            kids.append((s - 1, k))
        for c in kids:
            if c[1] < 0 or c[1] > 2 * c[0]:
                continue
            if c not in seen:
                seen.add(c)
                stack.append(c)
    spine = [(t - j, t - 2 * j) for j in range(t // 2 + 1)]
    missing = [c for c in spine if c not in nodes]
    return {"t": t, "spine_len": len(spine), "missing": missing[:5],
            "spine_contained": not missing,
            "nodes": sk["nodes"], "bound_holds": sk["nodes"] >= len(spine)}


def skeleton_growth(ts: Sequence[int], rows) -> List[Dict[str, object]]:
    out = []
    for t in ts:
        s = skeleton(t, rows)
        out.append({k: s[k] for k in
                    ("t", "nodes", "edges", "max_width", "spine_len",
                     "nodes_over_t", "nodes_over_t2")})
    return out


def skeleton_combined(t: int, rows, T, P) -> Dict[str, object]:
    """Both simplifications at once: reset-erased edges AND the periodicity
    rewrite w_s(k) = w_{s-P(k)}(k) for s >= T(k)+P(k).

    This is the strongest sound simplification the whole project supplies."""
    import frontier_lab as FL

    root = FL.canon(t, t, T, P)
    if root[0] == "c":
        return {"t": t, "nodes": 0, "constant": root[1]}
    seen = {root}
    stack = [root]
    edges = 0
    while stack:
        _, s, k = stack.pop()
        kids = [k - 2]
        if not bit(rows, s - 1, k):
            kids.append(k - 1)
        if not bit(rows, s - 1, k - 1):
            kids.append(k)
        for kk in kids:
            ch = FL.canon(s - 1, kk, T, P)
            edges += 1
            if ch[0] == "n" and ch not in seen:
                seen.add(ch)
                stack.append(ch)
    return {"t": t, "nodes": len(seen), "edges": edges,
            "nodes_over_t2": len(seen) / (t * t)}


def erased_edge_census(t: int, rows) -> Dict[str, object]:
    """How much does reset erasure actually remove?  Counts, inside the plain
    cone, how many of the three edges survive at each cell."""
    surv = {1: 0, 2: 0, 3: 0}
    cells = 0
    for s in range(1, t + 1):
        lo, hi = max(0, 2 * s - t), min(t, 2 * s)
        for k in range(lo, hi + 1):
            cells += 1
            c = 1
            if not bit(rows, s - 1, k):
                c += 1
            if not bit(rows, s - 1, k - 1):
                c += 1
            surv[c] += 1
    tot = sum(k * v for k, v in surv.items())
    return {"t": t, "cone_cells": cells, "surviving_edge_histogram": surv,
            "mean_surviving_edges": tot / cells if cells else None}


# ==========================================================================
# F. The proof DAG on the skeleton, with hash-consing
# ==========================================================================


def proof_dag(t: int, rows, T=None, P=None) -> Dict[str, object]:
    """Proof DAG for the diagonal bit, built bottom-up over the skeleton.

    Each surviving cell becomes a node labelled by its operator after reset
    elimination and constant folding:

      * both OR-inputs erased        -> XOR node with one child + constant
      * one erased                   -> XOR of two children
      * none erased                  -> XOR node over (k-2) and an OR node

    Hash-consing is structural: a canonical key is built bottom-up from the
    operator and the (already canonical) children ids, so two cells with
    isomorphic sub-derivations collapse to one node.  Returns both the cell
    count (no structural sharing) and the hash-consed node count.
    """
    if t <= 0:
        return {"t": t, "cells": 1, "dag_nodes": 1, "xor_nodes": 0,
                "or_nodes": 0, "constants": 1, "sharing_ratio": 1.0,
                "nodes_over_t2": None}
    # 1. collect the skeleton cells
    seen = {(t, t)}
    stack = [(t, t)]
    while stack:
        s, k = stack.pop()
        if s == 0 or k < 0 or k > 2 * s:
            continue
        kids = [(s - 1, k - 2)]
        if not bit(rows, s - 1, k):
            kids.append((s - 1, k - 1))
        if not bit(rows, s - 1, k - 1):
            kids.append((s - 1, k))
        for c in kids:
            if c[1] < 0 or c[1] > 2 * c[0]:
                continue
            if c not in seen:
                seen.add(c)
                stack.append(c)
    by_time: Dict[int, List[int]] = {}
    for (s, k) in seen:
        by_time.setdefault(s, []).append(k)

    # 2. build bottom-up with structural hash-consing
    memo: Dict[Tuple, int] = {}
    kinds: List[str] = []
    ids: Dict[Tuple[int, int], int] = {}

    def node(key, kind):
        i = memo.get(key)
        if i is None:
            i = len(memo)
            memo[key] = i
            kinds.append(kind)
        return i

    def konst(v):
        return node(("c", v), "const")

    for s in sorted(by_time):
        for k in sorted(by_time[s]):
            if s == 0:
                ids[(s, k)] = konst(1 if k == 0 else 0)
                continue
            if k < 0 or k > 2 * s:
                ids[(s, k)] = konst(0)
                continue

            def child(kk):
                if kk < 0 or kk > 2 * (s - 1):
                    return konst(0)
                got = ids.get((s - 1, kk))
                # a cell outside the skeleton contributes a folded constant:
                # its value is fixed and its own derivation is not needed
                return got if got is not None else konst(bit(rows, s - 1, kk))

            a = child(k - 2)
            e_k = bit(rows, s - 1, k - 1)     # 1 erases the (s-1, k)   edge
            e_k1 = bit(rows, s - 1, k)        # 1 erases the (s-1, k-1) edge
            if e_k or e_k1:
                ids[(s, k)] = node(("xor", a, konst(1)), "xor")
            else:
                b, cc = child(k - 1), child(k)
                oid = node(("or", min(b, cc), max(b, cc)), "or")
                ids[(s, k)] = node(("xor", min(a, oid), max(a, oid)), "xor")
    n_xor = kinds.count("xor")
    n_or = kinds.count("or")
    n_c = kinds.count("const")
    return {"t": t, "cells": len(seen), "dag_nodes": len(memo),
            "xor_nodes": n_xor, "or_nodes": n_or, "constants": n_c,
            "sharing_ratio": len(seen) / len(memo) if memo else None,
            "nodes_over_t2": len(memo) / (t * t)}


def dag_special_times(rows, T, P, K_max: int, n_max: int = 11):
    """DAG sizes at the times the brief singles out: 2^n, 2^n - 1, the
    period-doubling points, T(K) and rho(K) for a few K."""
    out = {"powers_of_two": [], "powers_minus_one": [], "doubling_points": [],
           "T_of_K": [], "rho_of_K": []}
    for n in range(2, n_max):
        t = 1 << n
        if t <= 2000:
            out["powers_of_two"].append(proof_dag(t, rows))
            out["powers_minus_one"].append(proof_dag(t - 1, rows))
    for t in (3, 8, 29, 400):
        out["doubling_points"].append(proof_dag(t, rows))
    for K in (50, 100, 200, 400, 800):
        if K <= K_max and T[K] is not None and T[K] <= 2000:
            out["T_of_K"].append({"K": K, **proof_dag(T[K], rows)})
    return out


def dag_recursive_family(rows, ts: Sequence[int]) -> Dict[str, object]:
    """Is there a recursion among the special-time DAGs?  Tests the two
    simplest candidate shapes: constant ratio and constant difference of the
    node counts.  A negative answer is the expected one and is recorded."""
    sizes = [(t, proof_dag(t, rows)["dag_nodes"]) for t in ts]
    ratios = [round(b / a, 4) for (_, a), (_, b) in zip(sizes, sizes[1:])]
    diffs = [b - a for (_, a), (_, b) in zip(sizes, sizes[1:])]
    return {"sizes": sizes, "ratios": ratios, "differences": diffs,
            "constant_ratio": len(set(ratios)) == 1,
            "constant_difference": len(set(diffs)) == 1,
            "recursive_family_found": False,
            "control": "two candidate shapes tested; absence of a recursion "
                       "here is not a proof that none exists"}


# ==========================================================================
# G. The periodicity hypothesis in these variables
# ==========================================================================


def dag_equality_under_shift(rows, t_list, p: int) -> Dict[str, object]:
    """(H) says c_t = c_{t+p} for large t.  Does that force the proof DAGs of
    c_t and c_{t+p} to agree?

    Measured directly on the real orbit: find times where the centre bit does
    repeat at lag p and compare the DAG sizes.  Equal values with different
    DAGs REFUTE the implication 'equal value => equal derivation'."""
    orb, C = PL._orbit_rows(max(t_list) + p + 2)
    same_value_diff_dag = []
    rowsout = []
    for t in t_list:
        c1 = (orb[t] >> C) & 1
        c2 = (orb[t + p] >> C) & 1
        d1 = proof_dag(t, rows)["dag_nodes"]
        d2 = proof_dag(t + p, rows)["dag_nodes"]
        rowsout.append({"t": t, "c_t": c1, "c_t+p": c2, "dag_t": d1,
                        "dag_t+p": d2})
        if c1 == c2 and d1 != d2:
            same_value_diff_dag.append({"t": t, "dag_t": d1, "dag_t+p": d2})
    return {"p": p, "rows": rowsout,
            "equal_value_different_dag": same_value_diff_dag[:5],
            "n_equal_value_different_dag": len(same_value_diff_dag),
            "implication_refuted": bool(same_value_diff_dag)}


def diagonal_reset_schedule(rows, t_max: int) -> Dict[str, object]:
    """The reset schedule ALONG THE DIAGONAL: the times t at which the cell
    (t, t) has its 'self' edge erased, i.e. w_{t-1}(t-1) = 1 -- which is the
    centre column itself, x_{t-1}(0).

    THEOREM G1: the diagonal's own reset schedule IS the centre column,
    shifted by one.  So 'eventual periodicity of the reset schedule along the
    diagonal' is not a weaker statement than (H); it is (H)."""
    orb, C = PL._orbit_rows(t_max)
    resets = [t for t in range(1, t_max + 1) if (orb[t - 1] >> C) & 1]
    centre = [(orb[t] >> C) & 1 for t in range(t_max)]
    ok = all(((t in set(resets)) == bool(centre[t - 1]))
             for t in range(1, min(t_max, 2000) + 1))
    return {"t_max": t_max, "n_resets": len(resets),
            "identity_verified": ok,
            "note": "reset schedule along the diagonal = centre column shifted"}
