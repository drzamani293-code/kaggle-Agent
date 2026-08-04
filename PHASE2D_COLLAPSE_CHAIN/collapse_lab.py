"""collapse_lab.py -- Phase 2D: the collapse chain of the edge-aligned prefix tower.

Builds on Phase 2C (prefix_lab.py).  Fibre at level K:

    u_{t+1} = a_t XOR ( c_t OR u_t ),    a_t = w_t(K-2), c_t = w_t(K-1), u_t = w_t(K)

COLLAPSING  <=> some c_t = 1 on the base cycle  (composition constant)
NEUTRAL     <=> c_t = 0 on the whole cycle and XOR_t a_t = 0  (identity)
DOUBLING    <=> c_t = 0 on the whole cycle and XOR_t a_t = 1  (negation)

Nothing here proves anything about the centre column.
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Sequence, Tuple
import prefix_lab as PL


# ---------------------------------------------------------------- levels ----
def level_data(K_max: int, T_max: int = None):
    """Exact per-level data: class, collapse times, reset value, cycle words."""
    T_max = T_max or int(1.7 * K_max) + 500
    T, P = PL.exact_TP_first_repeat(K_max, T_max)
    rows = PL.edge_rows(K_max, T_max)
    bit = lambda t, k: (rows[t] >> k) & 1
    out = []
    for K in range(2, K_max + 1):
        Tb, Pb = T[K - 1], P[K - 1]                  # base = (K-1)-prefix
        if Tb + Pb >= len(rows):
            break
        word = [(bit(t, K - 2), bit(t, K - 1)) for t in range(Tb, Tb + Pb)]
        cls = PL.classify_forcing_word(word)
        cpos = [j for j, (a, c) in enumerate(word) if c == 1]
        rec = {"K": K, "kind": cls["kind"], "T_base": Tb, "P_base": Pb,
               "T_K": T[K], "P_K": P[K],
               "first_c1_phase": cpos[0] if cpos else None,
               "last_c1_phase": cpos[-1] if cpos else None,
               "n_c1": len(cpos), "xor_a": cls["xor_of_a"]}
        if cpos:
            s_first = Tb + cpos[0]
            s_last = Tb + cpos[-1]
            rec["tau_K"] = s_first                    # first collapse time
            rec["erased_at"] = s_first + 1
            rec["reset_value_at_tau_plus_1"] = 1 ^ bit(s_first, K - 2)
            # exact one-period reset value:  NOT a_{s_last}  XOR  XOR of a after it
            v = 1 ^ bit(s_last, K - 2)
            for t in range(s_last + 1, Tb + Pb):
                v ^= bit(t, K - 2)
            rec["Phi_constant"] = v
            rec["measured_u_at_Tb_plus_Pb"] = bit(Tb + Pb, K)
        out.append(rec)
    return out, T, P, rows


def verify_collapse_formulas(levels, T, P, rows):
    """CHECK the exact formulas of COLLAPSE_RESET_THEORY.md against the orbit."""
    bit = lambda t, k: (rows[t] >> k) & 1
    n = bad_reset = bad_erase = bad_phi = bad_TK = 0
    for r in levels:
        if r["kind"] != "COLLAPSING":
            continue
        K, Tb, Pb = r["K"], r["T_base"], r["P_base"]
        n += 1
        # (i) value one step after the first collapse
        if bit(r["erased_at"], K) != r["reset_value_at_tau_plus_1"]:
            bad_reset += 1
        # (ii) Phi constant = the value one full period later
        if r["Phi_constant"] != r["measured_u_at_Tb_plus_Pb"]:
            bad_phi += 1
        # (iii) erasure: u is P_base-periodic from erased_at onwards
        if any(bit(t, K) != bit(t + Pb, K)
               for t in range(r["erased_at"], min(len(rows) - Pb, r["erased_at"] + 4 * Pb))):
            bad_erase += 1
        # (iv) sharpened transient bound  T(K) <= max(T(K-1), tau_K + 1)
        if r["T_K"] > max(Tb, r["tau_K"] + 1):
            bad_TK += 1
    return {"collapsing_levels_checked": n, "reset_value_mismatches": bad_reset,
            "phi_constant_mismatches": bad_phi, "erasure_failures": bad_erase,
            "sharp_transient_bound_violations": bad_TK,
            "all_exact": bad_reset == bad_phi == bad_erase == bad_TK == 0}


def eventually_zero_coordinates(K_max: int, T, P, rows):
    """Coordinates that are 0 throughout their own cycle."""
    return [k for k in range(K_max + 1)
            if all(not ((rows[t] >> k) & 1) for t in range(T[k], T[k] + P[k]))]


def permanently_zero_coordinates(K_max: int, rows):
    return [k for k in range(K_max + 1) if not any((W >> k) & 1 for W in rows)]


# ------------------------------------------------------------- chains ------
def cycle_word(rows, T_ref: int, P: int, k: int) -> Tuple[int, ...]:
    return tuple((rows[T_ref + j] >> k) & 1 for j in range(P))


def solve_fibre_cycle(a: Sequence[int], c: Sequence[int]):
    """Given one base period (a_j, c_j), return the P-periodic fibre word(s).

    COLLAPSING -> unique solution; NEUTRAL -> two; DOUBLING -> none of period P."""
    P = len(a)
    sols = []
    for u0 in (0, 1):
        u = u0
        seq = []
        for j in range(P):
            seq.append(u)
            u = a[j] ^ (c[j] | u)
        if u == u0:
            sols.append(tuple(seq))
    return sols


def chain_transducer_check(rows, T, P, K_lo: int, K_hi: int, T_ref: int):
    """THEOREM (chain transducer): along a run of COLLAPSING levels of common
    period P, the cycle word at level K is the unique P-periodic solution of the
    fibre recurrence driven by the cycle words at K-2 and K-1.  Memory = 2P bits."""
    Pc = P[K_lo]
    ok = True
    n = uniq = 0
    for K in range(K_lo, K_hi + 1):
        if P[K] != Pc or P[K - 1] != Pc or P[K - 2] != Pc:
            continue
        a = cycle_word(rows, T_ref, Pc, K - 2)
        c = cycle_word(rows, T_ref, Pc, K - 1)
        want = cycle_word(rows, T_ref, Pc, K)
        sols = solve_fibre_cycle(a, c)
        n += 1
        if len(sols) == 1:
            uniq += 1
        ok &= want in sols
    return {"levels_checked": n, "levels_with_unique_solution": uniq,
            "cycle_word_always_a_solution": bool(ok),
            "memory_bits": 2 * Pc, "period": Pc}


def chain_runs(levels):
    """Maximal runs of consecutive COLLAPSING levels."""
    runs = []
    start = None
    for r in levels:
        if r["kind"] == "COLLAPSING":
            if start is None:
                start = r["K"]
            last = r["K"]
        else:
            if start is not None:
                runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    return runs


# --------------------------------------------------------- diagonal cone ----
def periodic_frontier(T, K_max: int, s_max: int):
    """K_per(s) = max{ K : T(K) <= s }, for s = 0..s_max."""
    out = [-1] * (s_max + 1)
    K = 0
    for s in range(s_max + 1):
        while K + 1 <= K_max and T[K + 1] is not None and T[K + 1] <= s:
            K += 1
        out[s] = K if (T[0] is not None and T[0] <= s) else -1
    return out


def diagonal_cone(t: int, T, K_max: int):
    """Backward dependency cone of the diagonal point (t, t).

    One step back from (tau, k) reaches (tau-1, k-2), (tau-1, k-1), (tau-1, k),
    so at time s the cone is the index interval

        [ max(0, 2s - t) , min(t, 2s) ].

    Report, for each s, the cone width and how much of it lies in the already
    periodic region k <= K_per(s)."""
    kper = periodic_frontier(T, K_max, t)
    rows = []
    cross = None
    for s in range(t + 1):
        lo, hi = max(0, 2 * s - t), min(t, 2 * s)
        width = hi - lo + 1
        inper = max(0, min(hi, kper[s]) - lo + 1)
        rows.append({"s": s, "lo": lo, "hi": hi, "width": width,
                     "in_periodic": inper, "K_per": kper[s]})
        if inper == 0 and cross is None and s > 0:
            cross = s
    return {"t": t, "per_s": rows, "first_s_with_cone_entirely_transient": cross,
            "crossover_fraction": (cross / t) if (cross and t) else None}


# ------------------------------------------------------------------ ANF ----
def poly_xor(p: set, q: set) -> set:
    return p ^ q


def poly_mul(p: set, q: set) -> set:
    out = set()
    for m1 in p:
        for m2 in q:
            m = m1 | m2
            if m in out:
                out.discard(m)
            else:
                out.add(m)
    return out


def anf_rows(t_max: int, k_max: int):
    """ANF of w_t(k) in the free initial-row variables z_0..z_k.

    w_{t+1}(k) = w_t(k-2) + w_t(k-1) + w_t(k) + w_t(k-1)*w_t(k)   over GF(2)."""
    cur = [{1 << k} for k in range(k_max + 1)]        # w_0(k) = z_k
    hist = [cur]
    for _ in range(t_max):
        nxt = []
        for k in range(k_max + 1):
            A = cur[k - 2] if k - 2 >= 0 else set()
            B = cur[k - 1] if k - 1 >= 0 else set()
            C = cur[k]
            nxt.append(poly_xor(poly_xor(poly_xor(A, B), C), poly_mul(B, C)))
        cur = nxt
        hist.append(cur)
    return hist


def anf_stats(hist, t: int, k: int):
    p = hist[t][k]
    vars_used = 0
    for m in p:
        vars_used |= m
    deg = max((bin(m).count("1") for m in p), default=0)
    return {"t": t, "k": k, "monomials": len(p), "degree": deg,
            "variables_present": sorted(i for i in range(k + 1)
                                        if (vars_used >> i) & 1),
            "depends_on_z_k": any((m >> k) & 1 for m in p),
            "constant_term": int(0 in p)}


def anf_eval_seed(p: set) -> int:
    """Substitute the single-cell seed z_0 = 1, z_j = 0 for j >= 1."""
    v = 0
    for m in p:
        if m == 0 or m == 1:          # empty monomial or z_0 only
            v ^= 1
    return v


def verify_anf_against_orbit(t_max: int, k_max: int):
    hist = anf_rows(t_max, k_max)
    rows = PL.edge_rows(k_max, t_max)
    bad = n = 0
    for t in range(t_max + 1):
        for k in range(k_max + 1):
            n += 1
            bad += anf_eval_seed(hist[t][k]) != ((rows[t] >> k) & 1)
    return {"cells_checked": n, "mismatches": bad, "exact": bad == 0}
