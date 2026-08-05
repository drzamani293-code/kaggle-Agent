"""right_tower_lab.py -- Phase 2H section 6, hypothesis H3.

Section 2's candidate C' indexes the orbit by diagonals measured in from the
RIGHT light-cone edge.  Writing

        v_t(k) := x_t(t - k)                     (right-aligned coordinates)

the local rule becomes a ONE-SIDED recurrence in which every index on the
right-hand side is <= k:

        v_{t+1}(k) = v_t(k) XOR ( v_t(k-1) OR v_t(k-2) )               (RT)

(with the convention v_t(k) = 0 for k < 0).  So the prefix (v_t(0..K)) is
autonomous, exactly as the LEFT-aligned prefix of Phase 2E is autonomous under

        w_{t+1}(k) = w_t(k-2) XOR ( w_t(k-1) OR w_t(k) ).              (LT)

The two towers behave completely differently, and this file establishes why:
G_K (the right map) is a BIJECTION, F_K (the left map) is not.

The centre column is the moving diagonal of BOTH towers:
        x_t(0) = w_t(t) = v_t(t).
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def G_right(s: Tuple[int, ...]) -> Tuple[int, ...]:
    """One step of (RT) on a prefix of length K+1."""
    K = len(s) - 1
    return tuple(s[k] ^ ((s[k - 1] if k >= 1 else 0)
                         | (s[k - 2] if k >= 2 else 0)) for k in range(K + 1))


def F_left(s: Tuple[int, ...]) -> Tuple[int, ...]:
    """One step of (LT) -- Phase 2E's prefix map, for comparison."""
    K = len(s) - 1
    return tuple((s[k - 2] if k >= 2 else 0)
                 ^ ((s[k - 1] if k >= 1 else 0) | s[k]) for k in range(K + 1))


def G_inverse(u: Tuple[int, ...]) -> Tuple[int, ...]:
    """Explicit inverse of G_right, by induction on k:
           v(k) = u(k) XOR ( v(k-1) OR v(k-2) ).
    Its existence is the proof that G_K is a bijection."""
    v: List[int] = []
    for k in range(len(u)):
        a = v[k - 1] if k >= 1 else 0
        b = v[k - 2] if k >= 2 else 0
        v.append(u[k] ^ (a | b))
    return tuple(v)


def bijectivity_census(K_max: int = 12) -> List[Dict[str, object]]:
    """Exhaustive: is each map injective on {0,1}^(K+1)?"""
    out = []
    for K in range(0, K_max + 1):
        n = K + 1
        imL, imR = set(), set()
        collL = None
        for m in range(1 << n):
            s = tuple((m >> i) & 1 for i in range(n))
            a, b = F_left(s), G_right(s)
            if a in imL and collL is None:
                # find the earlier preimage for the record
                for m2 in range(m):
                    s2 = tuple((m2 >> i) & 1 for i in range(n))
                    if F_left(s2) == a:
                        collL = {"state_a": list(s2), "state_b": list(s),
                                 "common_image": list(a)}
                        break
            imL.add(a)
            imR.add(b)
        out.append({"K": K, "states": 1 << n,
                    "left_image_size": len(imL),
                    "left_bijective": len(imL) == (1 << n),
                    "right_image_size": len(imR),
                    "right_bijective": len(imR) == (1 << n),
                    "left_collision": collL})
    return out


def inverse_check(K_max: int = 14) -> Dict[str, object]:
    """G_inverse really inverts G_right, exhaustively."""
    bad = 0
    total = 0
    for K in range(0, K_max + 1):
        n = K + 1
        for m in range(1 << n):
            s = tuple((m >> i) & 1 for i in range(n))
            total += 1
            if G_inverse(G_right(s)) != s:
                bad += 1
    return {"K_max": K_max, "states_checked": total, "failures": bad}


def tower_periods(K_max: int, t_max: int, which: str = "right"):
    """Exact preperiod and period of each prefix orbit, by state hashing.
    Initial state: v_0(k) = x_0(-k or +k) = [k == 0] for both towers."""
    step = G_right if which == "right" else F_left
    out = []
    for K in range(0, K_max + 1):
        s = tuple(1 if k == 0 else 0 for k in range(K + 1))
        seen = {}
        t = 0
        while t <= t_max:
            if s in seen:
                out.append({"K": K, "preperiod": seen[s], "period": t - seen[s],
                            "period_is_power_of_two":
                                ((t - seen[s]) & (t - seen[s] - 1)) == 0})
                break
            seen[s] = t
            s = step(s)
            t += 1
        else:
            out.append({"K": K, "preperiod": None, "period": None,
                        "period_is_power_of_two": None})
    return out


def orbit_agreement(K: int, t_max: int) -> Dict[str, object]:
    """(RT) really is the rule, in the real single-cell orbit."""
    C = t_max + 2
    g = [[0] * (2 * C + 1) for _ in range(t_max + 1)]
    g[0][C] = 1
    for t in range(t_max):
        for j in range(1, 2 * C):
            g[t + 1][j] = g[t][j - 1] ^ (g[t][j] | g[t][j + 1])
    s = tuple(1 if k == 0 else 0 for k in range(K + 1))
    bad = 0
    for t in range(t_max):
        for k in range(K + 1):
            if s[k] != g[t][t - k + C]:
                bad += 1
        s = G_right(s)
    return {"K": K, "t_max": t_max, "cells_compared": t_max * (K + 1),
            "disagreements": bad}


def main():
    import json, os, time
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    out = {}

    print("=== H3a. (RT) reproduces the orbit ===")
    out["orbit_agreement"] = [orbit_agreement(24, 400), orbit_agreement(40, 300)]
    for r in out["orbit_agreement"]:
        print("  K=%d t_max=%d: %d cells, %d disagreements"
              % (r["K"], r["t_max"], r["cells_compared"], r["disagreements"]))

    print("\n=== H3b. bijectivity: right YES, left NO ===")
    out["bijectivity"] = bijectivity_census(12)
    for r in out["bijectivity"]:
        print("  K=%2d  left %6d/%6d bijective=%-5s | right %6d/%6d "
              "bijective=%s" % (r["K"], r["left_image_size"], r["states"],
                                r["left_bijective"], r["right_image_size"],
                                r["states"], r["right_bijective"]))
    out["inverse_check"] = inverse_check(14)
    print("  explicit inverse: %d states checked, %d failures"
          % (out["inverse_check"]["states_checked"],
             out["inverse_check"]["failures"]))

    print("\n=== H3c. preperiods: right tower is PURELY periodic ===")
    out["right_periods"] = tower_periods(20, 1 << 20, "right")
    out["left_periods"] = tower_periods(20, 1 << 20, "left")
    print("   K   right (pre, per)      left (pre, per)")
    for a, b in zip(out["right_periods"], out["left_periods"]):
        print("  %3d   (%4s, %6s)        (%4s, %6s)"
              % (a["K"], a["preperiod"], a["period"],
                 b["preperiod"], b["period"]))
    out["right_all_preperiod_zero"] = all(r["preperiod"] == 0
                                          for r in out["right_periods"])
    out["right_all_periods_powers_of_two"] = all(
        r["period_is_power_of_two"] for r in out["right_periods"])
    print("  right tower: every preperiod 0: %s ; every period a power of 2: %s"
          % (out["right_all_preperiod_zero"],
             out["right_all_periods_powers_of_two"]))

    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open("results/phase2h_right_tower.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/phase2h_right_tower.json (%.1f s)"
          % out["elapsed_sec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
