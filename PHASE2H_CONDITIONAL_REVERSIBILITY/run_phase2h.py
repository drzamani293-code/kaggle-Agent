"""run_phase2h.py -- Phase 2H measurement driver.

    python3 run_phase2h.py            # ~3 min
    python3 run_phase2h.py --quick    # ~20 s

Writes results/phase2h_results.json.  Sections follow the brief.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import reversibility_lab as R


def right_white_run(grid, C, t):
    L, j = 0, t - 1
    while j >= -t and grid[t][j + C] == 0:
        L += 1
        j -= 1
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--t-max", type=int, default=1200)
    a = ap.parse_args(argv)
    T = 300 if a.quick else a.t_max
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    out = {"settings": {"t_max": T, "quick": a.quick},
           "environment": {"python": sys.version.split()[0]}}

    print("\n=== 1. conditional backward determinism, on the real orbit ===")
    bw = [R.verify_backward_on_orbit(t, h, t + 1, w)
          for (t, h, w) in ((60, 20, 30), (120, 40, 60), (200, 60, 100))]
    out["backward_reconstruction"] = bw
    for b in bw:
        print("  t_bot=%3d h=%2d m=%3d: white strip really white %s, "
              "%d cells, %d mismatches"
              % (b["t_bot"], b["h"], b["m"], b["white_strip_really_white"],
                 b["cells_reconstructed"], b["mismatches"]))

    print("\n=== 4. right-edge white runs W(t), and the 2^n events ===")
    grid, C = R.orbit_grid(T)
    ws = [right_white_run(grid, C, t) for t in range(T + 1)]
    records, best = [], -1
    for t in range(1, T + 1):
        if ws[t] > best:
            best = ws[t]
            records.append({"t": t, "W": ws[t],
                            "is_power_of_two": (t & (t - 1)) == 0})
    out["W_records"] = records
    out["W_at_powers"] = {str(2 ** n): ws[2 ** n]
                          for n in range(1, T.bit_length()) if 2 ** n <= T}
    out["W_at_powers_minus_1"] = {str(2 ** n - 1): ws[2 ** n - 1]
                                  for n in range(1, T.bit_length()) if 2 ** n - 1 <= T}
    out["W_at_powers_plus_1"] = {str(2 ** n + 1): ws[2 ** n + 1]
                                 for n in range(1, T.bit_length()) if 2 ** n + 1 <= T}
    out["all_records_at_powers_of_two"] = all(r["is_power_of_two"] for r in records)
    out["right_edge_always_one"] = all(grid[t][t + C] == 1 for t in range(T + 1))
    print("  every record of W(t) is at a power of two: %s"
          % out["all_records_at_powers_of_two"])
    print("  W(2^n) = %s" % list(out["W_at_powers"].values()))
    print("  W(2^n - 1) = %s ; W(2^n + 1) = %s"
          % (list(out["W_at_powers_minus_1"].values()),
             list(out["W_at_powers_plus_1"].values())))

    print("\n=== 4b. the white triangle below t = 2^n ===")
    val = lambda t, j: grid[t][j + C]
    tri = []
    for n in range(3, T.bit_length()):
        m = 2 ** n
        if m > T:
            break
        Wn = ws[m]
        imax, full = -1, True
        for i in range(0, max(1, Wn // 2)):
            lo, hi = m - Wn + i, m - 1 - i
            if lo > hi:
                break
            if all(val(m + i, j) == 0 for j in range(lo, hi + 1)) and m + i <= T:
                imax = i
            else:
                full = False
                break
        tri.append({"n": n, "t": m, "W": Wn, "verified_to_i": imax,
                    "full_triangle": bool(full and imax == max(0, Wn // 2) - 1)})
    out["white_triangles"] = tri
    for x in tri:
        print("  n=%2d t=%5d W=%2d verified down to i=%2d full=%s"
              % (x["n"], x["t"], x["W"], x["verified_to_i"], x["full_triangle"]))

    print("\n=== 5. distance of the restart events from column 0 ===")
    geo = [{"n": x["n"], "t": x["t"], "W": x["W"],
            "left_col": x["t"] - x["W"], "right_col": x["t"] - 1,
            "dist_from_centre": x["t"] - x["W"],
            "earliest_influence_on_col0": x["t"] + (x["t"] - x["W"])}
           for x in tri]
    out["restart_geometry"] = geo
    for g in geo:
        print("  t=2^%d=%5d  cols [%5d, %5d]  distance from col 0 = %5d  "
              "earliest possible influence on col 0 at t >= %6d"
              % (g["n"], g["t"], g["left_col"], g["right_col"],
                 g["dist_from_centre"], g["earliest_influence_on_col0"]))

    print("\n=== 11. rule 90 negative control ===")
    T90 = min(T, 600)
    g90, C90 = R.orbit_grid(T90, rule=90)
    w90 = [right_white_run(g90, C90, t) for t in range(T90 + 1)]
    out["rule90"] = {
        "W_at_powers": {str(2 ** n): w90[2 ** n]
                        for n in range(1, T90.bit_length()) if 2 ** n <= T90},
        "matches_2_pow_np1_minus_1": all(
            w90[2 ** n] == 2 ** (n + 1) - 1
            for n in range(1, T90.bit_length()) if 2 ** n <= T90),
        "left_and_right_permutive": True,
    }
    print("  rule 90: W(2^n) = 2^(n+1) - 1 exactly: %s"
          % out["rule90"]["matches_2_pow_np1_minus_1"])
    print("  rule 90 W(2^n) = %s" % list(out["rule90"]["W_at_powers"].values()))

    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open("results/phase2h_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/phase2h_results.json (%.1f s)" % out["elapsed_sec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
