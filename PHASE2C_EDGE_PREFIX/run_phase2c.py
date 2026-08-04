"""run_phase2c.py -- Phase 2C measurement driver.

Writes results/phase2c_results.json and PREFIX_PERIOD_TABLE.csv.

    python3 run_phase2c.py                 # K <= 30000, ~3 min
    python3 run_phase2c.py --k-max 5000    # smaller
    python3 run_phase2c.py --quick

Sections
  1  formal verification of F_K, the projection relation and the skew form
  2  the fibre classification, exhaustive over short forcing words
  3  exact T(K), P(K) by three independent algorithms, cross-checked
  4  E1-E5 tested, with an aggressive counterexample search
  5  forced-bit census on the real orbit
  6  diagonal relation D1: where its range is non-empty, and verification there
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time

import prefix_lab as PL


def log(m):
    print(m, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-max", type=int, default=30000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args(argv)
    K_max = 2000 if a.quick else a.k_max
    T_max = int(1.6 * K_max) + 2000
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    out = {"settings": {"K_max": K_max, "T_max": T_max, "quick": a.quick},
           "environment": {"python": sys.version.split()[0],
                           "platform": platform.platform()}}

    # --- 1 -------------------------------------------------------------
    log("\n=== 1. formal prefix dynamics ===")
    out["F_forms"] = PL.verify_F_forms(10)
    out["projection"] = PL.verify_projection(9)
    out["skew_form"] = PL.verify_new_coordinate_drive(9)
    out["edge_vs_rule30"] = PL.verify_against_rule30(200, 300)
    log("  F list/bit-parallel forms agree: %s" % out["F_forms"]["forms_agree"])
    log("  P1 projection commutes: %s (%d assignments)"
        % (out["projection"]["projection_commutes"],
           out["projection"]["assignments_checked"]))
    log("  P2 skew form exact: %s" % out["skew_form"]["skew_form_exact"])
    log("  edge rows equal x_t(-t+k): %s (%d cells)"
        % (out["edge_vs_rule30"]["exact"], out["edge_vs_rule30"]["cells_checked"]))

    # --- 2 -------------------------------------------------------------
    log("\n=== 2. fibre classification ===")
    out["fibre_table"] = PL.fibre_table()
    out["classification"] = PL.verify_classification(9)
    log("  structural rule exact over %d forcing words: %s"
        % (out["classification"]["words_checked"],
           out["classification"]["structural_rule_exact"]))
    log("  class counts among sampled words: %s" % out["classification"]["counts"])

    # --- 3 -------------------------------------------------------------
    log("\n=== 3. exact T(K), P(K) ===")
    t1 = time.time()
    T, P = PL.exact_TP_first_repeat(K_max, T_max)
    gen = time.time() - t1
    undetermined = [K for K in range(K_max + 1) if T[K] is None]
    log("  first-repeat scan: %.1f s, undetermined K: %d" % (gen, len(undetermined)))
    if undetermined:
        log("  *** T_max too small for K >= %d" % undetermined[0])
        K_max = undetermined[0] - 1
    # cross-check 1: hash-based reference on the small range
    kref = 400 if not a.quick else 200
    ref = PL.exact_TP_reference(kref, T_max)
    agree_ref = all((T[K], P[K]) == ref[K] for K in range(kref + 1))
    # cross-check 2: incremental skew-product algorithm on an overlapping range
    kinc = 1500 if not a.quick else 600
    Ti, Pi, st = PL.exact_TP_incremental(kinc, T_max)
    agree_inc = all(Ti[K] == T[K] and Pi[K] == P[K] for K in range(kinc + 1))
    # cross-check 3: full periodicity+minimality verification at sampled K
    rows = PL.edge_rows(min(K_max, 12000), T_max)
    sample = sorted({0, 1, 5, 17, 18, 50, 400, 401, 1000, 2500, 5000,
                     min(K_max, 10000), min(K_max, 12000)})
    sample = [K for K in sample if K <= min(K_max, 12000)]
    fullv = {K: PL.full_verify_TP(K, T[K], P[K], rows) for K in sample}
    del rows
    out["exact_TP"] = {
        "K_max": K_max, "T_max": T_max, "seconds": round(gen, 1),
        "crosscheck_reference_hash": {"K_range": kref, "agree": agree_ref},
        "crosscheck_incremental_skew": {"K_range": kinc, "agree": agree_inc,
                                        "status": st},
        "full_verification_sample": {str(k): v for k, v in fullv.items()},
        "full_verification_all_pass": all(v["all"] for v in fullv.values()),
    }
    log("  vs hash reference (K<=%d): %s" % (kref, agree_ref))
    log("  vs incremental skew-product (K<=%d): %s" % (kinc, agree_inc))
    log("  full periodicity+minimality check at %d sampled K: %s"
        % (len(fullv), out["exact_TP"]["full_verification_all_pass"]))
    log("  %8s %9s %6s %8s" % ("K", "T(K)", "P(K)", "T/K"))
    for K in [10, 18, 100, 400, 1000, 5000, 10000, 20000, 30000]:
        if K <= K_max:
            log("  %8d %9d %6d %8.4f" % (K, T[K], P[K], T[K] / K))

    # --- 4 -------------------------------------------------------------
    log("\n=== 4. E1-E5 ===")
    e1_bad = [K for K in range(K_max + 1) if P[K] & (P[K] - 1)]
    e2_bad = [K for K in range(K_max) if P[K + 1] % P[K] or P[K + 1] > 2 * P[K]]
    e3_bad = [K for K in range(K_max) if T[K] > T[K + 1]]
    e4_bad = [K for K in range(K_max + 1) if T[K] <= K]
    diffs = [T[K] - K for K in range(K_max + 1)]
    doubling = sorted({(P[K], min(k for k in range(K_max + 1) if P[k] == P[K]))
                       for K in range(K_max + 1)})
    out["E_statements"] = {
        "E1_all_periods_powers_of_two": {"counterexamples": e1_bad[:20],
                                         "count": len(e1_bad)},
        "E2_P_divides_and_at_most_doubles": {"counterexamples": e2_bad[:20],
                                             "count": len(e2_bad)},
        "E3_T_nondecreasing": {"counterexamples": e3_bad[:20], "count": len(e3_bad)},
        "E4_T_greater_than_K": {"counterexamples": e4_bad,
                                "count": len(e4_bad),
                                "first_K_where_it_holds":
                                    (max(e4_bad) + 1) if e4_bad else 0},
        "E5_T_minus_K": {"min_over_K_ge_100": min(diffs[100:]) if K_max >= 100 else None,
                         "max": max(diffs),
                         "at_K_max": diffs[K_max]},
        "period_doubling_points": doubling,
    }
    log("  E1 counterexamples: %d" % len(e1_bad))
    log("  E2 counterexamples: %d" % len(e2_bad))
    log("  E3 counterexamples: %d" % len(e3_bad))
    log("  E4 counterexamples: %d  -> %s" % (len(e4_bad), e4_bad[:20]))
    log("  E5 T(K)-K: min over K>=100 = %s, max = %d, at K_max = %d"
        % (out["E_statements"]["E5_T_minus_K"]["min_over_K_ge_100"],
           max(diffs), diffs[K_max]))
    log("  period doubling points (P, first K): %s" % doubling)

    # --- 5 -------------------------------------------------------------
    log("\n=== 5. forced-bit census on the real orbit ===")
    cen = PL.forcing_word_census(T, P, min(K_max, 5000), T_max)
    out["forcing_census"] = cen
    log("  class counts: %s" % cen["counts"])
    log("  classification matches the measured period at every K: %s"
        % cen["classification_matches_measured_period"])
    log("  DOUBLING at K = %s" % cen["doubling_K"])
    ever = PL.coordinate_ever_one(min(K_max, 3000), min(T_max, 8000))
    out["coordinate_ever_one"] = ever
    log("  coordinates never equal to 1: %s" % ever["coordinates_never_one"])

    # --- 6 -------------------------------------------------------------
    log("\n=== 6. diagonal relation D1 ===")
    dr = PL.diagonal_relation_range(T, P, K_max)
    out["diagonal_range"] = dr
    log("  K with non-empty range for D1: %s (count %d)"
        % (dr["K_with_nonempty_range"][:20], dr["count"]))
    # verify D1 wherever it has content
    orb, R = PL._orbit_rows(400)
    ok = True
    pairs = 0
    for K in dr["K_with_nonempty_range"]:
        p = P[K]
        for t in range(T[K], K - p + 1):
            if t + p > 400:
                continue
            lhs = (orb[t + p] >> R) & 1
            rhs = (orb[t] >> (R + p)) & 1
            ok &= lhs == rhs
            pairs += 1
    out["diagonal_D1_verified"] = {"pairs_checked": pairs, "exact": bool(ok)}
    log("  D1 verified at %d (K,t) pairs where it has content: %s" % (pairs, ok))

    # --- CSV -----------------------------------------------------------
    csv_path = "PREFIX_PERIOD_TABLE.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "T_K", "P_K", "T_minus_K", "T_over_K"])
        for K in range(K_max + 1):
            w.writerow([K, T[K], P[K], T[K] - K,
                        "%.6f" % (T[K] / K) if K else ""])
    out["csv"] = {"path": csv_path, "rows": K_max + 1}
    log("\nwrote %s (%d rows)" % (csv_path, K_max + 1))

    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(a.outdir, "phase2c_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("wrote %s (%.1f s)" % (os.path.join(a.outdir, "phase2c_results.json"),
                               out["elapsed_sec"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
