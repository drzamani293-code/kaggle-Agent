"""run_phase2e.py -- Phase 2E measurement driver.

    python3 run_phase2e.py                  # K <= 30000, ~1 min
    python3 run_phase2e.py --k-max 5000     # smaller
    python3 run_phase2e.py --quick          # K <= 3000, ~15 s

Writes results/phase2e_results.json, TRANSIENT_CLASSIFICATION.csv and
RESET_SCHEDULE_RESULTS.csv.

Sections
  1  exact definitions: T, R, tau, A, K_per -- which coincide, which do not
  2  the exact recurrence for T(K), and the classification of every level
  3  the reset schedule rho(K)
  4  frontier dynamics: steps of K_per and of T
  5  the co-moving transient strip
  6  diagonal reset ancestry
  7  the hash-consed proof DAG for the diagonal
  8  consequences of the periodicity hypothesis
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time

import collapse_lab as CL
import frontier_lab as F


def log(m):
    print(m, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-max", type=int, default=30000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args(argv)
    K_max = 3000 if a.quick else a.k_max
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    out = {"settings": {"K_max": K_max, "quick": a.quick},
           "environment": {"python": sys.version.split()[0],
                           "platform": platform.platform()}}

    log("\n=== 0. level table ===")
    recs, T, P, rows, Rq = F.level_table(K_max)
    out["settings"]["T_max"] = len(rows) - 1
    out["settings"]["levels_tabulated"] = len(recs)
    log("  %d levels, T_max = %d, T(%d) = %d, P(%d) = %d"
        % (len(recs), len(rows) - 1, K_max, T[K_max], K_max, P[K_max]))

    # --- 1 ---------------------------------------------------------------
    log("\n=== 1. exact definitions ===")
    thA = F.verify_theorem_A(recs, T, P, Rq, K_max)
    two = F.verify_R_two_ways(rows, Rq, K_max, K_cross=min(2000, K_max),
                              spot=[(P[K_max], K_max), (P[K_max], K_max // 2),
                                    (P[min(300, K_max)], min(300, K_max))])
    tau2 = F.verify_tau_two_ways(recs, rows, T, P)
    dc = F.definition_comparison(recs)
    ns = F.diagonal_never_settled(T, K_max)
    out["definitions"] = {"theorem_A": thA, "R_implementations": two,
                          "tau_implementations": tau2, "comparison": dc,
                          "T_le_K_levels": ns}
    log("  Theorem A  T(K) = max_{k<=K} r_{P(K)}(k):  %s (%d levels)"
        % (thA["exact"], thA["levels_checked"]))
    log("  r_Q by three implementations agree: %s" % two["all_agree"])
    log("  tau_K by two implementations agree:  %s" % tau2["agree"])
    log("  T = R at %d levels, T > R at %d, T < R at %d"
        % (dc["counts"]["T_eq_R"], dc["counts"]["T_gt_R"], dc["counts"]["T_lt_R"]))
    log("  R = tau+1 at %d levels, R <= T(K-1) at %d, neither at %d"
        % (dc["counts"]["R_eq_tau_plus_1"], dc["counts"]["R_le_T_prev"],
           dc["counts"]["R_other"]))
    log("  T(K) <= K exactly for K in %s" % ns["levels_with_T_le_K"])

    # --- 2 ---------------------------------------------------------------
    log("\n=== 2. the exact recurrence for T(K) ===")
    thB = F.verify_theorem_B(recs)
    dich = F.verify_dichotomy(recs)
    ez = CL.eventually_zero_coordinates(K_max, T, P, rows)
    out["recurrence"] = {"theorem_B": thB, "dichotomy": dich,
                         "eventually_zero_coordinates": ez}
    log("  Theorem B predicts T(K) exactly at %d/%d levels (mismatches %d)"
        % (thB["levels_checked"] - thB["mismatches"], thB["levels_checked"],
           thB["mismatches"]))
    log("  classes: %s" % thB["class_counts"])
    log("  defect bit D predicts the class:      %s" % dich["D_bit_predicts_class"])
    log("  RESETTING => COLLAPSING:              %s" % dich["resetting_implies_collapsing"])
    log("  RESETTING => T(K) = tau_K + 1:        %s" % dich["resetting_T_equals_tau_plus_1"])
    log("  eventually-zero coordinates: %s" % ez)

    with open("TRANSIENT_CLASSIFICATION.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "P_prev", "P_K", "fibre_kind", "T_prev", "T_K",
                    "tau_K", "D_bit", "class", "T_predicted", "matches",
                    "R_K", "increment"])
        for r in recs:
            w.writerow([r["K"], r["P_prev"], r["P_K"], r["fibre_kind"],
                        r["T_prev"], r["T_K"], r["tau_K"], r["D_bit"],
                        r["cls"], r["T_pred"], int(r["T_pred"] == r["T_K"]),
                        r["R_K"], r["T_K"] - r["T_prev"]])
    log("  wrote TRANSIENT_CLASSIFICATION.csv (%d rows)" % len(recs))

    # --- 3 ---------------------------------------------------------------
    log("\n=== 3. reset schedule ===")
    rs = F.reset_schedule(recs)
    ri = F.reset_increments(recs)
    rg = F.resetting_gaps(recs)
    dens = rg["n_resetting"] / len(recs)
    out["reset_schedule"] = {"rho": rs, "increments": ri, "gaps": rg,
                             "resetting_density": dens,
                             "density_times_mean_increment":
                                 dens * ri["mean_increment"],
                             "T_over_K_at_K_max": T[K_max] / K_max}
    log("  T(K) = rho(K)+1 at %.4f of all levels" % rs["fraction"])
    log("    among RESETTING levels:  %.4f" % rs["by_class"]["RESETTING"]["fraction"])
    log("    among INHERITING levels: %.4f" % rs["by_class"]["INHERITING"]["fraction"])
    log("  rho undefined at %d levels: %s"
        % (rs["rho_not_found"], [r["K"] for r in recs if r["rho_K"] is None]))
    log("  RESETTING density %.6f, mean increment %.6f, product %.6f"
        % (dens, ri["mean_increment"], dens * ri["mean_increment"]))
    log("  T(K_max)/K_max = %.6f" % (T[K_max] / K_max))
    log("  gaps between RESETTING levels: max %s, mean %.4f"
        % (rg["max_gap"], rg["mean_gap"]))

    with open("RESET_SCHEDULE_RESULTS.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "class", "T_prev", "tau_K", "T_K", "rho_K",
                    "T_minus_rho_minus_1", "wait_tau_minus_T_prev"])
        for r in recs:
            rho = r["rho_K"]
            w.writerow([r["K"], r["cls"], r["T_prev"], r["tau_K"], r["T_K"],
                        rho, "" if rho is None else r["T_K"] - rho - 1,
                        "" if r["tau_K"] is None else r["tau_K"] - r["T_prev"]])
    log("  wrote RESET_SCHEDULE_RESULTS.csv (%d rows)" % len(recs))

    # --- 4 ---------------------------------------------------------------
    log("\n=== 4. frontier dynamics ===")
    fs = F.frontier_stats(T, P, K_max, T[K_max])
    ages = F.age_profile(T, [18, 100, 400, 1000, 5000, 10000, 20000, 30000])
    out["frontier"] = {"stats": fs, "ages": ages}
    log("  K_per implementations agree: %s" % fs["two_implementations_agree"])
    log("  max step of K_per: %s" % fs["max_K_per_step"])
    log("  T(K+1) - T(K) histogram: %s" % fs["dT_histogram"])
    log("  violations of T(K+1) - T(K) <= P(K): %d" % fs["n_bound_violations"])
    log("  monotonicity violations: %s" % fs["monotonicity_violations"])
    log("  %8s %9s %9s %8s" % ("K", "T(K)", "A(K,K)", "T/K"))
    for r in ages:
        log("  %8d %9d %9d %8.4f" % (r["K"], r["T"], r["A_diag"], r["T_over_K"]))

    # --- 5 ---------------------------------------------------------------
    log("\n=== 5. co-moving transient strip ===")
    ident = F.strip_identity(300 if not a.quick else 120, 8)
    cen = F.strip_census(min(20000, K_max), 12)
    dep = F.strip_dependency_growth(50, 4)
    out["strip"] = {"identity": ident, "census": cen, "dependency": dep}
    log("  q_t(r) = x_t(r) verified on %d cells, %d mismatches"
        % (ident["cells_checked"], ident["mismatches"]))
    log("  %3s %7s %9s %9s %s" % ("R", "width", "distinct", "possible", "autonomous"))
    for c in cen:
        log("  %3d %7d %9d %9d %s"
            % (c["R"], c["width"], c["distinct_words"], c["possible_words"],
               c["is_autonomous"]))
    log("  backward cone of the strip grows by %d column per side per step; "
        "after %d steps it is %s"
        % (dep["growth_per_step_each_side"], dep["steps"], dep["final_cone"]))

    # --- 6 ---------------------------------------------------------------
    log("\n=== 6. diagonal reset ancestry ===")
    anc = [F.ancestry_skeleton(recs, T, t)
           for t in ([200, 600] if a.quick else [200, 600, 1500, 3000])]
    out["ancestry"] = anc
    log("  %6s %10s %10s %10s %10s"
        % ("t", "allTransFrom", "fraction", "resetEvents", "maxLevel/t"))
    for r in anc:
        ev = r["reset_events"]
        log("  %6d %10s %10.4f %10d %10.4f"
            % (r["t"], r["first_s_with_cone_entirely_transient"],
               r["crossover_fraction"], ev["reset_events_in_cone"],
               ev["max_level_over_t"]))

    # --- 7 ---------------------------------------------------------------
    log("\n=== 7. hash-consed proof DAG for the diagonal ===")
    ts = [100, 300, 600] if a.quick else [100, 300, 600, 1200, 2000, 3000]
    dag = F.dag_growth(ts, T, P)
    out["proof_dag"] = dag
    log("  %6s %10s %12s %12s %8s %10s"
        % ("t", "DAGnodes", "naiveCone", "irreducible", "ratio", "nodes/t^2"))
    for d in dag:
        log("  %6d %10d %12d %12d %8.4f %10.4f"
            % (d["t"], d["nodes"], d["naive_cone_cells"],
               d["irreducible_cone_cells"], d["reduction_factor"],
               d["nodes_over_t_squared"]))

    # --- 8 ---------------------------------------------------------------
    log("\n=== 8. periodicity hypothesis ===")
    hc = F.hypothesis_consequences(T, P, K_max, [1, 2, 3, 7, 16, 100])
    dl = F.diagonal_lines(min(20000, K_max), [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                          q_max=32)
    out["hypothesis"] = {"D1_content": hc, "lines": dl}
    for r in hc["per_p"]:
        log("  p=%-4d Theorem D1 has content at %d levels"
            % (r["p"], r["levels_where_D1_has_content"]))
    log("  line refutations (min T+p over q <= 32):")
    for r, v in sorted(dl["lines"].items(), key=lambda kv: int(kv[0])):
        log("    r=%-3s length %d  min(T+p) = %d at p = %d"
            % (r, v["length"], v["min_T_plus_p"], v["argmin_p"]))

    out["elapsed_sec"] = round(time.time() - t0, 1)
    path = os.path.join(a.outdir, "phase2e_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log("\nwrote %s (%.1f s)" % (path, out["elapsed_sec"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
