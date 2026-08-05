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
  9  reset times sigma / tau / rho, and the fibre dependence horizon R(K)
 10  the five-way level classification (and the classification CSV)
 11  whether the frontier increments admit a finite-state description
 12  the co-moving strip of full width 2R+1: automaton, lookahead, memory
 13  the reset-erased dependency skeleton and its proof DAG
 14  the periodicity hypothesis in the transient-frontier variables
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
import transient_lab as X


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

    log("  (TRANSIENT_CLASSIFICATION.csv is written in section 10, with both "
        "the two-way class and the five-way case)")

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

    # --- 9 ---------------------------------------------------------------
    log("\n=== 9. reset times sigma / tau / rho, and the horizon R(K) ===")
    t_hi = min(len(rows), T[K_max] + 64)
    rt = X.reset_time_table(recs[:min(len(recs), 2000)], rows, t_hi)
    cmp_ = X.compare_reset_times(rt)
    hz = X.verify_horizon(rows, [2, 3, 5, 9, 17, 33, 100, 500, 1500,
                                 min(2999, K_max)], min(t_hi, 6000))
    out["reset_times"] = {"comparison": cmp_, "horizon_check": hz,
                          "sample": rt[:8]}
    log("  R(K) by perturbation = sigma(K) at every sampled level: %s" % hz["agree"])
    c = cmp_["counts"]
    log("  sigma < tau at %d levels, sigma = tau at %d  -> sigma and tau DISTINCT"
        % (c["sigma_lt_tau"], c["sigma_eq_tau"]))
    log("  tau = rho at %d levels, rho < tau at %d      -> tau and rho DISTINCT"
        % (c["tau_eq_rho"], c["rho_lt_tau"]))

    # --- 10 --------------------------------------------------------------
    log("\n=== 10. five-way level classification ===")
    ct = X.classification_table(recs, rows, ez)
    out["five_way"] = {k: v for k, v in ct.items() if k != "rows"}
    log("  counts: %s" % ct["counts"])
    log("  prediction failures: %d, independent disagreements: %d"
        % (ct["n_prediction_failures"], ct["n_independent_disagreements"]))
    log("  PERIOD_DOUBLING levels %s; ZERO_PREDECESSOR levels %s; identical: %s"
        % (ct["period_doubling_levels"], ct["zero_predecessor_levels"],
           ct["doubling_equals_zero_predecessor"]))
    byK5 = {r["K"]: r["class"] for r in ct["rows"]}
    with open("TRANSIENT_CLASSIFICATION.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "P_prev", "P_K", "fibre_kind", "T_prev", "T_K",
                    "tau_K", "D_bit", "class2", "case5", "T_predicted",
                    "matches", "R_K", "increment", "zero_predecessor"])
        for r, r5 in zip(recs, ct["rows"]):
            w.writerow([r["K"], r["P_prev"], r["P_K"], r["fibre_kind"],
                        r["T_prev"], r["T_K"], r["tau_K"], r["D_bit"],
                        r["cls"], r5["class"], r["T_pred"],
                        int(r["T_pred"] == r["T_K"]), r["R_K"],
                        r["T_K"] - r["T_prev"], r5["zero_predecessor"]])
    log("  rewrote TRANSIENT_CLASSIFICATION.csv with the five-way case column")

    # --- 11 --------------------------------------------------------------
    log("\n=== 11. is the frontier increment finite-state? ===")
    il = X.verify_increment_law(recs, rows)
    df = X.verify_defect_bit_formula(recs, rows)
    fl = X.verify_flip_flips_defect(recs, rows, 400)
    fs1 = X.defect_bit_is_finite_state(recs, rows, 1)
    fs2 = X.defect_bit_is_finite_state(recs, rows, 2)
    iw = X.increment_word_repeats(recs, 30)
    out["finite_state"] = {"increment_law": il, "defect_formula": df,
                           "flip": fl, "depth1": fs1, "depth2": fs2,
                           "increment_word": iw}
    log("  C1 increment = f(cycle word, D): exact %s (%d levels)"
        % (il["exact"], il["levels_checked"]))
    log("  C2 D(K) = w_{T(K-1)}(K) XOR Phi_K: exact %s (%d levels)"
        % (df["exact"], df["levels_checked"]))
    log("  flipping that one transient bit always flips D: %s"
        % fl["flip_always_flips_D"])
    log("  depth-1 cycle-word windows: %d distinct, %d counterexamples -> refuted %s"
        % (fs1["distinct_windows"], fs1["counterexamples"],
           fs1["finite_state_refuted"]))
    log("  depth-2: %d distinct windows over %d levels -- collision test has NO "
        "power here" % (fs2["distinct_windows"], len(recs)))

    # --- 12 --------------------------------------------------------------
    log("\n=== 12. co-moving strip of width 2R+1 ===")
    ss = X.verify_strip_step(2000, 6)
    pre = [X.verify_preimage(R) for R in (1, 2, 3, 4, 5, 6, 7)]
    t_strip = min(50000, 4 * K_max)
    wide = X.strip_states(t_strip, 8)
    oa = [X.strip_orbit_vs_automaton(t_strip, R,
                                     [s & ((1 << (2 * R + 1)) - 1) for s in wide])
          for R in (2, 4, 6, 8)]
    la = [X.strip_lookahead(min(20000, t_strip), R) for R in (1, 2, 3, 4, 6)]
    ch = [X.centre_history_determines_next(min(50000, t_strip), m)
          for m in (1, 2, 4, 8, 12, 16, 20)]
    me = X.strip_minimal_extension(4, 10)
    out["strip"]["step_law"] = ss
    out["strip"]["preimage"] = pre
    out["strip"]["orbit_vs_automaton"] = oa
    out["strip"]["lookahead"] = la
    out["strip"]["centre_history"] = ch
    out["strip"]["minimal_extension"] = me
    log("  strip step law exact against the orbit: %s" % ss["exact"])
    log("  every state has in-degree exactly 4 (Theorem D2), radii 1..7: %s"
        % all(p["every_state_has_indegree_4"] for p in pre))
    log("  recurrent core = whole state space at every radius: %s"
        % all(not o["automaton_excludes_anything"] for o in oa))
    log("  strip lookahead: theory R, first counterexample at R+1:")
    for r in la:
        log("    R=%d -> first counterexample at n=%s"
            % (r["radius"], r["first_n_with_counterexample"]))
    log("  centre column determined by its own last m bits: %s"
        % [(r["memory"], r["determined"]) for r in ch])

    # --- 13 --------------------------------------------------------------
    log("\n=== 13. reset-erased dependency skeleton and its proof DAG ===")
    sk_ts = [100, 300, 600] if a.quick else [100, 300, 600, 1200]
    sk = X.skeleton_growth(sk_ts, rows)
    sp = [X.verify_spine(t, rows) for t in ([100, 300] if a.quick else [100, 300, 600])]
    ec = X.erased_edge_census(400, rows)
    dg = [X.proof_dag(t, rows) for t in sk_ts]
    cb = [X.skeleton_combined(t, rows, T, P) for t in sk_ts]
    st_ = X.dag_special_times(rows, T, P, K_max)
    rf = X.dag_recursive_family(rows, [64, 128, 256, 512])
    out["skeleton"] = {"growth": sk, "spine": sp, "erased_edges": ec,
                       "proof_dag": dg, "combined": cb,
                       "special_times": st_, "recursive_family": rf}
    log("  %6s %10s %10s %10s %10s %10s"
        % ("t", "cone", "skeleton", "reset+per", "DAG", "width"))
    for s, c2, d in zip(sk, cb, dg):
        t_ = s["t"]
        cone = sum(min(t_, 2 * u) - max(0, 2 * u - t_) + 1 for u in range(t_ + 1))
        log("  %6d %10d %10d %10d %10d %10d"
            % (t_, cone, s["nodes"], c2["nodes"], d["dag_nodes"], s["max_width"]))
    log("  spine contained in the skeleton at every t tested: %s"
        % all(x["spine_contained"] for x in sp))
    log("  mean surviving edges per cell (of 3): %.4f" % ec["mean_surviving_edges"])
    log("  recursive family among special-time DAGs found: %s"
        % rf["recursive_family_found"])

    # --- 14 --------------------------------------------------------------
    log("\n=== 14. periodicity hypothesis in these variables ===")
    drs = X.diagonal_reset_schedule(rows, min(3000, len(rows) - 1))
    de = X.dag_equality_under_shift(rows, [200, 300, 400, 500], 29)
    out["hypothesis"]["diagonal_reset_schedule"] = drs
    out["hypothesis"]["dag_equality"] = de
    log("  reset schedule along the diagonal = centre column shifted: %s"
        % drs["identity_verified"])
    log("  equal centre value with different DAG size: %d cases -> "
        "'equal value => equal derivation' is REFUTED: %s"
        % (de["n_equal_value_different_dag"], de["implication_refuted"]))

    out["elapsed_sec"] = round(time.time() - t0, 1)
    path = os.path.join(a.outdir, "phase2e_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log("\nwrote %s (%.1f s)" % (path, out["elapsed_sec"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
