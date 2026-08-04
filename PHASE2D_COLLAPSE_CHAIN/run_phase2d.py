"""run_phase2d.py -- Phase 2D measurement driver.  Writes results/phase2d_results.json."""
from __future__ import annotations
import argparse, json, os, platform, sys, time
import collapse_lab as CL, prefix_lab as PL


def log(m): print(m, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-max", type=int, default=3000)
    ap.add_argument("--anf-t", type=int, default=13)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    K = 800 if a.quick else a.k_max
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    out = {"settings": {"K_max": K, "anf_t_max": a.anf_t, "quick": a.quick},
           "environment": {"python": sys.version.split()[0],
                           "platform": platform.platform()}}

    log("\n=== 1. level classification and collapse formulas ===")
    lv, T, P, rows = CL.level_data(K)
    counts = {}
    for r in lv:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    ver = CL.verify_collapse_formulas(lv, T, P, rows)
    out["level_counts"] = counts
    out["collapse_formula_verification"] = ver
    out["non_collapsing_levels"] = [r["K"] for r in lv if r["kind"] != "COLLAPSING"]
    out["levels_sample"] = lv[:6]
    log("  class counts: %s" % counts)
    log("  non-collapsing levels: %s" % out["non_collapsing_levels"])
    log("  formula verification: %s" % ver)

    log("\n=== 2. zero coordinates ===")
    ez = CL.eventually_zero_coordinates(K, T, P, rows)
    pz = CL.permanently_zero_coordinates(K, rows)
    out["eventually_zero_coordinates"] = ez
    out["permanently_zero_coordinates"] = pz
    out["nonc_equals_ez_plus_one"] = ([k + 1 for k in ez]
                                      == out["non_collapsing_levels"])
    log("  eventually zero (zero on their cycle): %s" % ez)
    log("  permanently zero (zero for all t):     %s" % pz)
    log("  non-collapsing levels == {k+1 : k eventually zero}: %s"
        % out["nonc_equals_ez_plus_one"])

    log("\n=== 3. collapse chains and the transducer ===")
    runs = CL.chain_runs(lv)
    out["chain_runs"] = runs
    out["longest_chain"] = max((b - a_ + 1, a_, b) for a_, b in runs)
    log("  maximal COLLAPSING runs: %s" % runs)
    log("  longest chain: length %d, K in [%d, %d]" % out["longest_chain"])
    lo, hi = runs[-1]
    tr = CL.chain_transducer_check(rows, T, P, max(lo + 2, 402), hi, T[hi])
    out["transducer"] = tr
    log("  transducer on the last chain: %s" % tr)

    log("\n=== 4. diagonal dependency cone ===")
    cones = {}
    for t in ([40, 120] if a.quick else [40, 120, 400, 1200, 2400]):
        if t > K:
            continue
        c = CL.diagonal_cone(t, T, K)
        cones[str(t)] = {k: v for k, v in c.items() if k != "per_s"}
        cones[str(t)]["sample_rows"] = c["per_s"][:: max(1, t // 8)]
        log("  t=%5d  cone entirely transient from s=%s  (fraction %.4f)"
            % (t, c["first_s_with_cone_entirely_transient"],
               c["crossover_fraction"] or 0.0))
    out["diagonal_cones"] = cones

    log("\n=== 5. diagonal ANF ===")
    tmax = a.anf_t
    hist = CL.anf_rows(tmax, tmax)
    diag = [CL.anf_stats(hist, t, t) for t in range(tmax + 1)]
    out["anf_diagonal"] = diag
    out["anf_verification"] = CL.verify_anf_against_orbit(min(tmax, 9),
                                                          min(tmax, 12))
    log("  %4s %12s %8s %10s %s" % ("t", "monomials", "degree", "dep on z_t",
                                    "vars"))
    for d in diag:
        log("  %4d %12d %8d %10s %d" % (d["t"], d["monomials"], d["degree"],
                                        d["depends_on_z_k"],
                                        len(d["variables_present"])))
    log("  ANF vs orbit under the seed: %s" % out["anf_verification"])
    # variable elimination: does w_t(k) stop depending on z_k?
    elim = []
    for k in range(min(tmax, 10) + 1):
        firsts = [t for t in range(tmax + 1)
                  if not CL.anf_stats(hist, t, k)["depends_on_z_k"]]
        elim.append({"k": k, "first_t_without_z_k": firsts[0] if firsts else None})
    out["variable_elimination"] = elim
    log("  first t at which w_t(k) no longer contains z_k: %s"
        % [(e["k"], e["first_t_without_z_k"]) for e in elim])

    log("\n=== 6. doubling points ===")
    dp = []
    for r in lv:
        if r["kind"] == "DOUBLING":
            Kd = r["K"]
            Tb, Pb = r["T_base"], r["P_base"]
            dp.append({"K": Kd, "T_base": Tb, "P_base": Pb,
                       "a_cycle_word_K_minus_2":
                           "".join(str((rows[t] >> (Kd - 2)) & 1)
                                   for t in range(Tb, Tb + Pb)),
                       "c_cycle_word_K_minus_1":
                           "".join(str((rows[t] >> (Kd - 1)) & 1)
                                   for t in range(Tb, Tb + Pb)),
                       "xor_a": r["xor_a"], "P_after": r["P_K"]})
    out["doubling_points"] = dp
    for d in dp:
        log("  K=%3d  P_base=%2d  a-word=%s  c-word=%s  parity=%d -> P=%d"
            % (d["K"], d["P_base"], d["a_cycle_word_K_minus_2"],
               d["c_cycle_word_K_minus_1"], d["xor_a"], d["P_after"]))
    out["doubling_K"] = [d["K"] for d in dp]
    out["eventually_zero_predecessors"] = [d["K"] - 1 for d in dp]

    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open("results/phase2d_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log("\nwrote results/phase2d_results.json (%.1f s)" % out["elapsed_sec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
