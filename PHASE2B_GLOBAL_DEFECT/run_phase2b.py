"""run_phase2b.py -- Phase 2B-GLOBAL measurement driver.

Writes results/phase2b_results.json plus the machine-readable geometry tables.

    python3 run_phase2b.py               # full run, ~20 min
    python3 run_phase2b.py --quick       # ~2 min

Sections
  1  exhaustive verification of every derived Boolean identity
  2  the identities re-checked against the real single-cell orbit
  3  defect geometry for the required lag set
  4  wall-run statistics and centre-word / crossing relations
  5  the zero-wall automaton, exhaustive for r = 1..R
  6  observed wall states in the real orbit vs the locally admissible set
  7  edge-aligned dynamics: one-sidedness, prefix periods, the diagonal
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import defect_lab as D
import rule30_lab as L

RESULTS = "results"
WITNESS_LAGS = [1, 2, 6, 27, 1324, 2750]          # distinct lags of the 8 witnesses
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
          67, 71, 73, 79, 83, 89, 97, 101, 127, 251, 509, 1021]
POWERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def log(m):
    print(m, flush=True)


def lag_set(quick):
    if quick:
        return sorted(set(list(range(1, 17)) + [2, 3, 5, 7, 16, 32] + [1, 2, 6, 27]))
    return sorted(set(list(range(1, 65)) + PRIMES + POWERS + WITNESS_LAGS))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--t-max", type=int, default=1500)
    ap.add_argument("--auto-radius", type=int, default=5)
    args = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    quick = args.quick
    t_max = 300 if quick else args.t_max
    r_max = 3 if quick else args.auto_radius
    lags = lag_set(quick)
    t0 = time.time()

    out = {
        "settings": {"t_max": t_max, "lags": lags, "automaton_radius_max": r_max,
                     "quick": quick},
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform()},
    }

    # --- 1. Boolean identities ------------------------------------------
    log("\n=== 1. exhaustive Boolean verification ===")
    idn = D.verify_defect_identities()
    wr = D.verify_zero_wall_reduction()
    out["identities"] = idn
    out["wall_reduction"] = wr
    log("  step identity over all 64 assignments: XOR/OR=%s  ANF=%s"
        % (idn["xor_or_form_exact"], idn["anf_form_exact"]))
    log("  Delta ANF matches truth table (16 assignments): %s" % idn["delta_anf_matches"])
    log("  wall reduction Delta(u,v,0,g) = g AND NOT u: %s"
        % wr["delta_with_e0_equals_g_and_not_u"])

    # --- 2. identities on the real orbit --------------------------------
    log("\n=== 2. identities re-checked on the real orbit ===")
    checks = {}
    for p in ([3, 7] if quick else [1, 3, 7, 16, 27]):
        checks["dynamics_p%d" % p] = D.verify_defect_dynamics_on_orbit(
            p, 200 if quick else 400, 30)
        checks["W1_p%d" % p] = D.check_wall_relation(p, 2000 if quick else 8000)
        checks["W3_p%d" % p] = D.check_ones_run_triangle(p, 4000 if quick else 20000, 2)
        checks["W4_p%d" % p] = D.check_left_slaving(p, 2000 if quick else 8000)
    out["orbit_checks"] = checks
    bad = [k for k, v in checks.items() if not v.get("exact")]
    log("  %d checks, all exact: %s%s" % (len(checks), not bad,
                                          (" FAILED: %s" % bad) if bad else ""))
    for k in sorted(checks):
        if k.startswith("W1") or k.startswith("W3"):
            v = checks[k]
            log("    %-10s hypothesis met %6s times, violations %d"
                % (k, v.get("hypothesis_holds_at", v.get("cells_checked")),
                   v["violations"]))

    # --- 3. defect geometry ---------------------------------------------
    log("\n=== 3. defect geometry over %d lags ===" % len(lags))
    geo = []
    edge_all = True
    for p in lags:
        g = D.defect_geometry(p, t_max, sample_every=max(1, t_max // 300))
        edge_all &= g["support_edge_defect_theorem_holds"]
        geo.append({k: v for k, v in g.items() if k != "per_row"})
        if p in (1, 7, 27, 1324, 2750):
            with open(os.path.join(RESULTS, "defect_rows_p%d.json" % p), "w") as f:
                json.dump(g["per_row"], f)
    out["geometry"] = geo
    out["support_edge_theorem_holds_for_all_lags"] = bool(edge_all)
    log("  support-edge defect theorem holds for every lag: %s" % edge_all)
    log("  %4s %10s %10s %12s %10s" % ("p", "meandens", "wallfrac", "runs/row", "cross/row"))
    for g in geo:
        if g["p"] in (1, 2, 7, 16, 27, 64, 1024, 1324, 2750):
            log("  %4d %10.5f %10.5f %12.3f %10.3f"
                % (g["p"], g["mean_defect_density"], g["zero_wall_fraction"],
                   g["total_runs"] / g["rows_sampled"],
                   g["runs_covering_column_0"] / g["rows_sampled"]))

    # --- 4. wall runs and crossing events -------------------------------
    log("\n=== 4. wall runs in the real orbit ===")
    walls = [D.wall_run_lengths(p, 4000 if quick else 30000) for p in lags]
    out["wall_runs"] = walls
    log("  %4s %8s %10s %12s" % ("p", "n_walls", "longest", "mean"))
    for w in walls:
        if w["p"] in (1, 2, 7, 16, 27, 64, 1024, 1324, 2750):
            log("  %4d %8d %10d %12.3f" % (w["p"], w["n_walls"], w["longest_wall"],
                                           w["mean_wall"]))
    out["longest_wall_over_all_lags"] = max(w["longest_wall"] for w in walls)
    out["argmax_lag"] = max(walls, key=lambda w: w["longest_wall"])["p"]
    log("  longest wall over all lags: %d (at p=%d)"
        % (out["longest_wall_over_all_lags"], out["argmax_lag"]))
    out["crossing_events"] = [D.crossing_events(p, 4000 if quick else 20000, 5)
                              for p in ([3, 7] if quick else [1, 7, 27])]

    # --- 5. the zero-wall automaton -------------------------------------
    log("\n=== 5. zero-wall automaton (exhaustive) ===")
    autos = []
    for r in range(1, r_max + 1):
        t1 = time.time()
        a = D.wall_automaton(r)
        rec = {k: a[k] for k in ("radius", "window_bits", "wall_states_total",
                                 "maximal_invariant_states", "invariant_is_empty")}
        rec["invariant_fraction"] = (a["maximal_invariant_states"]
                                     / a["wall_states_total"])
        rec["seconds"] = round(time.time() - t1, 1)
        autos.append(rec)
        log("  r=%d  wall states=%9d  maximal invariant=%8d (%.4f)  empty=%s  %.1fs"
            % (r, rec["wall_states_total"], rec["maximal_invariant_states"],
               rec["invariant_fraction"], rec["invariant_is_empty"], rec["seconds"]))
        if r == r_max:
            out["_last_automaton"] = a
    out["automaton"] = autos
    out["automaton_note"] = ("exhaustive enumeration is 2^(4r+1) wall states; "
                             "r>=6 is 3.4e7 states and was NOT computed here")

    # --- 6. observed vs admissible wall states --------------------------
    log("\n=== 6. observed wall states vs locally admissible ===")
    obs_rows = []
    a_last = out.pop("_last_automaton", None)
    for r in range(1, min(r_max, 4) + 1):
        a = D.wall_automaton(r) if (a_last is None or a_last["radius"] != r) else a_last
        inv = {a["states"][i] for i in range(len(a["states"])) if a["alive"][i]}
        obs = D.observed_wall_states(r, [1, 2, 3, 7, 16, 27], 4000 if quick else 20000)
        union = set().union(*obs.values())
        obs_rows.append({
            "radius": r,
            "admissible_wall_states": a["wall_states_total"],
            "maximal_invariant_states": len(inv),
            "observed_in_real_orbit": len(union),
            "observed_and_invariant": len(union & inv),
            "observed_not_invariant": len(union - inv),
            "invariant_not_observed": len(inv - union),
            "per_lag_observed": {str(p): len(s) for p, s in obs.items()},
        })
        log("  r=%d  admissible=%7d  invariant=%7d  observed=%7d  "
            "observed&invariant=%7d  invariant-not-observed=%7d"
            % (r, obs_rows[-1]["admissible_wall_states"], len(inv), len(union),
               obs_rows[-1]["observed_and_invariant"],
               obs_rows[-1]["invariant_not_observed"]))
    out["observed_vs_admissible"] = obs_rows

    # --- 7. edge-aligned dynamics ---------------------------------------
    log("\n=== 7. edge-aligned coordinates ===")
    ea = {"update_rule": D.verify_edge_update(200 if quick else 400,
                                              150 if quick else 300),
          "centre_is_diagonal": D.centre_is_diagonal(300 if quick else 800)}
    log("  one-sided update w_{t+1}(k) = w_t(k-2) XOR (w_t(k-1) OR w_t(k)): %s"
        % ea["update_rule"]["exact"])
    log("  centre column equals the diagonal w_t(t): %s"
        % ea["centre_is_diagonal"]["centre_equals_diagonal"])
    ks = ([1, 2, 4, 8, 16, 32] if quick
          else [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                320, 384, 448, 512, 640, 768, 1024, 1280])
    tm = 400 if quick else 2600
    prof = [D.prefix_period(K, tm) for K in ks]
    ea["prefix_periods"] = prof
    log("  %6s %10s %8s %8s" % ("K", "preperiod", "period", "T(K)/K"))
    for r in prof:
        if r["found"]:
            log("  %6d %10d %8d %8.3f" % (r["K"], r["preperiod"], r["period"],
                                          r["preperiod"] / max(r["K"], 1)))
        else:
            log("  %6d      not periodic within t_max=%d" % (r["K"], tm))
    found = [r for r in prof if r["found"] and r["K"] >= 64]
    if found:
        ea["late_T_over_K"] = sum(r["preperiod"] / r["K"] for r in found) / len(found)
        ea["periods_observed"] = sorted({r["period"] for r in prof if r["found"]})
        log("  mean T(K)/K for K>=64: %.3f   periods observed: %s"
            % (ea["late_T_over_K"], ea["periods_observed"]))
    out["edge_aligned"] = ea

    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "phase2b_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("\nwrote %s  (%.1f s)" % (os.path.join(RESULTS, "phase2b_results.json"),
                                 out["elapsed_sec"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
