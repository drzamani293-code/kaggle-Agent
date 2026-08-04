"""run_phase2a_search.py -- the Phase 2A systematic search.

Produces results/phase2a_results.json, the witness certificates, the unsat
cores, and WITNESS_REGISTRY.json.

    python3 run_phase2a_search.py                 # full run, ~12 min
    python3 run_phase2a_search.py --quick         # reduced bounds, ~1 min

Sections
  1  MODEL A sweep over (a, p), two CNF encodings x three SAT solvers + Z3
  2  solver-independent brute-force cross-check on small instances
  3  MODEL A certificates (patch saved and re-simulated independently)
  4  UNSAT control (Proposition 6 forcing case): cores and proofs
  5  MODEL B (full backward light cone to time 0) exhaustive real-orbit sweep
  6  MODEL B certification of the a = 0 witness
  7  MODEL B' certification of all eight Phase 1 witnesses
  8  real-orbit eligibility census for the deep look-backs
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import mb1loc_sat as M
import rule30_lab as L

A_LIST = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
RESULTS = "results"


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
def section1_model_a_sweep(p_max, solvers, z3_p_max):
    log("\n=== 1. MODEL A sweep: a in %s, p in 1..%d ===" % (A_LIST, p_max))
    rows = []
    disagreements = []
    t0 = time.time()
    for a in A_LIST:
        line = []
        for p in range(1, p_max + 1):
            statuses = {}
            for enc in ("enc1", "enc2"):
                cnf, var, patch, t_used, _ = M.build_cnf("A", a, p, encoding=enc)
                for sv in solvers:
                    r = M.solve_pysat(cnf, sv)
                    statuses["%s/%s" % (enc, sv)] = r["status"]
            if p <= z3_p_max:
                statuses["z3/native"] = M.solve_z3("A", a, p)["status"]
            uniq = set(statuses.values())
            if len(uniq) != 1:
                disagreements.append({"a": a, "p": p, "statuses": statuses})
            st = statuses["enc1/%s" % solvers[0]]
            line.append("S" if st == "SAT" else ("U" if st == "UNSAT" else "?"))
            rows.append({"a": a, "p": p, "status": st,
                         "n_configs": len(statuses),
                         "all_agree": len(uniq) == 1,
                         "cells": patch.n_cells()})
        log("  a=%2d  p=1..%d  %s" % (a, p_max, "".join(line)))
    n_unsat = sum(1 for r in rows if r["status"] == "UNSAT")
    log("  instances: %d   SAT: %d   UNSAT: %d   cross-config disagreements: %d   (%.1f s)"
        % (len(rows), len(rows) - n_unsat, n_unsat, len(disagreements), time.time() - t0))
    return {"grid": rows, "disagreements": disagreements,
            "p_max": p_max, "a_list": A_LIST,
            "solvers": solvers, "z3_p_max": z3_p_max,
            "n_instances": len(rows), "n_unsat": n_unsat}


# --------------------------------------------------------------------------
def section2_bruteforce(max_bits):
    log("\n=== 2. solver-independent brute force (base rows enumerated) ===")
    out = []
    for a in A_LIST:
        for p in range(1, 13):
            w = 2 * (a + p) + 2
            if w > max_bits:
                continue
            e = M.decide_by_enumeration(a, p, max_bits=max_bits)
            cnf, var, patch, t_used, _ = M.build_cnf("A", a, p)
            s = M.solve_pysat(cnf, "cadical153")["status"]
            agree = e["status"] == s
            out.append({"a": a, "p": p, "base_row_bits": w,
                        "enumeration": e["status"], "sat": s, "agree": agree,
                        "rows_enumerated": e["n_rows"]})
            if not agree:
                log("  *** DISAGREEMENT a=%d p=%d enum=%s sat=%s" % (a, p, e["status"], s))
    log("  %d instances decided twice; all agree: %s"
        % (len(out), all(o["agree"] for o in out)))
    return out


# --------------------------------------------------------------------------
def section3_certificates(cert_ps):
    log("\n=== 3. MODEL A certificates (independently re-simulated) ===")
    os.makedirs(os.path.join(RESULTS, "model_a_patches"), exist_ok=True)
    certs = []
    for a in A_LIST:
        for p in cert_ps:
            cnf, var, patch, t_used, _ = M.build_cnf("A", a, p)
            r = M.solve_pysat(cnf, "cadical153")
            if r["status"] != "SAT":
                certs.append({"a": a, "p": p, "status": r["status"]})
                continue
            base = M.extract_base_row(r["model"], var, patch)
            grid = M.simulate_patch(base, patch)      # independent evolution
            chk = M.check_patch_constraints(grid, a, p, t_used)
            name = "modelA_a%02d_p%02d.txt" % (a, p)
            path = os.path.join(RESULTS, "model_a_patches", name)
            with open(path, "w") as f:
                f.write("MODEL A certificate (unrestricted rule-30 patch)\n")
                f.write("a = %d   p = %d   normalised t = %d\n" % (a, p, t_used))
                f.write("STATUS: SAT.  This is NOT a counterexample to MB1-loc(%d):\n" % a)
                f.write("it is a rule-30 patch with a FREE base row, not the\n")
                f.write("single-cell orbit.  See PHASE2A_METHOD.md section 3.\n\n")
                f.write("base row (time %d): %s\n\n" % (
                    patch.t_base,
                    "".join(str(base[j]) for j in patch.sites(patch.t_base))))
                f.write(M.render_patch(grid, patch, a, p, t_used))
                f.write("\n\nconstraint re-check after independent simulation:\n")
                for k, v in chk.items():
                    f.write("  %-20s %s\n" % (k, v))
            certs.append({"a": a, "p": p, "status": "SAT",
                          "recheck_ok": chk["ALL"], "certificate": path,
                          "base_row": "".join(str(base[j]) for j in patch.sites(patch.t_base))})
    ok = all(c.get("recheck_ok", True) for c in certs)
    log("  %d certificates written; all re-simulate correctly: %s" % (len(certs), ok))
    return certs


# --------------------------------------------------------------------------
def section4_unsat_control(p_list, solvers):
    log("\n=== 4. UNSAT control: the Proposition 6 forcing case ===")
    os.makedirs(os.path.join(RESULTS, "unsat_cores"), exist_ok=True)
    out = []
    for p in p_list:
        for forced in (True, False):
            statuses, cores = {}, {}
            for enc in ("enc1", "enc2"):
                cnf, var, patch, t = M.build_cnf_prop6_control(p, encoding=enc, forced=forced)
                for sv in solvers:
                    r = M.solve_pysat(cnf, sv, want_proof=(forced and sv == "cadical153"))
                    statuses["%s/%s" % (enc, sv)] = r["status"]
                    if r["core"]:
                        cores["%s/%s" % (enc, sv)] = r["core"]
                    if r.get("proof") is not None:
                        proof_len = len(r["proof"])
            enum = M.decide_control_by_enumeration(p, forced=forced)
            # "SKIPPED" means the base row was too wide to enumerate; it is an
            # absence of evidence, not a disagreeing verdict, so it must not be
            # folded into the agreement set.
            enum_checked = enum["status"] in ("SAT", "UNSAT")
            uniq = set(statuses.values())
            if enum_checked:
                uniq |= {enum["status"]}
            rec = {"p": p, "centre_value": 1 if forced else 0,
                   "expected": "UNSAT" if forced else "SAT",
                   "statuses": statuses, "enumeration": enum["status"],
                   "enumeration_checked": enum_checked,
                   "rows_enumerated": enum.get("n_rows"),
                   "all_agree": len(uniq) == 1,
                   "cores": cores,
                   "drup_proof_lines": (proof_len if forced else None)}
            out.append(rec)
            log("  p=%d  a_t(0)=%d  expected %-5s  got %-5s (enum %s)  agree=%s  core=%s"
                % (p, rec["centre_value"], rec["expected"],
                   list(statuses.values())[0], enum["status"], rec["all_agree"],
                   list(cores.values())[0] if cores else "-"))
            if forced and cores:
                with open(os.path.join(RESULTS, "unsat_cores",
                                       "prop6_control_p%02d.txt" % p), "w") as f:
                    f.write("UNSAT control -- Proposition 6 forcing case\n")
                    f.write("p = %d, a_t(0) = 1\n\n" % p)
                    f.write("Constraints:\n")
                    f.write("  H1_agree_0 : col_0(t)   = col_0(t+p)\n")
                    f.write("  H1_agree_1 : col_0(t+1) = col_0(t+1+p)\n")
                    f.write("  H2_centre_one : col_0(t) = 1\n")
                    f.write("  C_left_disagree : col_-1(t) != col_-1(t+p)\n\n")
                    f.write("STATUS: UNSAT (label: THEOREM for this (p), see\n")
                    f.write("PHASE2A_RESULTS.md section 4 -- the forcing is uniform in p\n")
                    f.write("and is proved by hand in Proposition 6.)\n\n")
                    for k, v in cores.items():
                        f.write("unsat core [%s]: %s\n" % (k, v))
                    f.write("\nbrute-force enumeration over all %s base rows: %s\n"
                            % (enum.get("n_rows"), enum["status"]))
    return out


# --------------------------------------------------------------------------
def section5_model_b_sweep(t_max, p_max, a_vals):
    """MODEL B: full backward light cone to time 0, seed pinned.

    Model B has NO free variables, so each solve is a certificate check of one
    triple (a, t, p) against the real orbit.  Sweeping it over (t, p) is an
    exhaustive real-orbit search, done a second time by direct simulation."""
    log("\n=== 5. MODEL B sweep (full light cone to time 0): t<=%d, p<=%d, a in %s ==="
        % (t_max, p_max, a_vals))
    cols = L.columns(t_max + p_max + 4, (0, 1))
    c0, c1 = cols[0], cols[1]
    sat_hits, sim_hits, n = [], [], 0
    t0 = time.time()
    for a in a_vals:
        for t in range(a + 1, t_max + 1):
            for p in range(1, p_max + 1):
                # direct simulation verdict
                h1 = all(c0[s] == c0[s + p] for s in range(t - a, t + 1))
                sim = h1 and c0[t] == 0 and c1[t] != c1[t + p]
                if sim:
                    sim_hits.append((a, t, p))
                cnf, var, patch, t_used, _ = M.build_cnf("B", a, p, t=t)
                r = M.solve_pysat(cnf, "cadical153")
                n += 1
                if r["status"] == "SAT":
                    sat_hits.append((a, t, p))
    agree = sorted(sat_hits) == sorted(sim_hits)
    log("  %d instances solved (%.1f s)" % (n, time.time() - t0))
    log("  MODEL B SAT (i.e. genuine counterexamples): %d" % len(sat_hits))
    log("  direct-simulation counterexamples:          %d" % len(sim_hits))
    log("  the two searches agree: %s" % agree)
    for h in sorted(sat_hits)[:12]:
        log("    counterexample a=%d t=%d p=%d" % h)
    return {"t_max": t_max, "p_max": p_max, "a_vals": a_vals,
            "n_instances": n,
            "sat_hits": [list(h) for h in sorted(sat_hits)],
            "sim_hits": [list(h) for h in sorted(sim_hits)],
            "searches_agree": agree}


# --------------------------------------------------------------------------
def section6_model_b_witness():
    log("\n=== 6. MODEL B certification of the a=0 witness (t=17, p=1) ===")
    a, t, p = 0, 17, 1
    cnf, var, patch, t_used, _ = M.build_cnf("B", a, p, t=t)
    stats = cnf.stats()
    results = {}
    for sv in ("cadical153", "glucose42", "minisat22"):
        results[sv] = M.solve_pysat(cnf, sv)["status"]
    results["z3"] = M.solve_z3("B", a, p, t=t)["status"]
    r = M.solve_pysat(cnf, "cadical153")
    base = M.extract_base_row(r["model"], var, patch)
    grid = M.simulate_patch(base, patch)
    chk = M.check_patch_constraints(grid, a, p, t)
    seed_ok = all((base[j] == (1 if j == 0 else 0)) for j in patch.sites(patch.t_base))
    sim = M.real_orbit_witness_check(a, t, p)
    path = os.path.join(RESULTS, "modelB_witness_a00_t17_p1.txt")
    with open(path, "w") as f:
        f.write("MODEL B certificate -- FULL backward light cone to time 0\n")
        f.write("a = %d   t = %d   p = %d\n" % (a, t, p))
        f.write("patch cells: %d   CNF vars: %d   clauses: %d\n"
                % (patch.n_cells(), stats["vars"], stats["clauses"]))
        f.write("base row is time 0 over sites %d..%d, pinned to the single-cell seed\n"
                % (patch.width_at(0)[0], patch.width_at(0)[1]))
        f.write("\nsolver verdicts: %s\n" % results)
        f.write("recovered base row is the seed: %s\n" % seed_ok)
        f.write("constraint re-check on the recovered patch: %s\n" % chk)
        f.write("independent orbit re-simulation: %s\n" % sim)
        f.write("\nLABEL: COMPUTATIONAL CERTIFICATE -- (a,t,p) = (0,17,1) is a\n")
        f.write("genuine counterexample to MB1-loc(0) in the real orbit.\n\n")
        f.write(M.render_patch(grid, patch, a, p, t, max_width=60))
    log("  solvers: %s" % results)
    log("  recovered base row equals the single-cell seed: %s" % seed_ok)
    log("  constraints re-checked on the patch: %s" % chk["ALL"])
    log("  independent orbit simulation says counterexample: %s" % sim["is_counterexample"])
    log("  certificate: %s" % path)
    return {"a": a, "t": t, "p": p, "solvers": results, "cnf": stats,
            "cells": patch.n_cells(), "base_row_is_seed": seed_ok,
            "constraints_ok": chk, "orbit_simulation": sim, "certificate": path}


# --------------------------------------------------------------------------
def section7_model_bt_witnesses(witness_file):
    """MODEL B' (two-window): light cones truncated at verified orbit rows.

    A single cone spanning [t-a, t+p] costs ~(a+p)^2 cells -- 7.7 million for
    the a=28 witness.  Two cones of height a, each based on a verified orbit
    row, carry exactly the same constraints for ~2a^2 cells.  Both variants are
    run wherever the single-cone form is affordable."""
    log("\n=== 7. MODEL B' certification of the eight Phase 1 witnesses ===")
    os.makedirs(os.path.join(RESULTS, "model_bt_certificates"), exist_ok=True)
    data = json.load(open(witness_file))
    out = []
    for row in data["rows"]:
        w = row["witness"]
        if w is None:
            continue
        a, t, p = w["a"], w["t"], w["p"]
        t0 = time.time()
        patch1 = M.Patch(t - a, t)
        patch2 = M.Patch(t + p - a, t + p)
        base_rows = M.real_orbit_rows_pair(
            patch1.t_base, patch1.width_at(patch1.t_base),
            patch2.t_base, patch2.width_at(patch2.t_base))
        cnf, var, patches, br = M.build_cnf_bt2(a, t, p, base_rows=base_rows)
        verdicts = {sv: M.solve_pysat(cnf, sv)["status"]
                    for sv in ("cadical153", "glucose42", "minisat22")}
        cnf2, _, _, _ = M.build_cnf_bt2(a, t, p, encoding="enc2", base_rows=base_rows)
        verdicts["enc2/cadical153"] = M.solve_pysat(cnf2, "cadical153")["status"]

        grids = M.simulate_bt2(base_rows, patches)
        chk = M.check_bt2_constraints(grids, a, p, t)
        sim = M.real_orbit_witness_check(a, t, p)

        one_cone_cells = M.Patch(t - a, t + p).n_cells()
        two_window_cells = patches[0].n_cells() + patches[1].n_cells()
        single_cone = None
        if one_cone_cells <= 300000:
            lo, hi = M.Patch(t - a, t + p).width_at(t - a)
            brow = M.real_orbit_row(t - a, lo, hi)
            cnf3, var3, pt3, _, _ = M.build_cnf("Bt", a, p, t=t, base_row=brow)
            single_cone = {"cells": one_cone_cells,
                           "status": M.solve_pysat(cnf3, "cadical153")["status"]}

        name = "modelBt_a%02d_t%d_p%d.txt" % (a, t, p)
        path = os.path.join(RESULTS, "model_bt_certificates", name)
        with open(path, "w") as f:
            f.write("MODEL B' CERTIFICATE (two-window form)\n")
            f.write("=" * 70 + "\n")
            f.write("a = %d   t = %d   p = %d\n\n" % (a, t, p))
            f.write("LABEL: COMPUTATIONAL CERTIFICATE.\n")
            f.write("This certifies that the triple (a,t,p) violates MB1-loc(%d) in the\n" % a)
            f.write("actual single-cell rule-30 orbit.  Soundness is relative to the two\n")
            f.write("base rows, which come from the Phase 1 engine (cross-checked by four\n")
            f.write("independent implementations and by SHA-256 against stored sequences).\n\n")
            f.write("window 1: times %d..%d, base row over sites %d..%d\n"
                    % (patch1.t_base, patch1.t_top,
                       patch1.width_at(patch1.t_base)[0], patch1.width_at(patch1.t_base)[1]))
            f.write("window 2: times %d..%d, base row over sites %d..%d\n"
                    % (patch2.t_base, patch2.t_top,
                       patch2.width_at(patch2.t_base)[0], patch2.width_at(patch2.t_base)[1]))
            f.write("cells: two-window %d, single-cone equivalent %d\n"
                    % (two_window_cells, one_cone_cells))
            f.write("CNF: %s\n" % cnf.stats())
            f.write("single-cone cross-run: %s\n\n" % single_cone)
            f.write("solver verdicts (all must be SAT): %s\n\n" % verdicts)
            f.write("hypotheses re-checked on the simulated patch:\n")
            for k, v in chk.items():
                f.write("  %-20s %s\n" % (k, v))
            f.write("\nindependent full-orbit simulation from the single-cell seed:\n")
            for k, v in sim.items():
                f.write("  %-24s %s\n" % (k, v))
            f.write("\nwindow 1 (times %d..%d):\n" % (patch1.t_base, patch1.t_top))
            f.write(M.render_patch(grids[0], patch1, a, 0, t, max_width=110))
            f.write("\n\nwindow 2 (times %d..%d):\n" % (patch2.t_base, patch2.t_top))
            f.write(M.render_patch(grids[1], patch2, a, 0, t + p, max_width=110))
            f.write("\n")
        ok = (set(verdicts.values()) == {"SAT"} and chk["ALL"]
              and sim["is_counterexample"]
              and (single_cone is None or single_cone["status"] == "SAT"))
        out.append({"a": a, "t": t, "p": p, "verdicts": verdicts,
                    "two_window_cells": two_window_cells,
                    "single_cone_cells": one_cone_cells,
                    "single_cone_run": single_cone,
                    "cnf": cnf.stats(),
                    "constraints_ok": chk["ALL"],
                    "orbit_is_counterexample": sim["is_counterexample"],
                    "certified": bool(ok), "certificate": path,
                    "seconds": round(time.time() - t0, 1)})
        log("  a=%2d t=%7d p=%5d  cells=%5d (1-cone %d)  %s  certified=%s  %.1fs"
            % (a, t, p, two_window_cells, one_cone_cells,
               "SAT" if set(verdicts.values()) == {"SAT"} else str(verdicts),
               ok, out[-1]["seconds"]))
    log("  all eight certified: %s" % all(o["certified"] for o in out))
    return out


# --------------------------------------------------------------------------
def section8_eligibility_census(n_bits, p_max):
    """Why the real-orbit search finds nothing for a >= 32: count how many
    (t, p) pairs even satisfy the hypotheses."""
    log("\n=== 8. real-orbit eligibility census (N = %d, p <= %d) ===" % (n_bits, p_max))
    cols = L.columns(n_bits, (0, 1))
    N = n_bits + 1
    S0, S1 = M.L._to_int(cols[0]), M.L._to_int(cols[1])
    pc = lambda x: bin(x).count("1")
    # The eligible set for look-back a is nested inside the one for a-1, so
    # accumulate once per p across all look-backs instead of recomputing from
    # scratch for each a (13x fewer big-integer shifts).
    elig = {a: 0 for a in A_LIST}
    viol = {a: 0 for a in A_LIST}
    want = set(A_LIST)
    a_max = max(A_LIST)
    for p in range(1, p_max + 1):
        W = (1 << (N - p - 1)) - 1
        agree0 = (~(S0 ^ (S0 >> p))) & W
        dis1 = (S1 ^ (S1 >> p)) & W
        zero0 = (~S0) & W
        A = agree0
        for j in range(0, a_max + 1):
            if j:
                A = A & ((agree0 << j) & W)
            if j in want:
                e = A & zero0
                elig[j] += pc(e)
                viol[j] += pc(e & dis1)
            if A == 0:
                break
    rows = []
    for a in A_LIST:
        rows.append({"a": a, "eligible_positions": elig[a], "violations": viol[a],
                     "violation_rate": (viol[a] / elig[a]) if elig[a] else None})
        log("  a=%2d  eligible=%12d  violations=%12d  rate=%s"
            % (a, elig[a], viol[a],
               ("%.4f" % (viol[a] / elig[a])) if elig[a] else "n/a"))
    return {"n_bits": N, "p_max": p_max, "rows": rows}


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--witness-file", default="mb1loc_witnesses.json")
    args = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)

    quick = args.quick
    p_max = 16 if quick else 64
    z3_p_max = 4 if quick else 12
    solvers = ["cadical153"] if quick else ["cadical153", "glucose42", "minisat22"]
    max_bits = 18 if quick else 22
    cert_ps = [1] if quick else [1, 7]
    ctrl_ps = [1, 2, 3] if quick else [1, 2, 3, 5, 8, 13]
    b_t, b_p, b_a = (24, 8, [0, 4]) if quick else (48, 24, [0, 4, 8])
    census_bits, census_p = (200000, 2000) if quick else (200000, 20000)

    t_start = time.time()
    out = {
        "settings": {
            "a_list": A_LIST, "p_max": p_max, "z3_p_max": z3_p_max,
            "solvers": solvers, "bruteforce_max_bits": max_bits,
            "model_b_bounds": {"t_max": b_t, "p_max": b_p, "a_vals": b_a},
            "census": {"bits": census_bits, "p_max": census_p},
            "quick": quick,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    try:
        import z3
        out["environment"]["z3"] = z3.get_version_string()
    except Exception:
        pass
    try:
        import pysat
        out["environment"]["pysat"] = getattr(pysat, "__version__", "unknown")
    except Exception:
        pass

    out["model_a_sweep"] = section1_model_a_sweep(p_max, solvers, z3_p_max)
    out["bruteforce_crosscheck"] = section2_bruteforce(max_bits)
    out["model_a_certificates"] = section3_certificates(cert_ps)
    out["unsat_control"] = section4_unsat_control(ctrl_ps, solvers)
    out["model_b_sweep"] = section5_model_b_sweep(b_t, b_p, b_a)
    out["model_b_witness"] = section6_model_b_witness()
    if os.path.exists(args.witness_file):
        out["model_bt_witnesses"] = section7_model_bt_witnesses(args.witness_file)
    else:
        log("\n=== 7 SKIPPED: %s not found ===" % args.witness_file)
        out["model_bt_witnesses"] = None
    out["eligibility_census"] = section8_eligibility_census(census_bits, census_p)
    out["elapsed_sec"] = round(time.time() - t_start, 1)

    with open(os.path.join(RESULTS, "phase2a_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("\nwrote %s   (%.1f s total)" % (os.path.join(RESULTS, "phase2a_results.json"),
                                         out["elapsed_sec"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
