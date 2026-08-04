"""run_phase2a_tests.py -- the Phase 2A test battery, from a clean process.

Exits non-zero if anything fails.  This is what an independent reviewer should
run first.

    python3 run_phase2a_tests.py            # ~4 min (re-simulates every witness)
    python3 run_phase2a_tests.py --fast     # ~40 s (skips the deep witnesses)

Coverage
  1  rule-30 table agrees with the Phase 1 library
  2  Lemma C  (cone containment) verified cell by cell
  3  Lemma T  (translation invariance of MODEL A) verified empirically
  4  ENC1 vs ENC2 vs Z3 agreement over a grid
  5  SAT vs solver-free exhaustive enumeration on small instances
  6  the Proposition 6 UNSAT control behaves as predicted, with its core
  7  MODEL B (full light cone) certifies the a=0 witness, base row = the seed
  8  MODEL B' certifies every witness in WITNESS_REGISTRY.json
  9  MODEL B sweep agrees with direct simulation on a small range
 10  stored results JSON is internally consistent with the reported claims
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mb1loc_sat as M
import rule30_lab as L

PASSES, FAILURES = [], []
RESULTS = "results"


def check(name, cond, detail=""):
    (PASSES if cond else FAILURES).append(name)
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return bool(cond)


# --------------------------------------------------------------------------
def t1_rule_table():
    print("\n=== 1. rule-30 local rule ===")
    ok = all(M.rule30(l, c, r) == (l ^ (c | r))
             for l in (0, 1) for c in (0, 1) for r in (0, 1))
    check("rule30 table == l XOR (c OR r) on all 8 neighbourhoods", ok)
    check("table sourced from rule30_lab.rule_table(30)",
          M.RULE30_TABLE == L.rule_table(30), str(M.RULE30_TABLE))
    sub = [c for c in L.verify_local_rule()]
    check("Phase 1 local-rule checks still pass", all(c.passed for c in sub),
          "%d/%d" % (sum(c.passed for c in sub), len(sub)))


def t2_cone_containment():
    print("\n=== 2. Lemma C: cone containment ===")
    ok = True
    for h in range(1, 60):
        ok &= M.check_cone_containment(M.Patch(0, h))
    for lo, hi in ((-1, 0), (0, 1), (-2, 3)):
        ok &= M.check_cone_containment(M.Patch(0, 25, lo, hi))
    check("every cell's three predecessors lie inside the cone", ok,
          "checked heights 1..59 and three top-site ranges")


def t3_translation_invariance():
    print("\n=== 3. Lemma T: MODEL A does not depend on t ===")
    ok = True
    for a, p in ((0, 1), (4, 3), (8, 5), (12, 2)):
        norm = M.solve_pysat(M.build_cnf("A", a, p)[0], "cadical153")["status"]
        for t in (a + 1, a + 9, a + 137, a + 5000):
            cnf, _, _ = M.build_cnf_a_at_t(a, p, t)
            st = M.solve_pysat(cnf, "cadical153")["status"]
            ok &= (st == norm)
        check("A(a=%d,p=%d) verdict identical at t = a+1, a+9, a+137, a+5000" % (a, p),
              ok, norm)


def t4_encoding_agreement(fast):
    print("\n=== 4. ENC1 vs ENC2 vs Z3 ===")
    a_vals = [0, 8, 24] if fast else [0, 4, 8, 16, 24, 48]
    p_vals = [1, 3, 7] if fast else [1, 2, 3, 5, 9, 16]
    bad = []
    for a in a_vals:
        for p in p_vals:
            st = {}
            for enc in ("enc1", "enc2"):
                cnf = M.build_cnf("A", a, p, encoding=enc)[0]
                for sv in ("cadical153", "glucose42", "minisat22"):
                    st["%s/%s" % (enc, sv)] = M.solve_pysat(cnf, sv)["status"]
            st["z3"] = M.solve_z3("A", a, p)["status"]
            if len(set(st.values())) != 1:
                bad.append((a, p, st))
    check("all encodings and solvers agree on %d instances"
          % (len(a_vals) * len(p_vals)), not bad, str(bad[:2]))


def t5_bruteforce():
    print("\n=== 5. SAT vs solver-free exhaustive enumeration ===")
    bad = []
    n = 0
    for a in (0, 4):
        for p in range(1, 9):
            if 2 * (a + p) + 2 > 18:
                continue
            e = M.decide_by_enumeration(a, p, max_bits=18)
            if e["status"] == "SKIPPED":
                continue
            s = M.solve_pysat(M.build_cnf("A", a, p)[0], "cadical153")["status"]
            n += 1
            if e["status"] != s:
                bad.append((a, p, e["status"], s))
    check("enumeration and SAT agree on %d instances" % n, not bad, str(bad))


def t6_unsat_control():
    print("\n=== 6. Proposition 6 UNSAT control ===")
    ok_forced = ok_free = ok_core = True
    for p in (1, 2, 3, 5, 8):
        cnf = M.build_cnf_prop6_control(p, forced=True)[0]
        r = M.solve_pysat(cnf, "cadical153")
        ok_forced &= r["status"] == "UNSAT"
        ok_core &= (r["core"] is not None and "C_left_disagree" in r["core"]
                    and "H2_centre_one" in r["core"])
        ok_forced &= M.decide_control_by_enumeration(p, forced=True)["status"] == "UNSAT"
        cnf2 = M.build_cnf_prop6_control(p, forced=False)[0]
        ok_free &= M.solve_pysat(cnf2, "cadical153")["status"] == "SAT"
        ok_free &= M.decide_control_by_enumeration(p, forced=False)["status"] == "SAT"
    check("forced case (a_t(0)=1) is UNSAT for p in {1,2,3,5,8}, by SAT and by enumeration",
          ok_forced)
    check("unforced case (a_t(0)=0) is SAT for the same p", ok_free)
    check("unsat core names the disagreement and the centre-one hypothesis", ok_core)


def t7_model_b_full():
    print("\n=== 7. MODEL B, full backward light cone to time 0 ===")
    a, t, p = 0, 17, 1
    cnf, var, patch, _, _ = M.build_cnf("B", a, p, t=t)
    verdicts = {sv: M.solve_pysat(cnf, sv)["status"]
                for sv in ("cadical153", "glucose42", "minisat22")}
    verdicts["z3"] = M.solve_z3("B", a, p, t=t)["status"]
    check("all solvers say SAT for (a,t,p) = (0,17,1)",
          set(verdicts.values()) == {"SAT"}, str(verdicts))
    r = M.solve_pysat(cnf, "cadical153")
    base = M.extract_base_row(r["model"], var, patch)
    seed_ok = all(base[j] == (1 if j == 0 else 0) for j in patch.sites(patch.t_base))
    check("recovered base row is exactly the single-cell seed", seed_ok,
          "%d sites at time 0" % len(list(patch.sites(patch.t_base))))
    grid = M.simulate_patch(base, patch)
    chk = M.check_patch_constraints(grid, a, p, t)
    check("H1, H2 and not-C re-checked on the independently evolved patch", chk["ALL"])
    sim = M.real_orbit_witness_check(a, t, p)
    check("independent orbit simulation confirms the counterexample",
          sim["is_counterexample"], str(sim))


def t8_model_bt_witnesses(fast):
    print("\n=== 8. MODEL B' certification of the registered witnesses ===")
    path = "WITNESS_REGISTRY.json"
    if not os.path.exists(path):
        return check("WITNESS_REGISTRY.json present", False, "missing")
    reg = json.load(open(path))
    wits = [w for w in reg["witnesses"] if w["model"] == "real_orbit"]
    check("registry contains 8 real-orbit witnesses", len(wits) == 8,
          "%d found" % len(wits))
    n = 0
    for w in wits:
        a, t, p = w["a"], w["t"], w["p"]
        if fast and t > 20000:
            continue
        patch1 = M.Patch(t - a, t)
        patch2 = M.Patch(t + p - a, t + p)
        base_rows = M.real_orbit_rows_pair(
            patch1.t_base, patch1.width_at(patch1.t_base),
            patch2.t_base, patch2.width_at(patch2.t_base))
        cnf, var, patches, _ = M.build_cnf_bt2(a, t, p, base_rows=base_rows)
        st = M.solve_pysat(cnf, "cadical153")["status"]
        grids = M.simulate_bt2(base_rows, patches)
        chk = M.check_bt2_constraints(grids, a, p, t)
        sim = M.real_orbit_witness_check(a, t, p)
        n += 1
        check("witness a=%d t=%d p=%d: SAT + patch re-check + orbit re-simulation"
              % (a, t, p),
              st == "SAT" and chk["ALL"] and sim["is_counterexample"],
              "sat=%s recheck=%s orbit=%s" % (st, chk["ALL"], sim["is_counterexample"]))
    check("%d witnesses re-certified in this process" % n, n > 0)


def t9_model_b_sweep():
    print("\n=== 9. MODEL B sweep vs direct simulation ===")
    t_max, p_max, a = 20, 8, 0
    cols = L.columns(t_max + p_max + 4, (0, 1))
    c0, c1 = cols[0], cols[1]
    sat_hits, sim_hits = [], []
    for t in range(a + 1, t_max + 1):
        for p in range(1, p_max + 1):
            h1 = all(c0[s] == c0[s + p] for s in range(t - a, t + 1))
            if h1 and c0[t] == 0 and c1[t] != c1[t + p]:
                sim_hits.append((a, t, p))
            cnf = M.build_cnf("B", a, p, t=t)[0]
            if M.solve_pysat(cnf, "cadical153")["status"] == "SAT":
                sat_hits.append((a, t, p))
    check("MODEL B and simulation find the same counterexamples (t<=%d, p<=%d)"
          % (t_max, p_max), sorted(sat_hits) == sorted(sim_hits),
          "%d found by each" % len(sat_hits))


def t10_results_json():
    print("\n=== 10. stored results are consistent with the reported claims ===")
    path = os.path.join(RESULTS, "phase2a_results.json")
    if not os.path.exists(path):
        return check("phase2a_results.json present", False, "missing -- run the search")
    d = json.load(open(path))
    sw = d["model_a_sweep"]
    check("MODEL A sweep: every instance decided by every configuration",
          all(r["all_agree"] for r in sw["grid"]),
          "%d instances" % len(sw["grid"]))
    check("MODEL A sweep: no cross-configuration disagreements",
          len(sw["disagreements"]) == 0)
    check("MODEL A sweep: UNSAT count recorded", "n_unsat" in sw,
          "n_unsat = %d" % sw["n_unsat"])
    check("brute force agrees with SAT everywhere it was run",
          all(r["agree"] for r in d["bruteforce_crosscheck"]),
          "%d instances" % len(d["bruteforce_crosscheck"]))
    check("all MODEL A certificates re-simulate correctly",
          all(c.get("recheck_ok", False) for c in d["model_a_certificates"]))
    ctrl = d["unsat_control"]
    check("UNSAT control matched its prediction in every configuration",
          all(c["all_agree"] and set(c["statuses"].values()) == {c["expected"]}
              for c in ctrl), "%d control instances" % len(ctrl))
    n_enum = sum(1 for c in ctrl if c.get("enumeration_checked"))
    check("control instances also decided by solver-free enumeration",
          n_enum >= 10, "%d of %d (wider base rows are skipped by design)"
          % (n_enum, len(ctrl)))
    mb = d["model_b_sweep"]
    check("MODEL B sweep agrees with direct simulation",
          mb["searches_agree"], "%d instances" % mb["n_instances"])
    if d.get("model_bt_witnesses"):
        check("all eight witnesses certified by MODEL B'",
              all(w["certified"] for w in d["model_bt_witnesses"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    print("python %s   cwd %s" % (sys.version.split()[0], os.getcwd()))
    t1_rule_table()
    t2_cone_containment()
    t3_translation_invariance()
    t4_encoding_agreement(args.fast)
    t5_bruteforce()
    t6_unsat_control()
    t7_model_b_full()
    t8_model_bt_witnesses(args.fast)
    t9_model_b_sweep()
    t10_results_json()
    print("\n" + "=" * 70)
    print("%d passed, %d FAILED   (%.1f s)"
          % (len(PASSES), len(FAILURES), time.time() - t0))
    if FAILURES:
        print("failures: %s" % ", ".join(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
