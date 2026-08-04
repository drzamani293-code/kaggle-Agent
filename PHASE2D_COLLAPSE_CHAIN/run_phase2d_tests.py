"""run_phase2d_tests.py -- Phase 2D test battery, from a clean process."""
from __future__ import annotations
import argparse, json, os, sys, time
import collapse_lab as CL, prefix_lab as PL

PASSES, FAILURES = [], []


def check(name, cond, detail=""):
    (PASSES if cond else FAILURES).append(name)
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return bool(cond)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    K = 600 if a.fast else 3000
    t0 = time.time()
    print("python %s  cwd %s" % (sys.version.split()[0], os.getcwd()))

    print("\n=== 1. collapse reset formulas (R1-R3) ===")
    lv, T, P, rows = CL.level_data(K)
    v = CL.verify_collapse_formulas(lv, T, P, rows)
    check("R1 reset value exact", v["reset_value_mismatches"] == 0,
          "%d levels" % v["collapsing_levels_checked"])
    check("R2 one-period constant Phi exact", v["phi_constant_mismatches"] == 0)
    check("R1 erasure: P-periodic from tau+1", v["erasure_failures"] == 0)
    check("R3 sharp transient bound T(K) <= max(T(K-1), tau+1)",
          v["sharp_transient_bound_violations"] == 0)

    print("\n=== 2. zero coordinates and N1 ===")
    ez = CL.eventually_zero_coordinates(K, T, P, rows)
    pz = CL.permanently_zero_coordinates(K, rows)
    nonc = [r["K"] for r in lv if r["kind"] != "COLLAPSING"]
    check("eventually-zero coordinates are {2,7,28,399}", ez == [2, 7, 28, 399],
          str(ez))
    check("K=7 is the only permanently-zero coordinate", pz == [7], str(pz))
    check("N1: non-COLLAPSING levels == {k+1 : k eventually zero}",
          nonc == [k + 1 for k in ez], str(nonc))
    bit = lambda t, k: (rows[t] >> k) & 1
    n2 = all(bit(t, m - 2) == bit(t, m - 1)
             for m in ez for t in range(T[m], T[m] + P[m]))
    check("N2: w_t(m-2) = w_t(m-1) on a zero cycle", n2)
    par = all(sum(bit(t, m - 1) for t in range(T[m], T[m] + P[m])) % 2 == 1
              for m in ez)
    check("all four opportunities have ODD parity (hence DOUBLING, not NEUTRAL)",
          par)

    print("\n=== 3. chains and the transducer (CH1, CH2) ===")
    runs = CL.chain_runs(lv)
    check("chain runs are the gaps between doubling points",
          runs[:4] == [(2, 2), (4, 7), (9, 28), (30, 399)], str(runs[:5]))
    lo, hi = runs[-1]
    tr = CL.chain_transducer_check(rows, T, P, max(lo + 2, 402), hi, T[hi])
    check("CH1: cycle word is always a solution of the fibre recurrence",
          tr["cycle_word_always_a_solution"], "%d levels" % tr["levels_checked"])
    check("CH1: the solution is UNIQUE at every collapsing level",
          tr["levels_with_unique_solution"] == tr["levels_checked"])
    check("CH2: transducer memory is 2P bits", tr["memory_bits"] == 2 * tr["period"],
          "%d bits" % tr["memory_bits"])
    check("NEUTRAL words admit two solutions (relaxed-model counterexample)",
          len(CL.solve_fibre_cycle([0], [0])) == 2)

    print("\n=== 4. diagonal cone (DD1) ===")
    for t in ([60] if a.fast else [60, 400, 1200]):
        c = CL.diagonal_cone(t, T, K)
        rowsc = c["per_s"]
        check("DD1 cone at s=t is the single point (t,t) [t=%d]" % t,
              rowsc[t]["lo"] == t and rowsc[t]["hi"] == t)
        check("DD1 cone at s=t/2 is the whole row [t=%d]" % t,
              rowsc[t // 2]["lo"] == 0 and rowsc[t // 2]["hi"] >= t - 1)
        check("DD1 cone at s=0 is the seed cell [t=%d]" % t,
              rowsc[0]["lo"] == 0 and rowsc[0]["hi"] == 0)
        check("cone becomes entirely transient before s=t [t=%d]" % t,
              0 < (c["first_s_with_cone_entirely_transient"] or 0) < t,
              "s=%s (%.3f t)" % (c["first_s_with_cone_entirely_transient"],
                                 c["crossover_fraction"] or 0))

    print("\n=== 5. diagonal ANF ===")
    tm = 8 if a.fast else 11
    hist = CL.anf_rows(tm, tm)
    check("ANF reproduces the orbit under the single-cell seed",
          CL.verify_anf_against_orbit(min(tm, 9), min(tm, 11))["exact"])
    d = [CL.anf_stats(hist, t, t) for t in range(tm + 1)]
    check("monomial count is strictly increasing in t",
          all(d[i]["monomials"] < d[i + 1]["monomials"] for i in range(tm)),
          str([x["monomials"] for x in d]))
    check("NEGATIVE RESULT: w_t(k) never loses z_k (no variable elimination)",
          all(CL.anf_stats(hist, t, k)["depends_on_z_k"]
              for t in range(tm + 1) for k in range(min(tm, 8) + 1)))

    print("\n=== 6. stored results ===")
    p = "results/phase2d_results.json"
    if os.path.exists(p):
        j = json.load(open(p))
        check("stored run: all collapse formulas exact",
              j["collapse_formula_verification"]["all_exact"])
        check("stored run: N1 holds", j["nonc_equals_ez_plus_one"])
        check("stored run: 4 doubling points at 3,8,29,400",
              j["doubling_K"] == [3, 8, 29, 400], str(j["doubling_K"]))
        check("stored run: transducer unique everywhere",
              j["transducer"]["levels_with_unique_solution"]
              == j["transducer"]["levels_checked"])
        check("stored run: no variable elimination observed",
              all(e["first_t_without_z_k"] is None
                  for e in j["variable_elimination"]))
    else:
        check("results/phase2d_results.json present", False)

    print("\n" + "=" * 70)
    print("%d passed, %d FAILED   (%.1f s)" % (len(PASSES), len(FAILURES),
                                               time.time() - t0))
    if FAILURES:
        print("failures: %s" % ", ".join(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
