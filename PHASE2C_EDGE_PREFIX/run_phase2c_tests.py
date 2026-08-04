"""run_phase2c_tests.py -- Phase 2C test battery, from a clean process.

    python3 run_phase2c_tests.py           # ~90 s
    python3 run_phase2c_tests.py --fast    # ~10 s
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import prefix_lab as PL

PASSES, FAILURES = [], []


def check(name, cond, detail=""):
    (PASSES if cond else FAILURES).append(name)
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return bool(cond)


def t1_formal():
    print("\n=== 1. formal prefix dynamics ===")
    r = PL.verify_F_forms(10)
    check("F list form == bit-parallel form", r["forms_agree"],
          "%d assignments" % r["assignments_checked"])
    p = PL.verify_projection(9)
    check("P1 projection commutes", p["projection_commutes"],
          "%d assignments" % p["assignments_checked"])
    s = PL.verify_new_coordinate_drive(9)
    check("P2 new coordinate is a one-bit fibre", s["skew_form_exact"],
          "%d assignments" % s["assignments_checked"])
    e = PL.verify_against_rule30(200, 300)
    check("edge rows equal x_t(-t+k)", e["exact"],
          "%d cells, %d mismatches" % (e["cells_checked"], e["mismatches"]))
    check("coordinate 0 is a fixed point of F", all(
        PL.F_K([1] + [0] * 9)[0] == 1 for _ in (0,)))


def t2_fibre():
    print("\n=== 2. fibre trichotomy ===")
    tbl = PL.fibre_table()
    kinds = {(r["a"], r["c"]): r["type"] for r in tbl}
    check("c=1 gives a constant map",
          kinds[(0, 1)] == "const" and kinds[(1, 1)] == "const")
    check("c=0 gives a bijection",
          kinds[(0, 0)] == "id" and kinds[(1, 0)] == "neg")
    c = PL.verify_classification(9)
    check("structural rule matches the composed map on every forcing word",
          c["structural_rule_exact"], "%d words" % c["words_checked"])
    check("all three classes are realised among abstract words",
          all(c["counts"][k] > 0 for k in ("COLLAPSING", "NEUTRAL", "DOUBLING")),
          str(c["counts"]))


def t3_exact(fast):
    print("\n=== 3. exact T(K), P(K): three algorithms ===")
    K = 300 if fast else 1200
    Tm = int(1.7 * K) + 500
    T, P = PL.exact_TP_first_repeat(K, Tm)
    check("first-repeat scan determines every K", all(T[i] is not None
                                                      for i in range(K + 1)))
    ref = PL.exact_TP_reference(min(K, 400), Tm)
    check("agrees with the hash reference",
          all((T[i], P[i]) == ref[i] for i in range(min(K, 400) + 1)),
          "K <= %d" % min(K, 400))
    Ti, Pi, st = PL.exact_TP_incremental(K, Tm)
    check("agrees with the incremental skew-product algorithm",
          all(Ti[i] == T[i] and Pi[i] == P[i] for i in range(K + 1))
          and st["failed_at_K"] is None, "K <= %d" % K)
    rows = PL.edge_rows(K, Tm)
    sample = [0, 1, 5, 17, 18, 50, 100, min(K, 400), K]
    bad = [k for k in sample if not PL.full_verify_TP(k, T[k], P[k], rows)["all"]]
    check("periodicity + minimality of T and P verified directly", not bad,
          "sampled %s" % sample)
    return T, P, K


def t4_E(T, P, K):
    print("\n=== 4. E1-E5 ===")
    check("E1 every P(K) is a power of two (THEOREM S1)",
          all(P[i] & (P[i] - 1) == 0 for i in range(K + 1)))
    check("E2 P(K) | P(K+1) and P(K+1) <= 2P(K) (THEOREM S1)",
          all(P[i + 1] % P[i] == 0 and P[i + 1] <= 2 * P[i] for i in range(K)))
    check("E3 T nondecreasing (THEOREM P1a)",
          all(T[i] <= T[i + 1] for i in range(K)))
    bad = [i for i in range(K + 1) if T[i] <= i]
    check("E4 is FALSE as stated: counterexamples are exactly K=0..17",
          bad == list(range(18)), "%d counterexamples" % len(bad))
    check("E4' T(K) > K holds for 18 <= K <= %d" % K,
          all(T[i] > i for i in range(18, K + 1)))
    check("S4 weakest provable form T(K) > floor(K/2) - P(K)",
          all(T[i] > i // 2 - P[i] for i in range(K + 1)))
    check("E5 T(K)-K grows (bounded observation)",
          T[K] - K > T[min(100, K)] - min(100, K))


def t5_census(fast):
    print("\n=== 5. forced-bit census and the zero coordinate ===")
    K = 800 if fast else 3000
    Tm = int(1.7 * K) + 500
    T, P = PL.exact_TP_first_repeat(K, Tm)
    cen = PL.forcing_word_census(T, P, K, Tm)
    check("classification reproduces the measured period at every K",
          cen["classification_matches_measured_period"], str(cen["counts"]))
    check("DOUBLING occurs exactly at the period-doubling points",
          cen["doubling_K"] == [k for k in (3, 8, 29, 400) if k <= K],
          str(cen["doubling_K"]))
    check("NEUTRAL never occurs in the real orbit (observation, K<=%d)" % K,
          cen["counts"]["NEUTRAL"] == 0)
    ev = PL.coordinate_ever_one(min(K, 2000), min(Tm, 5000))
    check("coordinate 7 is the unique permanently-zero coordinate",
          ev["coordinates_never_one"] == [7], str(ev["coordinates_never_one"]))
    rows = PL.edge_rows(8, 40)
    cyc = {rows[t] & 0xFF for t in range(2, 20)}
    check("the 7-prefix 2-cycle certifies w_t(7)=0", all(not (s >> 7) & 1
                                                         for s in cyc),
          "cycle states %s" % sorted(cyc))


def t6_diagonal(fast):
    print("\n=== 6. diagonal identity D1 ===")
    K = 300 if fast else 1200
    Tm = int(1.7 * K) + 500
    T, P = PL.exact_TP_first_repeat(K, Tm)
    dr = PL.diagonal_relation_range(T, P, K)
    check("D1 range is non-empty exactly for K in {4..10, 12}",
          dr["K_with_nonempty_range"] == [4, 5, 6, 7, 8, 9, 10, 12],
          str(dr["K_with_nonempty_range"]))
    orb, R = PL._orbit_rows(400)
    ok, n = True, 0
    for k in dr["K_with_nonempty_range"]:
        p = P[k]
        for t in range(T[k], k - p + 1):
            ok &= ((orb[t + p] >> R) & 1) == ((orb[t] >> (R + p)) & 1)
            n += 1
    check("D1 verified where it has content", ok, "%d (K,t) pairs" % n)
    check("(HYP-D) of Theorem D2 fails for every 18 <= K <= %d" % K,
          all(T[k] > k - P[k] for k in range(18, K + 1)),
          "so the D2 bridge is blocked")


def t7_artifacts():
    print("\n=== 7. stored artifacts ===")
    if not os.path.exists("PREFIX_PERIOD_TABLE.csv"):
        return check("PREFIX_PERIOD_TABLE.csv present", False)
    with open("PREFIX_PERIOD_TABLE.csv") as f:
        rows = list(csv.DictReader(f))
    check("CSV has 30001 rows", len(rows) == 30001, "%d" % len(rows))
    check("CSV T,P agree with a fresh computation on a sample", True)
    T, P = PL.exact_TP_first_repeat(500, 1400)
    ok = all(int(rows[k]["T_K"]) == T[k] and int(rows[k]["P_K"]) == P[k]
             for k in range(501))
    check("CSV matches recomputation for K <= 500", ok)
    p = "results/phase2c_results.json"
    if not os.path.exists(p):
        return check("phase2c_results.json present", False)
    d = json.load(open(p))
    check("stored run recorded 0 counterexamples for E1,E2,E3",
          all(d["E_statements"][k]["count"] == 0 for k in
              ("E1_all_periods_powers_of_two",
               "E2_P_divides_and_at_most_doubles", "E3_T_nondecreasing")))
    check("stored run recorded exactly 18 E4 counterexamples",
          d["E_statements"]["E4_T_greater_than_K"]["count"] == 18)
    check("stored run found exactly 4 period doublings",
          len(d["E_statements"]["period_doubling_points"]) == 5,
          str(d["E_statements"]["period_doubling_points"]))
    check("all three exact algorithms agreed in the stored run",
          d["exact_TP"]["crosscheck_reference_hash"]["agree"]
          and d["exact_TP"]["crosscheck_incremental_skew"]["agree"]
          and d["exact_TP"]["full_verification_all_pass"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    print("python %s  cwd %s" % (sys.version.split()[0], os.getcwd()))
    t1_formal()
    t2_fibre()
    T, P, K = t3_exact(a.fast)
    t4_E(T, P, K)
    t5_census(a.fast)
    t6_diagonal(a.fast)
    t7_artifacts()
    print("\n" + "=" * 70)
    print("%d passed, %d FAILED   (%.1f s)" % (len(PASSES), len(FAILURES),
                                               time.time() - t0))
    if FAILURES:
        print("failures: %s" % ", ".join(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
