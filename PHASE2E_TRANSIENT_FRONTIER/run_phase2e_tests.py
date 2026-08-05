"""run_phase2e_tests.py -- Phase 2E test battery, from a clean process.

Exits non-zero if anything fails.

    python3 run_phase2e_tests.py            # ~60 s
    python3 run_phase2e_tests.py --fast     # ~10 s

Coverage
  1  the imported foundations are re-verified here, not trusted
  2  the exact (T, P) table, three independent computations
  3  Theorem A: T(K) = max_{k<=K} r_{P(K)}(k)
  4  Theorem B and the RESETTING / INHERITING dichotomy
  5  the reset schedule: rho, tau, increments
  6  frontier dynamics and the bound dT <= P
  7  the co-moving strip: identity, non-autonomy, cone growth
  8  the proof DAG: soundness of the rewrite, and its measured limits
  9  the stored results JSON matches what the documents claim
 11  reset times sigma/tau/rho and the fibre dependence horizon (Theorem 4.1)
 12  the five-way classification, cross-derived
 13  Theorems C1/C2/C3: the frontier increment is not finite-state
 14  the co-moving strip: step law, preimage (D2), lookahead (D4)
 15  the reset-erased skeleton, the XOR spine (E1), and its proof DAG
 16  Theorem G1 and the refutation of "equal value => equal derivation"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import collapse_lab as CL
import frontier_lab as F
import prefix_lab as PL
import transient_lab as X

PASSES, FAILURES = [], []
_CACHE = {}


def check(name, cond, detail=""):
    (PASSES if cond else FAILURES).append(name)
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return bool(cond)


def table(K):
    if K not in _CACHE:
        _CACHE[K] = F.level_table(K)
    return _CACHE[K]


def t1_foundations(fast):
    print("\n=== 1. foundations re-verified, not assumed ===")
    r = PL.verify_F_forms(10)
    check("F_K list and bit-parallel forms agree", r["forms_agree"])
    v = PL.verify_against_rule30(120 if fast else 300, 200 if fast else 400)
    check("edge rows equal x_t(-t+k)", v["exact"],
          "%d cells, %d mismatches" % (v["cells_checked"], v["mismatches"]))
    # support bound w_t(k) = 0 for k > 2t, used for constant folding in the DAG
    rows = PL.edge_rows(200, 120)
    bad = [(t, k) for t in range(121) for k in range(2 * t + 1, 201)
           if (rows[t] >> k) & 1]
    check("support bound w_t(k)=0 for k>2t", not bad, str(bad[:3]))
    # the seed
    check("w_0 is the single-cell seed", rows[0] == 1)


def t2_exact_TP(fast):
    print("\n=== 2. exact (T, P), three independent computations ===")
    K = 400 if fast else 1200
    T_max = int(1.5 * K) + 1000
    T, P = PL.exact_TP_first_repeat(K, T_max)
    ref = PL.exact_TP_reference(min(K, 300), T_max)
    check("first-repeat vs hash reference",
          all((T[k], P[k]) == ref[k] for k in range(min(K, 300) + 1)))
    Ti, Pi, st = PL.exact_TP_incremental(K, T_max)
    check("first-repeat vs incremental skew-product",
          all(Ti[k] == T[k] and Pi[k] == P[k] for k in range(K + 1)),
          str(st))
    rows = PL.edge_rows(K, T_max)
    sample = [0, 1, 17, 18, 50, 400] + ([] if fast else [800, 1200])
    fv = {k: PL.full_verify_TP(k, T[k], P[k], rows) for k in sample if k <= K}
    check("full periodicity + minimality at %d sampled K" % len(fv),
          all(v["all"] for v in fv.values()),
          str({k: v for k, v in fv.items() if not v["all"]}))


def t3_theorem_A(fast):
    print("\n=== 3. Theorem A: T(K) = max_{k<=K} r_{P(K)}(k) ===")
    K = 800 if fast else 4000
    recs, T, P, rows, Rq = table(K)
    a = F.verify_theorem_A(recs, T, P, Rq, K)
    check("Theorem A exact", a["exact"],
          "%d levels, %d mismatches" % (a["levels_checked"], a["mismatches"]))
    two = F.verify_R_two_ways(rows, Rq, K, K_cross=min(600, K),
                              spot=[(P[K], K), (P[K // 2], K // 2)])
    check("r_Q by three independent implementations agree", two["all_agree"],
          str(two["per_lag"]))
    dc = F.definition_comparison(recs)
    check("R(K) <= T(K) always (Corollary 1.2)", dc["T_never_below_R"],
          "T<R at %d levels" % dc["counts"]["T_lt_R"])
    check("R(K) is always tau+1 or <= T(K-1) (Corollary 2.2)",
          dc["R_always_classified"], "unclassified %d" % dc["counts"]["R_other"])
    check("R and T are genuinely different functions",
          dc["counts"]["T_gt_R"] > 0, "T > R at %d levels" % dc["counts"]["T_gt_R"])


def t4_theorem_B(fast):
    print("\n=== 4. Theorem B and the dichotomy ===")
    K = 800 if fast else 4000
    recs, T, P, rows, Rq = table(K)
    b = F.verify_theorem_B(recs)
    check("Theorem B predicts T(K) exactly at every level", b["exact"],
          "%d levels, %d mismatches, %s"
          % (b["levels_checked"], b["mismatches"], b["class_counts"]))
    d = F.verify_dichotomy(recs)
    check("defect bit D predicts RESETTING/INHERITING", d["D_bit_predicts_class"],
          str(d["D_violations"]))
    check("RESETTING implies COLLAPSING", d["resetting_implies_collapsing"],
          str(d["kind_violations"]))
    check("RESETTING implies T(K) = tau_K + 1", d["resetting_T_equals_tau_plus_1"],
          str(d["reset_violations"]))
    # Theorem B2 at the four measured doubling levels
    byK = {r["K"]: r for r in recs}
    dbl = [k for k in (3, 8, 29, 400) if k <= K]
    check("Theorem B2: T(K) = T(K-1) at every DOUBLING level",
          all(byK[k]["T_K"] == byK[k]["T_prev"] for k in dbl),
          str({k: (byK[k]["T_prev"], byK[k]["T_K"]) for k in dbl}))
    check("every DOUBLING level is classified INHERITING",
          all(byK[k]["cls"] == "INHERITING" for k in dbl))
    check("no NEUTRAL level observed (BOUNDED OBSERVATION, not a theorem)",
          all(r["fibre_kind"] != "NEUTRAL" for r in recs))
    # Lemma 2.1, directly on the orbit
    bad = 0
    n = 0
    for r in recs[:200]:
        Kk, p, Tp = r["K"], r["P_K"], r["T_prev"]
        if r["P_K"] != r["P_prev"]:
            continue
        for s in range(Tp, min(Tp + 3 * p, len(rows) - p - 1)):
            c = (rows[s] >> (Kk - 1)) & 1
            Ds = ((rows[s] ^ rows[s + p]) >> Kk) & 1
            Ds1 = ((rows[s + 1] ^ rows[s + 1 + p]) >> Kk) & 1
            n += 1
            bad += Ds1 != (0 if c else Ds)
    check("Lemma 2.1 (defect propagation) on the real orbit", bad == 0,
          "%d checks, %d violations" % (n, bad))


def t5_reset_schedule(fast):
    print("\n=== 5. reset schedule ===")
    K = 800 if fast else 4000
    recs, T, P, rows, Rq = table(K)
    rs = F.reset_schedule(recs)
    check("Theorem 3.1: T(K) = rho(K)+1 at EVERY resetting level",
          rs["by_class"]["RESETTING"]["fraction"] == 1.0,
          "%d levels" % rs["by_class"]["RESETTING"]["levels"])
    check("the INHERITING rate is NOT 1 (the aggregate figure is a mixture)",
          rs["by_class"]["INHERITING"]["fraction"] < 1.0,
          "%.4f" % rs["by_class"]["INHERITING"]["fraction"])
    # Proposition 3.2, checked directly
    bad = []
    for r in recs:
        if r["rho_K"] is None:
            continue
        lhs = (r["T_K"] == r["rho_K"] + 1)
        rhs = bool((rows[r["T_K"] - 1] >> (r["K"] - 1)) & 1)
        if lhs != rhs:
            bad.append(r["K"])
    check("Proposition 3.2: T=rho+1 iff w_{T-1}(K-1)=1", not bad, str(bad[:5]))
    ri = F.reset_increments(recs)
    check("every reset increment is >= 1", min(ri["increment_histogram"]) >= 1)
    check("Theorem 3.4: every increment <= P", max(ri["increment_histogram"]) <= P[K],
          "max %d, P %d" % (max(ri["increment_histogram"]), P[K]))
    # telescoping identity, Corollary B5
    tot = sum(k * v for k, v in ri["increment_histogram"].items())
    check("Corollary B5 telescoping: T(K_max) = T(1) + sum of increments",
          T[K] == T[1] + tot, "%d vs %d + %d" % (T[K], T[1], tot))


def t6_frontier(fast):
    print("\n=== 6. frontier dynamics ===")
    K = 800 if fast else 4000
    recs, T, P, rows, Rq = table(K)
    fs = F.frontier_stats(T, P, K, T[K])
    check("K_per by two independent implementations agree",
          fs["two_implementations_agree"])
    check("T is non-decreasing (Theorem 1.3)", not fs["monotonicity_violations"],
          str(fs["monotonicity_violations"]))
    check("Theorem 4.2: T(K+1)-T(K) <= P(K)", fs["n_bound_violations"] == 0,
          "%d violations" % fs["n_bound_violations"])
    ez = CL.eventually_zero_coordinates(K, T, P, rows)
    check("eventually-zero coordinates are a subset of {2,7,28,399}",
          set(ez) <= {2, 7, 28, 399}, str(ez))
    ns = F.diagonal_never_settled(T, K)
    check("T(K) <= K exactly for K <= 17 (Phase 2C E4 reproduced)",
          ns["first_K_from_which_T_gt_K"] == 18,
          str(ns["levels_with_T_le_K"]))


def t7_strip(fast):
    print("\n=== 7. co-moving transient strip ===")
    ident = F.strip_identity(120 if fast else 300, 6 if fast else 8)
    check("Theorem 5.1: q_t(r) = x_t(r)", ident["exact"],
          "%d cells, %d mismatches" % (ident["cells_checked"], ident["mismatches"]))
    cen = F.strip_census(3000 if fast else 20000, 8 if fast else 12)
    check("Theorem 5.4: a non-autonomy witness exists at EVERY radius",
          all(c["is_autonomous"] is False for c in cen),
          str([c["R"] for c in cen if c["is_autonomous"] is not False]))
    # re-verify one witness from scratch against the original-coordinate engine
    c = cen[0]
    w = c["autonomy_counterexample"]
    ws = F.strip_words(200, c["R"])
    check("the R=1 witness reproduces from a fresh simulation",
          ws[w["t1"]] == ws[w["t2"]] == w["state"]
          and ws[w["t1"] + 1] == w["succ1"] and ws[w["t2"] + 1] == w["succ2"],
          str(w))
    dep = F.strip_dependency_growth(50, 4)
    check("Theorem 5.5: cone opens by one column per side per step",
          dep["final_cone"] == [-50, 54] and dep["right_dependence_is_genuine"],
          str(dep["final_cone"]))
    # word coverage depends on how far the run goes; the documented claim
    # ("all words for R <= 10") is a t <= 20000 observation, so the fast run
    # asserts only what its shorter run can support.
    lim = 7 if fast else 10
    check("all 2^(R+1) strip words occur for R <= %d (t <= %d)"
          % (lim, 3000 if fast else 20000),
          all(c["all_words_seen"] for c in cen if c["R"] <= lim),
          str([(c["R"], c["distinct_words"], c["possible_words"])
               for c in cen if c["R"] <= lim and not c["all_words_seen"]]))


def t8_dag(fast):
    print("\n=== 8. proof DAG ===")
    K = 800 if fast else 4000
    recs, T, P, rows, Rq = table(K)
    # soundness of the periodicity rewrite, checked cell by cell
    bad = 0
    n = 0
    for k in range(0, min(K, 300) + 1):
        for s in range(T[k] + P[k], min(len(rows), T[k] + P[k] + 60)):
            n += 1
            bad += ((rows[s] >> k) & 1) != ((rows[s - P[k]] >> k) & 1)
    check("rewrite (R) is sound on the real orbit", bad == 0,
          "%d cells, %d violations" % (n, bad))
    # canon() agrees with the orbit
    bad = 0
    n = 0
    for s in range(0, 200):
        for k in range(0, 60):
            c = F.canon(s, k, T, P)
            v = (rows[s] >> k) & 1 if k >= 0 else 0
            got = c[1] if c[0] == "c" else (rows[c[1]] >> c[2]) & 1
            n += 1
            bad += got != v
    check("canon() preserves the value at every cell", bad == 0,
          "%d cells, %d violations" % (n, bad))
    ts = [50, 150] if fast else [100, 300, 600]
    dag = F.dag_growth(ts, T, P)
    check("DAG is smaller than the naive cone at every t",
          all(d["nodes"] < d["naive_cone_cells"] for d in dag),
          str([(d["t"], d["nodes"], d["naive_cone_cells"]) for d in dag]))
    check("DAG reduction factor stays below 2 (NEGATIVE RESULT)",
          all(d["reduction_factor"] < 2.0 for d in dag),
          str([round(d["reduction_factor"], 3) for d in dag]))
    check("nodes/t^2 does not fall by more than 20% over the range (quadratic)",
          min(d["nodes_over_t_squared"] for d in dag)
          > 0.8 * max(d["nodes_over_t_squared"] for d in dag),
          str([round(d["nodes_over_t_squared"], 4) for d in dag]))


def t9_results_json():
    print("\n=== 9. stored results match the documents ===")
    path = os.path.join("results", "phase2e_results.json")
    if not os.path.exists(path):
        return check("phase2e_results.json present", False, "run run_phase2e.py")
    d = json.load(open(path))
    check("Theorem A exact in the stored run",
          d["definitions"]["theorem_A"]["exact"])
    check("Theorem B exact in the stored run",
          d["recurrence"]["theorem_B"]["exact"])
    cc = d["recurrence"]["theorem_B"]["class_counts"]
    check("class counts are 16394 RESETTING / 13605 INHERITING",
          cc == {"RESETTING": 16394, "INHERITING": 13605}, str(cc))
    rsd = d["reset_schedule"]
    check("rho identity holds at every RESETTING level in the stored run",
          rsd["rho"]["by_class"]["RESETTING"]["fraction"] == 1.0)
    check("density x mean increment matches T/K to 1e-4",
          abs(rsd["density_times_mean_increment"] - rsd["T_over_K_at_K_max"])
          < 1e-4,
          "%.6f vs %.6f" % (rsd["density_times_mean_increment"],
                            rsd["T_over_K_at_K_max"]))
    check("no violation of dT <= P in the stored run",
          d["frontier"]["stats"]["n_bound_violations"] == 0)
    check("strip is non-autonomous at every radius in the stored run",
          all(c["is_autonomous"] is False for c in d["strip"]["census"]))
    check("strip identity exact in the stored run", d["strip"]["identity"]["exact"])
    check("DAG reduction factor below 2 at every t in the stored run",
          all(x["reduction_factor"] < 2.0 for x in d["proof_dag"]),
          str([round(x["reduction_factor"], 3) for x in d["proof_dag"]]))
    check("Theorem D1 has no content at p >= 7 (documented vacuity)",
          all(r["levels_where_D1_has_content"] == 0
              for r in d["hypothesis"]["D1_content"]["per_p"] if r["p"] >= 7))
    check("eventually-zero coordinates recorded as {2,7,28,399}",
          d["recurrence"]["eventually_zero_coordinates"] == [2, 7, 28, 399])


def t10_csv():
    print("\n=== 10. CSV files ===")
    for name, ncol in (("TRANSIENT_CLASSIFICATION.csv", 15),
                       ("RESET_SCHEDULE_RESULTS.csv", 8)):
        if not os.path.exists(name):
            check("%s present" % name, False, "run run_phase2e.py")
            continue
        with open(name) as f:
            head = f.readline().strip().split(",")
            rows = sum(1 for _ in f)
        check("%s has %d columns and %d rows" % (name, ncol, rows),
              len(head) == ncol and rows > 0, str(head))
    if os.path.exists("TRANSIENT_CLASSIFICATION.csv"):
        import csv as _csv
        with open("TRANSIENT_CLASSIFICATION.csv") as f:
            bad = [r["K"] for r in _csv.DictReader(f) if r["matches"] != "1"]
        check("every row of TRANSIENT_CLASSIFICATION.csv has matches=1",
              not bad, str(bad[:5]))


def t11_reset_times(fast):
    print("\n=== 11. reset times and the dependence horizon ===")
    K = 800 if fast else 3000
    recs, T, P, rows, Rq = table(K)
    ks = [k for k in (2, 3, 5, 9, 17, 33, 100, 500, 1500, 2999) if k <= K]
    hz = X.verify_horizon(rows, ks, min(len(rows), 6000))
    check("Theorem 4.1: R_dep(K) by perturbation = sigma(K)", hz["agree"],
          "%d levels, %d mismatches" % (hz["levels_checked"], hz["mismatches"]))
    tab = X.reset_time_table(recs[:400], rows, min(len(rows), 6000))
    c = X.compare_reset_times(tab)["counts"]
    check("sigma and tau are distinct (sigma < tau on most levels)",
          c["sigma_lt_tau"] > c["sigma_eq_tau"],
          "sigma<tau %d, sigma=tau %d" % (c["sigma_lt_tau"], c["sigma_eq_tau"]))
    check("tau and rho are distinct (rho < tau on the inheriting levels)",
          c["rho_lt_tau"] > 0, "rho<tau at %d levels" % c["rho_lt_tau"])
    check("tau never exceeds rho where both are defined", c["tau_lt_rho"] == 0)
    # sigma <= tau by definition, checked on the table
    bad = [r["K"] for r in tab
           if r["sigma"] is not None and r["tau"] is not None
           and r["sigma"] > r["tau"]]
    check("sigma <= tau at every level", not bad, str(bad[:5]))


def t12_five_way(fast):
    print("\n=== 12. the five-way classification ===")
    K = 800 if fast else 3000
    recs, T, P, rows, Rq = table(K)
    ez = CL.eventually_zero_coordinates(K, T, P, rows)
    ct = X.classification_table(recs, rows, ez)
    check("every level's predicted T matches", ct["n_prediction_failures"] == 0,
          str(ct["prediction_failures"][:3]))
    check("the two independent derivations of the case agree",
          ct["n_independent_disagreements"] == 0,
          str(ct["independent_disagreements"][:3]))
    check("the five cases partition the levels",
          sum(ct["counts"].values()) == len(recs), str(ct["counts"]))
    check("PERIOD_DOUBLING levels are exactly {3,8,29,400} (BOUNDED)",
          ct["period_doubling_levels"] == [k for k in (3, 8, 29, 400) if k <= K],
          str(ct["period_doubling_levels"]))
    check("no ZERO_PREDECESSOR level occurs (BOUNDED OBSERVATION, not a theorem)",
          ct["counts"]["ZERO_PREDECESSOR"] == 0)
    check("PHASE_DELAY is the largest resetting class",
          ct["counts"]["PHASE_DELAY"] > ct["counts"]["RESET_IMMEDIATE"],
          "%d vs %d" % (ct["counts"]["PHASE_DELAY"], ct["counts"]["RESET_IMMEDIATE"]))


def t13_finite_state(fast):
    print("\n=== 13. the frontier increment is not finite-state ===")
    K = 800 if fast else 3000
    recs, T, P, rows, Rq = table(K)
    il = X.verify_increment_law(recs, rows)
    check("Theorem C1: increment = f(cycle word, D)", il["exact"],
          "%d levels, %d mismatches" % (il["levels_checked"], il["mismatches"]))
    df = X.verify_defect_bit_formula(recs, rows)
    check("Theorem C2: D(K) = w_{T(K-1)}(K) XOR Phi_K", df["exact"],
          "%d levels, %d mismatches" % (df["levels_checked"], df["mismatches"]))
    fl = X.verify_flip_flips_defect(recs, rows, 200)
    check("Corollary C3: flipping the transient bit flips D",
          fl["flip_always_flips_D"], "%d levels" % fl["levels_checked"])
    fs1 = X.defect_bit_is_finite_state(recs, rows, 1)
    check("depth-1 cycle-word description refuted by explicit counterexamples",
          fs1["finite_state_refuted"],
          "%d counterexamples" % fs1["counterexamples"])
    fs2 = X.defect_bit_is_finite_state(recs, rows, 2)
    check("depth-2 collision test is reported as having no power",
          fs2["distinct_windows"] >= len(recs) - 10,
          "%d distinct windows over %d levels"
          % (fs2["distinct_windows"], len(recs)))


def t14_strip2(fast):
    print("\n=== 14. the co-moving strip of width 2R+1 ===")
    bad = 0
    for R in (1, 2, 3):
        for q in range(1 << (2 * R + 1)):
            for l in (0, 1):
                for r in (0, 1):
                    bad += X.strip_step(q, R, l, r) != X.strip_step_slow(q, R, l, r)
    check("bit-parallel and bit-by-bit strip steps agree exhaustively", bad == 0)
    ss = X.verify_strip_step(500 if fast else 2000, 6)
    check("strip step law reproduces the real orbit", ss["exact"],
          "%d steps, %d mismatches" % (ss["steps"], ss["mismatches"]))
    for R in ([1, 2, 3] if fast else [1, 2, 3, 4, 5, 6]):
        v = X.verify_preimage(R)
        check("Theorem D2 at R=%d: in-degree exactly 4, preimage exact" % R,
              v["every_state_has_indegree_4"] and v["preimage_failures"] == 0,
              "%d states" % v["states"])
    for R in ([1, 2] if fast else [1, 2, 3, 4]):
        la = X.strip_lookahead(5000 if fast else 20000, R)
        check("Theorem D4 at R=%d: first counterexample at n=R+1" % R,
              la["first_n_with_counterexample"] == R + 1,
              str(la["first_n_with_counterexample"]))
    for m in ([1, 4] if fast else [1, 4, 12, 20]):
        r = X.centre_history_determines_next(10000 if fast else 50000, m)
        check("centre column is not determined by its own last %d bits" % m,
              r["determined"] is False, str(r.get("witness")))


def t15_skeleton(fast):
    print("\n=== 15. the reset-erased skeleton and its DAG ===")
    K = 800 if fast else 3000
    recs, T, P, rows, Rq = table(K)
    for t in ([60, 150] if fast else [100, 300, 600]):
        v = X.verify_spine(t, rows)
        check("Theorem E1: the XOR spine survives at t=%d" % t,
              v["spine_contained"] and v["bound_holds"],
              "%d nodes, spine %d, missing %s"
              % (v["nodes"], v["spine_len"], v["missing"]))
    sk = X.skeleton_growth([60, 150] if fast else [100, 300, 600], rows)
    check("skeleton is smaller than the plain cone at every t",
          all(s["nodes"] < (s["t"] + 1) ** 2 for s in sk),
          str([(s["t"], s["nodes"]) for s in sk]))
    check("skeleton width grows with t",
          all(a["max_width"] < b["max_width"] for a, b in zip(sk, sk[1:])),
          str([s["max_width"] for s in sk]))
    ec = X.erased_edge_census(200 if fast else 400, rows)
    check("reset erasure removes edges but never all three",
          min(ec["surviving_edge_histogram"]) >= 1
          and ec["mean_surviving_edges"] < 3.0,
          "mean %.4f" % ec["mean_surviving_edges"])
    for t in ([60, 150] if fast else [100, 300]):
        d = X.proof_dag(t, rows)
        check("proof DAG at t=%d shares strictly (nodes < cells)" % t,
              d["dag_nodes"] < d["cells"] and d["constants"] <= 2,
              "%d nodes from %d cells, %d xor, %d or"
              % (d["dag_nodes"], d["cells"], d["xor_nodes"], d["or_nodes"]))
    cb = X.skeleton_combined(150 if fast else 600, rows, T, P)
    sk1 = X.skeleton(150 if fast else 600, rows)
    check("reset + periodicity beats reset alone", cb["nodes"] < sk1["nodes"],
          "%d vs %d" % (cb["nodes"], sk1["nodes"]))
    rf = X.dag_recursive_family(rows, [32, 64, 128] if fast else [64, 128, 256])
    check("no constant-ratio or constant-difference DAG family (NEGATIVE)",
          not rf["constant_ratio"] and not rf["constant_difference"],
          str(rf["ratios"]))


def t16_hypothesis2(fast):
    print("\n=== 16. the periodicity hypothesis in these variables ===")
    K = 800 if fast else 3000
    recs, T, P, rows, Rq = table(K)
    drs = X.diagonal_reset_schedule(rows, 1000 if fast else 3000)
    check("Theorem G1: diagonal reset schedule = centre column shifted",
          drs["identity_verified"], "%d resets" % drs["n_resets"])
    de = X.dag_equality_under_shift(rows, [200, 300] if fast else [200, 300, 400, 500], 29)
    check("'equal centre value => equal derivation' is REFUTED",
          de["implication_refuted"], str(de["equal_value_different_dag"][:1]))
    # F5's conclusion is impossible: a repeated prefix state forces T(K) <= t
    bad = []
    for k in (5, 20, 100):
        if k > K:
            continue
        mask = (1 << (k + 1)) - 1
        seen = {}
        for t in range(min(len(rows), T[k] + 3 * P[k] + 5)):
            st = rows[t] & mask
            if st in seen and T[k] > seen[st]:
                bad.append((k, seen[st], t))
            seen.setdefault(st, t)
    check("F5's configuration (repeat with t < T(K)) never occurs", not bad,
          str(bad[:3]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    print("python %s   cwd %s" % (sys.version.split()[0], os.getcwd()))
    t1_foundations(a.fast)
    t2_exact_TP(a.fast)
    t3_theorem_A(a.fast)
    t4_theorem_B(a.fast)
    t5_reset_schedule(a.fast)
    t6_frontier(a.fast)
    t7_strip(a.fast)
    t8_dag(a.fast)
    t9_results_json()
    t10_csv()
    t11_reset_times(a.fast)
    t12_five_way(a.fast)
    t13_finite_state(a.fast)
    t14_strip2(a.fast)
    t15_skeleton(a.fast)
    t16_hypothesis2(a.fast)
    print("\n" + "=" * 70)
    print("%d passed, %d FAILED   (%.1f s)"
          % (len(PASSES), len(FAILURES), time.time() - t0))
    if FAILURES:
        print("failures: %s" % ", ".join(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
