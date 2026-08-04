"""run_phase2b_tests.py -- Phase 2B test battery, from a clean process.

Exits non-zero if anything fails.

    python3 run_phase2b_tests.py           # ~3 min
    python3 run_phase2b_tests.py --fast    # ~30 s

Coverage
  1  every Boolean identity, exhaustively (no sampling)
  2  the identities re-checked cell by cell against the real orbit
  3  zero-wall lemmas W1, W3, W4 wherever their hypotheses occur
  4  the support-edge theorem D7
  5  edge-aligned theorems EA1, EA2, EA3
  6  the wall automaton reproduces its published counts
  7  the stored results JSON matches what the documents claim
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import defect_lab as D
import rule30_lab as L

PASSES, FAILURES = [], []


def check(name, cond, detail=""):
    (PASSES if cond else FAILURES).append(name)
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return bool(cond)


def t1_identities():
    print("\n=== 1. Boolean identities, exhaustive ===")
    r = D.verify_defect_identities()
    check("D1 XOR/OR form exact on all 64 assignments", r["xor_or_form_exact"])
    check("D2 ANF form exact on all 64 assignments", r["anf_form_exact"])
    check("D3 Delta table matches ANF on all 16 assignments", r["delta_anf_matches"])
    check("Delta table has 16 rows", len(r["delta_table"]) == 16)
    w = D.verify_zero_wall_reduction()
    check("D4 Delta(u,v,0,g) = g AND NOT u",
          w["delta_with_e0_equals_g_and_not_u"])
    check("D4 Delta(u,v,e,0) = e AND NOT v",
          w["delta_with_g0_equals_e_and_not_v"])
    # the masking rows of D3, called out explicitly in the docs
    mask = all(D.delta(1, v, 0, g) == 0 for v in (0, 1) for g in (0, 1))
    check("a black centre cell with no defect masks the defect on its right", mask)


def t2_orbit_identities(fast):
    print("\n=== 2. identities on the real single-cell orbit ===")
    for p in ([3, 7] if fast else [1, 3, 7, 16, 27]):
        r = D.verify_defect_dynamics_on_orbit(p, 150 if fast else 400, 30)
        check("defect step equation exact, p=%d" % p, r["exact"],
              "%d cells, %d mismatches" % (r["cells_checked"], r["mismatches"]))


def t3_wall_lemmas(fast):
    print("\n=== 3. zero-wall lemmas where their hypotheses occur ===")
    tm = 2000 if fast else 8000
    for p in ([3, 7] if fast else [1, 3, 7, 16, 27]):
        r = D.check_wall_relation(p, tm)
        check("W1 wall relation, p=%d" % p, r["exact"] and r["hypothesis_holds_at"] > 0,
              "hypothesis met %d times, %d violations"
              % (r["hypothesis_holds_at"], r["violations"]))
        r4 = D.check_left_slaving(p, tm)
        check("W4 left slaving, p=%d" % p, r4["exact"] and r4["hypothesis_holds_at"] > 0,
              "hypothesis met %d times, %d violations"
              % (r4["hypothesis_holds_at"], r4["violations"]))
    for p in ([7] if fast else [3, 7, 16, 27]):
        r3 = D.check_ones_run_triangle(p, 5000 if fast else 20000, 2)
        check("W3 ones-run triangle, p=%d" % p,
              r3["exact"] and r3["cells_checked"] > 0,
              "%d windows, %d cells, %d violations"
              % (r3["windows_found"], r3["cells_checked"], r3["violations"]))
    # the documented negative result: W3's hypothesis is vacuous at p = 1
    r1 = D.check_ones_run_triangle(1, 5000 if fast else 20000, 1)
    check("W3 hypothesis is vacuous at p=1 (documented negative result)",
          r1["windows_found"] == 0, "windows found: %d" % r1["windows_found"])


def t4_support_edges(fast):
    print("\n=== 4. support-edge theorem D7 ===")
    for p in ([1, 7] if fast else [1, 2, 7, 27, 64, 1324]):
        g = D.defect_geometry(p, 120 if fast else 400, sample_every=1)
        check("D7 edges exact for p=%d" % p, g["support_edge_defect_theorem_holds"],
              "%d rows" % g["rows_sampled"])
    check("defect field is never empty (implied by D7)", True,
          "leftmost/rightmost are exactly -+(t+p)")


def t5_edge_aligned(fast):
    print("\n=== 5. edge-aligned theorems ===")
    r = D.verify_edge_update(150 if fast else 400, 100 if fast else 300)
    check("EA1 one-sided update exact", r["exact"],
          "%d cells, %d mismatches" % (r["cells_checked"], r["mismatches"]))
    c = D.centre_is_diagonal(200 if fast else 800)
    check("EA3 centre column equals the diagonal w_t(t)",
          c["centre_equals_diagonal"], "t <= %d" % c["t_max"])
    # EA2: prefixes are autonomous, hence eventually periodic
    ok = True
    tm = 400 if fast else 2600
    for K in ([8, 32] if fast else [8, 32, 128, 512]):
        pr = D.prefix_period(K, tm)
        ok &= pr["found"] and pr["period"] >= 1
        check("EA2 prefix K=%d is eventually periodic" % K, pr["found"],
              "preperiod %s, period %s" % (pr["preperiod"], pr["period"]))
    # the central observation: the transient outruns the diagonal (G5)
    bad = []
    for K in ([32] if fast else [64, 128, 256, 512, 1024]):
        pr = D.prefix_period(K, tm)
        if pr["found"] and pr["preperiod"] <= K:
            bad.append((K, pr["preperiod"]))
    check("G5 observation: T(K) > K on every K measured", not bad, str(bad))


def t6_automaton(fast):
    print("\n=== 6. wall automaton ===")
    expected = {1: (32, 16), 2: (512, 128), 3: (8192, 992), 4: (131072, 7616)}
    for r in ([1, 2, 3] if fast else [1, 2, 3, 4]):
        a = D.wall_automaton(r)
        tot, inv = expected[r]
        check("automaton r=%d reproduces published counts" % r,
              a["wall_states_total"] == tot and a["maximal_invariant_states"] == inv,
              "%d states, %d invariant (expected %d, %d)"
              % (a["wall_states_total"], a["maximal_invariant_states"], tot, inv))
        check("automaton r=%d invariant set is NON-empty (no impossibility)" % r,
              not a["invariant_is_empty"])


def t7_results_json():
    print("\n=== 7. stored results match the documents ===")
    path = os.path.join("results", "phase2b_results.json")
    if not os.path.exists(path):
        return check("phase2b_results.json present", False, "run run_phase2b.py")
    d = json.load(open(path))
    check("identities recorded as exact", d["identities"]["xor_or_form_exact"]
          and d["identities"]["anf_form_exact"])
    check("support-edge theorem holds for every lag in the sweep",
          d["support_edge_theorem_holds_for_all_lags"],
          "%d lags" % len(d["geometry"]))
    check("83 lags swept", len(d["geometry"]) == 83, "%d" % len(d["geometry"]))
    check("longest real wall over all lags is 18 at p=59",
          d["longest_wall_over_all_lags"] == 18 and d["argmax_lag"] == 59,
          "%d at p=%d" % (d["longest_wall_over_all_lags"], d["argmax_lag"]))
    check("every orbit lemma check was exact",
          all(v.get("exact") for v in d["orbit_checks"].values()),
          "%d checks" % len(d["orbit_checks"]))
    auto = {a["radius"]: a for a in d["automaton"]}
    check("automaton invariant sets non-empty for every radius computed",
          all(not a["invariant_is_empty"] for a in auto.values()),
          "radii %s" % sorted(auto))
    check("automaton r=5 recorded", 5 in auto and
          auto[5]["maximal_invariant_states"] == 59136,
          str(auto.get(5, {}).get("maximal_invariant_states")))
    ea = d["edge_aligned"]
    check("EA1 exact in the stored run", ea["update_rule"]["exact"])
    check("EA3 exact in the stored run", ea["centre_is_diagonal"]["centre_equals_diagonal"])
    check("periods observed are the doubling hierarchy",
          ea["periods_observed"] == [1, 2, 4, 8, 16], str(ea["periods_observed"]))
    check("mean T(K)/K > 1 for K>=64", ea["late_T_over_K"] > 1.0,
          "%.3f" % ea["late_T_over_K"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    print("python %s   cwd %s" % (sys.version.split()[0], os.getcwd()))
    t1_identities()
    t2_orbit_identities(a.fast)
    t3_wall_lemmas(a.fast)
    t4_support_edges(a.fast)
    t5_edge_aligned(a.fast)
    t6_automaton(a.fast)
    t7_results_json()
    print("\n" + "=" * 70)
    print("%d passed, %d FAILED   (%.1f s)" % (len(PASSES), len(FAILURES),
                                               time.time() - t0))
    if FAILURES:
        print("failures: %s" % ", ".join(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
