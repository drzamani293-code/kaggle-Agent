"""run_phase2h_tests.py -- validation suite for Phase 2H.

    python3 run_phase2h_tests.py            # full
    python3 run_phase2h_tests.py --fast     # reduced ranges

Checks fall into four groups:
  A. the rule itself, and two independent implementations agreeing
  B. every theorem stated in the Phase 2H documents, re-derived here
  C. the recorded numbers in results/*.json matching the documents
  D. document hygiene: banned overclaims, required disclaimers, cross-refs

A passing suite is NOT evidence for any mathematical claim.  The proofs are in
the documents; this file checks that the code and the prose agree with them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import boundary_lab as BL          # noqa: E402
import reversibility_lab as RL     # noqa: E402
import right_tower_lab as RT       # noqa: E402
import triangle_enumerator as TE   # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          ("  -- " + detail) if detail and not cond else ""))


def load(fn):
    p = os.path.join(HERE, "results", fn)
    if not os.path.exists(p):
        p = os.path.join(HERE, fn)
    with open(p) as f:
        return json.load(f)


def plain(s: str) -> str:
    """Strip markdown emphasis/backticks so scanners see the bare words."""
    s = re.sub(r"[`*_]", "", s)
    return re.sub(r"\s+", " ", s)


def paragraphs(text: str):
    return [plain(p) for p in re.split(r"\n\s*\n", text)]


# =============================================================== group A ===
def group_A(fast):
    print("\n--- A. the rule, and two implementations ---")
    check("A1 rule 30 table is (0,1,1,1,1,0,0,0) under 4l+2c+r",
          [RL.f30(l, c, r) for l in (0, 1) for c in (0, 1) for r in (0, 1)]
          == [0, 1, 1, 1, 1, 0, 0, 0])
    check("A2 rule 90 table is (0,1,0,1,1,0,1,0)",
          [RL.f90(l, c, r) for l in (0, 1) for c in (0, 1) for r in (0, 1)]
          == [0, 1, 0, 1, 1, 0, 1, 0])
    t = 200 if fast else 400
    for rule in (30, 90):
        r = BL.crosscheck_implementations(t, rule)
        check("A3 explicit grid == bit-parallel, rule %d" % rule,
              r["disagreements"] == 0, str(r))
    # left-permutive, not right-permutive
    check("A4 rule 30 left-permutive",
          all(RL.f30(0, c, r) != RL.f30(1, c, r)
              for c in (0, 1) for r in (0, 1)))
    check("A5 rule 30 NOT right-permutive (f(l,1,0)==f(l,1,1))",
          all(RL.f30(l, 1, 0) == RL.f30(l, 1, 1) for l in (0, 1)))
    check("A6 rule 90 permutive on BOTH sides",
          all(RL.f90(0, c, r) != RL.f90(1, c, r)
              for c in (0, 1) for r in (0, 1))
          and all(RL.f90(l, c, 0) != RL.f90(l, c, 1)
                  for l in (0, 1) for c in (0, 1)))


# =============================================================== group B ===
def group_B(fast):
    print("\n--- B. the theorems, re-derived ---")

    # --- section 1: backward cone, and the exact count (star) ---
    cases = [(60, 20, 30)] if fast else [(60, 20, 30), (120, 40, 60),
                                         (200, 60, 100)]
    for (t_bot, h, w) in cases:
        r = RL.verify_backward_on_orbit(t_bot, h, t_bot + 1, w)
        b = (t_bot + 1) - w
        predicted = (h + 1) * ((t_bot + 1) - b + 2) + h * (h + 1) // 2
        check("B1 backward cone exact, t_bot=%d" % t_bot,
              r["exact"] and r["mismatches"] == 0, str(r))
        check("B1b white seed really white, t_bot=%d" % t_bot,
              r["white_strip_really_white"])
        check("B1c cell count matches formula (*), t_bot=%d" % t_bot,
              r["cells_reconstructed"] == predicted,
              "%d vs %d" % (r["cells_reconstructed"], predicted))
        check("B1d per-row widths match [b-i, m+1], t_bot=%d" % t_bot,
              all(r["cells_per_row"][t_bot - i] == ((t_bot + 1) - b + 2) + i
                  for i in range(h + 1)))

    # X-2H-01: the preserved counterexample must still fail
    bad = RL.verify_backward_on_orbit(60, 20, 40, 30)
    check("B2 X-2H-01 preserved: m=40 seed is NOT white and reconstruction fails",
          (not bad["white_strip_really_white"]) and bad["mismatches"] > 0,
          str({k: bad[k] for k in ("white_strip_really_white", "mismatches")}))

    # --- section 2 ---
    hs = range(2, 7) if fast else range(2, 11)
    for h in hs:
        r = BL.candidate_B_region(h)
        check("B3 candidate B envelope exact, h=%d" % h,
              not r["determination_violations"]
              and r["cells_claimed_determined"]
              == r["predicted_count_h_h_minus_1_over_2"]
              and r["envelope_is_sharp"], str(r))
        check("B3b W2 fibres have size >= 2 (top column 0 is free), h=%d" % h,
              r["min_tops_per_width2"] == 2)
        c1 = BL.candidate_C_right_edge(h, 30, 1)
        check("B4 candidate C (one diagonal) REFUTED, h=%d" % h,
              not c1["determines_centre_word_minus_apex"])
        c2 = BL.candidate_C_right_edge(h, 30, 2)
        check("B5 candidate C' (two diagonals) sufficient, fibres=2, h=%d" % h,
              c2["determines_centre_word_minus_apex"]
              and c2["max_tops_per_boundary_word"] == 2)

    # candidate A refuted, with the minimal counterexample
    def step3(a, b, c):
        return RL.f30(a, b, c)
    check("B6 X-2H-02 minimal counterexample 010 vs 011 share centre word 11",
          step3(0, 1, 0) == step3(0, 1, 1) == 1)

    # candidate D: the one-step map is not injective
    for rule in (30, 90):
        col = BL.forward_collisions(8, rule, limit=1)
        check("B7 one-step map NOT injective, rule %d" % rule,
              not col["map_is_injective"])

    # diagonal recursion on the real orbit
    for T in ((40,) if fast else (40, 120, 300)):
        r = BL.diagonal_recursion(T)
        check("B8 (DIAG) exact on the orbit, T=%d" % T,
              r["recursion_failures"] == 0 and r["centre_failures"] == 0,
              str(r))
        check("B8b E0 identically zero on the first half, T=%d" % T,
              r["E0_zero_on_first_half"])

    # --- section 3: OR-blindness and the reduced boundary ---
    tm = 3000 if fast else 20000
    r = BL.reduced_boundary_lemma(tm)
    check("B9 (BW) at j=0 holds on every row",
          r["identity_failures"] == 0, str(r))
    check("B9b OR-blindness: x_s(1) irrelevant when x_s(0)=1",
          r["irrelevance_failures"] == 0, str(r))
    check("B9c |Z| < rows tested (the reduction is non-trivial)",
          0 < r["|Z|"] < r["rows_tested"], str(r))

    # --- section 6 / H3: the two towers ---
    r = RT.orbit_agreement(24, 200 if fast else 400)
    check("B10 (RT) v_t(k)=x_t(t-k) reproduces the orbit",
          r["disagreements"] == 0, str(r))
    cen = RT.bijectivity_census(8 if fast else 12)
    check("B11 G_K bijective at every K",
          all(c["right_bijective"] for c in cen))
    check("B12 F_K NOT injective for every K >= 1",
          all(not c["left_bijective"] for c in cen if c["K"] >= 1))
    inv = RT.inverse_check(10 if fast else 14)
    check("B13 explicit inverse of G_K verified exhaustively",
          inv["failures"] == 0, str(inv))
    rp = RT.tower_periods(12 if fast else 20, 1 << 20, "right")
    check("B14 right tower preperiod is 0 at every level",
          all(x["preperiod"] == 0 for x in rp))
    check("B15 right tower periods are powers of two (bounded obs)",
          all(x["period_is_power_of_two"] for x in rp))
    lp = RT.tower_periods(12 if fast else 20, 1 << 20, "left")
    check("B16 left tower has NONZERO preperiods (the asymmetry)",
          any(x["preperiod"] > 0 for x in lp))

    # --- section 6 / H5: the rule 90 calibration ---
    t9 = 1000 if fast else 4000
    r = BL.rule90_centre_column(t9)
    check("B17 H5(a) rule 90 centre column is eventually periodic (zero)",
          r["centre_column_eventually_zero"]
          and r["centre_column_nonzero_times"] == [0], str(r)[:200])
    r1 = BL.rule90_column1_ones(t9)
    check("B18 H5(b) rule 90 column 1 ones exactly at 2^n - 1",
          r1["closed_form_matches"], str(r1)[:200])
    check("B18b H5(b) gaps unbounded in range => not eventually periodic",
          r1["largest_gap"] >= t9 // 4, str(r1["largest_gap"]))
    # T2.2 frozen edge holds for BOTH rules -- the reason T-02 transfers
    for rule in (30, 90):
        g, C = BL.orbit(200 if fast else 400, rule)
        check("B19 frozen edge x_t(-t)=1 for all t, rule %d" % rule,
              all(g[t][-t + C] == 1 for t in range(len(g))))

    # --- section 8: the complexity transfer ---
    tm = 8000 if fast else 60000
    fc = BL.column_factor_counts(tm, [4, 8, 12])
    tri = {e["h"]: e["max_width2_per_centre"]
           for e in load("SMALL_TRIANGLE_RESULTS.json")["rule30"]}
    for e in fc:
        check("B20 Theorem 8.7 p01(h) <= p0(h+1)*N(h), h=%d" % e["h"],
              e["p01(h)"] <= e["p0(h+1)"] * tri[e["h"]],
              "%d <= %d*%d" % (e["p01(h)"], e["p0(h+1)"], tri[e["h"]]))
    best = max(e["p01(h)"] // tri[e["h"]] for e in fc)
    check("B21 X-2H-06 preserved: derived bound is WEAKER than CT-01",
          best < 998140, "best %d vs CT-01 998140" % best)

    # --- section 9: Proposition 9.1, the balanced centre map ---
    for h in (range(1, 6) if fast else range(1, 9)):
        c, w2, rt, n = TE.enumerate_triangles(h, 30)
        import numpy as np
        cnt = np.bincount(c.astype(np.int64), minlength=1 << (h + 1))
        check("B22 Prop 9.1 centre map is exactly 2^h-to-one, h=%d" % h,
              cnt.min() == cnt.max() == (1 << h) and (cnt > 0).all(),
              "min %d max %d" % (cnt.min(), cnt.max()))

    # --- section 4: the restart observations, as recorded ---
    res = load("phase2h_results.json")
    check("B23 every record of W(t) is at a power of two (t <= 1200)",
          res["all_records_at_powers_of_two"] is True)
    check("B24 x_t(t) = 1 for every t in range",
          res["right_edge_always_one"] is True)
    check("B25 white triangle below 2^n full for 3 <= n <= 10",
          all(x["full_triangle"] for x in res["white_triangles"]))
    check("B26 rule 90 control W(2^n) = 2^(n+1) - 1 exactly",
          res["rule90"]["matches_2_pow_np1_minus_1"] is True)
    # section 5 inequalities
    for g in res["restart_geometry"]:
        check("B27 arrival bound 2^(n+1)-W_n, n=%d" % g["n"],
              g["earliest_influence_on_col0"] == 2 * g["t"] - g["W"]
              and g["dist_from_centre"] == g["t"] - g["W"], str(g))
    check("B28 inequality (III): 2^n - W_n > 0 for 3 <= n <= 10",
          all(g["dist_from_centre"] > 0 for g in res["restart_geometry"]))


# =============================================================== group C ===
def group_C(fast):
    print("\n--- C. recorded numbers match the documents ---")
    res = load("phase2h_results.json")
    doc = open(os.path.join(HERE, "CONDITIONAL_REVERSIBILITY_THEORY.md")).read()
    for b in res["backward_reconstruction"]:
        check("C1 cell count %d appears in the theory document"
              % b["cells_reconstructed"],
              str(b["cells_reconstructed"]) in doc)
    W = res["W_at_powers"]
    ev = open(os.path.join(HERE, "RESTART_EVENT_THEORY.md")).read()
    check("C2 W(2^n) values appear in RESTART_EVENT_THEORY.md",
          all(("| %s |" % v) in ev or (" %s " % v) in ev for v in W.values()))
    geo = open(os.path.join(HERE, "RESTART_CENTER_GEOMETRY.md")).read()
    for g in res["restart_geometry"]:
        check("C3 arrival time %d in RESTART_CENTER_GEOMETRY.md"
              % g["earliest_influence_on_col0"],
              str(g["earliest_influence_on_col0"]) in geo)
    tri = load("SMALL_TRIANGLE_RESULTS.json")
    an = open(os.path.join(HERE, "SMALL_TRIANGLE_ANALYSIS.md")).read()
    seq = ", ".join(str(e["max_width2_per_centre"]) for e in tri["rule30"])
    check("C4 the N(h) sequence appears verbatim in the analysis", seq in an,
          seq)
    bd = load("phase2h_boundary.json")
    pc = open(os.path.join(HERE, "PERIODIC_CENTER_TRIANGLES.md")).read()
    check("C5 |Z| = %d appears in PERIODIC_CENTER_TRIANGLES.md"
          % bd["reduced_boundary"]["|Z|"],
          str(bd["reduced_boundary"]["|Z|"]) in pc)
    rtj = load("phase2h_right_tower.json")
    wb = open(os.path.join(HERE, "WEAK_BRIDGE_THEOREMS.md")).read()
    check("C6 right tower preperiod-0 claim matches the data",
          rtj["right_all_preperiod_zero"] is True
          and "preperiod exactly 0" in wb)
    check("C7 the K=3 left-tower collision is recorded in WEAK_BRIDGE",
          "F_3(1,1,0,0) = F_3(1,0,1,0) = (1,1,0,1)" in wb)


# =============================================================== group D ===
BANNED = [
    "problem 1 solved", "problem 1 is solved", "solves problem 1",
    "we solve problem 1", "proved center aperiodic",
    "proved centre aperiodic", "prove the centre column is aperiodic",
    "prove the center column is aperiodic",
    "unique for all k", "proof of the rule 30 prize",
    "we have solved", "this settles problem 1",
]
UNSAT_OK = ["qualified", "uses of", "occurrence", "never as unsat",
            "not as unsat", "described as unsat", "call a bounded",
            "banned", "scanner", "search for"]

DOCS = ["CONDITIONAL_REVERSIBILITY_THEORY.md", "MINIMAL_BOUNDARY_DATA.md",
        "PERIODIC_CENTER_TRIANGLES.md", "RESTART_EVENT_THEORY.md",
        "RESTART_CENTER_GEOMETRY.md", "WEAK_BRIDGE_THEOREMS.md",
        "KOPRA_PROOF_ADAPTATION.md", "TRIANGLE_TO_TRACE_TRANSFER.md",
        "SMALL_TRIANGLE_ANALYSIS.md", "CONDITIONAL_BRIDGE_REGISTRY.md",
        "RETRIEVAL_NOTE.md", "README.md"]


def group_D(fast):
    print("\n--- D. document hygiene ---")
    for fn in DOCS:
        p = os.path.join(HERE, fn)
        check("D0 %s exists" % fn, os.path.exists(p))
        if not os.path.exists(p):
            continue
        text = open(p).read()
        low = plain(text).lower()
        hits = [b for b in BANNED if b in low]
        check("D1 no banned overclaim in %s" % fn, not hits, str(hits))
        bad = []
        for para in paragraphs(text):
            pl = para.lower()
            if "unsat" in pl and not any(q in pl for q in UNSAT_OK):
                bad.append(para[:120])
        check("D2 no unqualified UNSAT in %s" % fn, not bad, str(bad))

    reg = open(os.path.join(HERE, "CONDITIONAL_BRIDGE_REGISTRY.md")).read()
    check("D3 registry states Problem 1 is not solved",
          "is not solved" in plain(reg))
    check("D4 registry keeps BR-01 open", "BR-01` is open" in reg
          or "BR-01 is open" in plain(reg))
    kop = open(os.path.join(HERE, "KOPRA_PROOF_ADAPTATION.md")).read()
    check("D5 section 7 reported NOT COMPLETED", "NOT COMPLETED" in kop)
    check("D6 section 7 does not claim to reconstruct Theorem 3.5",
          "was never read" in kop)
    ret = open(os.path.join(HERE, "RETRIEVAL_NOTE.md")).read()
    check("D7 retrieval note records the 403s", "403" in ret)
    check("D8 no novelty claim anywhere",
          all("we are the first" not in plain(open(os.path.join(HERE, f)).read()).lower()
              and "novel result" not in plain(open(os.path.join(HERE, f)).read()).lower()
              for f in DOCS if os.path.exists(os.path.join(HERE, f))))
    # every failed conjecture is preserved and referenced
    allx = "".join(open(os.path.join(HERE, f)).read()
                   for f in DOCS if os.path.exists(os.path.join(HERE, f)))
    for xid in ("X-2H-01", "X-2H-02", "X-2H-03", "X-2H-04", "X-2H-05",
                "X-2H-06", "C-2H-01"):
        check("D9 %s preserved" % xid, xid in allx)
    # rule 90 control present in every document that draws a conclusion
    for fn in ("RESTART_EVENT_THEORY.md", "SMALL_TRIANGLE_ANALYSIS.md",
               "WEAK_BRIDGE_THEOREMS.md", "MINIMAL_BOUNDARY_DATA.md"):
        t = open(os.path.join(HERE, fn)).read().lower()
        check("D10 rule 90 control present in %s" % fn, "rule 90" in t)
    # bounded observations must be labelled
    ev = open(os.path.join(HERE, "RESTART_EVENT_THEORY.md")).read()
    check("D11 restart observations labelled BOUNDED OBSERVATION",
          "BOUNDED COMPUTATIONAL OBSERVATION" in ev
          and "BOUNDED OBSERVATION" in ev)
    sta = plain(open(os.path.join(HERE,
                                  "SMALL_TRIANGLE_ANALYSIS.md")).read()).lower()
    evp = plain(ev).lower()
    check("D12 N(h): no formula fitted, no growth rate claimed",
          "no formula is fitted and no growth rate is claimed" in sta, sta[:0])
    check("D13 W_n: no formula claimed",
          "not claimed to follow any formula" in evp, evp[:0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    print("Phase 2H validation suite%s" % ("  [FAST]" if a.fast else ""))
    group_A(a.fast)
    group_B(a.fast)
    group_C(a.fast)
    group_D(a.fast)
    n = len(PASS) + len(FAIL)
    print("\n%d/%d checks passed" % (len(PASS), n))
    if FAIL:
        print("\nFAILURES:")
        for name, detail in FAIL:
            print("  - %s  %s" % (name, detail))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
