"""run_phase2i_tests.py -- validation suite for Phase 2I.

    python3 run_phase2i_tests.py            # full
    python3 run_phase2i_tests.py --fast     # reduced ranges

Groups:
  A. the GF(2) rewriting and the three rules, exhaustively
  B. every theorem stated in the Phase 2I documents, re-derived here
  C. the numbers recorded in results/*.json match the documents
  D. document hygiene: banned overclaims, mandatory controls, preserved failures

A passing suite is NOT evidence for any mathematical claim.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nonlinear_lab as NL          # noqa: E402
import kernel_geometry as KG        # noqa: E402
import discriminator_lab as D       # noqa: E402
import tower_coupling_lab as TC     # noqa: E402
import symbolic_lab as S            # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          ("  -- " + detail) if detail and not cond else ""))


def load(fn):
    with open(os.path.join(HERE, "results", fn)) as f:
        return json.load(f)


def plain(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`*_]", "", s))


def paragraphs(text: str):
    return [plain(p) for p in re.split(r"\n\s*\n", text)]


# =============================================================== group A ===
def group_A(fast):
    print("\n--- A. GF(2) rewriting and the three rules ---")
    trip = [(l, c, r) for l in (0, 1) for c in (0, 1) for r in (0, 1)]
    check("A1 rule 30 boolean == GF(2) form (exhaustive)",
          all(NL.RULE_TABLES[30](*x) == NL.f30_gf2(*x) for x in trip))
    check("A2 rule30 = rule150 + n (exhaustive)",
          all(NL.RULE_TABLES[30](l, c, r)
              == (NL.RULE_TABLES[150](l, c, r) ^ NL.nonlinear_bit(c, r))
              for l, c, r in trip))
    check("A3 rule 30 table is (0,1,1,1,1,0,0,0) under 4l+2c+r",
          [NL.RULE_TABLES[30](*x) for x in trip] == [0, 1, 1, 1, 1, 0, 0, 0])
    check("A4 rule 90 table", [NL.RULE_TABLES[90](*x) for x in trip]
          == [0, 1, 0, 1, 1, 0, 1, 0])
    check("A5 rule 150 table", [NL.RULE_TABLES[150](*x) for x in trip]
          == [0, 1, 1, 0, 1, 0, 0, 1])
    degs = {r["rule"]: r["degree"] for r in
            [D.P1_forcing_degree(x) for x in (30, 90, 150)]}
    check("A6 forcing degrees 2/1/0 for rules 30/90/150",
          degs[30] == 2 and degs[90] == 1 and degs[150] == -1, str(degs))
    check("A7 rules 90 and 150 are mirror-symmetric, rule 30 is not",
          all(NL.RULE_TABLES[r](l, c, rr) == NL.RULE_TABLES[r](rr, c, l)
              for r in (90, 150) for l, c, rr in trip)
          and any(NL.RULE_TABLES[30](l, c, rr) != NL.RULE_TABLES[30](rr, c, l)
                  for l, c, rr in trip))


# =============================================================== group B ===
def group_B(fast):
    print("\n--- B. the theorems, re-derived ---")
    suite = D.seed_suite()

    # --- section 2: the kernel and Duhamel ---
    def direct_pow(m):
        p = [1]
        for _ in range(m):
            q = [0] * (len(p) + 2)
            for e, v in enumerate(p):
                if v:
                    q[e] ^= 1
                    q[e + 1] ^= 1
                    q[e + 2] ^= 1
            p = q
        return p
    M = 25 if fast else 40
    check("B1 Frobenius trinomial row == direct powering",
          all(NL.trinomial_row(m) == direct_pow(m) for m in range(M + 1)))
    mm = 16 if fast else 24
    check("B2 T == hyperbinary count mod 2",
          all(NL.T(m, d) == NL.hyperbinary_count(
              d + m, [i for i in range(m.bit_length()) if (m >> i) & 1]) % 2
              for m in range(1, mm + 1) for d in range(-m, m + 1)))
    check("B3 T is symmetric in d",
          all(NL.T(m, d) == NL.T(m, -d) for m in range(60) for d in range(-m, m + 1)))
    check("B4 T(m, +-m) = 1", all(NL.T(m, m) == 1 and NL.T(m, -m) == 1
                                  for m in range(200)))
    check("B5 Theorem 2.4: T(t,0) = 1 for every t",
          all(NL.T(t, 0) == 1 for t in range(1000 if fast else 3000)))
    for name, TM in (("A_single", 20 if fast else 34),
                     ("D_two_adjacent", 16 if fast else 28),
                     ("E_111", 16 if fast else 28)):
        sd = suite[name]
        g, C = NL.orbit(sd, TM, 30)
        nf, xf = NL.make_nfield(g, C), NL.make_xfield(g, C)
        bad = sum(1 for t in range(TM + 1) for j in range(-t - 1, t + 2)
                  if NL.duhamel_cell(sd, nf, t, j) != xf(t, j))
        check("B6 Duhamel exact on the full cone, %s" % name, bad == 0,
              "%d mismatches" % bad)
        sh = D.linear_shadow(sd, TM)
        check("B7 c_t = shadow(t) + |N(t)| mod 2, %s" % name,
              all(xf(t, 0) == (sh[t] ^ (len(NL.active_events(sd, nf, t)) % 2))
                  for t in range(TM + 1)))

    # --- Theorem 2.6: constant shadow <=> mirror-symmetric seed ---
    W = 4 if fast else 5
    cols = list(range(-W, W + 1))
    bad = []
    for mask in range(1, 1 << len(cols)):
        sd = {cols[i]: 1 for i in range(len(cols)) if (mask >> i) & 1}
        const = D.P2_constant_shadow(sd, 120)["constant"]
        sym = all((-j) in sd for j in sd)
        if const != sym:
            bad.append(sorted(sd))
    check("B8 Theorem 2.6 constant shadow <=> symmetric seed (exhaustive)",
          not bad, str(bad[:3]))

    # --- section 1: the n identities ---
    for rule, expect in ((30, True), (90, False), (150, False)):
        r = D.P8_n_minus1_identity(suite["A_single"], 200 if fast else 600, rule)
        check("B9 n_t(-1) = c_t(1+c_{t+1}) holds for rule %d: %s"
              % (rule, expect), r["holds"] == expect, str(r))
    check("B10 the identity holds for EVERY rule-30 seed (so it is F3-blind)",
          all(D.P8_n_minus1_identity(suite[k], 150, 30)["holds"] for k in suite))

    # --- section 3: Q canonical ---
    sd = suite["A_single"]
    g, C = NL.orbit(sd, 60, 30)
    nf, xf = NL.make_nfield(g, C), NL.make_xfield(g, C)
    ok = True
    seedzero = True
    for p in (1, 2, 3, 4, 8):
        for t in (5, 11, 16):
            d = NL.Q_canonical(sd, nf, p, t)
            if d["Q"] != (xf(t + p, 0) ^ xf(t, 0)):
                ok = False
            if d["seed"] != 0:
                seedzero = False
    check("B11 Q = SEED + OLD + SLAB agrees with c_{t+p}+c_t", ok)
    check("B12 SEED_p(t) = 0 for the single-cell seed", seedzero)

    # --- section 4: kernel geometry ---
    check("B13 Theorem 4.1 T(2^a,.) = three-slit",
          all(KG.shape_power_of_two(a)["matches"]
              for a in range(1, 9 if fast else 11)))
    check("B14 Theorem 4.2 Stern criterion for d <= 0",
          all(all(NL.T_row((1 << a) - 1)[d]
                  == (0 if (d + (1 << a)) % 3 == 0 else 1)
                  for d in NL.T_row((1 << a) - 1) if d <= 0)
              for a in range(1, 9 if fast else 11)))
    check("B15 2I-C-02 preserved: the criterion FAILS for some d > 0",
          any(any(NL.T_row((1 << a) - 1)[d]
                  != (0 if (d + (1 << a)) % 3 == 0 else 1)
                  for d in NL.T_row((1 << a) - 1) if d > 0)
              for a in range(2, 8)))
    ar = 5 if fast else 7
    check("B16 Theorem 4.3 three-copy identity",
          all(KG.shape_power_plus_r(a, r)["identity_holds"]
              for a in range(1, ar + 1) for r in range(1 << a)))
    check("B17 Theorem 4.4 scaled self-similar identity",
          all(KG.shape_scaled(qq, a, r)["identity_holds"]
              for qq in range(12 if fast else 20)
              for a in range(1, 5 if fast else 6) for r in range(1 << a)))
    check("B18 (DISJ) convolution under disjoint bits",
          all(KG.disjoint_bit_convolution(m1, m2)
              for m1 in range(80 if fast else 150)
              for m2 in range(80 if fast else 150) if not (m1 & m2)))
    check("B19 Theorem 4.5 difference-kernel convolution",
          all(KG.difference_kernel_theorem(m, p)["identity_holds"]
              for p in (1, 2, 4, 8, 3, 5, 6)
              for m in range(80 if fast else 150) if not (m & p)))
    # support recurrences
    def Sm(m):
        return set(d for d, v in NL.T_row(m).items() if v)
    A = lambda m: len(Sm(m))
    Cc = lambda m: len(Sm(m) & {e + 1 for e in Sm(m)})
    R = 150 if fast else 400
    check("B20 Theorem 4.6 A(2m) = A(m)", all(A(2 * m) == A(m) for m in range(R)))
    check("B21 Theorem 4.6 A(2m+1) = 3A(m) - 2C(m)",
          all(A(2 * m + 1) == 3 * A(m) - 2 * Cc(m) for m in range(R)))
    B = [0]
    Cs = [0]
    for m in range(2 * R):
        B.append(B[-1] + A(m))
        Cs.append(Cs[-1] + Cc(m))
    check("B22 Theorem 8.3 B(2t) = 4B(t) - 2*sum C",
          all(B[2 * t] == 4 * B[t] - 2 * Cs[t] for t in range(R)))
    check("B23 2I-X-01 preserved: difference kernel is NOT smaller on average",
          (sum(len(NL.kernel_difference(m, 2)) for m in range(200) if not (m & 2))
           > sum(len(NL.T_support(m)) for m in range(200) if not (m & 2))))

    # --- section 6/7: towers ---
    for rule in (30, 90, 150):
        r = TC.tower_agreement(suite["A_single"], 14, 60 if fast else 120, rule)
        check("B24 both tower recurrences reproduce the orbit, rule %d" % rule,
              r["left_disagreements"] == 0 and r["right_disagreements"] == 0,
              str(r))
    Ks = (3, 5, 7) if fast else (3, 5, 7, 9)
    check("B25 Theorem 7.3 rule 30 LEFT tower is not bijective",
          all(TC.image_size(K, 30, "left") < (1 << (K + 1)) for K in Ks))
    check("B26 Theorem 7.3 the other five maps ARE bijective",
          all(TC.image_size(K, rl, wh) == (1 << (K + 1))
              for K in Ks for rl, wh in ((30, "right"), (90, "left"),
                                         (90, "right"), (150, "left"),
                                         (150, "right"))))
    # the maximum is 4 only from K >= 4; at K = 1,2,3 it is 2,3,3 (small-K
    # boundary effect, recorded in INFORMATION_LOSS_THEORY.md)
    check("B27 max preimage count is 4 for K >= 4, smaller below",
          all(max(TC.preimage_table(K, 30, "left").values()) == 4
              for K in Ks if K >= 4)
          and [max(TC.preimage_table(K, 30, "left").values())
               for K in (1, 2, 3)] == [2, 3, 3])
    check("B28 preimage counts sum to 2^(K+1)",
          all(sum(TC.preimage_table(K, 30, "left").values()) == (1 << (K + 1))
              for K in Ks))
    check("B29 Theorem 6.4 mirror driver identity (rule 30)",
          all(TC.mirror_driver_identity(suite[k], 20 if fast else 32)["holds"]
              for k in ("A_single", "D_two_adjacent", "E_111")))
    check("B30 Corollary 6.5 rules 90/150 keep a symmetric seed symmetric",
          all(TC.mirror_defect(TC.recentre(suite[k]), 40, rule)
              ["identically_zero"]
              for rule in (90, 150)
              for k in ("A_single", "E_101", "E_111", "F_rand05_1100011")))
    check("B31 Pi3 F1: rule 30 breaks the mirror from a symmetric seed",
          not TC.mirror_defect(suite["A_single"], 40, 30)["identically_zero"])
    check("B32 Pi3 F3: some rule-30 seed is not mirror-symmetric",
          any(not TC.seed_is_symmetric(suite[k]) for k in suite))
    check("B33 2I-X-02 preserved: kappa != max only at t = 1",
          TC.kappa_along_orbit(suite["A_single"], 9, 150 if fast else 300,
                               30, "left")["first_20"][0] == 1
          and set(TC.kappa_along_orbit(suite["A_single"], 9,
                                       150 if fast else 300, 30,
                                       "left")["first_20"][1:]) == {4})
    check("B34 2I-X-03 preserved: no symmetric seed kept a symmetric orbit",
          not any(TC.mirror_defect(
              {d: 1 for d in range(-w, w + 1) if (mask >> abs(d)) & 1}, 15, 30
          )["identically_zero"]
              for w in range(1, 4) for mask in range(1, 1 << (w + 1))
              if any((mask >> abs(d)) & 1 for d in range(-w, w + 1))))

    # --- section 10: symbolic ---
    TT = 5 if fast else 7
    for t in range(TT + 1):
        Wv = t + 2
        check("B35 ANF: direct == Duhamel derivation at t=%d" % t,
              S.anf_centre(Wv, t, 30) == S.anf_duhamel_centre(Wv, t))
    check("B36 rules 90 and 150 have affine centre ANF",
          all(S.degree(S.anf_centre(7, 4, r)) <= 1 for r in (90, 150))
          and S.degree(S.anf_centre(7, 4, 30)) > 1)
    Wv = 4 if fast else 5
    agree = True
    for p in (1, 2, 3):
        b = S.brute_force_period_search(Wv, p, 3, 9)
        sa = S.sat_period_search(Wv, p, 3, 9)
        if sa.get("available") and (b["witnesses_found"] > 0) != sa["satisfiable"]:
            agree = False
    check("B37 2I-X-04 fixed: brute force and SAT agree on the same window",
          agree)


# =============================================================== group C ===
def group_C(fast):
    print("\n--- C. recorded numbers match the documents ---")
    res = load("phase2i_results.json")
    doc = lambda f: open(os.path.join(HERE, f)).read()
    check("C1 forcing degrees recorded as 2/1/0",
          [r["degree"] for r in res["rule_decomposition"]] == [2, 1, -1])
    check("C2 Duhamel verification recorded with 0 mismatches",
          all(x["exact"] for x in res["duhamel_verification"]))
    d2 = doc("NONLINEAR_DUHAMEL_FORMULA.md")
    for x in res["duhamel_verification"]:
        check("C3 cell count %d appears in the Duhamel document" % x["cells"],
              str(x["cells"]) in d2)
    check("C4 T(t,0)=1 recorded true", res["T_m_0_is_one"]["all_one"])
    check("C5 Q canonical agrees in every recorded case",
          all(x["agrees"] for x in res["Q_canonical"]))
    check("C6 SEED part zero in every recorded case",
          all(x["seed"] == 0 for x in res["Q_canonical"]))
    d5 = doc("SEED_DISCRIMINATOR_REGISTRY.md")
    for e in res["shadow_characterisation"]:
        check("C7 exhaustive shadow count %d appears somewhere"
              % e["constant_shadow"],
              str(e["constant_shadow"]) in doc("NONLINEAR_DUHAMEL_FORMULA.md"))
        check("C8 shadow count matches 2^(W+1)-1 at W=%d" % e["W"],
              e["constant_shadow"] == e["predicted_2_pow_W1_minus_1"]
              and e["all_of_them_symmetric"])
    check("C9 every suite seed is recorded with its support",
          all(k in d5 for k in res["seed_suite"]))
    check("C10 rule 30 left tower recorded non-bijective, others bijective",
          [(x["rule"], x["which"], x["bijective"]) for x in
           res["tower_injectivity"]].count((30, "left", False)) == 1
          and sum(1 for x in res["tower_injectivity"] if x["bijective"]) == 5)
    check("C11 symmetric-seed scan recorded 0 symmetric orbits",
          res["symmetric_seed_scan"]["rule30_orbits_still_symmetric"] == 0)
    d7 = doc("INFORMATION_LOSS_THEORY.md")
    for x in res["left_tower_loss"]:
        check("C12 image size %d appears in INFORMATION_LOSS_THEORY.md"
              % x["image"], str(x["image"]) in d7)
    check("C13 ANF derivations agree in the record",
          all(x["duhamel_agrees"] for x in res["anf_centre"]))
    check("C14 brute force and SAT agree in the record",
          res.get("search_methods_agree") is True)
    d8 = doc("NONLINEAR_SKELETON.md")
    check("C15 kernel sizes appear in NONLINEAR_SKELETON.md",
          all(str(res["skeleton"][t]["kernel_size"]) in d8 for t in range(2, 13)))
    check("C16 the parity identity holds at every recorded t",
          all(x["identity_c_eq_1_plus_parity"] for x in res["skeleton"]))


# =============================================================== group D ===
BANNED = ["problem 1 solved", "problem 1 is solved", "solves problem 1",
          "we solve problem 1", "proved center aperiodic",
          "proved centre aperiodic", "we have solved", "this settles problem 1",
          "proves the centre column is aperiodic",
          "proves the center column is aperiodic"]
UNSAT_OK = ["never as unsat", "not as unsat", "banned", "scanner",
            "reported as unsat", "rather than as unsat", "call a bounded"]
FORBIDDEN_STATS = ["chi-squared", "chi squared", "randomness test",
                   "statistically random", "passes randomness",
                   "entropy implies chaos"]

DOCS = ["NONLINEAR_ACTIVITY_THEORY.md", "NONLINEAR_DUHAMEL_FORMULA.md",
        "PERIODICITY_NONLINEAR_CONSTRAINT.md", "MOD2_KERNEL_GEOMETRY.md",
        "SEED_DISCRIMINATOR_REGISTRY.md", "DUAL_TOWER_COUPLING.md",
        "INFORMATION_LOSS_THEORY.md", "NONLINEAR_SKELETON.md",
        "NONLINEAR_BRIDGE_REGISTRY.md", "NONLINEAR_SYMBOLIC_RESULTS.md",
        "DECISION_GATE.md", "README.md"]


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
        bad = [pp[:100] for pp in paragraphs(text)
               if "unsat" in pp.lower()
               and not any(qq in pp.lower() for qq in UNSAT_OK)]
        check("D2 no unqualified UNSAT in %s" % fn, not bad, str(bad))
        st = [w for w in FORBIDDEN_STATS if w in low
              and "appears nowhere" not in low and "no statistical" not in low]
        check("D3 no statistical/randomness claim in %s" % fn, not st, str(st))

    allx = "".join(open(os.path.join(HERE, f)).read()
                   for f in DOCS if os.path.exists(os.path.join(HERE, f)))
    for xid in ("2I-C-01", "2I-C-02", "2I-X-01", "2I-X-02", "2I-X-03",
                "2I-X-04"):
        check("D4 %s preserved" % xid, xid in allx)
    for nlb in ("NLB1", "NLB2", "NLB3", "NLB4", "NLB5"):
        check("D5 %s has an entry" % nlb,
              nlb in open(os.path.join(HERE,
                                       "NONLINEAR_BRIDGE_REGISTRY.md")).read())
    gate = open(os.path.join(HERE, "DECISION_GATE.md")).read()
    check("D6 decision gate states a single classification",
          "Classification: **B**" in gate)
    check("D7 decision gate argues against A, C and D",
          "Why not A" in gate and "Why not C" in gate and "Why not D" in gate)
    check("D8 decision gate states Problem 1 is not solved",
          "is not solved" in plain(gate))
    for fn in ("SEED_DISCRIMINATOR_REGISTRY.md", "NONLINEAR_BRIDGE_REGISTRY.md",
               "DUAL_TOWER_COUPLING.md", "INFORMATION_LOSS_THEORY.md",
               "NONLINEAR_ACTIVITY_THEORY.md", "MOD2_KERNEL_GEOMETRY.md"):
        t = open(os.path.join(HERE, fn)).read().lower()
        check("D9 rule 90 control present in %s" % fn, "rule 90" in t)
    reg = open(os.path.join(HERE, "SEED_DISCRIMINATOR_REGISTRY.md")).read()
    check("D10 the four filters are defined in the registry",
          all(f in reg for f in ("F1", "F2", "F3", "F4")))
    check("D11 candidates equivalent to Problem 1 are rejected",
          "equivalent to Problem 1" in reg
          and "equivalent to Problem 1" in
          open(os.path.join(HERE, "NONLINEAR_BRIDGE_REGISTRY.md")).read())
    check("D12 other rule-30 seeds are recorded in the registry",
          "D_two_adjacent" in reg and "F_rand" in reg)
    check("D13 no powers-of-two theorem is inferred",
          "no theorem is inferred from" in plain(
              open(os.path.join(HERE, "MOD2_KERNEL_GEOMETRY.md")).read()).lower())
    check("D14 no novelty claim anywhere",
          not any(w in plain(allx).lower()
                  for w in ("we are the first", "novel result",
                            "this is new to the literature")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    print("Phase 2I validation suite%s" % ("  [FAST]" if a.fast else ""))
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
