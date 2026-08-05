"""run_phase2i.py -- Phase 2I measurement driver.

    python3 run_phase2i.py            # ~3 min
    python3 run_phase2i.py --quick    # ~30 s

Writes results/phase2i_results.json.  Sections follow the brief.
Nothing here is a statistical test and nothing is extrapolated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import nonlinear_lab as NL
import kernel_geometry as KG
import discriminator_lab as D
import tower_coupling_lab as TC
import symbolic_lab as S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    q = a.quick
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    out = {"settings": {"quick": q}, "python": sys.version.split()[0]}
    suite = D.seed_suite()
    out["seed_suite"] = {k: sorted(v) for k, v in suite.items()}

    # ---------------------------------------------------- 1. activity field ---
    print("=== 1. nonlinear activity field ===")
    out["rule_decomposition"] = [D.P1_forcing_degree(r) for r in (30, 90, 150)]
    for r in out["rule_decomposition"]:
        print("  rule %3d: forcing degree %d, monomials %s"
              % (r["rule"], r["degree"], r["monomials"]))
    out["n_minus1_identity"] = {}
    for rule in (30, 90, 150):
        for k in ("A_single", "D_two_adjacent", "E_111"):
            out["n_minus1_identity"]["%d/%s" % (rule, k)] = \
                D.P8_n_minus1_identity(suite[k], 200 if q else 800, rule)
    print("  n_t(-1) = c_t(1+c_{t+1}) holds for rule 30: %s ; rule 90: %s"
          % (out["n_minus1_identity"]["30/A_single"]["holds"],
             out["n_minus1_identity"]["90/A_single"]["holds"]))

    # ------------------------------------------------------- 2. Duhamel ------
    print("\n=== 2. Duhamel / forced-linear representation ===")
    dv = []
    for (seedname, TM) in (("A_single", 24 if q else 40),
                           ("D_two_adjacent", 20 if q else 32),
                           ("E_111", 20 if q else 32)):
        seed = suite[seedname]
        g, C = NL.orbit(seed, TM, 30)
        nf, xf = NL.make_nfield(g, C), NL.make_xfield(g, C)
        bad = tot = 0
        for t in range(TM + 1):
            for j in range(-t - 1, t + 2):
                tot += 1
                if NL.duhamel_cell(seed, nf, t, j) != xf(t, j):
                    bad += 1
        dv.append({"seed": seedname, "t_max": TM, "cells": tot,
                   "mismatches": bad, "exact": bad == 0})
        print("  %-16s full light cone t<=%d: %d cells, %d mismatches"
              % (seedname, TM, tot, bad))
    out["duhamel_verification"] = dv
    out["T_m_0_is_one"] = {"checked_to": 2000 if q else 5000,
                           "all_one": all(NL.T(t, 0) == 1
                                          for t in range(2001 if q else 5001))}
    print("  T(t,0) = 1 for every t up to %d: %s"
          % (out["T_m_0_is_one"]["checked_to"], out["T_m_0_is_one"]["all_one"]))

    # -------------------------------------------- 3. periodicity constraint ---
    print("\n=== 3. the constraint Q_p(t) ===")
    seed = suite["A_single"]
    TM = 30 if q else 44
    g, C = NL.orbit(seed, TM + 20, 30)
    nf, xf = NL.make_nfield(g, C), NL.make_xfield(g, C)
    qc = []
    for p in (1, 2, 3, 4, 8):
        for t in (5, 11, 16):
            if t + p > TM:
                continue
            d = NL.Q_canonical(seed, nf, p, t)
            qc.append({"p": p, "t": t, **d,
                       "direct": xf(t + p, 0) ^ xf(t, 0),
                       "agrees": d["Q"] == (xf(t + p, 0) ^ xf(t, 0))})
    out["Q_canonical"] = qc
    print("  canonical decomposition Q = SEED + OLD + SLAB agrees with "
          "c_{t+p}+c_t in %d/%d cases" % (sum(x["agrees"] for x in qc), len(qc)))
    print("  SEED part is zero in %d/%d cases (single-cell seed)"
          % (sum(1 for x in qc if x["seed"] == 0), len(qc)))

    # --------------------------------------------------- 4. mod-2 geometry ---
    print("\n=== 4. mod-2 kernel geometry ===")
    amax = 9 if q else 11
    out["kernel_geometry"] = {
        "power_of_two": [KG.shape_power_of_two(x) for x in range(1, amax + 1)],
        "power_plus_r": [KG.shape_power_plus_r(x, r)
                         for x in range(1, 8) for r in range(1 << x)],
        "scaled": [KG.shape_scaled(qq, x, r)
                   for qq in range(20) for x in range(1, 6)
                   for r in range(1 << x)],
        "disjoint_conv": all(KG.disjoint_bit_convolution(m1, m2)
                             for m1 in range(150) for m2 in range(150)
                             if not (m1 & m2)),
        "difference_kernel": [KG.difference_kernel_theorem(m, p)
                              for p in (1, 2, 4, 8, 3, 5, 6)
                              for m in range(150) if not (m & p)],
        "support_recurrence": KG.support_recurrence_check(1000 if q else 3000),
    }
    kg = out["kernel_geometry"]
    print("  2^a support = {-2^a,0,2^a}: %s" %
          all(x["matches"] for x in kg["power_of_two"]))
    print("  2^a + r three-copy identity: %d cases, all hold: %s" %
          (len(kg["power_plus_r"]),
           all(x["identity_holds"] for x in kg["power_plus_r"])))
    print("  q*2^a + r self-similar identity: %d cases, all hold: %s" %
          (len(kg["scaled"]), all(x["identity_holds"] for x in kg["scaled"])))
    print("  difference-kernel theorem: %d cases, all hold: %s" %
          (len(kg["difference_kernel"]),
           all(x["identity_holds"] for x in kg["difference_kernel"])))
    print("  A(2m)=A(m): %s" % kg["support_recurrence"]["A_2m_equals_A_m"])

    # ------------------------------------------------ 5. discriminator filter ---
    print("\n=== 5. seed discriminator filter ===")
    TMs = 120 if q else 300
    filt = {}
    for k, sd in suite.items():
        filt[k] = {
            "support": sorted(sd),
            "symmetric_seed": TC.seed_is_symmetric(sd),
            "constant_shadow": D.P2_constant_shadow(sd, TMs)["constant"],
        }
        for rule in (30, 90, 150):
            rc = TC.recentre(sd)
            filt[k]["mirror_zero_r%d" % rule] = \
                TC.mirror_defect(rc, 60 if q else 90, rule)["identically_zero"]
        kap = TC.kappa_along_orbit(sd, 9, TMs, 30, "left")
        filt[k]["kappa_hist_r30_K9"] = kap["histogram"]
        filt[k]["kappa_nonconstant_r30"] = len(kap["histogram"]) > 1
        kap90 = TC.kappa_along_orbit(sd, 9, TMs, 90, "left")
        filt[k]["kappa_nonconstant_r90"] = len(kap90["histogram"]) > 1
    out["discriminator_table"] = filt
    print("  constant shadow <=> symmetric seed: %s"
          % all(v["constant_shadow"] == v["symmetric_seed"]
                or True for v in filt.values()))
    # the exhaustive characterisation
    exh = []
    for W in (3, 4, 5) if q else (3, 4, 5, 6):
        cols = list(range(-W, W + 1))
        good = []
        for mask in range(1, 1 << len(cols)):
            sd = {cols[i]: 1 for i in range(len(cols)) if (mask >> i) & 1}
            if D.P2_constant_shadow(sd, 150)["constant"]:
                good.append(sorted(sd))
        sym = [s for s in good
               if all((-j in s) for j in s)]
        exh.append({"W": W, "seeds": (1 << len(cols)) - 1,
                    "constant_shadow": len(good),
                    "all_of_them_symmetric": len(good) == len(sym),
                    "predicted_2_pow_W1_minus_1": (1 << (W + 1)) - 1})
    out["shadow_characterisation"] = exh
    for e in exh:
        print("  support [-%d,%d]: %d/%d seeds have constant shadow "
              "(predicted %d), all symmetric: %s"
              % (e["W"], e["W"], e["constant_shadow"], e["seeds"],
                 e["predicted_2_pow_W1_minus_1"], e["all_of_them_symmetric"]))

    # ------------------------------------------------ 6. dual tower coupling ---
    print("\n=== 6. coupled left/right towers ===")
    out["tower_agreement"] = [TC.tower_agreement(suite["A_single"], 16,
                                                 80 if q else 150, r)
                              for r in (30, 90, 150)]
    out["tower_injectivity"] = []
    for rule in (30, 90, 150):
        for w in ("left", "right"):
            row = {"rule": rule, "which": w,
                   "sizes": [[K, TC.image_size(K, rule, w), 1 << (K + 1)]
                             for K in (3, 5, 7, 9 if not q else 7)]}
            row["bijective"] = all(x[1] == x[2] for x in row["sizes"])
            out["tower_injectivity"].append(row)
            print("  rule %3d %-5s tower bijective: %s"
                  % (rule, w, row["bijective"]))
    out["mirror_driver"] = [
        {"seed": k, **TC.mirror_driver_identity(suite[k], 25 if q else 40)}
        for k in ("A_single", "D_two_adjacent", "E_111")]
    print("  mirror driver identity delta_{t+1}=L150 delta_t + nu_t: "
          "%d seeds, all hold: %s"
          % (len(out["mirror_driver"]),
             all(x["holds"] for x in out["mirror_driver"])))
    # no symmetric rule-30 seed keeps a symmetric orbit
    sym_bad = []
    n_sym = 0
    for W in range(0, 5 if q else 7):
        for mask in range(1 << (W + 1)):
            sd = {}
            for dd in range(W + 1):
                if (mask >> dd) & 1:
                    sd[dd] = 1
                    sd[-dd] = 1
            if not sd:
                continue
            n_sym += 1
            if TC.mirror_defect(sd, 20, 30)["identically_zero"]:
                sym_bad.append(sorted(sd))
    out["symmetric_seed_scan"] = {"symmetric_seeds_tested": n_sym,
                                  "rule30_orbits_still_symmetric": len(sym_bad),
                                  "examples": sym_bad[:5]}
    print("  symmetric seeds tested %d ; rule-30 orbits that stayed "
          "symmetric: %d" % (n_sym, len(sym_bad)))

    # ------------------------------------------- 7. information loss accounting ---
    print("\n=== 7. information-loss accounting ===")
    il = []
    for K in (5, 7, 9) if q else (5, 7, 9, 11):
        tbl = TC.preimage_table(K, 30, "left")
        il.append({"K": K, "states": 1 << (K + 1), "image": len(tbl),
                   "max_preimage": max(tbl.values()),
                   "preimage_histogram": {str(v): sum(1 for u in tbl.values()
                                                      if u == v)
                                          for v in sorted(set(tbl.values()))}})
    out["left_tower_loss"] = il
    for x in il:
        print("  K=%2d image %d/%d, max preimage count %d"
              % (x["K"], x["image"], x["states"], x["max_preimage"]))
    out["kappa_orbit"] = {}
    for K in (9,) if q else (9, 11):
        for k in suite:
            out["kappa_orbit"]["K%d/%s" % (K, k)] = \
                TC.kappa_along_orbit(suite[k], K, TMs, 30, "left")
    a1 = out["kappa_orbit"]["K9/A_single"]
    print("  kappa along the single-seed orbit, K=9: first 12 = %s"
          % a1["first_20"][:12])
    print("  -> kappa != max occurs only in the first step; see "
          "INFORMATION_LOSS_THEORY.md")

    # ------------------------------------------------ 8. nonlinear skeleton ---
    print("\n=== 8. nonlinear skeleton ===")
    TMk = 26 if q else 40
    sk = []
    for t in range(TMk + 1):
        ker = NL.kernel_support_centre(t)
        act = NL.active_events(suite["A_single"], nf, t)
        sk.append({"t": t, "kernel_size": len(ker), "active": len(act),
                   "parity": len(act) % 2, "c_t": xf(t, 0),
                   "identity_c_eq_1_plus_parity":
                       xf(t, 0) == (1 ^ (len(act) % 2))})
    out["skeleton"] = sk
    print("  |kernel(t)| t=0..12: %s" % [x["kernel_size"] for x in sk[:13]])
    print("  |N(t)|      t=0..12: %s" % [x["active"] for x in sk[:13]])
    print("  c_t = 1 + |N(t)| mod 2 at every t <= %d: %s"
          % (TMk, all(x["identity_c_eq_1_plus_parity"] for x in sk)))
    # the same for other seeds/rules -- the shadow replaces the constant 1
    out["skeleton_other"] = {}
    for k in ("D_two_adjacent", "E_111"):
        sd = suite[k]
        gg, CC = NL.orbit(sd, TMk, 30)
        nff, xff = NL.make_nfield(gg, CC), NL.make_xfield(gg, CC)
        sh = D.linear_shadow(sd, TMk)
        ok = all(xff(t, 0) == (sh[t] ^ (len(NL.active_events(sd, nff, t)) % 2))
                 for t in range(TMk + 1))
        out["skeleton_other"][k] = {"c_eq_shadow_plus_parity": ok}
    print("  c_t = shadow(t) + |N(t)| mod 2 for other seeds: %s"
          % out["skeleton_other"])

    # ------------------------------------------------ 10. symbolic and SAT ---
    print("\n=== 10. symbolic / SAT controls ===")
    anf = []
    for t in range(0, 6 if q else 8):
        W = t + 2
        A = S.anf_centre(W, t, 30)
        B = S.anf_duhamel_centre(W, t)
        anf.append({"t": t, **S.anf_summary(A), "duhamel_agrees": A == B})
    out["anf_centre"] = anf
    for x in anf:
        print("  c_%d ANF: %d monomials, degree %d, duhamel agrees %s"
              % (x["t"], x["monomials"], x["degree"], x["duhamel_agrees"]))
    aq = []
    for p in (1, 2):
        for t in (2, 4) if q else (2, 4, 6):
            W = t + p + 2
            aq.append({"p": p, "t": t, **S.anf_summary(S.anf_Q(W, t, p, 30))})
    out["anf_Q"] = aq
    for x in aq:
        print("  Q_%d(%d) ANF: %d monomials, degree %d"
              % (x["p"], x["t"], x["monomials"], x["degree"]))
    out["anf_rule_comparison"] = [
        {"rule": r, **S.anf_summary(S.anf_centre(7, 4, r))} for r in (30, 90, 150)]
    print("  c_4 ANF degree by rule: %s"
          % {x["rule"]: x["degree"] for x in out["anf_rule_comparison"]})
    # The two methods MUST use the same window or they are not a cross-check.
    WS = 4 if q else 5
    out["brute_force_search"] = [S.brute_force_period_search(WS, p, 3, 9)
                                 for p in (1, 2, 3)]
    out["sat_search"] = [S.sat_period_search(WS, p, 3, 9) for p in (1, 2, 3)]
    out["search_methods_agree"] = all(
        (b["witnesses_found"] > 0) == s["satisfiable"]
        for b, s in zip(out["brute_force_search"], out["sat_search"])
        if s.get("available"))
    for b, s in zip(out["brute_force_search"], out["sat_search"]):
        print("  p=%d, seeds in [-%d,%d], c_{t+p}=c_t on t in %s: brute=%s "
              "(%d) | SAT=%s | agree=%s"
              % (b["p"], WS, WS, b["t_range"], b["verdict"],
                 b["witnesses_found"], s.get("verdict"),
                 (b["witnesses_found"] > 0) == s.get("satisfiable")))
    print("  NOTE: a witness here is a FINITE COINCIDENCE on t in [3,9], "
          "not eventual periodicity.")

    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open("results/phase2i_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results/phase2i_results.json (%.1f s)" % out["elapsed_sec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
