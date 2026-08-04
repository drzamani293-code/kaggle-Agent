"""build_witness_registry.py -- assemble WITNESS_REGISTRY.json.

Two kinds of entry, kept strictly apart:

  model = "real_orbit"        a genuine counterexample to MB1-loc(a) in the
                              single-cell rule-30 orbit.  Label:
                              COMPUTATIONAL CERTIFICATE.

  model = "A_relaxation"      a satisfying patch of MODEL A, the seed-free
                              relaxation.  **NOT** a counterexample to
                              MB1-loc.  Label: SAT WITNESS (RELAXATION).

Every real-orbit entry is re-verified here by simulating the single-cell orbit
from time 0, independently of the SAT machinery.

    python3 build_witness_registry.py
"""

from __future__ import annotations

import json
import os
import time

import mb1loc_sat as M

RESULTS = "results"


def main():
    res_path = os.path.join(RESULTS, "phase2a_results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    out = {
        "schema": {
            "a": "look-back depth of MB1-loc",
            "t": "time index of the queried cell (real orbit only)",
            "p": "lag",
            "model": "real_orbit | A_relaxation",
            "label": "COMPUTATIONAL CERTIFICATE | SAT WITNESS (RELAXATION)",
            "note": "A_relaxation entries are NOT counterexamples to MB1-loc; "
                    "see SAT_MODEL_SPECIFICATION.md Theorem A-incomplete.",
        },
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "witnesses": [],
    }

    # --- real-orbit witnesses -------------------------------------------
    src = "mb1loc_witnesses.json"
    if os.path.exists(src):
        phase1 = json.load(open(src))
        bt = {(w["a"], w["t"], w["p"]): w
              for w in (res.get("model_bt_witnesses") or [])}
        for row in phase1["rows"]:
            w = row["witness"]
            if w is None:
                out.setdefault("no_witness_found", []).append(
                    {"a": row["a"], "eligible_positions": row["eligible_positions"],
                     "violations": row["violations"],
                     "status": "BOUNDED UNSAT (search-bound limited; see "
                               "PHASE2A_RESULTS.md section 8)"})
                continue
            a, t, p = w["a"], w["t"], w["p"]
            sim = M.real_orbit_witness_check(a, t, p)
            cert = bt.get((a, t, p), {})
            out["witnesses"].append({
                "id": "real_a%02d_t%d_p%d" % (a, t, p),
                "model": "real_orbit",
                "label": "COMPUTATIONAL CERTIFICATE",
                "a": a, "t": t, "p": p,
                "hypotheses": {
                    "H1_col0_agrees_on_window": sim["H1_agreement_on_window"],
                    "H2_col0_zero_at_t": sim["H2_centre_zero"],
                },
                "conclusion_violated": sim["C_violated"],
                "values": {"a_t(0)": 0 if sim["H2_centre_zero"] else 1,
                           "a_t(1)": sim["col1_t"],
                           "a_{t+p}(1)": sim["col1_t_plus_p"]},
                "is_counterexample": sim["is_counterexample"],
                "verification": {
                    "orbit_resimulation_from_seed": sim["is_counterexample"],
                    "model_b_prime_solvers": cert.get("verdicts"),
                    "model_b_prime_constraints_ok": cert.get("constraints_ok"),
                    "two_window_cells": cert.get("two_window_cells"),
                    "single_cone_cells": cert.get("single_cone_cells"),
                    "single_cone_run": cert.get("single_cone_run"),
                    "certificate_file": cert.get("certificate"),
                },
                "source": "Phase 1 find_mb1loc_witnesses.py; re-certified in Phase 2A",
            })
        # the a = 0 witness additionally has a full MODEL B certificate
        mbw = res.get("model_b_witness")
        if mbw:
            for w in out["witnesses"]:
                if (w["a"], w["t"], w["p"]) == (mbw["a"], mbw["t"], mbw["p"]):
                    w["verification"]["model_b_full_light_cone"] = {
                        "solvers": mbw["solvers"],
                        "cells": mbw["cells"],
                        "base_row_is_seed": mbw["base_row_is_seed"],
                        "certificate_file": mbw["certificate"],
                    }

    # --- extra real-orbit counterexamples found by the MODEL B sweep -----
    mb = res.get("model_b_sweep")
    if mb:
        known = {(w["a"], w["t"], w["p"]) for w in out["witnesses"]}
        extra = []
        for a, t, p in mb["sat_hits"]:
            if (a, t, p) in known:
                continue
            sim = M.real_orbit_witness_check(a, t, p)
            extra.append({"a": a, "t": t, "p": p,
                          "is_counterexample": sim["is_counterexample"],
                          "values": {"a_t(1)": sim["col1_t"],
                                     "a_{t+p}(1)": sim["col1_t_plus_p"]}})
        out["model_b_sweep_counterexamples"] = {
            "label": "COMPUTATIONAL CERTIFICATE (exhaustive over the stated range)",
            "bounds": {"t_max": mb["t_max"], "p_max": mb["p_max"],
                       "a_vals": mb["a_vals"]},
            "n_instances_solved": mb["n_instances"],
            "count": len(extra),
            "all_reverified": all(e["is_counterexample"] for e in extra),
            "entries": extra,
        }

    # --- MODEL A relaxation patches -------------------------------------
    for c in res.get("model_a_certificates", []):
        if c.get("status") != "SAT":
            continue
        out["witnesses"].append({
            "id": "modelA_a%02d_p%02d" % (c["a"], c["p"]),
            "model": "A_relaxation",
            "label": "SAT WITNESS (RELAXATION) -- NOT a counterexample to MB1-loc",
            "a": c["a"], "p": c["p"], "t": None,
            "base_row": c.get("base_row"),
            "verification": {"patch_resimulated_and_rechecked": c.get("recheck_ok"),
                             "certificate_file": c.get("certificate")},
            "note": "MODEL A drops the single-cell seed; a satisfying patch "
                    "need not occur in the real orbit (Theorem A-incomplete).",
        })

    n_real = sum(1 for w in out["witnesses"] if w["model"] == "real_orbit")
    n_relax = sum(1 for w in out["witnesses"] if w["model"] == "A_relaxation")
    out["summary"] = {
        "real_orbit_counterexamples": n_real,
        "model_a_relaxation_patches": n_relax,
        "all_real_orbit_reverified": all(
            w["is_counterexample"] for w in out["witnesses"]
            if w["model"] == "real_orbit"),
    }
    with open("WITNESS_REGISTRY.json", "w") as f:
        json.dump(out, f, indent=2)
    print("real-orbit counterexamples : %d (all re-verified: %s)"
          % (n_real, out["summary"]["all_real_orbit_reverified"]))
    print("MODEL A relaxation patches : %d" % n_relax)
    if "model_b_sweep_counterexamples" in out:
        s = out["model_b_sweep_counterexamples"]
        print("MODEL B sweep extra        : %d (all re-verified: %s)"
              % (s["count"], s["all_reverified"]))
    print("wrote WITNESS_REGISTRY.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
