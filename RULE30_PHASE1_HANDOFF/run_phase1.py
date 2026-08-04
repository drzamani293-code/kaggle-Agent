"""run_phase1.py -- Phase 1 experiment driver for the Rule 30 audit.

Produces `phase1_results/phase1_results.json` and a human-readable
`phase1_results/phase1_report.txt`.  Everything reported in PHASE1_AUDIT.md
comes from this script; re-running it must reproduce the audit numbers
bit-for-bit (the computation is deterministic and uses no RNG).

    python3 run_phase1.py                 # default settings (N = 200001)
    python3 run_phase1.py --steps 50000   # cheaper run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict

import rule30_lab as L


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def residue_bias(bits, moduli=(2, 3, 4, 5, 6, 7, 8, 16)):
    """Frequency of 1 among t in each residue class mod m.  An anomaly probe;
    see the multiple-testing warning in PHASE1_AUDIT.md."""
    out = []
    N = len(bits)
    for m in moduli:
        for r in range(m):
            sub = bits[r::m]
            n = len(sub)
            ones = sum(sub)
            z = (ones - n / 2) / (math.sqrt(n) / 2)
            out.append({"m": m, "r": r, "n": n, "ones": ones,
                        "freq_one": ones / n, "z_fair_coin": z})
    return out


def autocorrelation(bits, max_lag=64):
    """Correlation of the +-1 valued sequence at small lags."""
    N = len(bits)
    x = [2 * b - 1 for b in bits]
    out = []
    for lag in range(1, max_lag + 1):
        s = sum(x[i] * x[i + lag] for i in range(N - lag))
        n = N - lag
        out.append({"lag": lag, "corr": s / n, "z_fair_coin": s / math.sqrt(n)})
    return out


def power_family_summary(rows):
    fams = {"2^k-1": [], "2^k": [], "2^k+1": []}
    for r in rows:
        fams["2^k-1"].append(r["c(2^k-1)"])
        fams["2^k"].append(r["c(2^k)"])
        fams["2^k+1"].append(r["c(2^k+1)"])
    out = {}
    for k, v in fams.items():
        n = len(v)
        ones = sum(v)
        # two-sided exact binomial p-value under a fair-coin null
        p = sum(
            math.comb(n, j) for j in range(n + 1)
            if abs(j - n / 2) >= abs(ones - n / 2)
        ) / 2 ** n
        out[k] = {"values": v, "n": n, "ones": ones,
                  "two_sided_binomial_p_fair_coin": p}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000,
                    help="number of CA steps; center column has steps+1 bits")
    ap.add_argument("--max-period", type=int, default=10000)
    ap.add_argument("--subword-max", type=int, default=24)
    ap.add_argument("--outdir", default="phase1_results")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()
    res = {
        "settings": {
            "steps": args.steps,
            "center_column_length": args.steps + 1,
            "max_period": args.max_period,
            "subword_max": args.subword_max,
            "convention": "c(t) = a_t(0), t = 0..steps, c(0) = 1; "
                          "rule a'(i) = a(i-1) XOR (a(i) OR a(i+1))",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": getattr(L.np, "__version__", "absent"),
            "git_commit": git_commit(),
            "script": os.path.basename(__file__),
        },
    }

    # ---- A. verification -------------------------------------------------
    print("[1/9] verification suite ...", flush=True)
    checks = L.run_all_verifications(fast=False)
    res["verification"] = [asdict(c) for c in checks]
    res["verification_summary"] = {
        "passed": sum(c.passed for c in checks),
        "total": len(checks),
        "all_passed": all(c.passed for c in checks),
    }

    # external reference prefix (OEIS A051023, obtained via web search summary,
    # NOT a direct download -- treated as weak external evidence only)
    ext = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1]
    print("[2/9] generating center column ...", flush=True)
    t0 = time.time()
    bits = L.center_column_bitwise(args.steps)
    res["timings_sec"] = {"center_column": round(time.time() - t0, 2)}
    res["external_reference_check"] = {
        "source": "OEIS A051023 first terms, retrieved via web search snippet "
                  "(oeis.org was blocked by the network proxy; snippet is a "
                  "secondary, non-authoritative source)",
        "reference_prefix": ext,
        "our_prefix": bits[: len(ext)],
        "match": bits[: len(ext)] == ext,
    }

    # ---- B. required experiments ----------------------------------------
    res["first_256_bits"] = "".join(map(str, bits[:256]))
    res["first_256_bits_list"] = bits[:256]

    print("[3/9] frequencies ...", flush=True)
    prefixes = [16, 64, 256, 1024, 4096, 16384, 65536, len(bits)]
    res["frequencies"] = L.bit_frequencies(bits, prefixes)

    print("[4/9] subword complexity ...", flush=True)
    res["subword_complexity"] = L.subword_complexity(bits, args.subword_max)

    print("[5/9] Berlekamp-Massey linear complexity ...", flush=True)
    t0 = time.time()
    checkpoints = [16, 64, 256, 1024, 4096, 16384, 65536, len(bits)]
    res["linear_complexity"] = L.linear_complexity_profile(bits, checkpoints)
    res["timings_sec"]["berlekamp_massey"] = round(time.time() - t0, 2)

    print("[6/9] powers of two ...", flush=True)
    p2 = L.powers_of_two_behaviour(bits)
    res["powers_of_two"] = p2
    res["powers_of_two_summary"] = power_family_summary(p2)

    print("[7/9] candidate-period scan ...", flush=True)
    t0 = time.time()
    scan = L.period_scan_fast(bits, args.max_period)
    # cross-check the fast scan against the naive scan on a small range
    naive = L.period_scan(bits[:20000], 64)
    agree = all(
        naive["first_mismatch"][p] == L.period_scan_fast(bits[:20000], 64)["first_mismatch"][p]
        for p in range(1, 65)
    )
    res["period_scan_crosscheck_naive_vs_fast"] = agree
    res["period_scan"] = {
        "N": scan["N"],
        "max_period": scan["max_period"],
        "min_last_mismatch_over_p": scan["min_last_mismatch_over_p"],
        "argmin_p": scan["argmin_p"],
        "periods_with_no_mismatch_in_window": scan["periods_with_no_mismatch_in_window"],
        "table_small_p": [
            {"p": p,
             "first_mismatch": scan["first_mismatch"][p],
             "last_mismatch": scan["last_mismatch"][p],
             "window_end": scan["N"] - p - 1,
             "slack": (scan["N"] - p - 1) - (scan["last_mismatch"][p] or -1)}
            for p in range(1, 33)
        ],
        "max_first_mismatch": max(
            (v for v in scan["first_mismatch"].values() if v is not None), default=None
        ),
        "argmax_first_mismatch": max(
            (p for p, v in scan["first_mismatch"].items() if v is not None),
            key=lambda p: scan["first_mismatch"][p], default=None
        ),
    }
    res["timings_sec"]["period_scan"] = round(time.time() - t0, 2)

    # ---- C. extra diagnostics -------------------------------------------
    print("[8/9] entropy, runs, correlations, residue bias ...", flush=True)
    res["run_lengths"] = L.run_length_stats(bits)
    res["block_entropy"] = L.block_entropy(bits[:100000], 16)
    res["autocorrelation"] = autocorrelation(bits, 32)
    res["residue_bias"] = residue_bias(bits)

    L_full = res["linear_complexity"][-1]["L"]
    res["rigorous_lower_bounds"] = L.periodicity_lower_bounds(
        bits, res["subword_complexity"], L_full
    )

    print("[9/9] left-region geometry ...", flush=True)
    t0 = time.time()
    res["left_region"] = L.left_region_shift_symmetry(steps=4000)
    res["timings_sec"]["left_region"] = round(time.time() - t0, 2)
    res["timings_sec"]["total"] = round(time.time() - t_start, 2)

    with open(os.path.join(args.outdir, "phase1_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(args.outdir, "center_column_%d.txt" % (args.steps + 1)), "w") as f:
        f.write("".join(map(str, bits)) + "\n")

    write_report(res, os.path.join(args.outdir, "phase1_report.txt"))
    print("wrote", args.outdir)
    return 0


def write_report(res, path):
    o = []
    w = o.append
    w("RULE 30 PHASE 1 -- RAW RESULTS")
    w("=" * 78)
    w("settings: " + json.dumps(res["settings"]))
    w("environment: " + json.dumps(res["environment"]))
    w("timings: " + json.dumps(res["timings_sec"]))
    w("")
    w("VERIFICATION: %d/%d passed" % (res["verification_summary"]["passed"],
                                      res["verification_summary"]["total"]))
    for c in res["verification"]:
        w("  [%s] %s" % ("PASS" if c["passed"] else "FAIL", c["name"]))
    w("")
    w("EXTERNAL REFERENCE CHECK: match=%s" % res["external_reference_check"]["match"])
    w("")
    w("FIRST 256 CENTER BITS (t = 0..255), rows of 64:")
    s = res["first_256_bits"]
    for i in range(0, 256, 64):
        w("  t=%3d  %s" % (i, s[i:i + 64]))
    w("")
    w("FREQUENCIES")
    w("  %10s %10s %10s %10s %10s %10s" % ("N", "ones", "zeros", "freq1", "excess", "z"))
    for d in res["frequencies"]:
        w("  %10d %10d %10d %10.6f %10d %10.3f" %
          (d["n"], d["ones"], d["zeros"], d["freq_one"], d["excess_ones"], d["z_fair_coin"]))
    w("")
    w("SUBWORD (FACTOR) COMPLEXITY of the observed prefix")
    w("  %4s %12s %14s %14s" % ("n", "p_obs(n)", "2^n", "N-n+1"))
    for d in res["subword_complexity"]:
        w("  %4d %12d %14s %14d" % (d["n"], d["p_obs"], d["max_possible_2^n"],
                                    d["saturation_bound_N-n+1"]))
    w("")
    w("BERLEKAMP-MASSEY LINEAR COMPLEXITY over GF(2)")
    w("  %10s %10s %10s" % ("N", "L(N)", "L/N"))
    for d in res["linear_complexity"]:
        w("  %10d %10d %10.5f" % (d["N"], d["L"], d["L_over_N"]))
    w("")
    w("BEHAVIOUR AT t = 2^k-1, 2^k, 2^k+1")
    w("  %3s %10s %6s %10s %6s %10s %6s" % ("k", "2^k-1", "c", "2^k", "c", "2^k+1", "c"))
    for d in res["powers_of_two"]:
        w("  %3d %10d %6d %10d %6d %10d %6d" %
          (d["k"], d["2^k-1"], d["c(2^k-1)"], d["2^k"], d["c(2^k)"],
           d["2^k+1"], d["c(2^k+1)"]))
    for k, v in res["powers_of_two_summary"].items():
        w("  family %-7s ones=%d/%d  two-sided binomial p (fair coin) = %.4f"
          % (k, v["ones"], v["n"], v["two_sided_binomial_p_fair_coin"]))
    w("")
    w("CANDIDATE PERIODS (first and last mismatch within the window)")
    ps = res["period_scan"]
    w("  %4s %14s %14s %14s %8s" % ("p", "first_mismatch", "last_mismatch",
                                    "window_end", "slack"))
    for d in ps["table_small_p"]:
        w("  %4d %14s %14s %14d %8d" % (d["p"], d["first_mismatch"],
                                        d["last_mismatch"], d["window_end"], d["slack"]))
    w("  over all p <= %d: min last_mismatch = %s (at p = %s)"
      % (ps["max_period"], ps["min_last_mismatch_over_p"], ps["argmin_p"]))
    w("  largest first_mismatch = %s (at p = %s)"
      % (ps["max_first_mismatch"], ps["argmax_first_mismatch"]))
    w("  periods with NO mismatch in the window: %s"
      % ps["periods_with_no_mismatch_in_window"])
    w("")
    w("RIGOROUS LOWER BOUNDS ON T + p  (T = preperiod, p = period)")
    w("  " + json.dumps(res["rigorous_lower_bounds"], indent=2).replace("\n", "\n  "))
    w("")
    w("RUN LENGTHS: " + json.dumps(res["run_lengths"]))
    w("")
    w("BLOCK ENTROPY (first 100000 bits)")
    for d in res["block_entropy"]:
        w("  n=%2d  H_n = %9.5f bits   H_n/n = %.5f   distinct = %d"
          % (d["n"], d["H_n_bits"], d["H_n_over_n"], d["distinct"]))
    w("")
    w("AUTOCORRELATION (+-1 sequence)")
    for d in res["autocorrelation"][:16]:
        w("  lag %3d  corr %+0.6f  z %+8.3f" % (d["lag"], d["corr"], d["z_fair_coin"]))
    w("")
    w("RESIDUE-CLASS BIAS (|z| > 3 only)")
    flagged = [d for d in res["residue_bias"] if abs(d["z_fair_coin"]) > 3]
    for d in flagged:
        w("  t = %d mod %d: n=%d ones=%d freq=%.6f z=%+.3f"
          % (d["r"], d["m"], d["n"], d["ones"], d["freq_one"], d["z_fair_coin"]))
    if not flagged:
        w("  none")
    w("")
    w("LEFT-REGION SHIFT SYMMETRY (best (tau, sigma))")
    w("  " + json.dumps(res["left_region"]))
    with open(path, "w") as f:
        f.write("\n".join(o) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
