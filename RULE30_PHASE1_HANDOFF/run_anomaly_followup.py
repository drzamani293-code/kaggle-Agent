"""run_anomaly_followup.py -- independent-sample re-tests of the Phase 1 anomalies.

Phase 1 (N = 200001) produced two things that could be mistaken for signal:

  1. a missing length-11 factor at N = 20001 ("10000010011");
  2. a +3.31-sigma excess of 1s among t == 1 (mod 3).

Both were found *by searching*, so their nominal significance is inflated by
multiple testing.  The honest re-test is on a fresh, disjoint sample.  This
script extends the center column to N bits (default 10^6) and re-tests the
mod-3 statistic on t in [200001, N) only -- data that played no part in the
discovery -- and recomputes the rigorous lower bound on T + p at the larger N.

    python3 run_anomaly_followup.py --steps 1000000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import rule30_lab as L


def factor_count(bits, n):
    """Number of distinct length-n factors, using rolling integer windows
    (memory-lean compared with slicing strings)."""
    mask = (1 << n) - 1
    v = 0
    seen = set()
    for i, b in enumerate(bits):
        v = ((v << 1) | b) & mask
        if i >= n - 1:
            seen.add(v)
    return len(seen)


def residue_test(bits, m, lo, hi):
    out = []
    for r in range(m):
        sub = [bits[t] for t in range(lo, hi) if t % m == r]
        n = len(sub)
        ones = sum(sub)
        z = (ones - n / 2) / (math.sqrt(n) / 2)
        # two-sided normal p-value
        p = math.erfc(abs(z) / math.sqrt(2))
        out.append({"m": m, "r": r, "n": n, "ones": ones,
                    "freq_one": ones / n, "z": z, "p_two_sided": p})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--split", type=int, default=200001,
                    help="bits before this index were used for discovery")
    ap.add_argument("--outdir", default="phase1_results")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    print("generating %d center bits ..." % (args.steps + 1), flush=True)
    bits = L.center_column_bitwise(args.steps)
    gen = time.time() - t0
    N = len(bits)

    res = {"N": N, "generation_sec": round(gen, 1), "split": args.split}

    # consistency with the shorter run
    prev = os.path.join(args.outdir, "center_column_200001.txt")
    if os.path.exists(prev):
        with open(prev) as f:
            old = f.read().strip()
        res["prefix_consistent_with_200001_run"] = (
            "".join(map(str, bits[: len(old)])) == old
        )

    ones = sum(bits)
    res["global_frequency"] = {
        "N": N, "ones": ones, "zeros": N - ones, "freq_one": ones / N,
        "excess_ones": 2 * ones - N,
        "z_fair_coin": (ones - N / 2) / (math.sqrt(N) / 2),
    }

    # --- anomaly 1: forbidden factors -----------------------------------
    print("factor counts ...", flush=True)
    res["missing_factor_retest"] = {
        "word": "10000010011",
        "occurrences_in_full_sequence": "".join(map(str, bits)).count("10000010011"),
        "note": "absent from the first 20001 bits only; a sampling artefact",
    }
    fc = {}
    for n in (20, 24, 26, 28):
        fc[n] = factor_count(bits, n)
        print("  p_obs(%d) = %d" % (n, fc[n]), flush=True)
    res["factor_counts"] = fc
    res["rigorous_T_plus_p_lower_bound"] = {
        "value": max(fc.values()),
        "witness_n": max(fc, key=lambda n: fc[n]),
        "argument": "an eventually periodic word with preperiod T and period p "
                    "has at most T + p distinct factors of any fixed length",
        "ceiling_from_this_N": N - max(fc, key=lambda n: fc[n]) + 1,
    }

    # --- anomaly 2: residue-class bias ----------------------------------
    print("residue tests ...", flush=True)
    res["mod3_discovery_sample"] = residue_test(bits, 3, 0, args.split)
    res["mod3_fresh_sample"] = residue_test(bits, 3, args.split, N)
    res["mod3_full_sample"] = residue_test(bits, 3, 0, N)
    for m in (2, 4, 5, 7, 8, 16):
        res["mod%d_fresh_sample" % m] = residue_test(bits, m, args.split, N)

    # --- period scan at the larger N ------------------------------------
    print("period scan ...", flush=True)
    t1 = time.time()
    scan = L.period_scan_fast(bits, 20000)
    res["period_scan"] = {
        "max_period": 20000,
        "min_last_mismatch_over_p": scan["min_last_mismatch_over_p"],
        "argmin_p": scan["argmin_p"],
        "periods_with_no_mismatch_in_window":
            scan["periods_with_no_mismatch_in_window"],
        "sec": round(time.time() - t1, 1),
    }

    with open(os.path.join(args.outdir, "anomaly_followup.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(json.dumps({k: v for k, v in res.items()
                      if not k.startswith("mod")}, indent=2))
    print("\nmod 3, discovery sample t < %d:" % args.split)
    for d in res["mod3_discovery_sample"]:
        print("  r=%d n=%d freq=%.6f z=%+.3f p=%.4f"
              % (d["r"], d["n"], d["freq_one"], d["z"], d["p_two_sided"]))
    print("mod 3, FRESH sample t >= %d:" % args.split)
    for d in res["mod3_fresh_sample"]:
        print("  r=%d n=%d freq=%.6f z=%+.3f p=%.4f"
              % (d["r"], d["n"], d["freq_one"], d["z"], d["p_two_sided"]))
    print("wrote", os.path.join(args.outdir, "anomaly_followup.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
