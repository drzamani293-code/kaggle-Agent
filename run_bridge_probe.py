"""run_bridge_probe.py -- experiments targeting the width-2 -> width-1 gap.

The width-2 theorem (WIDTH2_PROOF_RECONSTRUCTION.md) shows that at most one
column of the rule-30 space-time diagram can be eventually periodic.  To close
the gap to Prize Problem 1 one needs a bridging lemma of the form

    MB1:  if column 0 is eventually periodic, then so is column -1

which, by the exact identity  a_t(-1) = a_{t+1}(0) XOR (a_t(0) OR a_t(1)),
is equivalent to

    MB1*: ... then column 1 agrees at lag p at every large t with a_t(0) = 0.

MB1 is conditional on a hypothesis that is conjecturally false, so it cannot
be tested directly -- it is vacuously true if Problem 1 has the expected
answer.  What *can* be tested is its finitary shadow: does a long stretch of
lag-p agreement in column 0 force agreement in column 1?  This script measures
exactly that.

    python3 run_bridge_probe.py --steps 200000 --max-period 4096
"""

from __future__ import annotations

import argparse
import json
import os
import time

import rule30_lab as L


def two_sided_table(steps, max_period, avals=(0, 1, 2, 4, 8), bvals=(0, 1, 2, 4, 8)):
    """P[col1 agrees at lag p at t | col0 agrees on [t-a, t+b], a_t(0) = 0]."""
    cols = L.columns(steps, (0, 1))
    N = steps + 1
    S0, S1 = L._to_int(cols[0]), L._to_int(cols[1])
    pc = lambda x: bin(x).count("1")
    num = {(a, b): 0 for a in avals for b in bvals}
    den = {(a, b): 0 for a in avals for b in bvals}
    for p in range(1, max_period + 1):
        W = (1 << (N - p - max(bvals) - 1)) - 1
        agree0 = (~(S0 ^ (S0 >> p))) & W
        agree1 = (~(S1 ^ (S1 >> p))) & W
        zero0 = (~S0) & W
        for a in avals:
            for b in bvals:
                A = W
                for j in range(-a, b + 1):
                    A &= (agree0 >> j) if j >= 0 else (agree0 << (-j))
                d = A & zero0 & W
                den[(a, b)] += pc(d)
                num[(a, b)] += pc(d & agree1)
    return [
        {"lookback_a": a, "lookahead_b": b, "n": den[(a, b)],
         "P_col1_agrees": (num[(a, b)] / den[(a, b)]) if den[(a, b)] else None}
        for a in avals for b in bvals
    ]


def period_band_check(steps, bands=((1, 64), (65, 512), (513, 4096)), avals=(4, 8, 12)):
    cols = L.columns(steps, (0, 1))
    N = steps + 1
    S0, S1 = L._to_int(cols[0]), L._to_int(cols[1])
    pc = lambda x: bin(x).count("1")
    out = []
    for lo, hi in bands:
        for a in avals:
            num = den = 0
            for p in range(lo, hi + 1):
                W = (1 << (N - p - 1)) - 1
                agree0 = (~(S0 ^ (S0 >> p))) & W
                agree1 = (~(S1 ^ (S1 >> p))) & W
                zero0 = (~S0) & W
                A = W
                for j in range(a + 1):
                    A &= (agree0 << j) if j else agree0
                d = A & W & zero0
                den += pc(d)
                num += pc(d & agree1)
            out.append({"p_lo": lo, "p_hi": hi, "lookback_a": a, "n": den,
                        "P_col1_agrees": num / den if den else None})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--max-period", type=int, default=4096)
    ap.add_argument("--outdir", default="phase1_results")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    res = {"settings": vars(args)}
    print("locality probe (maximal windows) ...", flush=True)
    res["locality_probe"] = L.locality_probe(
        steps=args.steps, periods=range(1, 257), min_window=8)
    print("forward transfer profile ...", flush=True)
    res["transfer_profile_forward"] = L.transfer_profile(
        steps=args.steps, periods=range(1, 257), kmax=16)
    print("two-sided table ...", flush=True)
    res["two_sided_table"] = two_sided_table(args.steps, 256)
    print("backward transfer depth profile ...", flush=True)
    res["backward_transfer"] = L.backward_transfer(
        steps=args.steps, max_period=args.max_period)
    print("period-band homogeneity ...", flush=True)
    res["period_band_check"] = period_band_check(args.steps)
    res["elapsed_sec"] = round(time.time() - t0, 1)

    with open(os.path.join(args.outdir, "bridge_probe.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n--- forward conditioning: P[col1 agrees | col0 agrees on [t, t+b]] ---")
    print("baseline = %.6f" % res["transfer_profile_forward"]["baseline_P_col1_agrees"])
    for d in res["transfer_profile_forward"]["profile"][:10]:
        print("  window %2d  n=%10d  P=%.5f  (zero set: %.5f)"
              % (d["window_len"], d["n"], d["P_col1_agrees_given_col0_window"],
                 d["P_col1_agrees_given_col0_window_and_col0_zero"]))
    print("\n--- two-sided: rows = lookback a, cols = lookahead b ---")
    avals = sorted({d["lookback_a"] for d in res["two_sided_table"]})
    bvals = sorted({d["lookahead_b"] for d in res["two_sided_table"]})
    idx = {(d["lookback_a"], d["lookahead_b"]): d for d in res["two_sided_table"]}
    print("      " + "".join("  b=%-8d" % b for b in bvals))
    for a in avals:
        print("a=%-3d " % a + "".join("  %.4f  " % idx[(a, b)]["P_col1_agrees"]
                                      for b in bvals))
    print("\n--- backward depth profile (b = 0) ---")
    for d in res["backward_transfer"]["profile"]:
        print("  lookback %2d  n=%10d  P=%.5f" % (d["lookback_a"], d["n"],
                                                  d["P_col1_agrees"]))
    print("\nwrote", os.path.join(args.outdir, "bridge_probe.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
