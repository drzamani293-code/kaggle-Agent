"""find_mb1loc_witnesses.py -- explicit refutation of the local bridging lemma.

MB1-loc(a):  for all t, p:  if  a_s(0) = a_{s+p}(0)  for every s in [t-a, t]
             and a_t(0) = 0,  then  a_t(1) = a_{t+p}(1).

MB1-loc(a) is a universally quantified statement, so a single (t, p) pair
violating it refutes it outright -- finite computation *can* do that (it is
the reverse direction, proving non-periodicity, that it cannot do).  This
script exhibits explicit witnesses for as deep a look-back `a` as the data
supports, and re-verifies each one from scratch against a freshly simulated
space-time diagram.

    python3 find_mb1loc_witnesses.py --steps 200000 --max-period 20000
"""
from __future__ import annotations
import argparse, json, os
import rule30_lab as L


def find_witnesses(steps, max_period, avals):
    cols = L.columns(steps, (0, 1))
    N = steps + 1
    S0, S1 = L._to_int(cols[0]), L._to_int(cols[1])
    found = {a: None for a in avals}
    counts = {a: 0 for a in avals}
    trials = {a: 0 for a in avals}
    pc = lambda x: bin(x).count("1")
    for p in range(1, max_period + 1):
        W = (1 << (N - p - 1)) - 1
        agree0 = (~(S0 ^ (S0 >> p))) & W
        dis1 = (S1 ^ (S1 >> p)) & W          # column 1 DISAGREES at lag p
        zero0 = (~S0) & W
        A = W
        prev = 0
        for a in sorted(avals):
            for j in range(prev, a + 1):
                A &= (agree0 << j) if j else agree0
            A &= W
            prev = a + 1
            elig = A & zero0
            trials[a] += pc(elig)
            bad = elig & dis1
            counts[a] += pc(bad)
            if bad and found[a] is None:
                t = (bad & -bad).bit_length() - 1
                found[a] = {"a": a, "t": t, "p": p}
    return found, counts, trials


def reverify(w, pad=64):
    """Re-check a witness against an independently simulated diagram."""
    a, t, p = w["a"], w["t"], w["p"]
    steps = t + p + pad
    cols = L.columns(steps, (0, 1))
    c0, c1 = cols[0], cols[1]
    window_ok = all(c0[s] == c0[s + p] for s in range(t - a, t + 1))
    return {
        **w,
        "col0_window_agrees_on_[t-a,t]": window_ok,
        "a_t(0)": c0[t],
        "a_t(1)": c1[t],
        "a_{t+p}(1)": c1[t + p],
        "violates_MB1_loc": window_ok and c0[t] == 0 and c1[t] != c1[t + p],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--max-period", type=int, default=20000)
    ap.add_argument("--outdir", default="phase1_results")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    avals = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    found, counts, trials = find_witnesses(args.steps, args.max_period, avals)
    out = {"settings": vars(args), "rows": []}
    for a in avals:
        row = {"a": a, "eligible_positions": trials[a], "violations": counts[a],
               "violation_rate": counts[a] / trials[a] if trials[a] else None,
               "witness": None}
        if found[a]:
            row["witness"] = reverify(found[a])
        out["rows"].append(row)
        print(json.dumps(row))
    with open(os.path.join(args.outdir, "mb1loc_witnesses.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
