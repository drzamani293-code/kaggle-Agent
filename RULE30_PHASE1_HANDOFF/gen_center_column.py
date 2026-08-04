"""gen_center_column.py -- generate and checksum the rule-30 center column.

This is the canonical generator for the sequence files used by every other
script in the handoff package.  It writes the center column as an ASCII string
of '0'/'1' (one line, no separators) and prints a SHA-256 of the file contents,
so that any later analysis can be tied to an exact byte sequence.

    python3 gen_center_column.py --steps 200000
    python3 gen_center_column.py --steps 1000000

Implementation: engine C (big-integer bit-parallel) from rule30_lab.py.
For an *independent* regeneration with a different engine, see
verify_sequence_independent.py.

Cost is O(N^2 / 64): ~9 s at N = 2*10^5, ~350 s at N = 10^6.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time

import rule30_lab as L


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000000,
                    help="number of CA steps; the file holds steps+1 bits")
    ap.add_argument("--outdir", default="phase1_results")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    bits = L.center_column_bitwise(args.steps)
    gen = time.time() - t0

    path = os.path.join(args.outdir, "center_column_%d.txt" % (args.steps + 1))
    with open(path, "w") as f:
        f.write("".join(map(str, bits)) + "\n")

    digest = sha256_of_file(path)
    ones = sum(bits)
    print("file            : %s" % path)
    print("bits            : %d  (t = 0..%d)" % (len(bits), args.steps))
    print("ones / zeros    : %d / %d" % (ones, len(bits) - ones))
    print("first 32 bits   : %s" % "".join(map(str, bits[:32])))
    print("last 32 bits    : %s" % "".join(map(str, bits[-32:])))
    print("sha256(file)    : %s" % digest)
    print("generation time : %.1f s" % gen)

    with open(os.path.join(args.outdir, "SHA256SUMS.txt"), "a") as f:
        f.write("%s  %s\n" % (digest, os.path.basename(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
