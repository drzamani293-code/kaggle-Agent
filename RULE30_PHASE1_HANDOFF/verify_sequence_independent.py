"""verify_sequence_independent.py -- regenerate the center column with a
different engine and compare hashes.

The sequence files written by gen_center_column.py come from engine C, the
big-integer bit-parallel evaluation of

    new = (row << 1) ^ (row | (row >> 1))          [rule 30]

A bug in the *direction* of those shifts would produce a self-consistent but
wrong sequence, and would not be caught by re-running the same engine.  This
script regenerates the same sequence two other ways:

  1. **Mirror engine (rule 86).**  Rule 86 is the left-right mirror of rule 30
     (verified exhaustively in rule30_lab.verify_local_rule).  Its bit-parallel
     form has the shifts *reversed*:

         new = (row >> 1) ^ (row | (row << 1))      [rule 86]

     Because the single-cell seed is mirror-symmetric, the rule-86 diagram is
     the mirror image of the rule-30 diagram, so its centre column must be
     bit-for-bit identical.  Any shift-direction error breaks this.

  2. **numpy engine (engine B).**  A table-lookup implementation sharing no
     code with either big-integer engine, run over a prefix (it is O(N^2)
     in *cells*, so only a prefix is affordable).

Both are compared against the stored file by SHA-256.

    python3 verify_sequence_independent.py --file phase1_results/center_column_1000001.txt
    python3 verify_sequence_independent.py --file ... --numpy-prefix 50000
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time

import rule30_lab as L


def center_column_mirror86(steps: int):
    """Engine C', the mirror of engine C: rule 86 with reversed shifts."""
    R = steps + 2
    width = 2 * R + 1
    mask = (1 << width) - 1
    row = 1 << R
    center = [1]
    for _ in range(steps):
        row = ((row >> 1) ^ (row | (row << 1))) & mask
        center.append((row >> R) & 1)
    return center


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--numpy-prefix", type=int, default=50000,
                    help="how many steps to cross-check with the numpy engine")
    args = ap.parse_args(argv)

    with open(args.file) as f:
        stored = f.read().strip()
    steps = len(stored) - 1
    stored_hash = sha256_text(stored + "\n")
    print("stored file      : %s" % args.file)
    print("stored bits      : %d" % len(stored))
    print("stored sha256    : %s" % stored_hash)

    ok = True

    print("\n[1] mirror engine (rule 86, reversed shifts), %d steps ..." % steps,
          flush=True)
    t0 = time.time()
    mirror = "".join(map(str, center_column_mirror86(steps)))
    dt = time.time() - t0
    same = mirror == stored
    print("    sha256         : %s" % sha256_text(mirror + "\n"))
    print("    identical      : %s   (%.1f s)" % (same, dt))
    ok &= same

    n = min(args.numpy_prefix, steps)
    print("\n[2] numpy engine (engine B), first %d steps ..." % n, flush=True)
    t0 = time.time()
    npcol = "".join(map(str, L.center_column_numpy(n)))
    dt = time.time() - t0
    same2 = npcol == stored[: n + 1]
    print("    prefix match   : %s   (%.1f s)" % (same2, dt))
    ok &= same2

    print("\nRESULT: %s" % ("ALL ENGINES AGREE" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
