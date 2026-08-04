"""verify_factor_bound.py -- the factor-complexity lower bound on T + p,
recomputed by three independent algorithms.

THE LEMMA
---------
Lemma A (factor counting).  Let x = (x(0), x(1), ...) be an infinite sequence
over any alphabet.  If x is eventually periodic with preperiod T >= 0 and
period p >= 1 -- i.e. x(t + p) = x(t) for every t >= T -- then for every
n >= 1 the number of distinct factors (contiguous blocks) of x of length n is
at most T + p.

Proof.  Let w_i = x(i) x(i+1) ... x(i+n-1) denote the factor starting at
position i, so the set of length-n factors is { w_i : i >= 0 }.  Split the
starting positions:

  * i < T.  At most T such positions, hence at most T factors from them.
  * i >= T.  Write i = T + qp + r with 0 <= r < p and q >= 0.  For every
    j >= 0 we have x(i + j) = x(T + r + j), because i + j >= T and periodicity
    lets us subtract p exactly q times without leaving the region t >= T.
    Hence w_i = w_{T+r}, and there are at most p distinct values of r.

So the total is at most T + p.  QED

CONSEQUENCE
-----------
Every factor observed in a finite prefix is a genuine factor of the infinite
sequence.  So if a prefix of the rule-30 centre column contains K distinct
factors of some length n, then any (T, p) making the centre column eventually
periodic must satisfy

        T + p >= K.

This is a *lower bound only*.  It cannot show that no (T, p) exists.  Note the
ceiling: a prefix of length N contains at most N - n + 1 factors of length n,
so this method can never yield a bound larger than about N.

THREE INDEPENDENT COUNTS
------------------------
  A. rolling n-bit integer window into a Python `set`      (hash-based)
  B. numpy bit-packing into an array + `np.unique`         (sort-based)
  C. string slicing into a `set` of `str`                  (hash-based, but a
                                                            different key type
                                                            and code path)

A and B use different data structures *and* different algorithmic principles
(hashing vs sorting), so agreement between them is a genuine cross-check of the
count, not just a rerun.

    python3 verify_factor_bound.py --file phase1_results/center_column_1000001.txt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time

try:
    import numpy as np
except ImportError:
    np = None


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_A_rolling_int(bits, n):
    """Rolling n-bit integer window, hashed into a set."""
    mask = (1 << n) - 1
    v = 0
    seen = set()
    for i, b in enumerate(bits):
        v = ((v << 1) | b) & mask
        if i >= n - 1:
            seen.add(v)
    return len(seen)


def count_B_numpy_unique(bits, n):
    """Bit-pack every window into an integer array, then sort-and-unique."""
    if np is None:
        raise RuntimeError("numpy required for method B")
    a = np.frombuffer(bytes(bits), dtype=np.uint8).astype(np.uint64)
    N = len(a)
    m = N - n + 1
    vals = np.zeros(m, dtype=np.uint64)
    for j in range(n):
        vals = (vals << np.uint64(1)) | a[j : j + m]
    return int(np.unique(vals).size)


def count_C_string_set(bits, n):
    """String slicing into a set of str."""
    s = "".join(map(str, bits))
    return len({s[i : i + n] for i in range(len(s) - n + 1)})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[20, 24, 26, 28])
    ap.add_argument("--skip-c-above", type=int, default=28,
                    help="method C is memory-hungry; skip it for larger n")
    args = ap.parse_args(argv)

    digest = sha256_of_file(args.file)
    with open(args.file) as f:
        s = f.read().strip()
    bits = [1 if ch == "1" else 0 for ch in s]
    N = len(bits)

    print("sequence file      : %s" % args.file)
    print("sha256(file)       : %s" % digest)
    print("length N           : %d bits (t = 0..%d)" % (N, N - 1))
    print("ones / zeros       : %d / %d" % (sum(bits), N - sum(bits)))
    print()
    print("%4s %14s %14s %14s %10s %14s"
          % ("n", "A rolling-int", "B numpy-unique", "C string-set",
             "agree", "ceiling N-n+1"))

    best = (0, None)
    all_agree = True
    for n in args.lengths:
        t0 = time.time()
        a = count_A_rolling_int(bits, n)
        b = count_B_numpy_unique(bits, n) if np is not None else None
        c = count_C_string_set(bits, n) if n <= args.skip_c_above else None
        vals = [v for v in (a, b, c) if v is not None]
        agree = len(set(vals)) == 1
        all_agree &= agree
        print("%4d %14d %14s %14s %10s %14d   (%.1f s)"
              % (n, a, b if b is not None else "-", c if c is not None else "-",
                 "YES" if agree else "*** NO ***", N - n + 1, time.time() - t0))
        if a > best[0]:
            best = (a, n)

    print()
    print("Lemma A  =>  T + p >= %d   (witness length n = %d)" % (best[0], best[1]))
    print("ceiling of this method at N = %d is %d" % (N, N - best[1] + 1))
    print()
    print("RESULT: %s" % ("all methods agree" if all_agree else "*** DISAGREEMENT ***"))
    return 0 if all_agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
