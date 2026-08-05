"""triangle_enumerator.py -- Phase 2H section 9.

Exhaustive enumeration of locally valid rule-30 (and rule-90) light-cone
triangles of height h, and the question the brief poses:

    does the CENTRE boundary word of a triangle determine anything else?

Geometry (original coordinates, absolute columns 0 .. 2h):

    row 0 : columns 0 .. 2h            (the free top row, 2h+1 bits)
    row s : columns s .. 2h-s
    row h : column  h                  (the apex)

        col:  0 1 2 3 4 5 6            (h = 3)
        s=0:  # # # # # # #
        s=1:    # # # # #
        s=2:      # # #
        s=3:        #                  <- apex, column h

    centre word   C = ( x_s(h)   )_{s=0..h}        h+1 bits
    right word    Rt= ( x_s(2h-s))_{s=0..h}        h+1 bits   (the right edge)
    width-2 word  W2= ( x_s(h), x_s(h+1) )_{s=0..h-1}   2h bits

Every one of the 2^(2h+1) top rows gives one triangle, so the enumeration is
exhaustive, not a sample.

Used ONLY to discover and refute conjectures (brief section 9).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def enumerate_triangles(h: int, rule: int = 30):
    """Return (centre, width2, right, n) as numpy arrays over all top rows."""
    w = 2 * h + 1
    n = 1 << w
    rows = np.arange(n, dtype=np.uint64)
    centre = np.zeros(n, dtype=np.uint64)
    width2 = np.zeros(n, dtype=np.uint64)
    right = np.zeros(n, dtype=np.uint64)
    for s in range(h + 1):
        cbit = (rows >> np.uint64(h)) & np.uint64(1)
        centre |= cbit << np.uint64(s)
        rbit = (rows >> np.uint64(2 * h - s)) & np.uint64(1)
        right |= rbit << np.uint64(s)
        if s < h:
            nbit = (rows >> np.uint64(h + 1)) & np.uint64(1)
            width2 |= (cbit << np.uint64(2 * s)) | (nbit << np.uint64(2 * s + 1))
        if s == h:
            break
        if rule == 30:
            rows = (rows << np.uint64(1)) ^ (rows | (rows >> np.uint64(1)))
        else:                                   # rule 90
            rows = (rows << np.uint64(1)) ^ (rows >> np.uint64(1))
        rows &= np.uint64((1 << w) - 1)
    return centre, width2, right, n


def analyse(h: int, rule: int = 30):
    t0 = time.time()
    centre, width2, right, n = enumerate_triangles(h, rule)
    nb_c = h + 1
    res = {"h": h, "rule": rule, "triangles": int(n),
           "centre_bits": nb_c, "width2_bits": 2 * h}

    # how many distinct centre words occur at all
    res["distinct_centre_words"] = int(np.unique(centre).size)
    res["possible_centre_words"] = 1 << nb_c

    # does the centre word determine the width-2 word?
    key = (centre << np.uint64(2 * h)) | width2
    pairs = np.unique(key)
    res["distinct_(centre,width2)_pairs"] = int(pairs.size)
    cs = (pairs >> np.uint64(2 * h))
    cnt = np.bincount(cs.astype(np.int64), minlength=1 << nb_c)
    nz = cnt[cnt > 0]
    res["max_width2_per_centre"] = int(nz.max())
    res["mean_width2_per_centre"] = float(nz.mean())
    res["centre_determines_width2"] = bool(nz.max() == 1)
    del key, pairs, cs

    # does the centre word determine the right edge?
    key = (centre << np.uint64(nb_c)) | right
    pairs = np.unique(key)
    cs = (pairs >> np.uint64(nb_c))
    cnt = np.bincount(cs.astype(np.int64), minlength=1 << nb_c)
    nz = cnt[cnt > 0]
    res["max_right_per_centre"] = int(nz.max())
    res["centre_determines_right"] = bool(nz.max() == 1)
    del key, pairs, cs

    # the converse direction: does the width-2 word determine the centre?
    # (trivially yes -- the centre is a sub-word -- recorded as a sanity check)
    res["width2_contains_centre"] = True

    # how many triangles share the most popular centre word
    cnt_all = np.bincount(centre.astype(np.int64), minlength=1 << nb_c)
    res["max_triangles_per_centre"] = int(cnt_all.max())
    res["min_triangles_per_nonempty_centre"] = int(cnt_all[cnt_all > 0].min())
    res["seconds"] = round(time.time() - t0, 2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-max", type=int, default=11)
    ap.add_argument("--rule", type=int, default=30)
    ap.add_argument("--out", default="SMALL_TRIANGLE_RESULTS.json")
    a = ap.parse_args(argv)
    out = {"rule30": [], "rule90": []}
    for h in range(1, a.h_max + 1):
        r30 = analyse(h, 30)
        out["rule30"].append(r30)
        print("h=%2d rule30  triangles=%10d  centre->width2 determined: %-5s "
              "max width2/centre=%d  (%.1fs)"
              % (h, r30["triangles"], r30["centre_determines_width2"],
                 r30["max_width2_per_centre"], r30["seconds"]))
        if h <= 9:
            r90 = analyse(h, 90)
            out["rule90"].append(r90)
            print("      rule90 (control)         centre->width2 determined: "
                  "%-5s  max width2/centre=%d"
                  % (r90["centre_determines_width2"],
                     r90["max_width2_per_centre"]))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
