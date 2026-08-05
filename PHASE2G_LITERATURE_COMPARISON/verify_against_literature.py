"""verify_against_literature.py -- Phase 2G.

Checks OUR computed quantities against numerical data attributed to the
literature.  The attributed data was obtained from search-engine summaries and
figure captions, NOT from the papers themselves (see RETRIEVAL_LOG.md), so a
match is evidence about our own conventions and about the correspondence of
objects -- it is not a reading of any paper.

    python3 verify_against_literature.py
"""
from __future__ import annotations

import json
import os

import prefix_lab as PL

# ---------------------------------------------------------------------------
# Attributed data.  SOURCE STATUS: secondary (search-engine summary of
# Rowland 2006).  Not read from the paper.
# ---------------------------------------------------------------------------
ROWLAND_PERIODS = [1, 1, 1, 2, 1, 2, 2, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                   4, 4, 4, 2, 4, 4, 4, 4, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                   8, 8, 8, 8, 8, 4, 8, 8]
# "the boundary between the two regions moving to the left, on average, by
# about 0.252 cells every evolution step" -- same source status.
BOUNDARY_SLOPE = 0.252


def coord_period_and_preperiod(rows, k, guard=50):
    """Exact minimal eventual period of the single diagonal k, and its
    preperiod.  Two independent computations of the period are compared."""
    n = len(rows)
    for q in (1, 2, 4, 8, 16, 32, 64, 128):
        if q >= n - guard:
            break
        last = -1
        for t in range(n - q):
            if ((rows[t + q] >> k) ^ (rows[t] >> k)) & 1:
                last = t
        if last + 1 < n - q - guard:
            return q, last + 1
    return None, None


def coord_period_by_cycle_detect(rows, k, guard=50):
    """Independent method: read the tail of diagonal k as a word and find its
    minimal period directly."""
    n = len(rows)
    tail = [(rows[t] >> k) & 1 for t in range(n - guard - 256, n - guard)]
    for q in (1, 2, 4, 8, 16, 32, 64, 128):
        if all(tail[i] == tail[i + q] for i in range(len(tail) - q)):
            return q
    return None


def main():
    os.makedirs("results", exist_ok=True)
    K_max, T_max = 60, 400
    rows = PL.edge_rows(K_max, T_max)

    per, pre, per2 = [], [], []
    for k in range(len(ROWLAND_PERIODS)):
        q, p = coord_period_and_preperiod(rows, k)
        per.append(q)
        pre.append(p)
        per2.append(coord_period_by_cycle_detect(rows, k))

    match = per == ROWLAND_PERIODS
    agree = per == per2

    # prefix periods P(K) should be the running maximum of the diagonal periods
    T, P = PL.exact_TP_first_repeat(60, 400)
    runmax, m = [], 0
    for q in per:
        m = max(m, q)
        runmax.append(m)
    prefix_ok = all(P[k] == runmax[k] for k in range(len(per)))

    # the frontier slope against the attributed boundary speed
    T2, P2 = PL.exact_TP_first_repeat(30000, 46000)
    ratio = T2[30000] / 30000
    ours_boundary = 1.0 - 1.0 / ratio

    out = {
        "source_status": "attributed data is SECONDARY (search-engine summary "
                         "and figure captions); the papers were NOT retrieved",
        "diagonal_periods_ours": per,
        "diagonal_periods_attributed_to_rowland": ROWLAND_PERIODS,
        "exact_match": match,
        "terms_compared": len(ROWLAND_PERIODS),
        "two_independent_period_methods_agree": agree,
        "diagonal_preperiods_ours": pre,
        "prefix_period_is_running_max_of_diagonal_periods": prefix_ok,
        "doubling_positions_ours": [k for k in range(1, len(per))
                                    if per[k] > max(per[:k])],
        "period_one_positions_ours": [k for k in range(len(per)) if per[k] == 1],
        "T_over_K_at_30000": ratio,
        "implied_boundary_speed_ours": ours_boundary,
        "boundary_speed_attributed": BOUNDARY_SLOPE,
        "boundary_relative_difference": abs(ours_boundary - BOUNDARY_SLOPE)
                                        / BOUNDARY_SLOPE,
    }
    with open(os.path.join("results", "literature_check.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("diagonal periods, k = 0 .. %d" % (len(per) - 1))
    print("  ours      :", per)
    print("  attributed:", ROWLAND_PERIODS)
    print("  EXACT MATCH on all %d terms: %s" % (len(per), match))
    print("  two independent period methods agree: %s" % agree)
    print("  prefix P(K) = running max of diagonal periods: %s" % prefix_ok)
    print("  first-doubling positions (ours): %s" % out["doubling_positions_ours"])
    print("  period-1 diagonals (ours):       %s" % out["period_one_positions_ours"])
    print("  diagonal preperiods (ours):      %s" % pre)
    print()
    print("frontier slope vs attributed boundary speed")
    print("  T(30000)/30000        = %.6f" % ratio)
    print("  implied boundary speed = 1 - K_per/t = %.6f" % ours_boundary)
    print("  attributed             = %.3f" % BOUNDARY_SLOPE)
    print("  relative difference    = %.3f %%"
          % (100 * out["boundary_relative_difference"]))
    return 0 if (match and agree and prefix_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
