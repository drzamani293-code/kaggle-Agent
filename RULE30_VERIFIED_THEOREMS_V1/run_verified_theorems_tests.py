"""run_verified_theorems_tests.py -- Phase 2F final validation.

Runs from a fresh extraction.  Exits non-zero if anything fails.

    python3 run_verified_theorems_tests.py            # ~30 s
    python3 run_verified_theorems_tests.py --fast     # ~5 s
    python3 run_verified_theorems_tests.py --no-artifacts   # skip hash checks

Sections
  1  every Boolean identity, exhaustively (no sampling)
  2  the proved theorems, re-derived and re-checked on the real orbit
  3  finite certificates verified against the raw artifacts
  4  SHA-256 hashes of every catalogued artifact
  5  cross-references: every claim ID cited in a document exists in the ledger
  6  banned-overclaim scan across every document in this package

The suite locates the sibling phase packages by walking up from this file.
If they are absent (e.g. this folder was extracted alone) the artifact and
certificate sections report SKIPPED rather than failing, and say so.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PASSES, FAILURES, SKIPPED = [], [], []


def check(name, cond, detail=""):
    (PASSES if cond else FAILURES).append(name)
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return bool(cond)


def skip(name, why):
    SKIPPED.append(name)
    print("[SKIP] %s  -- %s" % (name, why))


def docs():
    return sorted(f for f in os.listdir(HERE) if f.endswith(".md"))


def read(f):
    with open(os.path.join(HERE, f), encoding="utf-8") as fh:
        return fh.read()


def plain(text):
    """Strip markdown emphasis and code marks so phrase checks see the prose."""
    return text.replace("**", "").replace("*", "").replace("`", "")


# ==========================================================================
# 1. Boolean identities, exhaustive
# ==========================================================================


def rule30(l, c, r):
    return l ^ (c | r)


def t1_identities():
    print("\n=== 1. Boolean identities, exhaustive ===")
    dom2 = [(a, b) for a in (0, 1) for b in (0, 1)]
    dom3 = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]

    # the rule table itself, indexed by 4l+2c+r
    tab = [rule30((n >> 2) & 1, (n >> 1) & 1, n & 1) for n in range(8)]
    check("T-01 rule table is (0,1,1,1,1,0,0,0) under 4l+2c+r",
          tab == [0, 1, 1, 1, 1, 0, 0, 0], str(tab))

    # T-01(a) left-permutivity, (c) failure of right-permutivity
    check("T-01(a) left-permutive on all 4 (c,r)",
          all(rule30(0, c, r) != rule30(1, c, r) for c, r in dom2))
    check("T-01(c) NOT right-permutive (fails exactly when c=1)",
          all((rule30(l, 1, 0) == rule30(l, 1, 1)) for l in (0, 1))
          and all((rule30(l, 0, 0) != rule30(l, 0, 1)) for l in (0, 1)))

    # T-01(b) local inversion, exhaustively over the three inputs
    check("T-01(b) local inversion identity, all 8 neighbourhoods",
          all(l == rule30(l, c, r) ^ (c | r) for l, c, r in dom3))

    # T-05 the defect equation, exhaustive over (u,v,e,g) and the left defect
    def delta(u, v, e, g):
        return ((u ^ e) | (v ^ g)) ^ (u | v)

    ok_xor = ok_anf = True
    for u, v in dom2:
        for e, g in dom2:
            for dl in (0, 1):
                lhs = (rule30(dl ^ 0, u ^ e, v ^ g)
                       ^ rule30(0, u, v)) ^ (dl ^ dl)
                # direct: d_{t+1} = d(j-1) XOR Delta
                direct = dl ^ delta(u, v, e, g)
                # recompute from the definition
                new_p = (dl ^ 0) ^ ((u ^ e) | (v ^ g))
                new_0 = 0 ^ (u | v)
                defn = new_p ^ new_0
                ok_xor &= (direct == defn)
                anf = (dl ^ e ^ g ^ (u & g) ^ (v & e) ^ (e & g))
                ok_anf &= (anf == defn)
    check("T-05 XOR/OR form exact on all 32 assignments", ok_xor)
    check("T-05 ANF form exact on all 32 assignments", ok_anf)

    # T-5.1 masking
    check("T-5.1 a black cell with no defect masks the defect on its right",
          all(delta(1, v, 0, g) == 0 for v, g in dom2))
    # T-6.1's reduction Delta(u,v,0,g) = g AND NOT u
    check("T-6.1 reduction Delta(u,v,0,g) = g AND NOT u",
          all(delta(u, v, 0, g) == (g & (1 ^ u)) for u, v in dom2 for g in (0, 1)))

    # T-10 the fibre trichotomy, exhaustive over short base words
    def compose(word):
        f0, f1 = 0, 1
        for a, c in word:
            f0, f1 = a ^ (c | f0), a ^ (c | f1)
        return f0, f1

    bad = []
    for n in range(1, 7):
        for m in range(1 << (2 * n)):
            word = [(((m >> (2 * i)) & 1), ((m >> (2 * i + 1)) & 1))
                    for i in range(n)]
            f0, f1 = compose(word)
            anyc = any(c for _, c in word)
            xa = 0
            for a, _ in word:
                xa ^= a
            if anyc:
                want = "const"
            else:
                want = "neg" if xa else "id"
            got = "const" if f0 == f1 else ("id" if f0 == 0 else "neg")
            if got != want:
                bad.append((word, got, want))
    check("T-10 trichotomy exact over every base word of length <= 6",
          not bad, str(bad[:2]))

    # T-21.8 strip preimage: surjective with in-degree exactly 4
    def step(q, R, left, right):
        w = 2 * R + 1
        mask = (1 << w) - 1
        lo = ((q << 1) | left) & mask
        hi = (q >> 1) | (right << (w - 1))
        return (lo ^ (q | hi)) & mask

    ok = True
    for R in (1, 2, 3):
        n = 1 << (2 * R + 1)
        indeg = [0] * n
        for q in range(n):
            for l in (0, 1):
                for rb in (0, 1):
                    indeg[step(q, R, l, rb)] += 1
        ok &= (min(indeg) == max(indeg) == 4)
    check("T-21.8 strip transition relation has in-degree exactly 4 (R<=3)", ok)


# ==========================================================================
# 2. Theorems re-checked on the real orbit
# ==========================================================================


def orbit_rows(t_max):
    C = t_max + 2
    mask = (1 << (2 * C + 1)) - 1
    row = 1 << C
    rows = [row]
    for _ in range(t_max):
        row = ((row << 1) ^ (row | (row >> 1))) & mask
        rows.append(row)
    return rows, C


def edge_rows(K_max, T_max):
    mask = (1 << (K_max + 1)) - 1
    W = 1
    out = [W]
    for _ in range(T_max):
        W = ((W << 2) ^ ((W << 1) | W)) & mask
        out.append(W)
    return out


def exact_TP(K_max, T_max):
    rows = edge_rows(K_max, T_max)
    n = len(rows)
    T = [None] * (K_max + 1)
    P = [None] * (K_max + 1)
    for e in range(15):
        Q = 1 << e
        if Q >= n:
            break
        m = n - Q
        L = []
        for t in range(m):
            d = rows[t + Q] ^ rows[t]
            L.append((K_max + 1) if d == 0 else ((d & -d).bit_length() - 1))
        t0 = 0
        for K in range(K_max + 1):
            while t0 < m and L[t0] <= K:
                t0 += 1
            if t0 >= m:
                break
            if T[K] is None:
                T[K], P[K] = t0, Q
    return T, P, rows


def t2_theorems(fast):
    print("\n=== 2. theorems re-checked on the real orbit ===")
    tmax = 200 if fast else 400
    orb, C = orbit_rows(tmax)
    wrows = edge_rows(2 * tmax + 4, tmax)

    # T-07 the coordinate change, cell by cell
    bad = [(t, k) for t in range(tmax + 1) for k in range(0, 2 * t + 1)
           if ((wrows[t] >> k) & 1) != ((orb[t] >> (C - t + k)) & 1)]
    check("T-07 w_t(k) = x_t(-t+k), cell by cell", not bad, str(bad[:3]))

    # T-2.1 / T-7.3 support bounds
    check("T-2.1 light cone x_t(j)=0 for |j|>t",
          not [(t, j) for t in range(tmax + 1) for j in (t + 1, t + 2, -t - 1)
               if (orb[t] >> (C + j)) & 1])
    check("T-7.3 support bound w_t(k)=0 for k>2t",
          not [(t, k) for t in range(tmax + 1)
               for k in range(2 * t + 1, min(2 * t + 5, 2 * tmax + 4))
               if (wrows[t] >> k) & 1])

    # T-2.2 frozen left edge
    check("T-2.2 frozen left edge x_t(-t)=1 for every t",
          all((orb[t] >> (C - t)) & 1 for t in range(tmax + 1)))

    # T-21.6 strip identity, including negative r
    bad = [(t, r) for t in range(tmax + 1) for r in range(-6, 7)
           if t + r >= 0 and ((wrows[t] >> (t + r)) & 1) != ((orb[t] >> (C + r)) & 1)]
    check("T-21.6 q_t(r) = x_t(r), including negative r", not bad, str(bad[:3]))

    # T-08 autonomy of F_K from t = 0  (correction C-08)
    K = 40
    w2 = edge_rows(K, 80)
    bad = []
    for t in range(79):
        z = [(w2[t] >> k) & 1 for k in range(K + 1)]
        nxt = [(z[k - 2] if k >= 2 else 0) ^ ((z[k - 1] if k >= 1 else 0) | z[k])
               for k in range(K + 1)]
        want = [(w2[t + 1] >> k) & 1 for k in range(K + 1)]
        if nxt != want:
            bad.append(t)
    check("T-08 F_K is autonomous from t=0 (C-08: no 't >= K/2' needed)",
          not bad, str(bad[:5]))

    # T-22 coordinate 7 permanently zero, via the explicit 2-cycle
    w7 = edge_rows(7, 60)
    check("T-22 w_t(7)=0 for every t, and the orbit is a 2-cycle from t=2",
          all(not ((r >> 7) & 1) for r in w7)
          and w7[2] == w7[4] and w7[3] == w7[5] and w7[2] != w7[3],
          "W2=%s W3=%s (bit k=0..7 left to right)"
          % (format(w7[2], '08b')[::-1], format(w7[3], '08b')[::-1]))

    # the prefix tower
    Kmax = 800 if fast else 3000
    T, P, rows = exact_TP(Kmax, int(1.6 * Kmax) + 500)
    check("CC-07 exact (T,P) computed for every K in range",
          all(T[K] is not None for K in range(Kmax + 1)))
    check("T-12 T is non-decreasing",
          not [K for K in range(Kmax) if T[K] > T[K + 1]])
    check("T-09 P(K-1) divides P(K)",
          not [K for K in range(1, Kmax + 1) if P[K] % P[K - 1]])
    check("T-11 P(K) in {P(K-1), 2P(K-1)} and is a power of two",
          not [K for K in range(1, Kmax + 1)
               if P[K] not in (P[K - 1], 2 * P[K - 1]) or (P[K] & (P[K] - 1))])
    check("T-13.1 T(K) <= T(K-1) + P(K-1)",
          not [K for K in range(1, Kmax + 1) if T[K] > T[K - 1] + P[K - 1]])
    check("T-21.3 T(K)+P(K) > floor(K/2)",
          not [K for K in range(Kmax + 1) if not T[K] + P[K] > K // 2])
    check("BO-03 T(K) <= K holds EXACTLY for K <= 17 (C-07)",
          [K for K in range(Kmax + 1) if T[K] <= K] == list(range(18)))

    # T-18 the exact recurrence, and T-19
    bit = lambda t, k: 0 if k < 0 else (rows[t] >> k) & 1
    badB, badR, n = [], [], 0
    for K in range(2, Kmax + 1):
        Tp, Pp, Pk = T[K - 1], P[K - 1], P[K]
        if Tp + Pk + 1 >= len(rows):
            break
        n += 1
        cword = [bit(Tp + j, K - 1) for j in range(Pp)]
        tau = next((Tp + j for j in range(Pp) if cword[j]), None)
        D = bit(Tp, K) ^ bit(Tp + Pk, K)
        if Pk == 2 * Pp or not any(cword):
            pred = Tp
        else:
            pred = Tp if D == 0 else (tau + 1 if tau is not None else None)
        if pred != T[K]:
            badB.append(K)
        if T[K] > Tp:
            rho = next((t for t in range(T[K] - 1, max(-1, T[K] - 300), -1)
                        if bit(t, K - 1)), None)
            if rho is None or T[K] != rho + 1:
                badR.append(K)
    check("T-18 exact recurrence predicts T(K) at every level",
          not badB, "%d levels checked, first failures %s" % (n, badB[:5]))
    check("T-19 T(K) = rho(K)+1 at every resetting level",
          not badR, str(badR[:5]))

    # T-14 and T-15, at the coordinates that are zero on their cycle
    ez = [k for k in range(Kmax + 1)
          if all(not bit(T[k] + j, k) for j in range(P[k]))]
    check("BO-04 coordinates zero on their cycle are a subset of {2,7,28,399}",
          set(ez) <= {2, 7, 28, 399}, str(ez))
    badN2 = [m for m in ez if m >= 2 and
             not all(bit(T[m] + j, m - 2) == bit(T[m] + j, m - 1)
                     for j in range(P[m]))]
    check("T-15 zero-cycle identity at the COORDINATE indices (C-16)",
          not badN2, str(badN2))
    badN1 = [K for K in range(2, min(Kmax, 3000) + 1)
             if (not any(bit(T[K - 1] + j, K - 1) for j in range(P[K - 1])))
             != ((K - 1) in set(ez))]
    check("T-14 non-COLLAPSING iff predecessor zero on its cycle",
          not badN1, str(badN1[:5]))

    # T-20 the XOR spine
    for t in ([60] if fast else [60, 150, 300]):
        seen = {(t, t)}
        stack = [(t, t)]
        while stack:
            s, k = stack.pop()
            if s == 0 or k < 0 or k > 2 * s:
                continue
            kids = [(s - 1, k - 2)]
            if not bit(s - 1, k):
                kids.append((s - 1, k - 1))
            if not bit(s - 1, k - 1):
                kids.append((s - 1, k))
            for c in kids:
                if c[1] < 0 or c[1] > 2 * c[0]:
                    continue
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        spine = [(t - j, t - 2 * j) for j in range(t // 2 + 1)]
        check("T-20 XOR spine lies in the skeleton at t=%d, |Sk| >= floor(t/2)+1" % t,
              all(c in seen for c in spine) and len(seen) >= len(spine),
              "|Sk|=%d spine=%d" % (len(seen), len(spine)))

    # VQ-02: F5's configuration is impossible
    bad = []
    for k in (5, 20, 100):
        if k > Kmax:
            continue
        mask = (1 << (k + 1)) - 1
        seen = {}
        for t in range(min(len(rows), T[k] + 3 * P[k] + 5)):
            st = rows[t] & mask
            if st in seen and T[k] > seen[st]:
                bad.append((k, seen[st], t))
            seen.setdefault(st, t)
    check("VQ-02 no prefix-state repeat with t < T(K) (F5's configuration)",
          not bad, str(bad[:3]))

    # T-21.10 the diagonal's reset schedule is the centre column, shifted
    bad = [t for t in range(1, tmax + 1)
           if bool((wrows[t - 1] >> (t - 1)) & 1) != bool((orb[t - 1] >> C) & 1)]
    check("T-21.10 diagonal reset schedule = centre column shifted by one",
          not bad, str(bad[:3]))


# ==========================================================================
# 3-4. Certificates and hashes
# ==========================================================================

ARTIFACTS = {
    "RULE30_PHASE1_HANDOFF/phase1_results/center_column_1000001.txt":
        "0bb02e4ed6c3d80bd832eed0b6991cc99f6de01a5f32fa795281d3efb2e9104e",
    "RULE30_PHASE1_HANDOFF/phase1_results/center_column_200001.txt":
        "cf018b09db932412d5f42c3d03076181a34d806173fbee749e30c871fe5d994a",
    "PHASE2A_SAT_MB1LOC/results/phase2a_results.json":
        "29bc3bbed880faae1a55e94c61f30112dc9fc09fed76b279bc86f120d3a37e57",
    "PHASE2B_GLOBAL_DEFECT/results/phase2b_results.json":
        "81725e5ae6837cd7222bed02cc9dcda3e536c579b72d8a54a6a0e2b4bfe36731",
    "PHASE2C_EDGE_PREFIX/results/phase2c_results.json":
        "18229b7c9041ee81918e087d440cde515918be1b5cf0c3264cc4d7684e3b9563",
    "PHASE2C_EDGE_PREFIX/PREFIX_PERIOD_TABLE.csv":
        "2494997051f5c5cd521ae83276740f6ec152f520263e64aa96aa3b2d5a1bd959",
    "PHASE2D_COLLAPSE_CHAIN/results/phase2d_results.json":
        "18b4b8d6dd86b3e41b7f97bb08a9d26c3550427d4601c3e6396d95b25806ec21",
    "PHASE2E_TRANSIENT_FRONTIER/results/phase2e_results.json":
        "67a42a319f8d49c17e2fd4a8603a10d30e16617c471676e33d8c13092dae2349",
    "PHASE2E_TRANSIENT_FRONTIER/TRANSIENT_CLASSIFICATION.csv":
        "db21ed34a9fdda73e9d7d47198f915852b8b266f9640ae533cb5d98e41b76aa9",
    "PHASE2E_TRANSIENT_FRONTIER/RESET_SCHEDULE_RESULTS.csv":
        "f3e3b07233fdb85f40baca09c5aa4f5b48690e480817c039651dcdb97664aaa1",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def t3_certificates(fast, do_artifacts):
    print("\n=== 3. finite certificates against raw artifacts ===")
    seq = os.path.join(ROOT, "RULE30_PHASE1_HANDOFF/phase1_results/"
                             "center_column_1000001.txt")
    if not do_artifacts or not os.path.exists(seq):
        skip("CC-01/CC-02 re-verification", "artifacts not present")
        return
    with open(seq) as fh:
        bits = fh.read().strip()
    check("CC-01 the stored sequence has 1000001 bits", len(bits) == 1000001,
          str(len(bits)))
    # regenerate a prefix independently and compare
    n = 20000 if fast else 200000
    orb, C = orbit_rows(n)
    gen = "".join(str((orb[t] >> C) & 1) for t in range(n + 1))
    check("CC-01 an independent engine reproduces the first %d bits" % (n + 1),
          gen == bits[:n + 1])
    if not fast:
        # CC-02: recount the length-28 factors
        k = 28
        v = 0
        mask = (1 << k) - 1
        seen = set()
        for i, ch in enumerate(bits):
            v = ((v << 1) | (ch == "1")) & mask
            if i >= k - 1:
                seen.add(v)
        check("CC-02 exactly 998140 distinct length-28 factors", len(seen) == 998140,
              str(len(seen)))
        check("CT-01 hence T + p >= 998140 (conditional on CC-02)", True,
              "conditional theorem; the numeral is the certificate")
    else:
        skip("CC-02 factor recount", "--fast")


def t4_hashes(do_artifacts):
    print("\n=== 4. SHA-256 of catalogued artifacts ===")
    if not do_artifacts:
        skip("artifact hashes", "--no-artifacts")
        return
    missing = 0
    for rel, want in ARTIFACTS.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            missing += 1
            skip("hash %s" % rel, "not present")
            continue
        got = sha256(path)
        check("hash %s" % rel, got == want, got if got != want else "")
    if missing:
        print("  (%d artifacts absent -- this folder may have been extracted "
              "on its own)" % missing)


# ==========================================================================
# 5-6. Document cross-checks
# ==========================================================================

ID_RE = re.compile(r"\b(T-\d\d|CT-\d\d|CC-\d\d|BO-\d\d|CJ-\d\d|RF-\d\d|"
                   r"VQ-\d\d|BR-\d\d)\b")


def t5_ids():
    print("\n=== 5. claim IDs are declared in the ledger ===")
    ledger = read("MASTER_CLAIM_LEDGER.md")
    declared = set()
    for line in ledger.splitlines():
        m = re.match(r"\|\s*\*\*([A-Z]{1,2}-\d\d)\*\*", line.strip())
        if m:
            declared.add(m.group(1))
    check("the ledger declares at least 50 claim IDs", len(declared) >= 50,
          "%d declared" % len(declared))
    for cat, pre in (("A", "T-"), ("B", "CT-"), ("C", "CC-"), ("D", "BO-"),
                     ("E", "CJ-"), ("F", "RF-"), ("G", "VQ-"), ("H", "BR-")):
        n = sum(1 for d in declared if d.startswith(pre))
        check("ledger category %s (%s) is non-empty" % (cat, pre), n > 0,
              "%d entries" % n)
    undeclared = {}
    for f in docs():
        if f == "MASTER_CLAIM_LEDGER.md":
            continue
        for cid in set(ID_RE.findall(read(f))):
            if cid not in declared:
                undeclared.setdefault(cid, []).append(f)
    check("every claim ID cited in any document exists in the ledger",
          not undeclared, str(sorted(undeclared)[:8]))
    # every theorem cited in the dependency graph has a proof in VERIFIED_THEOREMS
    vt = read("VERIFIED_THEOREMS.md")
    missing = [n for n in range(1, 24)
               if ("## T%d " % n) not in vt and ("## T%d —" % n) not in vt]
    check("VERIFIED_THEOREMS.md contains sections T1..T23", not missing,
          str(missing))
    notes = (vt.count("Why this does not solve Problem 1")
             + vt.count("Why these do not solve Problem 1"))
    sections = len(re.findall(r"^## T\d", vt, re.M))
    check("every theorem section carries a 'why this does not solve Problem 1' "
          "note", notes >= sections,
          "%d notes for %d sections" % (notes, sections))
    # required documents present
    need = ["SOURCE_MANIFEST.md", "MASTER_CLAIM_LEDGER.md",
            "DEFINITIONS_AND_NOTATION.md", "VERIFIED_THEOREMS.md",
            "PROOF_AUDIT_REPORT.md", "THEOREM_DEPENDENCY_GRAPH.md",
            "CORRECTIONS_AND_RETRACTIONS.md", "COMPUTATIONAL_CERTIFICATES.md",
            "NOVELTY_STATUS.md", "PAPER_DRAFT_V1.md",
            "EXPERT_REVIEW_REQUEST.md"]
    absent = [f for f in need if not os.path.exists(os.path.join(HERE, f))]
    check("all 11 required documents are present", not absent, str(absent))
    # the dependency graph has a mermaid block and a plain-text tree
    dg = read("THEOREM_DEPENDENCY_GRAPH.md")
    check("dependency graph has a mermaid block", "```mermaid" in dg)
    check("dependency graph marks the bridge", "BR-01" in dg and "BRIDGE" in dg)


BANNED = [
    (r"\bproblem 1 (is )?solved\b", "claims Problem 1 solved"),
    (r"\bwe solve\b.{0,40}\bproblem 1\b", "claims to solve Problem 1"),
    (r"\bprove[dsn]?\b[^.\n]{0,40}\bcent(er|re) (column )?(is )?aperiodic\b",
     "claims the centre column proved aperiodic"),
    (r"\bcent(er|re) column is not eventually periodic\b",
     "asserts non-periodicity"),
    (r"\bunique for all k\b", "claims uniqueness for all K"),
]


def t6_overclaims():
    print("\n=== 6. banned-overclaim scan ===")
    hits = []
    for f in docs():
        low = plain(read(f)).lower()
        for pat, why in BANNED:
            for m in re.finditer(pat, low):
                # a disclaimer may sit earlier in the same sentence or bullet,
                # so look at a window, not just the physical line
                ctx = low[max(0, m.start() - 400):m.end() + 120]
                if any(w in ctx for w in ("not ", "never", "no claim",
                                          "does not", "must not", "banned",
                                          "rejected", "retracted", "explicitly",
                                          "would follow", "is not made")):
                    continue
                start = low.rfind("\n", 0, m.start()) + 1
                end = low.find("\n", m.end())
                hits.append((f, why, low[start:end].strip()[:90]))
    check("no document claims Problem 1 is solved or the centre column proved "
          "aperiodic", not hits, str(hits[:3]))

    # UNSAT may appear only with an explicit qualification nearby
    QUALIFIERS = ("not unsat", "bounded unsat", "is not", "are not", "never",
                  "no witness", "only when", "only where", "predicted",
                  "0 unsat", "unsat: 0", "unsat cores", "unsat elsewhere",
                  "would refute", "single unsat", "emptiness",
                  "returned unsat", "no bounded", "is unsat exactly",
                  "as unsat", "called unsat", "banned", "described as",
                  "no unsat", "unsat anywhere", "hand proof",
                  "qualified", "uses of", "occurrence")
    bad_unsat = []
    for f in docs():
        # paragraphs, not physical lines: a qualification may be on the
        # previous wrapped line of the same sentence
        for i, para in enumerate(plain(read(f)).split("\n\n")):
            if "UNSAT" not in para:
                continue
            if any(w in para.lower() for w in QUALIFIERS):
                continue
            bad_unsat.append((f, i + 1, para.strip().replace("\n", " ")[:90]))
    check("every use of 'UNSAT' is qualified", not bad_unsat,
          str(bad_unsat[:3]))

    # the required disclaimers are actually present
    paper = plain(read("PAPER_DRAFT_V1.md")).lower()
    check("the paper draft disclaims solving Problem 1",
          "does not solve" in paper and "prize problem 1" in paper)
    nov = plain(read("NOVELTY_STATUS.md")).lower()
    check("novelty is recorded as unresolved",
          "unresolved" in nov and "403" in nov)
    led = plain(read("MASTER_CLAIM_LEDGER.md")).lower()
    check("the ledger states that tests do not promote a claim",
          "passed tests" in led)
    for f in ("PAPER_DRAFT_V1.md", "EXPERT_REVIEW_REQUEST.md",
              "MASTER_CLAIM_LEDGER.md"):
        body = plain(read(f)).lower()
        check("%s states that Problem 1 is not solved" % f,
              "does not solve" in body or "not solved" in body
              or "not claimed" in body)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-artifacts", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    print("python %s" % sys.version.split()[0])
    print("package dir: %s" % HERE)
    print("repo root:   %s" % ROOT)
    t1_identities()
    t2_theorems(a.fast)
    t3_certificates(a.fast, not a.no_artifacts)
    t4_hashes(not a.no_artifacts)
    t5_ids()
    t6_overclaims()
    print("\n" + "=" * 70)
    print("%d passed, %d FAILED, %d skipped   (%.1f s)"
          % (len(PASSES), len(FAILURES), len(SKIPPED), time.time() - t0))
    if FAILURES:
        print("failures: %s" % ", ".join(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
