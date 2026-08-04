"""run_all_tests.py -- the whole test battery, from a clean process.

Runs every check that can be re-executed in a couple of minutes and exits
non-zero if anything fails.  This is the script an independent reviewer should
run first.

    python3 run_all_tests.py                # fast battery (~2 min)
    python3 run_all_tests.py --with-1m      # also re-checks the 10^6-bit file

What it covers
  1. the 21-check implementation/lemma verification suite
  2. SHA-256 of every stored sequence file against SHA256SUMS.txt
  3. independent regeneration of the 200001-bit sequence (mirror engine + numpy)
  4. factor-complexity counts by three independent algorithms
  5. re-verification of every claimed MB1-loc counterexample, by re-simulating
     the diagram around each witness from scratch
  6. the headline numbers quoted in PHASE1_AUDIT.md, against the stored JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import rule30_lab as L

RESULTS = "phase1_results"
FAILURES = []
PASSES = []


def check(name, cond, detail=""):
    (PASSES if cond else FAILURES).append(name)
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return cond


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
def test_verification_suite():
    print("\n=== 1. implementation + lemma verification suite ===")
    checks = L.run_all_verifications(fast=False)
    for c in checks:
        check(c.name, c.passed, c.detail[:90])
    return all(c.passed for c in checks)


def test_hashes():
    print("\n=== 2. sequence file checksums ===")
    sums = os.path.join(RESULTS, "SHA256SUMS.txt")
    if not os.path.exists(sums):
        return check("SHA256SUMS.txt present", False, "missing")
    seen = {}
    for line in open(sums):
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        seen[name] = digest          # later entries win (regeneration)
    for name, digest in sorted(seen.items()):
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            check("sha256 %s" % name, False, "file missing")
            continue
        got = sha256_of_file(path)
        check("sha256 %s" % name, got == digest, got[:16] + "...")
    return True


def test_independent_regeneration():
    print("\n=== 3. independent regeneration (mirror engine + numpy) ===")
    path = os.path.join(RESULTS, "center_column_200001.txt")
    if not os.path.exists(path):
        return check("200001-bit file present", False, "missing")
    rc = subprocess.call([sys.executable, "verify_sequence_independent.py",
                          "--file", path, "--numpy-prefix", "20000"],
                         stdout=subprocess.DEVNULL)
    return check("verify_sequence_independent (200001 bits)", rc == 0)


def test_factor_bound(with_1m):
    print("\n=== 4. factor-complexity counts, three algorithms ===")
    files = [("center_column_200001.txt", 199900)]
    if with_1m:
        files.append(("center_column_1000001.txt", None))
    ok = True
    for name, expected in files:
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            ok &= check("factor bound %s" % name, False, "file missing")
            continue
        out = subprocess.run([sys.executable, "verify_factor_bound.py",
                              "--file", path, "--lengths", "24", "28"],
                             capture_output=True, text=True)
        agree = "all methods agree" in out.stdout
        bound = None
        for line in out.stdout.splitlines():
            if line.startswith("Lemma A"):
                bound = int(line.split(">=")[1].split("(")[0].strip())
        ok &= check("factor bound %s (three algorithms agree)" % name, agree,
                    "T + p >= %s" % bound)
        if expected is not None:
            ok &= check("factor bound %s value" % name, bound == expected,
                        "expected %d, got %s" % (expected, bound))
    return ok


def test_mb1loc_witnesses():
    print("\n=== 5. re-verification of MB1-loc counterexamples ===")
    path = os.path.join(RESULTS, "mb1loc_witnesses.json")
    if not os.path.exists(path):
        return check("mb1loc_witnesses.json present", False, "missing")
    data = json.load(open(path))
    ok = True
    n_witnesses = 0
    for row in data["rows"]:
        w = row["witness"]
        if w is None:
            check("a=%d: no witness claimed" % row["a"], row["violations"] == 0,
                  "%d violations reported" % row["violations"])
            continue
        n_witnesses += 1
        a, t, p = w["a"], w["t"], w["p"]
        # re-simulate from scratch around the witness
        cols = L.columns(t + p + 8, (0, 1))
        c0, c1 = cols[0], cols[1]
        window = all(c0[s] == c0[s + p] for s in range(t - a, t + 1))
        viol = window and c0[t] == 0 and c1[t] != c1[t + p]
        ok &= check("MB1-loc(a=%d) refuted by (t=%d, p=%d)" % (a, t, p), viol,
                    "col0 window agrees=%s, a_t(0)=%d, a_t(1)=%d, a_{t+p}(1)=%d"
                    % (window, c0[t], c1[t], c1[t + p]))
    ok &= check("witness count", n_witnesses == 8,
                "%d re-verified witnesses" % n_witnesses)
    return ok


def test_audit_numbers():
    print("\n=== 6. headline numbers quoted in PHASE1_AUDIT.md ===")
    p1 = os.path.join(RESULTS, "phase1_results.json")
    af = os.path.join(RESULTS, "anomaly_followup.json")
    if not (os.path.exists(p1) and os.path.exists(af)):
        return check("result JSONs present", False, "missing")
    d = json.load(open(p1))
    a = json.load(open(af))
    check("verification 21/21 recorded", d["verification_summary"] ==
          {"passed": 21, "total": 21, "all_passed": True})
    check("first 256 bits prefix", d["first_256_bits"][:14] == "11011100110001")
    check("external reference matched", d["external_reference_check"]["match"] is True)
    f = {x["n"]: x for x in d["frequencies"]}
    check("ones at N=200001 is 100073", f[200001]["ones"] == 100073)
    lc = {x["N"]: x["L"] for x in d["linear_complexity"]}
    check("linear complexity L(200001) = 100001", lc[200001] == 100001)
    sc = {x["n"]: x["p_obs"] for x in d["subword_complexity"]}
    check("p_obs(24) = 198744", sc[24] == 198744)
    ps = d["period_scan"]
    check("min last_mismatch 189998 at p=10000",
          ps["min_last_mismatch_over_p"] == 189998 and ps["argmin_p"] == 10000)
    check("largest first_mismatch is 11", ps["max_first_mismatch"] == 11)
    check("no period p<=10000 survives the window",
          ps["periods_with_no_mismatch_in_window"] == [])
    check("1M factor bound 998140", a["rigorous_T_plus_p_lower_bound"]["value"] == 998140)
    check("1M period scan T > 979998",
          a["period_scan"]["min_last_mismatch_over_p"] == 979998)
    check("mod-3 anomaly did not replicate",
          abs(a["mod3_fresh_sample"][1]["z"]) < 2.0,
          "fresh-sample z = %+.3f" % a["mod3_fresh_sample"][1]["z"])
    check("missing-factor anomaly resolved",
          a["missing_factor_retest"]["occurrences_in_full_sequence"] == 458)
    check("1M prefix consistent with 200k run",
          a["prefix_consistent_with_200001_run"] is True)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-1m", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    print("python %s   cwd %s" % (sys.version.split()[0], os.getcwd()))

    test_verification_suite()
    test_hashes()
    test_independent_regeneration()
    test_factor_bound(args.with_1m)
    test_mb1loc_witnesses()
    test_audit_numbers()

    print("\n" + "=" * 70)
    print("%d passed, %d FAILED   (%.1f s)" % (len(PASSES), len(FAILURES),
                                               time.time() - t0))
    if FAILURES:
        print("failures: %s" % ", ".join(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
