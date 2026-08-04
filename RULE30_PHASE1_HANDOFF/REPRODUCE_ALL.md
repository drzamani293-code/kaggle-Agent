# REPRODUCE_ALL — exact commands, from a clean machine

Every numerical result in this package is deterministic: there is no RNG, no
sampling, no parallelism-dependent ordering, and no timing-dependent logic.
The same commands on any machine must produce byte-identical output (timings
excepted).

---

## 0. Environment

Reference environment (the one that produced the stored artifacts):

```
Python   3.11.15
numpy    2.4.6
OS       Linux-6.18.5-fc-v18-x86_64-with-glibc2.39
CPU      4 cores (single-threaded workloads throughout)
git commit  b42ce46d480d750c4e779f48f415dde6f9b292c9
```

Setup on a clean machine:

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt          # numpy only
cd RULE30_PHASE1_HANDOFF                 # all commands below run from here
```

**numpy is required** for engine B and for the sort-based factor count.
Everything else is pure standard-library Python. No network access is needed;
no data is downloaded.

**Run everything from inside `RULE30_PHASE1_HANDOFF/`.** Scripts import
`rule30_lab` from the current directory and write to `./phase1_results/`,
which already contains the stored artifacts — re-running overwrites them with
(identical) regenerated versions. Copy the directory first if you want to
diff old against new.

---

## 1. The full test battery (do this first) — ~5 minutes

```bash
python3 run_all_tests.py
```

Runs, and exits non-zero on any failure:

1. the 21-check implementation and lemma verification suite;
2. SHA-256 of every stored sequence file against `phase1_results/SHA256SUMS.txt`;
3. independent regeneration of the 200001-bit sequence (mirror engine + numpy);
4. factor-complexity counts by three independent algorithms;
5. re-verification of all 8 claimed MB1-loc counterexamples, each by
   re-simulating the diagram around the witness from scratch;
6. the headline numbers quoted in `PHASE1_AUDIT.md`, against the stored JSON.

Expected final line:

```
51 passed, 0 FAILED   (~300 s)
```

Add `--with-1m` to include the 10⁶-bit file in step 4 (adds ~10 s, requires
`center_column_1000001.txt` to be present).

Most of the runtime is step 5: the deepest MB1-loc witness sits at
`t = 850603`, and re-simulating the diagram to that time from scratch costs
about 4 minutes. That is deliberate — the witnesses are re-derived rather than
read back from the stored JSON.

---

## 2. Verification suite alone — ~3 seconds

```bash
python3 rule30_lab.py verify          # full, 21 checks
python3 rule30_lab.py verify --fast   # reduced sizes, ~0.5 s
```

Expected: `21/21 checks passed`.

---

## 3. Regenerating the sequences

| Command | Time | Output | SHA-256 |
|---|---|---|---|
| `python3 gen_center_column.py --steps 200000` | ~9 s | `phase1_results/center_column_200001.txt` | `cf018b09db932412d5f42c3d03076181a34d806173fbee749e30c871fe5d994a` |
| `python3 gen_center_column.py --steps 1000000` | ~334 s | `phase1_results/center_column_1000001.txt` | `0bb02e4ed6c3d80bd832eed0b6991cc99f6de01a5f32fa795281d3efb2e9104e` |

Both append their digest to `phase1_results/SHA256SUMS.txt`.

Generation is `O(N²/64)`; `N = 10⁷` would take roughly 9 hours with this
generator and is not part of the package.

### Independent regeneration (different engines)

```bash
python3 verify_sequence_independent.py \
        --file phase1_results/center_column_200001.txt --numpy-prefix 20000
```

~13 s. Regenerates via the **mirror engine** (rule 86, reversed shifts — a
shift-direction bug cannot survive this) and cross-checks a prefix with the
**numpy engine**. Expected: `RESULT: ALL ENGINES AGREE`.

For the 10⁶-bit file the mirror engine takes ~350 s:

```bash
python3 verify_sequence_independent.py \
        --file phase1_results/center_column_1000001.txt --numpy-prefix 20000
```

---

## 4. The main experiment set — ~57 seconds

```bash
python3 run_phase1.py --steps 200000 --max-period 10000 --subword-max 24
```

Writes `phase1_results/phase1_results.json`, `phase1_report.txt`,
`center_column_200001.txt`.

Produces: first 256 centre bits, prefix frequencies, subword complexity
`n = 1..24`, Berlekamp–Massey linear complexity profile, behaviour at
`2^k−1 / 2^k / 2^k+1`, the candidate-period mismatch scan, run lengths, block
entropy, autocorrelation, residue-class bias, and the left-region geometry.

Cheaper variant for a smoke test: `--steps 20000 --max-period 2000` (~4 s).

---

## 5. The 10⁶-bit anomaly follow-up — ~7 minutes

```bash
python3 run_anomaly_followup.py --steps 1000000
```

Writes `phase1_results/anomaly_followup.json`. Produces the independent-sample
re-tests of both anomalies, the factor counts at `n = 20, 24, 26, 28`, and the
period scan to `p = 20000`.

---

## 6. The bridging-lemma probes — ~4 minutes

```bash
python3 run_bridge_probe.py --steps 200000 --max-period 4096
```

Writes `phase1_results/bridge_probe.json`. Produces the locality probe, the
forward transfer profile, the two-sided `(a, b)` table, the backward depth
profile, and the period-band homogeneity check.

---

## 7. The MB1-loc refutation witnesses — ~5 min (200k) / ~25 min (1M)

```bash
python3 find_mb1loc_witnesses.py --steps 200000  --max-period 20000
python3 find_mb1loc_witnesses.py --steps 1000000 --max-period 20000   # the run quoted in the audit
```

Writes `phase1_results/mb1loc_witnesses.json`. Each witness is re-verified
inside the script against a freshly simulated diagram; `run_all_tests.py`
step 5 re-verifies them again in a separate process.

---

## 8. The factor-complexity bound — ~10 seconds

```bash
python3 verify_factor_bound.py --file phase1_results/center_column_1000001.txt \
        --lengths 20 24 26 28
```

Expected tail:

```
Lemma A  =>  T + p >= 998140   (witness length n = 28)
RESULT: all methods agree
```

See `FACTOR_COMPLEXITY_BOUND.md` for the lemma, its proof, and the three
algorithms.

---

## 9. Full reproduction, in order

```bash
python3 rule30_lab.py verify                                       #   ~3 s
python3 gen_center_column.py --steps 200000                        #   ~9 s
python3 gen_center_column.py --steps 1000000                       # ~334 s
python3 verify_sequence_independent.py --file phase1_results/center_column_200001.txt   #  ~13 s
python3 verify_sequence_independent.py --file phase1_results/center_column_1000001.txt  # ~370 s
python3 verify_factor_bound.py --file phase1_results/center_column_1000001.txt --lengths 20 24 26 28  # ~10 s
python3 run_phase1.py --steps 200000 --max-period 10000 --subword-max 24                #  ~57 s
python3 run_anomaly_followup.py --steps 1000000                                          # ~430 s
python3 run_bridge_probe.py --steps 200000 --max-period 4096                             # ~240 s
python3 find_mb1loc_witnesses.py --steps 1000000 --max-period 20000                      # ~25 min
python3 run_all_tests.py --with-1m                                                       # ~300 s
```

Total ≈ 45 minutes single-threaded.

---

## 10. Where each published number comes from

| Number | Document | Command | Artifact key |
|---|---|---|---|
| 21/21 verification checks | AUDIT §2.9 | `rule30_lab.py verify` | `phase1_results.json: verification_summary` |
| first 256 centre bits | AUDIT §5.1 | `run_phase1.py` | `first_256_bits` |
| freq(1) = 0.500362 at N=200001 | AUDIT §5.2 | `run_phase1.py` | `frequencies` |
| freq(1) = 0.500767 at N=10⁶ | AUDIT §5.2 | `run_anomaly_followup.py` | `global_frequency` |
| `p_obs(24) = 198744` | AUDIT §5.3 | `run_phase1.py` | `subword_complexity` |
| `p_obs(28) = 998140` | AUDIT §5.3, FACTOR §3 | `verify_factor_bound.py` | `factor_counts` |
| `L(200001) = 100001` | AUDIT §5.4 | `run_phase1.py` | `linear_complexity` |
| powers-of-two table + binomial p | AUDIT §5.6 | `run_phase1.py` | `powers_of_two`, `powers_of_two_summary` |
| period scan, largest first mismatch = 11 | AUDIT §5.7 | `run_phase1.py` | `period_scan` |
| `T > 979998` for `p ≤ 20000` | AUDIT §5.7 | `run_anomaly_followup.py` | `period_scan` |
| **`T + p ≥ 998140`** | AUDIT §5.8, FACTOR | `verify_factor_bound.py` | — (printed) |
| longest runs 19 / 21 | AUDIT §5.9 | `run_phase1.py` | `run_lengths` |
| missing factor → 458 occurrences | AUDIT §6.1 | `run_anomaly_followup.py` | `missing_factor_retest` |
| mod-3 z: +3.311 → +1.408 | AUDIT §6.2 | `run_anomaly_followup.py` | `mod3_discovery_sample`, `mod3_fresh_sample` |
| left-region τ = 16, slopes | AUDIT §7.3 | `run_phase1.py` | `left_region` |
| two-sided transfer table | AUDIT §7.4 | `run_bridge_probe.py` | `two_sided_table` |
| backward depth profile | AUDIT §7.4 | `run_bridge_probe.py` | `backward_transfer` |
| MB1-loc witnesses (8) | AUDIT §8 | `find_mb1loc_witnesses.py` | `mb1loc_witnesses.json` |

---

## 11. Known reproduction hazards

* **Working directory.** Scripts resolve `rule30_lab` and `phase1_results/`
  relative to the current directory. Run from inside the handoff folder.
* **`run_phase1.py` overwrites `center_column_200001.txt`** with an identical
  file; the digest in `SHA256SUMS.txt` still matches.
* **`gen_center_column.py` appends** to `SHA256SUMS.txt` rather than replacing
  it, so repeated runs add duplicate lines. `run_all_tests.py` takes the last
  entry per filename.
* **Memory.** `run_phase1.py` holds ~4000 rows of ~8000 cells for the
  left-region measurement (~30 MB). The 10⁶-bit generation peaks around
  250 MB (big integers of 2·10⁶ bits). Nothing needs more than ~1 GB.
* **Timings vary** by a factor of 2–3 across machines; big-integer performance
  is CPython-version-sensitive. No result depends on timing.
* **Non-determinism: none known.** If any number differs from the stored
  artifacts, that is a genuine discrepancy and should be reported, not
  attributed to environment.
