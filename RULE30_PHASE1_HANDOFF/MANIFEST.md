# File Manifest — RULE30_PHASE1_HANDOFF

Generated 2026-08-04 14:38:17 UTC.


**Phase 1 git commit:** `b42ce46d480d750c4e779f48f415dde6f9b292c9`  
**Commit at manifest generation:** `b42ce46d480d750c4e779f48f415dde6f9b292c9`  
**Repository:** `drzamani293-code/kaggle-agent`  
**Branch:** `claude/rule30-prize-problem-1-qr1omb`

Environment that produced the artifacts: Python 3.11.15, numpy 2.4.6, 
Linux-6.18.5-fc-v18-x86_64-with-glibc2.39.

Every file in the package is listed. SHA-256 is over the exact bytes shipped.


## Documents

| File | Lines | Bytes | SHA-256 | Description |
|---|---:|---:|---|---|
| `README.md` | 134 | 6871 | `c4d7bd424190425bbf8b62b85847659b...` | entry point and orientation |
| `MANIFEST.md` | — | — | *(self-referential: a file cannot contain its own digest)* | this manifest |
| `REPRODUCE_ALL.md` | 253 | 9806 | `3a64a9e69c8598b468d7d11cdf48920d...` | exact commands from a clean machine |
| `CLAIM_LEDGER.md` | 114 | 19067 | `310ebd25b0e7dc911a30c2794aa2a9f9...` | one row per substantive claim |
| `PROOF_DEPENDENCY_GRAPH.md` | 312 | 12426 | `a71db5f1a69d7b81ad2005d8f0cf2a20...` | logical dependency graphs for the four load-bearing results |
| `PHASE1_AUDIT.md` | 697 | 31198 | `b76eefaefac3943fa589d629dab72baf...` | the experimental audit (settings, results, anomalies, warnings) |
| `WIDTH2_PROOF_RECONSTRUCTION.md` | 623 | 28015 | `92821ef44ed4fc32ee0ac2adab65a88d...` | expository proof of W2/W2' plus the gap analysis |
| `FORMAL_PROOF_AT_MOST_ONE_COLUMN.md` | 421 | 17917 | `e3afa1d56204275ec4212d593473c513...` | fully quantified proof of Theorem W2' |
| `FACTOR_COMPLEXITY_BOUND.md` | 270 | 10026 | `1646a9d1e9e48213840af563094518cf...` | the T+p >= 998140 bound: lemma, proof, parameters, recomputation |
| `CONJECTURE_REGISTRY.md` | 353 | 16270 | `29140d4280316ecced380bba17aad605...` | C1-C9 with evidence and falsification tests |
| `research_notes.md` | 106 | 5065 | `360bc9fc85a06b1af03ce4c7f88d3f6a...` | session log including mistakes made and corrected |

## Library and generators

| File | Lines | Bytes | SHA-256 | Description |
|---|---:|---:|---|---|
| `rule30_lab.py` | 1183 | 43290 | `17ce530b700991a497d18f7993700e86...` | 4 engines, 21-check verification suite, all statistics, bridging probes |
| `gen_center_column.py` | 68 | 2270 | `8aa1f424515b4183320b24fa36a35ba3...` | canonical sequence generator, writes SHA-256 |

## Verification scripts

| File | Lines | Bytes | SHA-256 | Description |
|---|---:|---:|---|---|
| `verify_sequence_independent.py` | 101 | 3413 | `63b5cb8d4c49127cb11f4716c54c53bd...` | regenerates a sequence via mirror engine (rule 86) + numpy |
| `verify_factor_bound.py` | 152 | 5452 | `1017efa273b41ae6d5b40f6ad50f3bb9...` | three independent factor-complexity counts |
| `run_all_tests.py` | 210 | 8312 | `0d39b2283f2e1eb6897a1b638dc0cb60...` | whole battery from a clean process; exits non-zero on failure |

## Experiment drivers

| File | Lines | Bytes | SHA-256 | Description |
|---|---:|---:|---|---|
| `run_phase1.py` | 314 | 12602 | `3c9a4dcf95fb372a656d011d4ecbaa08...` | main experiment driver |
| `run_anomaly_followup.py` | 147 | 5459 | `771f5ca327e4e370ee71d0b71749799f...` | 10^6-bit independent-sample anomaly re-tests |
| `run_bridge_probe.py` | 137 | 5678 | `2942fe610b57c689924fe070bacd583c...` | transfer-profile experiments for the bridging lemma |
| `find_mb1loc_witnesses.py` | 91 | 3336 | `92f9ef27c99f4ec0357a2f84ecc4b73b...` | MB1-loc counterexample search with re-verification |

## Environment

| File | Lines | Bytes | SHA-256 | Description |
|---|---:|---:|---|---|
| `requirements.txt` | 560 | 10925 | `c9d2a7e1d13b66e596935c5be0ac76a8...` | python dependencies (numpy only) |

## Result artifacts (`phase1_results/`)

| File | Bytes | SHA-256 | Description |
|---|---:|---|---|
| `SHA256SUMS.txt` | 183 | `faeb20f22815fb168bcd850bfeff656b...` | digests of the sequence files |
| `anomaly_followup.json` | 10562 | `b8026fe2dbc1aef5106a9cfdeaf12349...` | 10^6-bit re-tests, factor counts, period scan to p=20000 |
| `anomaly_followup.log` | 1555 | `1dc6eb8237cca7beddae66e8266ce518...` | stdout of run_anomaly_followup.py |
| `bridge_probe.json` | 10953 | `c547b61e9eb5d9de62003f617e3e6fa4...` | locality probe, transfer profiles, two-sided table, period bands |
| `bridge_probe.log` | 1741 | `6d25b9a52622504ea529a186d2a4f8b5...` | stdout of run_bridge_probe.py |
| `center_column_1000001.txt` | 1000002 | `0bb02e4ed6c3d80bd832eed0b6991cc9...` | centre column c(0..1000000), one line of ASCII 0/1 |
| `center_column_200001.txt` | 200002 | `cf018b09db932412d5f42c3d03076181...` | centre column c(0..200000), one line of ASCII 0/1 |
| `gen_1M.log` | 348 | `657d166878c085ff575dd0ffffe21b2d...` | stdout of the 10^6-bit generation |
| `mb1loc.log` | 1925 | `2a31b631e564be5e11684f9bc8212f15...` | stdout of the earlier 200001-bit witness search (superseded, retained) |
| `mb1loc_1M.log` | 2093 | `ef6b39c60d5923f998c47c90e229937b...` | stdout of the 10^6-bit witness search (quoted in AUDIT section 8) |
| `mb1loc_witnesses.json` | 3155 | `cd37adcdb3cf0e7bd1a49a366279968f...` | MB1-loc witness table (10^6-bit run) -- THE REFUTATION DATA |
| `phase1_report.txt` | 11913 | `4160534b590d5bb4c49bebe75221c8d1...` | human-readable rendering of phase1_results.json |
| `phase1_results.json` | 40096 | `2d43e42420f18b9beae13922208ffb0a...` | main experiment output (all Phase 1 B results) |
| `phase1_run.log` | 320 | `5072f20cd0c0d482b63ab72b65e46ed2...` | stdout of the run_phase1.py run |
| `verify_factor_1M.log` | 788 | `f0303c6cb1768e31ae6c71895f570fd2...` | three-algorithm factor counts on the 10^6-bit file |
| `verify_seq_1M.log` | 475 | `7e6ae5bcc35045d4861b522358947427...` | mirror-engine + numpy regeneration of the 10^6-bit file |
| `verify_seq_200001.log` | 470 | `89658483d1eab35555985d71120d06a6...` | mirror-engine + numpy regeneration of the 200001-bit file |

## Totals

* files hashed: **37**
* plus `MANIFEST.md` itself: **38 files total**
* bytes: **1543979** (1.5 MB)

## Full digests of the sequence files

```
cf018b09db932412d5f42c3d03076181a34d806173fbee749e30c871fe5d994a  center_column_200001.txt
0bb02e4ed6c3d80bd832eed0b6991cc99f6de01a5f32fa795281d3efb2e9104e  center_column_1000001.txt
```

## Files deliberately NOT included

* Nothing from the host repository (`kaggle-Agent`) — it is unrelated to
  this work and none of it was modified.
* No virtualenv, no `__pycache__`, no editor state.
* No superseded *results* were deleted: `mb1loc.log` (the earlier 200001-bit
  witness search, superseded by `mb1loc_1M.log`) is retained deliberately.
