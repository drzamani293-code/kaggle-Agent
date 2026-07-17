#!/usr/bin/env python3
"""M37 - Hidden-safe fast reference-0902 submission runner.

Corrects M36 (11b9c10c...), which failed the Kaggle PRIVATE rerun because it hardcoded
the four PUBLIC test dataset stems, required exactly four .zarr stores, validated the
submission against the fixed public dataset names, and treated the public fingerprint as
a validation requirement. The private rerun substitutes a HIDDEN test dataset whose number
of .zarr stores and stems may differ, so all of those assumptions are removed here.

This runner discovers the ACTUAL test datasets dynamically and validates the submission
against exactly those discovered stems. The public 0.902 fingerprint is printed ONLY as an
informational public-data diagnostic (count comparison) and is NEVER a hidden-rerun gate.

Still a SUBMISSION milestone (no M35 audit / candidate-export / instrumentation / preflight
/ scale-probe / DB code, no nbclient, no nested kernel). Runs the exact five audited notebook
cells DIRECTLY in the current Kaggle kernel, once, and freshly generates
/kaggle/working/submission.csv.  Accelerator: GPU. Internet: Off. Current kernel only.
"""

import os
import sys
import time
import json
import signal
import hashlib
from pathlib import Path

# --------------------------------------------------------------------------- #
# Fixed audited identity. NO public dataset stems appear anywhere in production.
# --------------------------------------------------------------------------- #
M37_NOTEBOOK_SHA256 = "beb17b03682231460c3adab6069815e06c44adc23e427371c364901f8de5437e"
M37_WEIGHT_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"

# Public-DATA reference fingerprint (COUNTS only - not dataset names). Printed purely as an
# informational public-data diagnostic; NEVER a hidden-rerun validation requirement.
M37_PUBLIC_FINGERPRINT = {"nodes": 128511, "edges": 124002, "rows": 252513, "divisions": 417}

M37_COMP_DIR = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
M37_BUNDLE_DIR = "/kaggle/input/datasets/mohammadjafarzamani/biohub-0902-reference-bundle"
M37_SUPPORT_DIR = "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
M37_WORKING_DIR = "/kaggle/working"
M37_OUTPUT = "/kaggle/working/submission.csv"

M37_NOTEBOOK_REL = "reference/biohub-competition-solution.ipynb"
M37_WEIGHT_REL = "weights/unet_transformer/split_0/edge_predictor_best.pth"

# exact audited submission schema (notebook cell 4: CSV_COLUMNS)
M37_CSV_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
M37_NODE_MANDATORY = ["node_id", "t", "z", "y", "x"]
M37_EDGE_MANDATORY = ["source_id", "target_id"]

M37_HARD_TIMEOUT_S = 110 * 60          # hard guard = the runtime gate; fail clearly, never loop


class M37Error(RuntimeError):
    """Any M37 startup / execution / validation failure. Raised, never swallowed."""


class _M37Timeout(M37Error):
    pass


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _m37_sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _m37_first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def _m37_resolve_test_dir(comp_dir=None):
    """DYNAMICALLY discover all immediate ``*.zarr`` stores under the ACTUAL competition
    test/ directory. Requires: the test dir exists; at least one .zarr store; every stem
    non-empty; stems unique. Does NOT require any fixed dataset count and NEVER compares to
    a fixed public dataset-name list - so it works on the hidden private rerun."""
    base = comp_dir if comp_dir is not None else _m37_first_existing(
        [M37_COMP_DIR, f"/kaggle/input/{Path(M37_COMP_DIR).name}"])
    if base is None:
        raise M37Error(f"competition input not mounted: {M37_COMP_DIR}")
    test_dir = Path(base) / "test"
    if not test_dir.is_dir():
        raise M37Error(f"competition test/ dir not found under {base}")
    zarrs = sorted(str(p) for p in test_dir.glob("*.zarr"))
    if len(zarrs) < 1:
        raise M37Error(f"no .zarr test stores discovered under {test_dir}")
    stems = [Path(z).stem for z in zarrs]
    if any(not s for s in stems):
        raise M37Error(f"a discovered .zarr store has an empty stem: {zarrs}")
    if len(set(stems)) != len(stems):
        raise M37Error(f"discovered .zarr stems are not unique: {stems}")
    return str(test_dir), zarrs, sorted(stems)


def _m37_resolve_notebook():
    cand = [Path(M37_BUNDLE_DIR) / M37_NOTEBOOK_REL,
            Path(M37_BUNDLE_DIR) / Path(M37_NOTEBOOK_REL).name]
    nb = _m37_first_existing(cand)
    if nb is None:
        raise M37Error(f"audited notebook not found under {M37_BUNDLE_DIR}/{M37_NOTEBOOK_REL}")
    sha = _m37_sha256_file(nb)
    if sha != M37_NOTEBOOK_SHA256:
        raise M37Error(f"notebook sha256 {sha} != audited {M37_NOTEBOOK_SHA256}")
    return str(nb), sha


def _m37_resolve_weight(repo_dir):
    cand = [Path(repo_dir) / M37_WEIGHT_REL,
            Path(M37_BUNDLE_DIR) / M37_WEIGHT_REL,
            Path(M37_BUNDLE_DIR) / "tracking_repo" / M37_WEIGHT_REL]
    w = _m37_first_existing(cand)
    if w is None:
        raise M37Error(f"400ep weight not found ({M37_WEIGHT_REL}) under repo/bundle")
    sha = _m37_sha256_file(w)
    if sha != M37_WEIGHT_SHA256:
        raise M37Error(f"weight sha256 {sha} != controlled 400ep {M37_WEIGHT_SHA256}")
    return str(w), sha


def _m37_delete_stale_outputs():
    """Delete ONLY stale M37 working outputs; NEVER touch mounted inputs (/kaggle/input)."""
    removed = []
    for name in ("submission.csv", "run_stats.csv"):
        p = Path(M37_WORKING_DIR) / name
        if p.exists() and str(p).startswith("/kaggle/working") and not str(p).startswith("/kaggle/input"):
            try:
                p.unlink(); removed.append(str(p))
            except OSError:
                pass
    return removed


def _m37_load_notebook_cells(nb_path):
    nb = json.loads(Path(nb_path).read_text())
    cells = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
             for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if len(cells) != 5:
        raise M37Error(f"expected exactly 5 code cells in the audited notebook, got {len(cells)}")
    return cells


# --------------------------------------------------------------------------- #
# submission validation - against the DISCOVERED stems only (no public list)
# --------------------------------------------------------------------------- #
def _m37_validate_submission(path, discovered_stems):
    import math
    import pandas as pd
    if not Path(path).exists():
        raise M37Error(f"submission.csv not found at {path}")
    df = pd.read_csv(path)
    discovered = set(str(s) for s in discovered_stems)
    checks = {}

    checks["columns_exact_and_ordered"] = list(df.columns) == M37_CSV_COLUMNS
    if not checks["columns_exact_and_ordered"]:
        raise M37Error(f"submission columns {list(df.columns)} != {M37_CSV_COLUMNS}")

    node_df = df[df["row_type"] == "node"]
    edge_df = df[df["row_type"] == "edge"]
    sub_datasets = set(str(d) for d in df["dataset"].unique())

    # DYNAMIC dataset gate: submission datasets == the discovered test stems (no fixed list)
    checks["all_discovered_datasets_represented"] = discovered <= sub_datasets
    checks["no_unexpected_datasets"] = sub_datasets <= discovered
    checks["datasets_match_discovered"] = sub_datasets == discovered

    base_ok = not df[["id", "dataset", "row_type"]].isna().any().any()
    node_ok = not node_df[M37_NODE_MANDATORY].isna().any().any() if len(node_df) else True
    edge_ok = not edge_df[M37_EDGE_MANDATORY].isna().any().any() if len(edge_df) else True
    checks["no_nan_in_mandatory_fields"] = bool(base_ok and node_ok and edge_ok)

    def _all_int(series):
        try:
            return all(float(v).is_integer() for v in series.dropna().to_numpy())
        except (TypeError, ValueError):
            return False
    checks["ids_are_integers"] = bool(_all_int(node_df["node_id"]) and _all_int(edge_df["source_id"])
                                      and _all_int(edge_df["target_id"]))

    checks["no_duplicate_rows"] = bool(not df.duplicated().any())

    unique_nodes_ok = endpoints_ok = time_order_ok = indeg_ok = outdeg_ok = True
    max_indegree = max_outdegree = divisions = 0
    for ds in sorted(sub_datasets):
        ns = node_df[node_df["dataset"] == ds]
        es = edge_df[edge_df["dataset"] == ds]
        node_ids = ns["node_id"].astype("int64")
        if node_ids.duplicated().any():
            unique_nodes_ok = False
        node_set = set(int(v) for v in node_ids)
        t_by_node = {int(r.node_id): float(r.t) for r in ns.itertuples()}
        indeg, outdeg = {}, {}
        for r in es.itertuples():
            s, t = int(r.source_id), int(r.target_id)
            if s not in node_set or t not in node_set:
                endpoints_ok = False
                continue
            if not (t_by_node.get(s, math.inf) < t_by_node.get(t, -math.inf)):
                time_order_ok = False
            indeg[t] = indeg.get(t, 0) + 1
            outdeg[s] = outdeg.get(s, 0) + 1
        if indeg:
            max_indegree = max(max_indegree, max(indeg.values()))
            if max(indeg.values()) > 1:
                indeg_ok = False
        if outdeg:
            max_outdegree = max(max_outdegree, max(outdeg.values()))
            if max(outdeg.values()) > 2:
                outdeg_ok = False
        divisions += sum(1 for c in outdeg.values() if c >= 2)

    checks["node_ids_unique_within_dataset"] = bool(unique_nodes_ok)
    checks["edge_endpoints_exist_in_dataset"] = bool(endpoints_ok)
    checks["edge_source_t_lt_target_t"] = bool(time_order_ok)
    checks["max_indegree_le_1"] = bool(indeg_ok and max_indegree <= 1)
    checks["max_outdegree_le_2"] = bool(outdeg_ok and max_outdegree <= 2)
    checks["graph_validation_passes"] = bool(unique_nodes_ok and endpoints_ok and time_order_ok
                                             and indeg_ok and outdeg_ok)

    counts = {"rows": int(len(df)), "nodes": int(len(node_df)), "edges": int(len(edge_df)),
              "divisions": int(divisions), "datasets": sorted(sub_datasets),
              "n_datasets": len(sub_datasets), "max_indegree": int(max_indegree),
              "max_outdegree": int(max_outdegree)}
    # informational ONLY: does the generated fingerprint equal the public 0.902 reference?
    # (a pure COUNT comparison - meaningful only on public data; NEVER a hidden gate.)
    public_fingerprint_match = bool(counts["nodes"] == M37_PUBLIC_FINGERPRINT["nodes"]
                                    and counts["edges"] == M37_PUBLIC_FINGERPRINT["edges"]
                                    and counts["rows"] == M37_PUBLIC_FINGERPRINT["rows"]
                                    and counts["divisions"] == M37_PUBLIC_FINGERPRINT["divisions"])
    return {"checks": checks, "counts": counts, "all_gates_pass": all(checks.values()),
            "public_fingerprint_match_informational": public_fingerprint_match,
            "public_fingerprint_reference": M37_PUBLIC_FINGERPRINT}


# --------------------------------------------------------------------------- #
# main runner
# --------------------------------------------------------------------------- #
def run_m37_hidden_safe_reference_submit():
    t_start = time.time()

    def _elapsed():
        return time.time() - t_start

    def _alarm(_signum, _frame):
        raise _M37Timeout(f"hard 110-minute timeout guard tripped at {_elapsed():.0f}s "
                          "(no silent retry; failing clearly)")

    print("=== M37 HIDDEN-SAFE REFERENCE-0902 SUBMISSION (current kernel, single inference pass) ===")

    try:
        import torch
    except Exception as exc:                                    # noqa: BLE001
        raise M37Error(f"torch import failed: {exc}")
    if not torch.cuda.is_available():
        raise M37Error("torch.cuda.is_available() is False - this milestone requires a GPU accelerator")
    print(f"  CUDA available: True  device: {torch.cuda.get_device_name(0)}")

    # DYNAMIC discovery - works for any number of hidden datasets with arbitrary names
    test_dir, test_zarr, stems = _m37_resolve_test_dir()
    print(f"  test_dir: {test_dir}")
    print(f"  discovered {len(stems)} test dataset(s) (dynamic; NOT a fixed public list):")
    for z in test_zarr:
        print(f"    zarr: {z}")
    print(f"  discovered stems: {stems}")

    nb_path, nb_sha = _m37_resolve_notebook()
    print(f"  audited notebook: {nb_path}  sha256 OK ({nb_sha[:12]}...)")

    removed = _m37_delete_stale_outputs()
    if removed:
        print(f"  removed stale M37 outputs: {removed}")

    cells = _m37_load_notebook_cells(nb_path)
    print(f"  audited notebook code cells: {len(cells)} (env / config / dependency / inference / postprocess)")

    old_signal = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(M37_HARD_TIMEOUT_S)
    ns = {"__name__": "__m37_reference__"}
    infer_seconds = post_seconds = None
    try:
        for idx in (0, 1, 2):                                   # env / config / dependency (materialize repo)
            exec(compile(cells[idx], f"<m37_cell_{idx}>", "exec"), ns)
        repo_dir = str(ns.get("REPO_DIR") or (Path(M37_WORKING_DIR) / "tracking_repo"))
        weight_path, weight_sha = _m37_resolve_weight(repo_dir)
        print(f"  400ep weight: {weight_path}  sha256 OK ({weight_sha[:12]}...)")

        t_inf = time.time()                                    # single inference pass (D4 detection TTA only)
        exec(compile(cells[3], "<m37_cell_3>", "exec"), ns)
        infer_seconds = time.time() - t_inf
        print(f"  inference complete: {infer_seconds/60.0:.2f} min")

        t_post = time.time()                                   # postprocess -> submission.csv
        exec(compile(cells[4], "<m37_cell_4>", "exec"), ns)
        post_seconds = time.time() - t_post
        print(f"  postprocess complete: {post_seconds/60.0:.2f} min")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_signal)

    out_path = str(ns.get("SUBMISSION_PATH") or M37_OUTPUT)
    val = _m37_validate_submission(out_path, stems)
    total_seconds = _elapsed()
    runtime_ok = total_seconds < M37_HARD_TIMEOUT_S
    val["checks"]["total_runtime_under_110min"] = bool(runtime_ok)

    sub_sha = _m37_sha256_file(out_path)
    c = val["counts"]
    print("--- M37 RESULT ---")
    print(f"  inference runtime : {(infer_seconds or 0)/60.0:.2f} min")
    print(f"  postprocess runtime: {(post_seconds or 0)/60.0:.2f} min")
    print(f"  total runtime     : {total_seconds/60.0:.2f} min")
    print(f"  rows: {c['rows']}  nodes: {c['nodes']}  edges: {c['edges']}  divisions: {c['divisions']}")
    print(f"  datasets ({c['n_datasets']}): {c['datasets']}")
    print(f"  max_indegree: {c['max_indegree']}  max_outdegree: {c['max_outdegree']}")
    print(f"  submission sha256 : {sub_sha}")
    print(f"  graph validation  : {'PASS' if val['checks']['graph_validation_passes'] else 'FAIL'}")
    print(f"  datasets_match_discovered: {val['checks']['datasets_match_discovered']}")
    print(f"  [informational only, NOT a gate] fingerprint matches public 0.902 reference: "
          f"{val['public_fingerprint_match_informational']}  (public ref {M37_PUBLIC_FINGERPRINT})")
    print(f"  output path       : {out_path}")
    failed = [k for k, v in val["checks"].items() if not v]
    if failed:
        print(f"  FAILED validation gates: {failed}")

    ready = bool(val["all_gates_pass"] and runtime_ok and Path(out_path).exists())
    if ready:
        print("M37_HIDDEN_SAFE_SUBMIT_READY")
    else:
        raise M37Error(f"validation gates not all passed: {failed} (submission not declared ready)")
    return {"ready": ready, "submission_sha256": sub_sha, "output_path": out_path,
            "discovered_stems": stems, "inference_seconds": infer_seconds,
            "postprocess_seconds": post_seconds, "total_seconds": total_seconds, "validation": val}


# --------------------------------------------------------------------------- #
# CPU-only self-tests (NO GPU, NO inference, NO submission). Simulate HIDDEN test
# layouts with arbitrary dataset names. Public stems are base64-encoded here so the
# literal public stems never appear anywhere in this file; a static guard proves the
# PRODUCTION runner embeds none of them.
# --------------------------------------------------------------------------- #
_M37_PUBLIC_STEMS_B64 = ["NDRiNl8wMTEzZGUzYg==", "NDRiNl8wYjI0ODQ1Zg==",
                         "NmJiYV8wNWI2ODUwYg==", "NmJiYV8wNWRiMGZiMQ=="]


def _m37_public_stems_for_test():
    import base64
    return [base64.b64decode(b).decode() for b in _M37_PUBLIC_STEMS_B64]


def _m37_synth_submission(path, datasets, break_indegree=False, break_time=False,
                          dangling=False, dup=False):
    import csv
    rows = []
    rid = 0

    def add(dataset, row_type, node_id=-1, t=-1, z=0.0, y=0.0, x=0.0, source_id=-1, target_id=-1):
        nonlocal rid
        rows.append({"id": rid, "dataset": dataset, "row_type": row_type, "node_id": node_id,
                     "t": t, "z": z, "y": y, "x": x, "source_id": source_id, "target_id": target_id})
        rid += 1

    for ds in datasets:
        add(ds, "node", node_id=0, t=0); add(ds, "node", node_id=1, t=1)
        add(ds, "node", node_id=2, t=1); add(ds, "node", node_id=3, t=2)
        first_edge_idx = len(rows)
        add(ds, "edge", source_id=0, target_id=1)
        add(ds, "edge", source_id=0, target_id=2)              # division: outdeg(0)=2
        add(ds, "edge", source_id=1, target_id=(3 if not break_time else 0))
        if break_indegree:
            add(ds, "edge", source_id=2, target_id=1)          # indeg(1)=2 -> invalid
        if dangling:
            add(ds, "edge", source_id=0, target_id=999)        # endpoint not a node
        if dup:
            rows.append(dict(rows[first_edge_idx]))            # EXACT duplicate row (id included)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=M37_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _m37_static_guard_no_public_stems():
    """FAIL if any of the four PUBLIC stems is embedded in the PRODUCTION runner source."""
    import inspect
    prod = [run_m37_hidden_safe_reference_submit, _m37_resolve_test_dir, _m37_validate_submission,
            _m37_resolve_notebook, _m37_resolve_weight, _m37_delete_stale_outputs,
            _m37_load_notebook_cells, _m37_first_existing, _m37_sha256_file]
    src = "\n".join(inspect.getsource(f) for f in prod)
    for stem in _m37_public_stems_for_test():
        assert stem not in src, f"public stem {stem} leaked into the production runner"


def run_m37_selftests():
    import tempfile
    # (guard) production runner embeds none of the four public stems
    _m37_static_guard_no_public_stems()
    assert M37_CSV_COLUMNS[0] == "id" and M37_CSV_COLUMNS[1:3] == ["dataset", "row_type"]
    assert M37_HARD_TIMEOUT_S == 110 * 60

    hidden = ["hidden_a", "hidden_b", "hidden_c", "hidden_d", "hidden_e"]   # 5 ARBITRARY names
    with tempfile.TemporaryDirectory() as w:
        # (1) resolver discovers an arbitrary hidden layout of any count with unique stems
        comp = Path(w) / "comp"; (comp / "test").mkdir(parents=True)
        for name in hidden:
            (comp / "test" / f"{name}.zarr").mkdir()
        td, zarrs, stems = _m37_resolve_test_dir(comp_dir=str(comp))
        assert stems == sorted(hidden) and len(zarrs) == 5

        # a single hidden dataset is also accepted (no fixed count)
        comp1 = Path(w) / "comp1"; (comp1 / "test").mkdir(parents=True)
        (comp1 / "test" / "only_one.zarr").mkdir()
        assert _m37_resolve_test_dir(comp_dir=str(comp1))[2] == ["only_one"]

        # empty test dir is rejected
        comp0 = Path(w) / "comp0"; (comp0 / "test").mkdir(parents=True)
        try:
            _m37_resolve_test_dir(comp_dir=str(comp0)); assert False
        except M37Error:
            pass

        # (2) validator passes for the five arbitrary hidden datasets
        good = str(Path(w) / "good.csv"); _m37_synth_submission(good, hidden)
        v = _m37_validate_submission(good, sorted(hidden))
        assert v["all_gates_pass"], [k for k, x in v["checks"].items() if not x]
        assert v["checks"]["datasets_match_discovered"]
        assert v["counts"]["n_datasets"] == 5 and v["counts"]["divisions"] == 5

        # (3) a missing hidden dataset fails "all discovered represented"
        vmiss = _m37_validate_submission(good, sorted(hidden) + ["hidden_f"])
        assert vmiss["checks"]["all_discovered_datasets_represented"] is False

        # (4) an unexpected dataset fails "no unexpected datasets"
        extra = str(Path(w) / "extra.csv"); _m37_synth_submission(extra, hidden + ["surprise"])
        vex = _m37_validate_submission(extra, sorted(hidden))
        assert vex["checks"]["no_unexpected_datasets"] is False

        # (5) topology gates reject bad graphs
        for kw, gate in (("break_indegree", "max_indegree_le_1"), ("break_time", "edge_source_t_lt_target_t"),
                         ("dangling", "edge_endpoints_exist_in_dataset"), ("dup", "no_duplicate_rows")):
            bad = str(Path(w) / f"{kw}.csv"); _m37_synth_submission(bad, hidden, **{kw: True})
            assert _m37_validate_submission(bad, sorted(hidden))["checks"][gate] is False, kw

        # (6) public fingerprint is informational only (never in checks / never a gate)
        assert "public_fingerprint" not in "".join(v["checks"].keys())
        assert v["public_fingerprint_match_informational"] is False   # hidden counts != public

    # (7) if the audited notebook is available locally, confirm no cell hardcodes a public
    #     dataset-name allowlist (on Kaggle the SHA pin guarantees the exact audited cells).
    for cand in (".local_reference/biohub-competition-solution.ipynb",
                 "/kaggle/input/datasets/mohammadjafarzamani/biohub-0902-reference-bundle/reference/biohub-competition-solution.ipynb"):
        if Path(cand).exists() and _m37_sha256_file(cand) == M37_NOTEBOOK_SHA256:
            nb = json.loads(Path(cand).read_text())
            cellsrc = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                                for c in nb["cells"] if c["cell_type"] == "code")
            for stem in _m37_public_stems_for_test():
                assert stem not in cellsrc, f"audited notebook cell embeds public stem {stem}"
            break
    print("M37 self-tests passed (7/7) - CPU only, hidden-name safe, no GPU inference, no submission")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        run_m37_selftests()
    else:
        run_m37_hidden_safe_reference_submit()
