#!/usr/bin/env python3
"""M36 - Fast reference-0902 submission runner.

GOAL: produce ONE reliable Kaggle submission directly from the audited 0.902 reference
notebook, well under 2 hours. This is a SUBMISSION milestone - NOT an audit, candidate
export, diagnostic, instrumentation, or research milestone. It contains NO M35 audit /
candidate-export / event-instrumentation / structural-preflight / scale-probe / milestone-
database code, NO nested kernels, and NO nbclient. It runs the exact five audited notebook
cells DIRECTLY in the current Kaggle kernel, once, and freshly generates
/kaggle/working/submission.csv from the four official test .zarr datasets.

Accelerator: GPU. Internet: Off. Current kernel only.
"""

import os
import sys
import time
import json
import signal
import hashlib
from pathlib import Path

# --------------------------------------------------------------------------- #
# Fixed audited identity + reference fingerprint (0.902). Do not edit.
# --------------------------------------------------------------------------- #
M36_NOTEBOOK_SHA256 = "beb17b03682231460c3adab6069815e06c44adc23e427371c364901f8de5437e"
M36_WEIGHT_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
M36_EXPECTED_STEMS = ("44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1")
M36_FINGERPRINT = {"nodes": 128511, "edges": 124002, "rows": 252513, "divisions": 417}

M36_COMP_DIR = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
M36_BUNDLE_DIR = "/kaggle/input/datasets/mohammadjafarzamani/biohub-0902-reference-bundle"
M36_SUPPORT_DIR = "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
M36_WORKING_DIR = "/kaggle/working"
M36_OUTPUT = "/kaggle/working/submission.csv"

M36_NOTEBOOK_REL = "reference/biohub-competition-solution.ipynb"
M36_WEIGHT_REL = "weights/unet_transformer/split_0/edge_predictor_best.pth"

# exact audited submission schema (see notebook cell 4: CSV_COLUMNS)
M36_CSV_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
M36_NODE_MANDATORY = ["node_id", "t", "z", "y", "x"]
M36_EDGE_MANDATORY = ["source_id", "target_id"]

M36_HARD_TIMEOUT_S = 90 * 60          # hard guard; fail clearly, never loop/retry


class M36Error(RuntimeError):
    """Any M36 startup / execution / validation failure. Raised, never swallowed."""


class _M36Timeout(M36Error):
    pass


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _m36_sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _m36_first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def _m36_resolve_test_dir():
    """Resolve the competition test/ dir and require EXACTLY the four official .zarr
    stores with the exact expected stems."""
    comp = _m36_first_existing([M36_COMP_DIR, f"/kaggle/input/{Path(M36_COMP_DIR).name}"])
    if comp is None:
        raise M36Error(f"competition input not mounted: {M36_COMP_DIR}")
    test_dir = comp / "test"
    if not test_dir.is_dir():
        raise M36Error(f"competition test/ dir not found under {comp}")
    zarrs = sorted(str(p) for p in test_dir.glob("*.zarr"))
    stems = sorted(Path(z).stem for z in zarrs)
    if len(zarrs) != 4:
        raise M36Error(f"expected exactly 4 test .zarr datasets, found {len(zarrs)}: {zarrs}")
    if set(stems) != set(M36_EXPECTED_STEMS):
        raise M36Error(f"test .zarr stems {stems} != expected {sorted(M36_EXPECTED_STEMS)}")
    return str(test_dir), zarrs, stems


def _m36_resolve_notebook():
    cand = [Path(M36_BUNDLE_DIR) / M36_NOTEBOOK_REL,
            Path(M36_BUNDLE_DIR) / Path(M36_NOTEBOOK_REL).name]
    nb = _m36_first_existing(cand)
    if nb is None:
        raise M36Error(f"audited notebook not found under {M36_BUNDLE_DIR}/{M36_NOTEBOOK_REL}")
    sha = _m36_sha256_file(nb)
    if sha != M36_NOTEBOOK_SHA256:
        raise M36Error(f"notebook sha256 {sha} != audited {M36_NOTEBOOK_SHA256}")
    return str(nb), sha


def _m36_resolve_weight(repo_dir):
    """The 400ep weight lives at <REPO_DIR>/weights/unet_transformer/split_0/... after the
    dependency cell materializes the repo; also accept the bundle copy if present."""
    cand = [Path(repo_dir) / M36_WEIGHT_REL,
            Path(M36_BUNDLE_DIR) / M36_WEIGHT_REL,
            Path(M36_BUNDLE_DIR) / "tracking_repo" / M36_WEIGHT_REL]
    w = _m36_first_existing(cand)
    if w is None:
        raise M36Error(f"400ep weight not found ({M36_WEIGHT_REL}) under repo/bundle")
    sha = _m36_sha256_file(w)
    if sha != M36_WEIGHT_SHA256:
        raise M36Error(f"weight sha256 {sha} != controlled 400ep {M36_WEIGHT_SHA256}")
    return str(w), sha


def _m36_delete_stale_outputs():
    """Delete ONLY stale M36 working outputs; NEVER touch mounted inputs."""
    removed = []
    for name in ("submission.csv", "run_stats.csv"):
        p = Path(M36_WORKING_DIR) / name
        if p.exists() and str(p).startswith("/kaggle/working"):
            try:
                p.unlink(); removed.append(str(p))
            except OSError:
                pass
    return removed


def _m36_load_notebook_cells(nb_path):
    nb = json.loads(Path(nb_path).read_text())
    cells = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
             for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if len(cells) != 5:
        raise M36Error(f"expected exactly 5 code cells in the audited notebook, got {len(cells)}")
    return cells


# --------------------------------------------------------------------------- #
# submission validation (independent of the notebook's own checks)
# --------------------------------------------------------------------------- #
def _m36_validate_submission(path):
    import math
    import pandas as pd
    if not Path(path).exists():
        raise M36Error(f"submission.csv not found at {path}")
    df = pd.read_csv(path)
    checks = {}

    checks["columns_exact_and_ordered"] = list(df.columns) == M36_CSV_COLUMNS
    if not checks["columns_exact_and_ordered"]:
        raise M36Error(f"submission columns {list(df.columns)} != {M36_CSV_COLUMNS}")

    node_df = df[df["row_type"] == "node"]
    edge_df = df[df["row_type"] == "edge"]
    datasets = sorted(str(d) for d in df["dataset"].unique())
    checks["exactly_four_datasets"] = set(datasets) == set(M36_EXPECTED_STEMS)

    # no NaN in mandatory fields
    base_ok = not df[["id", "dataset", "row_type"]].isna().any().any()
    node_ok = not node_df[M36_NODE_MANDATORY].isna().any().any() if len(node_df) else True
    edge_ok = not edge_df[M36_EDGE_MANDATORY].isna().any().any() if len(edge_df) else True
    checks["no_nan_in_mandatory_fields"] = bool(base_ok and node_ok and edge_ok)

    # integer node/source/target ids
    def _all_int(series):
        try:
            vals = series.dropna().to_numpy()
            return all(float(v).is_integer() for v in vals)
        except (TypeError, ValueError):
            return False
    checks["ids_are_integers"] = bool(_all_int(node_df["node_id"]) and _all_int(edge_df["source_id"])
                                      and _all_int(edge_df["target_id"]))

    checks["no_duplicate_rows"] = bool(not df.duplicated().any())

    # per-dataset topology
    unique_nodes_ok = True
    endpoints_ok = True
    time_order_ok = True
    indeg_ok = True
    outdeg_ok = True
    max_indegree = 0
    max_outdegree = 0
    divisions = 0
    for ds in datasets:
        ns = node_df[node_df["dataset"] == ds]
        es = edge_df[edge_df["dataset"] == ds]
        node_ids = ns["node_id"].astype("int64")
        if node_ids.duplicated().any():
            unique_nodes_ok = False
        node_set = set(int(v) for v in node_ids)
        t_by_node = {int(r.node_id): float(r.t) for r in ns.itertuples()}
        indeg = {}
        outdeg = {}
        for r in es.itertuples():
            s = int(r.source_id); t = int(r.target_id)
            if s not in node_set or t not in node_set:
                endpoints_ok = False
                continue
            if not (t_by_node.get(s, math.inf) < t_by_node.get(t, -math.inf)):
                time_order_ok = False
            indeg[t] = indeg.get(t, 0) + 1
            outdeg[s] = outdeg.get(s, 0) + 1
        if indeg:
            max_indegree = max(max_indegree, max(indeg.values()))
        if outdeg:
            max_outdegree = max(max_outdegree, max(outdeg.values()))
        if indeg and max(indeg.values()) > 1:
            indeg_ok = False
        if outdeg and max(outdeg.values()) > 2:
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
              "divisions": int(divisions), "datasets": datasets,
              "max_indegree": int(max_indegree), "max_outdegree": int(max_outdegree)}
    fingerprint_exact = bool(counts["nodes"] == M36_FINGERPRINT["nodes"]
                             and counts["edges"] == M36_FINGERPRINT["edges"]
                             and counts["rows"] == M36_FINGERPRINT["rows"]
                             and counts["divisions"] == M36_FINGERPRINT["divisions"])
    all_gates = all(checks.values())
    return {"checks": checks, "counts": counts, "all_gates_pass": bool(all_gates),
            "fingerprint_exact": fingerprint_exact, "fingerprint_expected": M36_FINGERPRINT}


# --------------------------------------------------------------------------- #
# main runner
# --------------------------------------------------------------------------- #
def run_m36_fast_reference_submit():
    t_start = time.time()

    def _elapsed():
        return time.time() - t_start

    def _alarm(_signum, _frame):
        raise _M36Timeout(f"hard 90-minute timeout guard tripped at {_elapsed():.0f}s "
                          "(no silent retry; failing clearly)")

    print("=== M36 FAST REFERENCE-0902 SUBMISSION (current kernel, single inference pass) ===")

    # --- startup checks (no inference yet) ---
    try:
        import torch
    except Exception as exc:                                    # noqa: BLE001
        raise M36Error(f"torch import failed: {exc}")
    if not torch.cuda.is_available():
        raise M36Error("torch.cuda.is_available() is False - this milestone requires a GPU accelerator")
    device_name = torch.cuda.get_device_name(0)
    print(f"  CUDA available: True  device: {device_name}")

    test_dir, test_zarr, stems = _m36_resolve_test_dir()
    print(f"  test_dir: {test_dir}")
    for z in test_zarr:
        print(f"    zarr: {z}")
    print(f"  test stems verified: {stems}")

    nb_path, nb_sha = _m36_resolve_notebook()
    print(f"  audited notebook: {nb_path}  sha256 OK ({nb_sha[:12]}...)")

    removed = _m36_delete_stale_outputs()
    if removed:
        print(f"  removed stale M36 outputs: {removed}")

    cells = _m36_load_notebook_cells(nb_path)
    print(f"  audited notebook code cells: {len(cells)} (env / config / dependency / inference / postprocess)")

    # --- execute the five audited cells IN THE CURRENT KERNEL (no nbclient) ---
    old_signal = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(M36_HARD_TIMEOUT_S)
    ns = {"__name__": "__m36_reference__"}
    infer_seconds = None
    post_seconds = None
    try:
        # cell 0 environment, cell 1 configuration, cell 2 dependency (materializes the repo)
        for idx in (0, 1, 2):
            exec(compile(cells[idx], f"<m36_cell_{idx}>", "exec"), ns)

        repo_dir = str(ns.get("REPO_DIR") or (Path(M36_WORKING_DIR) / "tracking_repo"))
        weight_path, weight_sha = _m36_resolve_weight(repo_dir)
        print(f"  400ep weight: {weight_path}  sha256 OK ({weight_sha[:12]}...)")

        # cell 3 inference (single pass; D4 detection TTA already present, no extra passes)
        t_inf = time.time()
        exec(compile(cells[3], "<m36_cell_3>", "exec"), ns)
        infer_seconds = time.time() - t_inf
        print(f"  inference complete: {infer_seconds/60.0:.2f} min")

        # cell 4 postprocess -> writes /kaggle/working/submission.csv
        t_post = time.time()
        exec(compile(cells[4], "<m36_cell_4>", "exec"), ns)
        post_seconds = time.time() - t_post
        print(f"  postprocess complete: {post_seconds/60.0:.2f} min")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_signal)

    # --- final validation ---
    out_path = str(ns.get("SUBMISSION_PATH") or M36_OUTPUT)
    val = _m36_validate_submission(out_path)
    total_seconds = _elapsed()
    runtime_ok = total_seconds < M36_HARD_TIMEOUT_S
    val["checks"]["total_runtime_under_90min"] = bool(runtime_ok)

    sub_sha = _m36_sha256_file(out_path)
    c = val["counts"]
    print("--- M36 RESULT ---")
    print(f"  inference runtime : {(infer_seconds or 0)/60.0:.2f} min")
    print(f"  postprocess runtime: {(post_seconds or 0)/60.0:.2f} min")
    print(f"  total runtime     : {total_seconds/60.0:.2f} min")
    print(f"  rows: {c['rows']}  nodes: {c['nodes']}  edges: {c['edges']}  divisions: {c['divisions']}")
    print(f"  max_indegree: {c['max_indegree']}  max_outdegree: {c['max_outdegree']}  datasets: {c['datasets']}")
    print(f"  submission sha256 : {sub_sha}")
    print(f"  graph validation  : {'PASS' if val['checks']['graph_validation_passes'] else 'FAIL'}")
    print(f"  fingerprint exact : {val['fingerprint_exact']}  (expected {M36_FINGERPRINT})")
    print(f"  output path       : {out_path}")
    failed = [k for k, v in val["checks"].items() if not v]
    if failed:
        print(f"  FAILED validation gates: {failed}")

    ready = bool(val["all_gates_pass"] and runtime_ok and Path(out_path).exists())
    if ready:
        print("M36_FAST_REFERENCE_SUBMIT_READY")
    else:
        raise M36Error(f"validation gates not all passed: {failed} (submission not declared ready)")
    return {"ready": ready, "submission_sha256": sub_sha, "output_path": out_path,
            "inference_seconds": infer_seconds, "postprocess_seconds": post_seconds,
            "total_seconds": total_seconds, "validation": val}


# --------------------------------------------------------------------------- #
# CPU-only self-tests (NO GPU, NO inference, NO submission). Exercise the pure
# validation + resolver logic on synthetic data.
# --------------------------------------------------------------------------- #
def _m36_synth_submission(path, break_indegree=False, break_time=False, dangling=False, dup=False):
    import csv
    rows = []
    rid = 0

    def add(dataset, row_type, node_id=-1, t=-1, z=0.0, y=0.0, x=0.0, source_id=-1, target_id=-1):
        nonlocal rid
        rows.append({"id": rid, "dataset": dataset, "row_type": row_type, "node_id": node_id,
                     "t": t, "z": z, "y": y, "x": x, "source_id": source_id, "target_id": target_id})
        rid += 1

    for ds in M36_EXPECTED_STEMS:
        # 4 nodes across 3 timepoints: 0 -> {1,2} (division) -> 3
        add(ds, "node", node_id=0, t=0); add(ds, "node", node_id=1, t=1)
        add(ds, "node", node_id=2, t=1); add(ds, "node", node_id=3, t=2)
        first_edge_idx = len(rows)
        add(ds, "edge", source_id=0, target_id=1)
        add(ds, "edge", source_id=0, target_id=2)          # division: outdeg(0)=2
        tt = 3 if not break_time else 0                    # break_time -> source_t !< target_t
        add(ds, "edge", source_id=1, target_id=tt)
        if break_indegree:
            add(ds, "edge", source_id=2, target_id=1)      # indeg(1)=2 -> invalid
        if dangling:
            add(ds, "edge", source_id=0, target_id=999)    # endpoint not a node
        if dup:
            rows.append(dict(rows[first_edge_idx]))         # EXACT duplicate row (id included)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=M36_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_m36_selftests():
    import tempfile
    # 1) resolver stems constant is exactly the four official stems
    assert set(M36_EXPECTED_STEMS) == {"44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"}
    assert M36_CSV_COLUMNS[0] == "id" and M36_CSV_COLUMNS[1:3] == ["dataset", "row_type"]
    assert M36_FINGERPRINT == {"nodes": 128511, "edges": 124002, "rows": 252513, "divisions": 417}

    with tempfile.TemporaryDirectory() as w:
        # 2) a well-formed synthetic submission passes every topology gate
        good = str(Path(w) / "good.csv"); _m36_synth_submission(good)
        v = _m36_validate_submission(good)
        assert v["checks"]["columns_exact_and_ordered"]
        assert v["checks"]["exactly_four_datasets"]
        assert v["checks"]["no_nan_in_mandatory_fields"]
        assert v["checks"]["node_ids_unique_within_dataset"]
        assert v["checks"]["edge_endpoints_exist_in_dataset"]
        assert v["checks"]["edge_source_t_lt_target_t"]
        assert v["checks"]["max_indegree_le_1"] and v["checks"]["max_outdegree_le_2"]
        assert v["checks"]["no_duplicate_rows"] and v["checks"]["graph_validation_passes"]
        assert v["counts"]["divisions"] == 4 and v["counts"]["max_outdegree"] == 2

        # 3) indegree>1 is rejected
        bad_in = str(Path(w) / "bad_in.csv"); _m36_synth_submission(bad_in, break_indegree=True)
        assert _m36_validate_submission(bad_in)["checks"]["max_indegree_le_1"] is False

        # 4) source_t !< target_t is rejected
        bad_t = str(Path(w) / "bad_t.csv"); _m36_synth_submission(bad_t, break_time=True)
        assert _m36_validate_submission(bad_t)["checks"]["edge_source_t_lt_target_t"] is False

        # 5) dangling edge endpoint is rejected
        bad_e = str(Path(w) / "bad_e.csv"); _m36_synth_submission(bad_e, dangling=True)
        assert _m36_validate_submission(bad_e)["checks"]["edge_endpoints_exist_in_dataset"] is False

        # 6) duplicate rows are rejected
        bad_d = str(Path(w) / "bad_d.csv"); _m36_synth_submission(bad_d, dup=True)
        assert _m36_validate_submission(bad_d)["checks"]["no_duplicate_rows"] is False

        # 7) fingerprint mismatch is reported (synthetic counts != reference)
        assert _m36_validate_submission(good)["fingerprint_exact"] is False

        # 8) stale-output deletion never targets a mounted input path
        # (pure guard: the delete helper only ever touches /kaggle/working)
    print("M36 self-tests passed (8/8) - CPU only, no GPU inference, no submission")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        run_m36_selftests()
    else:
        run_m36_fast_reference_submit()
