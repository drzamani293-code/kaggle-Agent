#!/usr/bin/env python3
"""M38 - Hidden-resilient fast reference-0902 submission runner.

Corrects M37 (5bab5865...), which passed the public test but failed the Kaggle HIDDEN
rerun with "Notebook Threw Exception". Root causes removed here:
  1. M37's self-imposed 110-minute SIGALRM raised an unhandled _M37Timeout -> Kaggle
     reports "Notebook Threw Exception". M38 has NO signal.alarm / SIGALRM / hard-timeout
     exception at all - only a SOFT runtime target that stops launching new inference and
     NEVER raises.
  2. The audited notebook's single `subprocess.run(predict_cmd, check=True)` over ALL
     datasets, and cell 4's `if len(geffs) != len(test_stems): raise`, are all-or-nothing:
     one hidden dataset failing aborts the whole notebook. M38 runs inference PER DATASET
     with `check=False`, checkpoints each success, and postprocesses only the datasets that
     succeeded, merging fallback rows for the rest.

Still a SUBMISSION milestone (exact audited 0.902 scientific config, notebook SHA, 400ep
weight SHA, D4 detection TTA, ILP weights, postprocess behavior). No M35 instrumentation,
no nbclient, no nested kernel, no Kaggle-API submission. Accelerator: GPU. Internet: Off.
Current kernel only. Official Kaggle GPU limit is 12h; M38 targets < 2h softly.
"""

import os
import sys
import csv
import time
import json
import shutil
import hashlib
import traceback
from pathlib import Path

# --------------------------------------------------------------------------- #
# Fixed audited identity. NO public dataset stems anywhere.
# --------------------------------------------------------------------------- #
M38_NOTEBOOK_SHA256 = "beb17b03682231460c3adab6069815e06c44adc23e427371c364901f8de5437e"
M38_WEIGHT_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"

# public-DATA reference (COUNTS + SHA only - not dataset names). Used ONLY for the
# public development-run exactness gate; NEVER a hidden validation gate.
M38_PUBLIC_FINGERPRINT = {"nodes": 128511, "edges": 124002, "rows": 252513, "divisions": 417}
M38_PUBLIC_REFERENCE_SHA256 = "8c73d776abca799a37c2bd24768a1bbb510418c3327192430730540d56cea698"

M38_COMP_DIR = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
M38_BUNDLE_DIR = "/kaggle/input/datasets/mohammadjafarzamani/biohub-0902-reference-bundle"
M38_SUPPORT_DIR = "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
M38_WORKING_DIR = "/kaggle/working"
M38_OUTPUT = "/kaggle/working/submission.csv"
M38_CHECKPOINT_DIR = "/kaggle/working/m38_geff_checkpoint"
M38_STATUS_JSON = "/kaggle/working/m38_status.json"

M38_NOTEBOOK_REL = "reference/biohub-competition-solution.ipynb"
M38_WEIGHT_REL = "weights/unet_transformer/split_0/edge_predictor_best.pth"
M38_METHOD = "unet_transformer"
M38_PRED_NAMESPACE = "unknown"        # predictions/<ns>/unet_transformer/split_0/<ds>.geff

M38_CSV_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
M38_NODE_MANDATORY = ["node_id", "t", "z", "y", "x"]
M38_EDGE_MANDATORY = ["source_id", "target_id"]

# SOFT runtime policy (targets, never exceptions)
M38_SOFT_TOTAL_TARGET_S = 120 * 60            # target total < 2h
M38_SOFT_INFERENCE_DEADLINE_S = 95 * 60       # stop LAUNCHING new inference around 95 min
M38_POSTPROCESS_RESERVE_S = 15 * 60           # reserve >= 15 min for postprocess + write


class M38Error(RuntimeError):
    """Internal error. Caught by the top-level runner once a fallback exists; never escapes."""


# --------------------------------------------------------------------------- #
# small pure helpers
# --------------------------------------------------------------------------- #
def _m38_sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _m38_sha256_tree(path):
    """SHA of a file OR a directory tree (GEFF stores are directories)."""
    p = Path(path)
    if p.is_file():
        return _m38_sha256_file(p)
    h = hashlib.sha256()
    for sub in sorted(p.rglob("*")):
        if sub.is_file():
            h.update(str(sub.relative_to(p)).encode())
            with open(sub, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


def _m38_first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def _m38_resolve_test_dir(comp_dir=None):
    """DYNAMICALLY discover every immediate *.zarr under the ACTUAL competition test/ dir.
    Requires: test dir exists; >= 1 store; non-empty, unique stems. No fixed count; never a
    fixed public dataset-name list - so it works on the hidden rerun."""
    base = comp_dir if comp_dir is not None else _m38_first_existing(
        [M38_COMP_DIR, f"/kaggle/input/{Path(M38_COMP_DIR).name}"])
    if base is None:
        raise M38Error(f"competition input not mounted: {M38_COMP_DIR}")
    test_dir = Path(base) / "test"
    if not test_dir.is_dir():
        raise M38Error(f"competition test/ dir not found under {base}")
    zarrs = sorted(str(p) for p in test_dir.glob("*.zarr"))
    if not zarrs:
        raise M38Error(f"no .zarr test stores discovered under {test_dir}")
    stems = [Path(z).stem for z in zarrs]
    if any(not s for s in stems):
        raise M38Error(f"a discovered .zarr store has an empty stem: {zarrs}")
    if len(set(stems)) != len(stems):
        raise M38Error(f"discovered .zarr stems are not unique: {stems}")
    return str(test_dir), sorted(stems)


def _m38_reassign_ids(rows):
    """id becomes exactly 0..N-1, consecutive, in row order."""
    for i, r in enumerate(rows):
        r["id"] = i
    return rows


def _m38_fallback_rows(stems):
    """One valid node row per dataset at t=0,z=0,y=0,x=0,node_id=0, no edges. Consecutive id."""
    rows = []
    for ds in stems:
        rows.append({"id": 0, "dataset": ds, "row_type": "node", "node_id": 0,
                     "t": 0, "z": 0.0, "y": 0.0, "x": 0.0, "source_id": -1, "target_id": -1})
    return _m38_reassign_ids(rows)


def _m38_try_sample_submission_rows(stems, sample_path):
    """Return the competition sample_submission rows ONLY if its schema and dataset names
    exactly match the discovered test datasets; else None."""
    if not sample_path or not Path(sample_path).exists():
        return None
    try:
        with open(sample_path, newline="") as fh:
            reader = csv.DictReader(fh)
            if list(reader.fieldnames or []) != M38_CSV_COLUMNS:
                return None
            rows = [dict(r) for r in reader]
    except (OSError, csv.Error):
        return None
    if set(str(r["dataset"]) for r in rows) != set(stems):
        return None
    return _m38_reassign_ids(rows)


def _m38_write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=M38_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in M38_CSV_COLUMNS})


def _m38_read_rows(path):
    with open(path, newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _m38_write_fallback_submission(out_path, stems, sample_path=None):
    """Create a valid emergency submission covering EVERY discovered dataset. Prefer the
    competition sample_submission.csv when its schema+datasets match; else one node row per
    dataset. Returns the fallback rows (kept in memory so it stays available if later fails)."""
    rows = _m38_try_sample_submission_rows(stems, sample_path)
    source = "sample_submission"
    if rows is None:
        rows = _m38_fallback_rows(stems)
        source = "synthetic_node_per_dataset"
    _m38_write_rows(out_path, rows)
    return rows, source


def _m38_merge_postprocess_with_fallback(postproc_rows, fallback_rows, successful_stems, all_stems):
    """Rows from successful datasets come from postprocess; missing datasets get their
    fallback rows. id reassigned consecutively. Datasets never overlap between the two."""
    succ = set(successful_stems)
    fb_by_ds = {}
    for r in fallback_rows:
        fb_by_ds.setdefault(str(r["dataset"]), []).append(r)
    merged = [r for r in postproc_rows if str(r["dataset"]) in succ]
    for ds in all_stems:
        if ds not in succ:
            merged.extend(dict(r) for r in fb_by_ds.get(ds, []))
    return _m38_reassign_ids(merged)


# --------------------------------------------------------------------------- #
# validation - dynamic, against the discovered stems only, incl. id == 0..N-1
# --------------------------------------------------------------------------- #
def _m38_validate_submission(path, discovered_stems):
    import math
    import pandas as pd
    if not Path(path).exists():
        raise M38Error(f"submission.csv not found at {path}")
    df = pd.read_csv(path)
    discovered = set(str(s) for s in discovered_stems)
    checks = {}

    checks["columns_exact_and_ordered"] = list(df.columns) == M38_CSV_COLUMNS
    if not checks["columns_exact_and_ordered"]:
        return {"checks": checks, "counts": {}, "all_gates_pass": False,
                "public_fingerprint_match_informational": False}

    node_df = df[df["row_type"] == "node"]
    edge_df = df[df["row_type"] == "edge"]
    sub_datasets = set(str(d) for d in df["dataset"].unique())
    checks["all_discovered_datasets_represented"] = discovered <= sub_datasets
    checks["no_unexpected_datasets"] = sub_datasets <= discovered
    checks["datasets_match_discovered"] = sub_datasets == discovered

    base_ok = not df[["id", "dataset", "row_type"]].isna().any().any()
    node_ok = not node_df[M38_NODE_MANDATORY].isna().any().any() if len(node_df) else True
    edge_ok = not edge_df[M38_EDGE_MANDATORY].isna().any().any() if len(edge_df) else True
    checks["no_nan_in_mandatory_fields"] = bool(base_ok and node_ok and edge_ok)

    def _all_int(series):
        try:
            return all(float(v).is_integer() for v in series.dropna().to_numpy())
        except (TypeError, ValueError):
            return False
    checks["ids_are_integers"] = bool(_all_int(node_df["node_id"]) and _all_int(edge_df["source_id"])
                                      and _all_int(edge_df["target_id"]))

    ids = df["id"].tolist()
    checks["id_is_0_to_n_minus_1"] = bool(ids == list(range(len(df))))

    semantic = df.drop(columns=["id"])
    checks["no_duplicate_semantic_rows"] = bool(not semantic.duplicated().any())

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

    counts = {"rows": int(len(df)), "nodes": int(len(node_df)), "edges": int(len(edge_df)),
              "divisions": int(divisions), "datasets": sorted(sub_datasets),
              "n_datasets": len(sub_datasets), "max_indegree": int(max_indegree),
              "max_outdegree": int(max_outdegree)}
    public_fp = bool(counts["nodes"] == M38_PUBLIC_FINGERPRINT["nodes"]
                     and counts["edges"] == M38_PUBLIC_FINGERPRINT["edges"]
                     and counts["rows"] == M38_PUBLIC_FINGERPRINT["rows"]
                     and counts["divisions"] == M38_PUBLIC_FINGERPRINT["divisions"])
    return {"checks": checks, "counts": counts, "all_gates_pass": all(checks.values()),
            "public_fingerprint_match_informational": public_fp}


def _m38_public_exactness_gate(sha256, counts):
    """PUBLIC development-run policy ONLY (never a hidden gate): the generated submission
    must byte-reproduce the M37 public SHA and fingerprint. Reported informationally on
    hidden data; a dev/public run must fail if this is False when running the public set."""
    return {"public_sha_exact": bool(sha256 == M38_PUBLIC_REFERENCE_SHA256),
            "public_fingerprint_exact": bool(counts.get("nodes") == M38_PUBLIC_FINGERPRINT["nodes"]
                                             and counts.get("edges") == M38_PUBLIC_FINGERPRINT["edges"]
                                             and counts.get("rows") == M38_PUBLIC_FINGERPRINT["rows"]
                                             and counts.get("divisions") == M38_PUBLIC_FINGERPRINT["divisions"]),
            "public_reference_sha256": M38_PUBLIC_REFERENCE_SHA256}


# --------------------------------------------------------------------------- #
# real Kaggle engine - exec the audited cells, per-dataset predict, postprocess
# --------------------------------------------------------------------------- #
class _M38KaggleEngine:
    """Drives the exact audited notebook in the CURRENT kernel, but per-dataset and
    check=False. Injectable so CPU tests can replace it with a fake."""

    def __init__(self):
        self.ns = None
        self.predict_cmd = None
        self.repo_dir = None
        self.test_dir = None
        self._real_subprocess = None

    def discover(self):
        return _m38_resolve_test_dir()

    def resolve_notebook(self):
        nb = _m38_first_existing([Path(M38_BUNDLE_DIR) / M38_NOTEBOOK_REL,
                                  Path(M38_BUNDLE_DIR) / Path(M38_NOTEBOOK_REL).name])
        if nb is None:
            raise M38Error(f"audited notebook not found under {M38_BUNDLE_DIR}/{M38_NOTEBOOK_REL}")
        sha = _m38_sha256_file(nb)
        if sha != M38_NOTEBOOK_SHA256:
            raise M38Error(f"notebook sha256 {sha} != audited {M38_NOTEBOOK_SHA256}")
        return str(nb), sha

    def _load_cells(self, nb_path):
        nb = json.loads(Path(nb_path).read_text())
        cells = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                 for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        if len(cells) != 5:
            raise M38Error(f"expected 5 audited code cells, got {len(cells)}")
        return cells

    def prepare(self):
        """exec cells 0,1,2 (env/config/dependency -> materialize repo), then exec cell 3
        with subprocess CAPTURED (not run) so the exact D4-TTA patch is applied and the
        exact predict_cmd is built without launching the all-datasets inference."""
        import subprocess as _sp
        self._real_subprocess = _sp
        nb_path, _sha = self.resolve_notebook()
        self._cells = self._load_cells(nb_path)
        self.ns = {"__name__": "__m38_reference__"}
        for idx in (0, 1, 2):
            exec(compile(self._cells[idx], f"<m38_cell_{idx}>", "exec"), self.ns)
        self.repo_dir = Path(self.ns.get("REPO_DIR") or (Path(M38_WORKING_DIR) / "tracking_repo"))
        self.test_dir = Path(self.ns.get("TEST_DIR"))

        # verify controlled 400ep weight (before any inference)
        w = _m38_first_existing([self.repo_dir / M38_WEIGHT_REL, Path(M38_BUNDLE_DIR) / M38_WEIGHT_REL,
                                 Path(M38_BUNDLE_DIR) / "tracking_repo" / M38_WEIGHT_REL])
        if w is None or _m38_sha256_file(w) != M38_WEIGHT_SHA256:
            raise M38Error("controlled 400ep weight missing or sha256 mismatch")

        captured = {}

        class _Shim:
            def __getattr__(self, name):
                return getattr(_sp, name)

            def run(self, cmd, *a, **kw):        # CAPTURE the predict cmd, do NOT run it
                captured["cmd"] = list(cmd)
                captured["cwd"] = kw.get("cwd")
                captured["env"] = kw.get("env")
                return _sp.CompletedProcess(cmd, 0, b"", b"")
        self.ns["subprocess"] = _Shim()
        exec(compile(self._cells[3], "<m38_cell_3>", "exec"), self.ns)     # applies D4-TTA patch, builds predict_cmd
        self.ns["subprocess"] = _sp                                        # restore real subprocess for later
        self.predict_cmd = captured.get("cmd") or list(self.ns.get("predict_cmd") or [])
        self._pred_cwd = captured.get("cwd") or str(self.repo_dir)
        self._pred_env = captured.get("env") or {**os.environ, "PYTHONPATH": "src"}
        if not self.predict_cmd:
            raise M38Error("could not capture the audited predict command from cell 3")

    def _predictions_glob(self):
        return sorted((self.repo_dir / "predictions").glob(f"*/{M38_METHOD}/split_0/*.geff"))

    def infer_one(self, dataset):
        """Run the EXACT audited predict command for ONE dataset (own split JSON), check=False."""
        splits_path = self.repo_dir / f"m38_split_{dataset}.json"
        splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": [dataset]}]))
        cmd = list(self.predict_cmd)
        # replace the value following --splits with this per-dataset json filename
        for i, tok in enumerate(cmd):
            if tok == "--splits" and i + 1 < len(cmd):
                cmd[i + 1] = splits_path.name
                break
        t0 = time.time()
        proc = self._real_subprocess.run(cmd, cwd=self._pred_cwd, env=self._pred_env,
                                         check=False, capture_output=True, text=True)
        runtime = time.time() - t0
        geffs = [p for p in self._predictions_glob() if p.stem == dataset]
        return {"returncode": int(proc.returncode), "runtime": runtime,
                "stdout": proc.stdout or "", "stderr": proc.stderr or "",
                "geff_src": str(geffs[0]) if geffs else None}

    def checkpoint(self, dataset, geff_src):
        """Copy the freshly produced GEFF store to the checkpoint dir and verify readable."""
        if not geff_src or not Path(geff_src).exists():
            raise M38Error(f"no GEFF produced for {dataset}")
        ck = Path(M38_CHECKPOINT_DIR) / f"{dataset}.geff"
        if ck.exists():
            shutil.rmtree(ck, ignore_errors=True)
        ck.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(geff_src, ck)
        # verify the checkpoint is readable as a graph
        graph_from_geff = self.ns.get("graph_from_geff")
        if graph_from_geff is not None:
            graph_from_geff(ck)                # raises if unreadable
        return str(ck), _m38_sha256_tree(ck)

    def postprocess(self, successful_stems):
        """Restore every successful checkpoint into the predictions dir the audited cell 4
        expects, then run the EXACT audited postprocess (cell 4) over that subset."""
        preds = self.repo_dir / "predictions" / M38_PRED_NAMESPACE / M38_METHOD / "split_0"
        if (self.repo_dir / "predictions").exists():
            shutil.rmtree(self.repo_dir / "predictions", ignore_errors=True)
        preds.mkdir(parents=True, exist_ok=True)
        for ds in successful_stems:
            ck = Path(M38_CHECKPOINT_DIR) / f"{ds}.geff"
            if not ck.exists():
                raise M38Error(f"checkpoint missing for {ds}")
            shutil.copytree(ck, preds / f"{ds}.geff")
        self.ns["test_stems"] = list(successful_stems)     # so cell 4's len-check matches the subset
        self.ns.setdefault("predict_seconds", 0.0)
        exec(compile(self._cells[4], "<m38_cell_4>", "exec"), self.ns)
        return str(self.ns.get("SUBMISSION_PATH") or M38_OUTPUT)


# --------------------------------------------------------------------------- #
# top-level orchestrator - NO exception escapes once a fallback exists
# --------------------------------------------------------------------------- #
def run_m38_hidden_resilient_reference_submit(engine=None, sample_submission_path=None, now=None):
    t_start = (now() if now else time.time())

    def _elapsed():
        return (now() if now else time.time()) - t_start

    eng = engine if engine is not None else _M38KaggleEngine()
    if sample_submission_path is None:
        sample_submission_path = str(Path(M38_COMP_DIR) / "sample_submission.csv")
    status = {"milestone": "M38_HIDDEN_RESILIENT_REFERENCE_0902_SUBMIT", "mode": None,
              "discovered_stems": None, "per_dataset": {}, "successful": [], "failed": [],
              "fallback_source": None, "postprocess_ok": None, "label": None,
              "submission_sha256": None, "reasons": [], "exception": None}
    print("=== M38 HIDDEN-RESILIENT REFERENCE-0902 SUBMISSION (current kernel, per-dataset) ===")

    fallback_rows = None
    stems = None
    try:
        test_dir, stems = eng.discover()
        status["discovered_stems"] = stems
        print(f"  discovered {len(stems)} dataset(s): {stems}")
        # (4) build a valid emergency fallback covering EVERY discovered dataset, BEFORE inference
        fallback_rows, fb_source = _m38_write_fallback_submission(M38_OUTPUT, stems, sample_submission_path)
        status["fallback_source"] = fb_source
        print(f"  emergency fallback submission written ({fb_source}); every dataset covered")
    except Exception as exc:                                     # noqa: BLE001 (before fallback exists)
        status["exception"] = f"{type(exc).__name__}: {exc}"
        status["label"] = "M38_NO_SUBMISSION"
        _m38_write_status(status)
        print(f"  FATAL before fallback (no submission possible): {status['exception']}")
        traceback.print_exc()
        return status

    # From here on NOTHING may escape: a valid submission.csv already exists.
    successful, failed = [], []
    try:
        eng.prepare()
        print("  audited cells prepared (repo materialized, D4-TTA patch applied, predict cmd captured)")
        for ds in stems:
            rec = {"started": False, "completed": False, "returncode": None, "runtime": None,
                   "checkpoint_path": None, "checkpoint_sha256": None, "error_type": None,
                   "stderr_tail": None, "fallback_used": True}
            # (3) SOFT deadline: stop LAUNCHING new inference near 95 min, reserving 15 min
            if _elapsed() > M38_SOFT_INFERENCE_DEADLINE_S:
                rec["error_type"] = "soft_deadline_skip"
                status["per_dataset"][ds] = rec
                failed.append(ds)
                print(f"  [{ds}] SKIPPED (soft {M38_SOFT_INFERENCE_DEADLINE_S//60}min inference deadline; fallback kept)")
                continue
            rec["started"] = True
            try:
                res = eng.infer_one(ds)
                rec["returncode"] = res["returncode"]
                rec["runtime"] = res["runtime"]
                rec["stderr_tail"] = (res.get("stderr") or "")[-2000:]
                if res["returncode"] != 0 or not res.get("geff_src"):
                    rec["error_type"] = "predict_returncode" if res["returncode"] != 0 else "no_geff_output"
                    failed.append(ds)
                    print(f"  [{ds}] inference FAILED rc={res['returncode']} (kept fallback)")
                else:
                    ck_path, ck_sha = eng.checkpoint(ds, res["geff_src"])
                    rec["checkpoint_path"] = ck_path
                    rec["checkpoint_sha256"] = ck_sha
                    rec["completed"] = True
                    rec["fallback_used"] = False
                    successful.append(ds)
                    print(f"  [{ds}] inference OK in {res['runtime']/60.0:.2f} min; checkpoint verified")
            except Exception as exc:                             # noqa: BLE001 (per-dataset isolation)
                rec["error_type"] = type(exc).__name__
                rec["stderr_tail"] = (str(exc))[-2000:]
                failed.append(ds)
                print(f"  [{ds}] inference EXCEPTION {type(exc).__name__}: {exc} (kept fallback)")
            status["per_dataset"][ds] = rec

        status["successful"] = successful
        status["failed"] = failed

        # (7) postprocess the successful subset; merge fallback rows for the rest
        postproc_ok = False
        if successful:
            try:
                pp_path = eng.postprocess(successful)
                pp_rows = _m38_read_rows(pp_path)
                merged = _m38_merge_postprocess_with_fallback(pp_rows, fallback_rows, successful, stems)
                _m38_write_rows(M38_OUTPUT, merged)
                postproc_ok = True
            except Exception as exc:                             # noqa: BLE001 (retain fallback)
                status["reasons"].append(f"postprocess failed: {type(exc).__name__}: {exc}")
                print(f"  postprocess FAILED: {type(exc).__name__}: {exc} (retaining fallback)")
                _m38_write_rows(M38_OUTPUT, fallback_rows)       # ensure a valid submission remains
        else:
            status["reasons"].append("no dataset succeeded; retaining fallback")
            _m38_write_rows(M38_OUTPUT, fallback_rows)
        status["postprocess_ok"] = postproc_ok
    except Exception as exc:                                     # noqa: BLE001 (top-level guard)
        status["exception"] = f"{type(exc).__name__}: {exc}"
        status["reasons"].append("top-level exception caught; fallback preserved")
        print(f"  TOP-LEVEL EXCEPTION caught (fallback preserved): {status['exception']}")
        traceback.print_exc()
        _m38_write_rows(M38_OUTPUT, fallback_rows)

    # --- validate + label (never raises) ---
    try:
        val = _m38_validate_submission(M38_OUTPUT, stems)
    except Exception as exc:                                     # noqa: BLE001
        val = {"checks": {}, "counts": {}, "all_gates_pass": False,
               "public_fingerprint_match_informational": False}
        status["reasons"].append(f"validation error: {exc}")
    status["submission_sha256"] = _m38_sha256_file(M38_OUTPUT) if Path(M38_OUTPUT).exists() else None
    status["validation"] = val
    pub = _m38_public_exactness_gate(status["submission_sha256"], val.get("counts", {}))
    status["public_exactness_informational"] = pub

    n_ok, n_all = len(successful), len(stems or [])
    if status.get("postprocess_ok") and n_ok == n_all and n_all > 0:
        label = "M38_FULL_SUBMIT_READY"
    elif status.get("postprocess_ok") and n_ok >= 1:
        label = "M38_PARTIAL_SUBMIT_READY"
    else:
        label = "M38_FALLBACK_SUBMIT_READY"
    status["label"] = label
    status["mode"] = {"M38_FULL_SUBMIT_READY": "FULL", "M38_PARTIAL_SUBMIT_READY": "PARTIAL",
                      "M38_FALLBACK_SUBMIT_READY": "FALLBACK"}[label]

    valid_and_present = bool(val.get("all_gates_pass") and Path(M38_OUTPUT).exists())
    _m38_write_status(status)

    c = val.get("counts", {})
    print("--- M38 RESULT ---")
    print(f"  mode: {status['mode']}  successful: {n_ok}/{n_all}  failed: {failed}")
    print(f"  rows: {c.get('rows')}  nodes: {c.get('nodes')}  edges: {c.get('edges')}  divisions: {c.get('divisions')}")
    print(f"  datasets ({c.get('n_datasets')}): {c.get('datasets')}")
    print(f"  submission sha256: {status['submission_sha256']}")
    print(f"  validation all_gates_pass: {val.get('all_gates_pass')}")
    print(f"  [informational, NOT a hidden gate] public exact reproduction: {pub['public_sha_exact']} "
          f"(sha) / {pub['public_fingerprint_exact']} (fingerprint)")
    print(f"  total runtime: {_elapsed()/60.0:.2f} min (soft target < {M38_SOFT_TOTAL_TARGET_S//60} min)")
    print(f"  status json: {M38_STATUS_JSON}   output: {M38_OUTPUT}")
    if valid_and_present:
        print(label)
    else:
        # even here we do NOT raise - the fallback submission remains on disk
        print(f"  WARNING: validation gates not all passed ({[k for k,v in val.get('checks',{}).items() if not v]}); "
              f"submission.csv retained")
        print(label)
    return status


def _m38_write_status(status):
    try:
        Path(M38_STATUS_JSON).parent.mkdir(parents=True, exist_ok=True)
        Path(M38_STATUS_JSON).write_text(json.dumps(status, indent=2, default=str))
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# CPU-only self-tests (NO GPU, NO inference, NO submission). A FakeEngine simulates
# per-dataset success / failure / timeout / postprocess failure. Public stems never
# appear as literals.
# --------------------------------------------------------------------------- #
class _M38FakeEngine:
    """Simulates the Kaggle engine deterministically for CPU tests."""

    def __init__(self, stems, fail=(), missing_checkpoint=(), pp_raises=False, per_infer_seconds=1.0):
        self.stems = stems
        self.fail = set(fail)
        self.missing_checkpoint = set(missing_checkpoint)
        self.pp_raises = pp_raises
        self.per_infer_seconds = per_infer_seconds
        self.tmp = None

    def discover(self):
        return "/fake/test", sorted(self.stems)

    def prepare(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="m38fake_")

    def infer_one(self, ds):
        if ds in self.fail:
            return {"returncode": 1, "runtime": self.per_infer_seconds, "stdout": "",
                    "stderr": f"boom for {ds}", "geff_src": None}
        p = Path(self.tmp) / f"{ds}.geff"; p.mkdir(parents=True, exist_ok=True)
        (p / "data").write_text(f"geff:{ds}")
        return {"returncode": 0, "runtime": self.per_infer_seconds, "stdout": "ok",
                "stderr": "", "geff_src": str(p)}

    def checkpoint(self, ds, geff_src):
        if ds in self.missing_checkpoint:
            raise M38Error(f"checkpoint verify failed for {ds}")
        ck = Path(self.tmp) / "ck" / f"{ds}.geff"; ck.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(geff_src, ck)
        return str(ck), _m38_sha256_tree(ck)

    def postprocess(self, successful):
        if self.pp_raises:
            raise M38Error("simulated postprocess failure")
        # emit a valid postprocess submission for the successful datasets (2 nodes + 1 edge each)
        rows = []
        rid = 0
        for ds in successful:
            for nid, t in ((0, 0), (1, 1)):
                rows.append({"id": rid, "dataset": ds, "row_type": "node", "node_id": nid,
                             "t": t, "z": 0.0, "y": 0.0, "x": 0.0, "source_id": -1, "target_id": -1}); rid += 1
            rows.append({"id": rid, "dataset": ds, "row_type": "edge", "node_id": -1,
                         "t": -1, "z": 0.0, "y": 0.0, "x": 0.0, "source_id": 0, "target_id": 1}); rid += 1
        out = Path(self.tmp) / "pp_submission.csv"
        _m38_write_rows(out, rows)
        return str(out)


def _m38_clock():
    """A controllable clock for the soft-deadline test."""
    state = {"t": 0.0}

    def now():
        return state["t"]
    return state, now


def run_m38_selftests():
    import tempfile
    hidden = ["hidden_a", "hidden_b", "hidden_c", "hidden_d", "hidden_e"]   # arbitrary names/count

    # 1) fallback rows are valid and cover every discovered dataset; id consecutive
    with tempfile.TemporaryDirectory() as w:
        fb, src = _m38_write_fallback_submission(str(Path(w) / "s.csv"), hidden)
        assert src == "synthetic_node_per_dataset"
        v = _m38_validate_submission(str(Path(w) / "s.csv"), hidden)
        assert v["all_gates_pass"], [k for k, x in v["checks"].items() if not x]
        assert v["checks"]["id_is_0_to_n_minus_1"] and v["counts"]["n_datasets"] == 5

    # 2) FULL run: every dataset succeeds -> FULL, all gates pass, no exception
    with tempfile.TemporaryDirectory() as w:
        _redirect_outputs(w)
        st = run_m38_hidden_resilient_reference_submit(engine=_M38FakeEngine(hidden))
        assert st["label"] == "M38_FULL_SUBMIT_READY" and st["mode"] == "FULL"
        assert st["validation"]["all_gates_pass"] and set(st["successful"]) == set(hidden)

    # 3) one subprocess failure among several -> PARTIAL, merged with fallback, gates pass
    with tempfile.TemporaryDirectory() as w:
        _redirect_outputs(w)
        st = run_m38_hidden_resilient_reference_submit(engine=_M38FakeEngine(hidden, fail={"hidden_c"}))
        assert st["label"] == "M38_PARTIAL_SUBMIT_READY" and "hidden_c" in st["failed"]
        assert st["validation"]["all_gates_pass"]                       # fallback row covers hidden_c
        assert st["validation"]["checks"]["datasets_match_discovered"]

    # 4) soft deadline reached -> remaining datasets skipped, no exception, still valid
    with tempfile.TemporaryDirectory() as w:
        _redirect_outputs(w)
        state, now = _m38_clock()

        class _SlowEngine(_M38FakeEngine):
            def infer_one(self, ds):
                state["t"] += 40 * 60          # each dataset advances the clock 40 min
                return super().infer_one(ds)
        st = run_m38_hidden_resilient_reference_submit(engine=_SlowEngine(hidden), now=now)
        assert any(r.get("error_type") == "soft_deadline_skip" for r in st["per_dataset"].values())
        assert st["validation"]["all_gates_pass"] and Path(_redir_output()).exists()

    # 5) postprocess failure -> FALLBACK, no exception, fallback valid
    with tempfile.TemporaryDirectory() as w:
        _redirect_outputs(w)
        st = run_m38_hidden_resilient_reference_submit(engine=_M38FakeEngine(hidden, pp_raises=True))
        assert st["label"] == "M38_FALLBACK_SUBMIT_READY" and st["postprocess_ok"] is False
        assert st["validation"]["all_gates_pass"]

    # 6) missing/unverifiable checkpoint -> that dataset fails, others survive -> PARTIAL
    with tempfile.TemporaryDirectory() as w:
        _redirect_outputs(w)
        st = run_m38_hidden_resilient_reference_submit(engine=_M38FakeEngine(hidden, missing_checkpoint={"hidden_b"}))
        assert st["label"] == "M38_PARTIAL_SUBMIT_READY" and "hidden_b" in st["failed"]
        assert st["validation"]["all_gates_pass"]

    # 7) a hard exception inside prepare() must NOT escape once a fallback exists
    with tempfile.TemporaryDirectory() as w:
        _redirect_outputs(w)

        class _ExplodingEngine(_M38FakeEngine):
            def prepare(self):
                raise RuntimeError("kaboom in prepare")
        st = run_m38_hidden_resilient_reference_submit(engine=_ExplodingEngine(hidden))
        assert st["label"] == "M38_FALLBACK_SUBMIT_READY"
        assert st["exception"] and "kaboom" in st["exception"]
        assert st["validation"]["all_gates_pass"]                       # fallback still valid

    # 8) partial-merge integrity: successful rows + fallback rows, id 0..N-1, datasets exact
    fb = _m38_fallback_rows(hidden)
    pp = [{"id": 0, "dataset": "hidden_a", "row_type": "node", "node_id": 0, "t": 0, "z": 0.0,
           "y": 0.0, "x": 0.0, "source_id": -1, "target_id": -1}]
    merged = _m38_merge_postprocess_with_fallback(pp, fb, ["hidden_a"], hidden)
    assert [r["id"] for r in merged] == list(range(len(merged)))
    assert set(str(r["dataset"]) for r in merged) == set(hidden)

    # 9) public-exactness gate STRUCTURE (never a hidden gate): compares SHA to the M37 SHA
    g = _m38_public_exactness_gate(M38_PUBLIC_REFERENCE_SHA256, M38_PUBLIC_FINGERPRINT)
    assert g["public_sha_exact"] is True and g["public_fingerprint_exact"] is True
    assert _m38_public_exactness_gate("deadbeef", {})["public_sha_exact"] is False
    # and it is NOT one of the validation checks
    vk = _m38_validate_submission_keys()
    assert not any("public" in k for k in vk)

    print("M38 self-tests passed (9/9) - CPU only, resilient, no GPU inference, no submission")


# small test utilities (redirect the fixed output paths into a temp dir)
_M38_REDIR = {"root": None}


def _redirect_outputs(root):
    global M38_OUTPUT, M38_CHECKPOINT_DIR, M38_STATUS_JSON
    _M38_REDIR["root"] = root
    M38_OUTPUT = str(Path(root) / "submission.csv")
    M38_CHECKPOINT_DIR = str(Path(root) / "ck")
    M38_STATUS_JSON = str(Path(root) / "m38_status.json")


def _redir_output():
    return M38_OUTPUT


def _m38_validate_submission_keys():
    import tempfile
    with tempfile.TemporaryDirectory() as w:
        p = str(Path(w) / "k.csv")
        _m38_write_rows(p, _m38_fallback_rows(["a", "b"]))
        return list(_m38_validate_submission(p, ["a", "b"])["checks"].keys())


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        run_m38_selftests()
    else:
        run_m38_hidden_resilient_reference_submit()
