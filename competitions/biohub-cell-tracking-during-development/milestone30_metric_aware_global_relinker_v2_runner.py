"""
Biohub - Cell Tracking During Development
Milestone 30 - METRIC-AWARE GLOBAL RELINKER v2 (re-solve linking from candidates).

M29/v1 patched the ILP solution after linking. M30/v2 RE-SOLVES unit-timepoint
linking from the RAW GEFF candidate graph using a fused cost:
    cost = -log(edge_prob) + w_dist*(distance_um/7.0) + w_motion*motion_deviation_um.
It is embedded self-contained so each runner is a standalone one-cell Kaggle
script needing only the competition dataset + the mounted 400ep Pilkwang support
pack. M19-C 0.880 stays best/final unless an M30 variant beats it.

Critical engineering corrections implemented here:
  1. RAW CANDIDATE PRESERVATION - read_candidate_geff reads ALL unit-timepoint
     candidate edges WITHOUT the edge solution mask (the reused read_geff_graph
     applies that mask and returns only the ILP solution). build_base_graph is
     NEVER run before v2. edge_prob is never silently defaulted to 1.0; if it is
     missing / constant / degenerate the run reports
     DO_NOT_SUBMIT_EDGE_PROB_UNAVAILABLE.
  2. EXPLICIT NO-LINK OPTION - each Hungarian stage has per-source no-link dummy
     columns (LinkConfig.null_link_cost, default 4.5); a real edge is taken only
     if its fused cost beats no-link AND passes min-prob / max-distance / max-cost.
  3. TWO-STAGE LINKING - Phase 1 primary (one slot/source, each target <=1 parent,
     real targets + no-link dummies) then Phase 2 division (only sources that got
     a primary child and still have out-degree 1, only currently-unassigned
     targets, gates applied BEFORE an independent Hungarian with no-division
     dummies + a global cap; a rejected division never steals another source's
     target).
  4. SPARSE-CANDIDATE AUGMENTATION - if the raw graph is sparse, add tight
     geometric candidates (top-k<=3 within knn_max_um, conservative fixed prob,
     no duplicates) for EVERY source; raw vs augmented counts reported.
  5. REAL v2 LOCAL CV - a non-submit harness discovers train GT GEFF, runs the v2
     engine, and attempts the official metric (tracking_cellmot.metrics /
     scripts/evaluate.py). It NEVER fabricates scores; if wiring is impossible it
     reports CV_NOT_WIRED with the exact missing function/import/GT format.

Stage 0 (non-submit): M30_CANDIDATE_GRAPH_DIAGNOSTIC_NOT_SUBMIT profiles the raw
candidate graph (edge_prob availability + quantiles, candidates/source & /target
distributions, per-frame & per-dataset stats) and classifies it
FULL_CANDIDATE_GRAPH / MODERATELY_SPARSE / ILP_SOLUTION_LIKE / EDGE_PROB_UNAVAILABLE.
It writes m30_candidate_graph_report.json + m30_candidate_graph_table.csv and NO
submission.

Submit-capable variants (candidate-mode-gated):
  - A v2_full_candidates_balanced : raw candidates, balanced. FULL graph only.
  - B v2_full_candidates_gap123   : A + gap3. FULL graph only.
  - C v2_auto_sparse_knn_tight     : raw + tight kNN. Sparse / ILP-like graphs.
  - D v2_auto_det095               : auto mode (raw if FULL, kNN if sparse), det 0.95.
  - E v2_division_relaxed          : auto mode, relaxed division gates.
Plus M30_V2_LOCAL_CV_HARNESS_NOT_SUBMIT (non-submit).

Submit order depends on Stage 0: FULL -> A, B, E, D; sparse/ILP-like -> C, D, E
(skip A/B unless their guard says they are valid for the class). Never blindly
submit all variants.

OK_TO_SUBMIT_EXPERIMENTAL requires: artifact guard passes, edge_prob present +
non-degenerate, selected candidate mode matches the Stage-0 class, valid, no
fallback, no NaN, consecutive ids, no dangling, every edge exactly t->t+1,
max_in<=1 / max_out<=2, all dynamic test datasets present + non-empty,
110000<=n_node_rows<=165000, 100000<=n_edge_rows<=155000,
synthetic_nodes_added<=12000, divisions_total<=4000,
final_source=metric_aware_global_relinker_v2. Otherwise a specific DO_NOT_SUBMIT_*
is reported and a valid hidden-safe fallback is still written.

No supplied CSV is ever submitted or hardcoded; no Kaggle-API submit; dynamic
hidden-rerun-safe test discovery; off Kaggle it is a clean self-test-only dry
run. No prior milestone file is modified.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import blosc2
except ImportError:
    blosc2 = None

try:
    import zstandard
except ImportError:
    zstandard = None


# --------------------------------------------------------------------------- #
# 0. Constants: artifact locations, competition paths, hidden-name guard
# --------------------------------------------------------------------------- #
KAGGLE_COMPETITION_INPUT_DIR = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
KAGGLE_WORKING_DIR = "/kaggle/working"
TRACKING_REPO_DIR = "/kaggle/working/tracking_repo"

# Candidate mount points for the uploaded support-pack artifact, in
# preference order. The resolver requires the manifest + predict script +
# weights + wheels to be present before accepting a candidate.
ARTIFACT_CANDIDATE_DIRS = [
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1",
    "/kaggle/input/datasets/tom99763/biohub-tracking-support-pack-50ep-v1",
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-v1",
    # Broader fallbacks for other Kaggle mount layouts (dataset slug only).
    "/kaggle/input/biohub-tracking-support-pack-50ep-v1",
    "/kaggle/input/biohub-tracking-support-pack-v1",
]

ARTIFACT_REQUIRED_ENTRIES = [
    "ARTIFACT_MANIFEST.json",
    "repo/scripts/predict_unet_transformer.py",
    "weights/unet_transformer/split_0/edge_predictor_best.pth",
    "wheels",
]

# Core scientific packages pip must NEVER replace when installing the
# artifact's bundled wheels (Kaggle's preinstalled builds are ABI-matched
# to its CUDA/BLAS stack; letting a wheel swap them breaks torch/numba).
PROTECTED_PACKAGES = ["numpy", "scipy", "numba", "llvmlite", "torch", "pandas"]

# Two-phase offline install order (polars first so the Float16 shim below
# is importable before anything that depends on it).
WHEEL_INSTALL_PHASES = [
    ["polars"],
    ["tracksdata", "zarr", "pyscipopt", "geff", "geff-spec", "ilpy", "imagecodecs",
     "rustworkx", "numcodecs", "donfig", "bidict"],
]

# Fresh-subprocess import smoke test after install.
IMPORT_CHECK_MODULES = ["zarr", "tracksdata", "geff", "polars", "numcodecs", "rustworkx", "biohub_tracking"]

# Extra sys.path entries every subprocess needs (the reference repo layout).
SUBPROCESS_PATH_ENTRIES = [
    "/kaggle/working/tracking_repo/src",
    "/kaggle/working/tracking_repo",
    "/kaggle/working",
]

# Public dataset names that MUST NOT appear hardcoded in any submitted
# output - used only by a self-test that asserts we never emit them.
PUBLIC_TEST_DATASET_NAMES = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]

SUBMISSION_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]

# The polars Float16 compatibility shim written into sitecustomize.py at
# several locations so every subprocess picks it up on interpreter start.
SITECUSTOMIZE_CONTENT = (
    "try:\n"
    "    import polars as pl\n"
    "    if not hasattr(pl, \"Float16\"):\n"
    "        pl.Float16 = pl.Float32\n"
    "except Exception:\n"
    "    pass\n"
)

# --------------------------------------------------------------------------- #
# 1. Manual zarr v3 reader (copied from Milestone 15 - reads GEFF prediction
#    arrays without depending on a particular installed zarr version)
# --------------------------------------------------------------------------- #
DTYPE_CODES = {
    "bool": "b1",
    "int8": "i1", "int16": "i2", "int32": "i4", "int64": "i8",
    "uint8": "u1", "uint16": "u2", "uint32": "u4", "uint64": "u8",
    "float16": "f2", "float32": "f4", "float64": "f8",
}

NODE_PROP_NAMES = ("t", "z", "y", "x")

# Physical voxel scale for this competition's volumes (µm/voxel).
VOXEL_SIZE_UM = {"z": 1.625, "y": 0.40625, "x": 0.40625}
_VOXEL_SCALE = np.array([VOXEL_SIZE_UM["z"], VOXEL_SIZE_UM["y"], VOXEL_SIZE_UM["x"]])


# --------------------------------------------------------------------------- #
# 1. zarr.json metadata parsing (manual offline reader, no zarr/numcodecs)
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_array_metadata(array_dir: Path) -> dict:
    """Parse a zarr v3 array's `zarr.json` into shape/dtype/chunking/codecs."""
    meta_path = array_dir / "zarr.json"
    doc = read_json(meta_path)

    if doc.get("zarr_format") != 3:
        raise ValueError(f"{meta_path}: only zarr v3 is supported, got zarr_format={doc.get('zarr_format')!r}")
    if doc.get("node_type") != "array":
        raise ValueError(f"{meta_path}: not a zarr array (node_type={doc.get('node_type')!r})")

    shape = tuple(int(s) for s in doc["shape"])

    data_type = doc["data_type"]
    if isinstance(data_type, dict):  # some writers nest it as {"name": "uint16"}
        data_type = data_type.get("name")
    code = DTYPE_CODES.get(data_type)
    if code is None:
        raise ValueError(f"{meta_path}: unsupported data_type {data_type!r}")

    codecs = doc.get("codecs", [])
    endian = "little"
    for codec in codecs:
        if codec.get("name") in ("bytes", "endian"):
            endian = codec.get("configuration", {}).get("endian", "little")
            break
    dtype = np.dtype("bool") if code == "b1" else np.dtype(("<" if endian == "little" else ">") + code)

    chunk_grid = doc.get("chunk_grid", {})
    if chunk_grid.get("name", "regular") != "regular":
        raise NotImplementedError(f"{meta_path}: unsupported chunk_grid {chunk_grid.get('name')!r}")
    chunk_shape = tuple(int(c) for c in chunk_grid.get("configuration", {}).get("chunk_shape", shape))

    chunk_key_encoding = doc.get(
        "chunk_key_encoding", {"name": "default", "configuration": {"separator": "/"}}
    )
    fill_value = doc.get("fill_value", 0) or 0

    return {
        "path": array_dir,
        "shape": shape,
        "dtype": dtype,
        "chunk_shape": chunk_shape,
        "codecs": codecs,
        "chunk_key_encoding": chunk_key_encoding,
        "fill_value": fill_value,
    }


# --------------------------------------------------------------------------- #
# 2. Chunk key construction + codec decoding
# --------------------------------------------------------------------------- #
def build_chunk_path(array_dir: Path, index: Sequence[int], chunk_key_encoding: dict) -> Path:
    """Build the on-disk chunk file path for a given chunk index tuple."""
    name = chunk_key_encoding.get("name", "default")
    separator = chunk_key_encoding.get("configuration", {}).get("separator", "/")
    index_strs = [str(i) for i in index]

    if name == "v2":
        key = separator.join(index_strs) if index_strs else "0"
    else:
        key = separator.join(["c", *index_strs]) if index_strs else "c"

    if separator == "/":
        return array_dir.joinpath(*key.split("/"))
    return array_dir / key


def decode_bytes_codecs(raw: bytes, codecs: list[dict], expected_nbytes: int) -> bytes:
    """Reverse the bytes->bytes compressor chain, then hand back raw array bytes.

    `codecs[0]` is the array->bytes codec (must be "bytes"/"endian" - a plain
    passthrough once endianness is known); `codecs[1:]` are bytes->bytes
    compressors applied in encode order, so they're undone in reverse order.
    """
    if not codecs:
        return raw

    array_bytes_codec = codecs[0]
    if array_bytes_codec.get("name") not in ("bytes", "endian"):
        raise NotImplementedError(
            f"unsupported array->bytes codec: {array_bytes_codec.get('name')!r} "
            "(only the plain 'bytes' codec is supported - no sharding)"
        )

    data = raw
    for codec in reversed(codecs[1:]):
        codec_name = codec.get("name")
        if codec_name == "blosc":
            if blosc2 is None:
                raise RuntimeError("chunk uses the 'blosc' codec but the blosc2 package is not installed")
            data = blosc2.decompress(data)
        elif codec_name == "zstd":
            if zstandard is None:
                raise RuntimeError("chunk uses the 'zstd' codec but the zstandard package is not installed")
            data = zstandard.ZstdDecompressor().decompress(data, max_output_size=expected_nbytes)
        else:
            raise NotImplementedError(f"unsupported bytes->bytes codec: {codec_name!r}")

    return data


def decode_chunk_bytes(raw: bytes, codecs: list[dict], dtype: np.dtype, chunk_shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(chunk_shape)) if chunk_shape else 1
    expected_nbytes = count * dtype.itemsize
    decoded = decode_bytes_codecs(raw, codecs, expected_nbytes)
    arr = np.frombuffer(decoded, dtype=dtype, count=count)
    return arr.reshape(chunk_shape)


def read_chunk_at(meta: dict, index: Sequence[int]) -> np.ndarray:
    """Read+decode a single chunk. Missing chunk files are sparse and decode
    to an array filled with the array's fill_value (never raises for that case).
    """
    chunk_path = build_chunk_path(meta["path"], index, meta["chunk_key_encoding"])
    chunk_shape = meta["chunk_shape"]
    if not chunk_path.exists():
        return np.full(chunk_shape, meta["fill_value"], dtype=meta["dtype"])
    raw = chunk_path.read_bytes()
    return decode_chunk_bytes(raw, meta["codecs"], meta["dtype"], chunk_shape)


def read_full_zarr_array(array_dir: Path) -> np.ndarray:
    """Read an entire zarr v3 array (stitching together every chunk)."""
    meta = parse_array_metadata(array_dir)
    shape, chunk_shape = meta["shape"], meta["chunk_shape"]

    if not shape:  # 0-d array
        return read_chunk_at(meta, ())

    n_chunks = [math.ceil(shape[d] / chunk_shape[d]) if shape[d] else 0 for d in range(len(shape))]
    full = np.full(shape, meta["fill_value"], dtype=meta["dtype"])
    for index in itertools.product(*(range(n) for n in n_chunks)):
        chunk = read_chunk_at(meta, index)
        starts = [index[d] * chunk_shape[d] for d in range(len(shape))]
        ends = [min(starts[d] + chunk_shape[d], shape[d]) for d in range(len(shape))]
        dst = tuple(slice(s, e) for s, e in zip(starts, ends))
        src = tuple(slice(0, e - s) for s, e in zip(starts, ends))
        full[dst] = chunk[src]
    return full

# --------------------------------------------------------------------------- #
# 2. Environment detection + artifact resolution + repo materialization
# --------------------------------------------------------------------------- #
def is_kaggle_env() -> bool:
    """True only when the definitive Kaggle marker `/kaggle/input` exists.
    Deliberately NOT keyed on `/kaggle/working` - a local dry run can
    create that directory, and keying on it would flip an off-Kaggle
    execution into attempting a real (impossible) artifact run, breaking
    clean notebook execution off-Kaggle.
    """
    return Path("/kaggle/input").exists()


def _artifact_entry_status(artifact_dir: Path) -> dict:
    return {entry: (artifact_dir / entry).exists() for entry in ARTIFACT_REQUIRED_ENTRIES}


def resolve_artifact(candidate_dirs: Sequence[str] = ARTIFACT_CANDIDATE_DIRS) -> dict:
    """Returns the first candidate artifact mount that contains every
    required entry (manifest, predict script, weights, wheels). Never
    assumes availability - records which candidates were checked and why
    each was rejected, so a hidden rerun with a differently-named mount
    is diagnosable.
    """
    checked = []
    for candidate in candidate_dirs:
        artifact_dir = Path(candidate)
        if not artifact_dir.exists():
            checked.append({"path": candidate, "exists": False, "entries": {}})
            continue
        status = _artifact_entry_status(artifact_dir)
        checked.append({"path": candidate, "exists": True, "entries": status})
        if all(status.values()):
            return {"resolved": True, "artifact_dir": str(artifact_dir), "checked": checked}
    return {"resolved": False, "artifact_dir": None, "checked": checked}


def materialize_repo(artifact_dir: Path, working_dir: Path = Path(KAGGLE_WORKING_DIR)) -> dict:
    """Copies the artifact's `repo/` to `/kaggle/working/tracking_repo` and
    places the weights under `tracking_repo/weights` (symlink preferred,
    copy fallback). Idempotent: a re-run replaces an existing tracking_repo
    rather than erroring.
    """
    repo_src = artifact_dir / "repo"
    repo_dst = working_dir / "tracking_repo"
    if repo_dst.exists():
        shutil.rmtree(repo_dst)
    shutil.copytree(repo_src, repo_dst)

    weights_src = artifact_dir / "weights"
    weights_dst = repo_dst / "weights"
    weights_mode = None
    if weights_dst.exists() or weights_dst.is_symlink():
        if weights_dst.is_symlink() or weights_dst.is_file():
            weights_dst.unlink()
        else:
            shutil.rmtree(weights_dst)
    try:
        weights_dst.symlink_to(weights_src, target_is_directory=True)
        weights_mode = "symlink"
    except OSError:
        shutil.copytree(weights_src, weights_dst)
        weights_mode = "copy"

    return {
        "repo_src": str(repo_src), "repo_dst": str(repo_dst),
        "weights_src": str(weights_src), "weights_dst": str(weights_dst), "weights_mode": weights_mode,
    }


def write_sitecustomize(locations: Sequence[Path] | None = None) -> list[str]:
    """Writes the polars Float16 compatibility shim as sitecustomize.py at
    each location so every subprocess launched with these dirs on its path
    applies the patch on interpreter start.
    """
    if locations is None:
        locations = [Path(KAGGLE_WORKING_DIR), Path(TRACKING_REPO_DIR), Path(TRACKING_REPO_DIR) / "src"]
    written = []
    for loc in locations:
        loc.mkdir(parents=True, exist_ok=True)
        target = loc / "sitecustomize.py"
        target.write_text(SITECUSTOMIZE_CONTENT)
        written.append(str(target))
    return written


def build_subprocess_env(base_env: dict | None = None, path_entries: Sequence[str] = SUBPROCESS_PATH_ENTRIES) -> dict:
    """Returns an environment dict whose PYTHONPATH is prefixed with the
    reference repo's three required roots (preserving any existing
    PYTHONPATH), for every subprocess this runner launches.
    """
    env = dict(os.environ if base_env is None else base_env)
    existing = env.get("PYTHONPATH", "")
    parts = list(path_entries) + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env

# --------------------------------------------------------------------------- #
# 3. Dependency installation (offline wheels only) + import verification
# --------------------------------------------------------------------------- #
def build_pip_install_command(packages: Sequence[str], wheels_dir: Path) -> list[str]:
    """A single offline pip install command: `--no-index --find-links
    <wheels>` (only the bundled wheels are ever considered) and ALWAYS
    `--no-deps` (pip may never pull or resolve a replacement for a
    protected core package - or anything else - from an index).
    """
    return [
        sys.executable, "-m", "pip", "install", "--no-index",
        "--find-links", str(wheels_dir), "--no-deps", *packages,
    ]


def install_dependencies(wheels_dir: Path, env: dict, phases: Sequence[Sequence[str]] = WHEEL_INSTALL_PHASES) -> dict:
    """Runs the two-phase offline install (polars first, then the rest).
    Every command is `--no-index --find-links <wheels> --no-deps`, so no
    protected core package (numpy/scipy/numba/llvmlite/torch/pandas) can
    ever be swapped by dependency resolution. Records each phase's command
    + return code + captured output for the dependency report.
    """
    phase_results = []
    all_ok = True
    for i, packages in enumerate(phases):
        cmd = build_pip_install_command(packages, wheels_dir)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        ok = proc.returncode == 0
        all_ok = all_ok and ok
        phase_results.append({
            "phase": i, "packages": list(packages), "command": cmd, "returncode": proc.returncode,
            "ok": ok, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:],
        })
        print(f"  [install phase {i}] {' '.join(packages)} -> rc={proc.returncode}")
        if not ok:
            print(f"  [install phase {i}] FAILED:\n{proc.stderr[-2000:]}")
    return {
        "wheels_dir": str(wheels_dir), "protected_packages": PROTECTED_PACKAGES,
        "phases": phase_results, "all_ok": all_ok,
    }


def verify_imports(env: dict, modules: Sequence[str] = IMPORT_CHECK_MODULES) -> dict:
    """Runs a FRESH subprocess (so it picks up sitecustomize + the newly-
    installed wheels + PYTHONPATH) that imports every required module,
    printing OK/FAIL per module. Returns the parsed per-module result.
    """
    check_src = (
        "import importlib, json\n"
        f"mods = {list(modules)!r}\n"
        "res = {}\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "        res[m] = True\n"
        "    except Exception as e:\n"
        "        res[m] = f'ERROR: {type(e).__name__}: {e}'\n"
        "print('IMPORT_CHECK_JSON=' + json.dumps(res))\n"
    )
    proc = subprocess.run([sys.executable, "-c", check_src], env=env, capture_output=True, text=True)
    parsed = {}
    for line in proc.stdout.splitlines():
        if line.startswith("IMPORT_CHECK_JSON="):
            parsed = json.loads(line[len("IMPORT_CHECK_JSON="):])
            break
    all_ok = bool(parsed) and all(v is True for v in parsed.values())
    return {"returncode": proc.returncode, "modules": parsed, "all_ok": all_ok,
            "stderr_tail": proc.stderr[-2000:]}

# --------------------------------------------------------------------------- #
# 4. Dynamic split generation (hidden-rerun safe - never hardcodes public
#    dataset names)
# --------------------------------------------------------------------------- #
def _strip_zarr(name: str) -> str:
    return name[:-5] if name.endswith(".zarr") else name


def discover_test_datasets_from_dir(test_dir: Path) -> list[str]:
    """Discovers currently-mounted test dataset names from the test dir -
    either `<name>.zarr` stores or `<name>/` dataset directories - with
    the `.zarr` suffix stripped. Returns a sorted, de-duplicated list.
    """
    if not test_dir.exists():
        return []
    names = set()
    for child in test_dir.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            names.add(_strip_zarr(child.name))
    return sorted(names)


def discover_test_datasets_from_sample_submission(sample_submission_path: Path) -> list[str]:
    """Discovers test dataset names from the mounted sample_submission.csv's
    `dataset` column (the competition's own source of truth), `.zarr`
    stripped. Returns a sorted, de-duplicated list.
    """
    if not sample_submission_path.exists():
        return []
    df = pd.read_csv(sample_submission_path)
    if "dataset" not in df.columns:
        return []
    return sorted({_strip_zarr(str(d)) for d in df["dataset"].dropna().unique()})


def discover_test_datasets(test_dir: Path, sample_submission_path: Path) -> dict:
    """Prefers sample_submission's dataset column (competition source of
    truth); falls back to the mounted test dir. Returns both sources plus
    the chosen list, so the choice is auditable in the diagnostic.
    """
    from_submission = discover_test_datasets_from_sample_submission(sample_submission_path)
    from_dir = discover_test_datasets_from_dir(test_dir)
    if from_submission:
        chosen, source = from_submission, "sample_submission"
    elif from_dir:
        chosen, source = from_dir, "test_dir"
    else:
        chosen, source = [], "none"
    return {"chosen": chosen, "source": source, "from_sample_submission": from_submission, "from_test_dir": from_dir}


def build_canonical_splits(dataset_names: Sequence[str]) -> list:
    """The canonical splits schema the predict script accepts: a LIST whose
    element 0 is the fold-0 dict with `test` set to the discovered dataset
    names. predict_unet_transformer.py does `folds[fold]["test"]` with an
    INTEGER `fold` (0), so the top-level JSON MUST be a list - a
    `{"0": {...}}` dict would json.load into STRING keys and raise
    `KeyError: 0` at predict time.
    """
    return [{"test": list(dataset_names), "train": [], "val": []}]


def rewrite_template_splits(template, dataset_names: Sequence[str]) -> list | None:
    """Rewrites a template into a LIST whose element 0's `test` is the
    discovered names (predict indexes `folds[0]` with an integer). Handles:
      - list template:          [{"test": [...]}, ...]
      - splits-list template:   {"splits": [{"test": [...]}, ...]}
      - numbered-dict template: {"0": {"test": [...]}, "1": {...}}
    Fold 0's other keys (train/val) are preserved where present. Returns
    None for an unrecognized shape so the caller rebuilds canonically. The
    result is ALWAYS a list - never a `{"0": ...}` dict.
    """
    names = list(dataset_names)
    if isinstance(template, list) and template and isinstance(template[0], dict):
        new = [dict(e) if isinstance(e, dict) else e for e in template]
        new[0] = dict(new[0])
        new[0]["test"] = names
        return new
    if isinstance(template, dict) and isinstance(template.get("splits"), list) and template["splits"] and isinstance(template["splits"][0], dict):
        new = [dict(e) if isinstance(e, dict) else e for e in template["splits"]]
        new[0] = dict(new[0])
        new[0]["test"] = names
        return new
    if isinstance(template, dict) and isinstance(template.get("0"), dict):
        split0 = dict(template["0"])
        split0["test"] = names
        return [split0]
    return None


def _dataset_present_in_test_dir(test_dir: Path, name: str) -> bool:
    return (test_dir / name).exists() or (test_dir / f"{name}.zarr").exists()


def prepare_splits(
    test_dir: Path, sample_submission_path: Path, template_path: Path | None,
    out_splits_path: Path, diagnostic_path: Path,
) -> dict:
    """Builds a splits file for the CURRENTLY MOUNTED test data:
      1. discover the mounted dataset names,
      2. rewrite a template's split-0 test list with them (or build
         canonically if there's no template / an unrecognized shape),
      3. VERIFY every split-0 name actually exists in the test dir -
         rebuild canonically from only the verified-present names if not,
      4. write the splits file the predict script will read, plus a
         diagnostic recording exactly what was discovered and written.
    Public dataset names are never hardcoded; the file only ever contains
    names that exist on disk right now.
    """
    discovery = discover_test_datasets(test_dir, sample_submission_path)
    names = discovery["chosen"]

    template = None
    template_used = False
    if template_path is not None and template_path.exists():
        try:
            template = json.loads(template_path.read_text())
        except Exception as exc:
            print(f"  [splits] could not parse template {template_path}: {exc!r} - building canonically")

    splits = None
    if template is not None:
        splits = rewrite_template_splits(template, names)
        template_used = splits is not None
    if splits is None:
        splits = build_canonical_splits(names)

    # `splits` is ALWAYS a list now - fold 0 is `splits[0]` (predict indexes
    # folds[0] with an integer). Verify split-0 test names exist on disk; if
    # any is missing (e.g. a template nested names somewhere we didn't
    # rewrite), rebuild canonically from only the verified-present names.
    split0 = splits[0] if isinstance(splits, list) and splits and isinstance(splits[0], dict) else {}
    split0_test = list(split0.get("test", []))

    verified_present = [n for n in split0_test if _dataset_present_in_test_dir(test_dir, n)]
    missing = [n for n in split0_test if not _dataset_present_in_test_dir(test_dir, n)]
    rebuilt_canonical = False
    if missing or not split0_test:
        present_names = [n for n in names if _dataset_present_in_test_dir(test_dir, n)] or names
        splits = build_canonical_splits(present_names)
        rebuilt_canonical = True
        template_used = False
        split0_test = present_names
        verified_present = [n for n in present_names if _dataset_present_in_test_dir(test_dir, n)]
        missing = [n for n in present_names if not _dataset_present_in_test_dir(test_dir, n)]

    out_splits_path.parent.mkdir(parents=True, exist_ok=True)
    out_splits_path.write_text(json.dumps(splits, indent=2))

    # Verify the WRITTEN file matches EXACTLY how predict_unet_transformer.py
    # reads it: `folds = json.load(f); test_names = folds[fold]["test"]` with
    # `fold == 0` (an integer). The file MUST be a list and folds[0]["test"]
    # MUST equal the discovered names - the exact regression the KeyError: 0
    # fallback surfaced (a {"0": ...} dict json.loads to string keys).
    with open(out_splits_path) as f:
        folds = json.load(f)
    assert isinstance(folds, list), "splits file must be a JSON list (predict indexes folds[0] with an integer)"
    assert len(folds) > 0, "splits list must be non-empty"
    assert isinstance(folds[0], dict), "splits[0] must be a dict"
    assert folds[0]["test"] == split0_test, "folds[0]['test'] must equal the discovered test names"

    diagnostic = {
        "discovered": discovery, "template_path": str(template_path) if template_path else None,
        "template_used": template_used, "rebuilt_canonical": rebuilt_canonical,
        "split0_test_names": split0_test, "verified_present": verified_present, "missing": missing,
        "out_splits_path": str(out_splits_path),
        "final_schema": "list", "predict_index_check": "folds[0]['test'] OK",
    }
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    print(f"  [splits] wrote {out_splits_path} (list schema) with {len(split0_test)} test dataset(s): {split0_test}")
    return {"splits": splits, "split0_test_names": split0_test, "diagnostic": diagnostic}

# --------------------------------------------------------------------------- #
# 5. Prediction command builder + runner
# --------------------------------------------------------------------------- #
PREDICT_SCRIPT_REL = "scripts/predict_unet_transformer.py"
WEIGHTS_REL = "weights/unet_transformer/split_0/edge_predictor_best.pth"
SPLITS_FILENAME = "kaggle_test_splits_50ep.json"

PREDICT_HYPERPARAMS = {
    "unet_batch_size": "4", "det_threshold": "0.99", "ilp_edge_weight": "-1.0",
    "ilp_appearance_weight": "0.1", "ilp_disappearance_weight": "0.1", "ilp_division_weight": "1.0",
}


def build_predict_command(
    data_dir: str = f"{KAGGLE_COMPETITION_INPUT_DIR}/test", splits_filename: str = SPLITS_FILENAME,
    split: str = "0", weights_rel: str = WEIGHTS_REL, hyperparams: dict = PREDICT_HYPERPARAMS,
) -> list[str]:
    """Builds the exact GPU predict command (run with cwd=tracking_repo),
    mirroring the successful manual public run. `--use-ilp` is a bare flag.
    """
    return [
        sys.executable, PREDICT_SCRIPT_REL,
        "--data-dir", data_dir,
        "--splits", splits_filename,
        "--split", split,
        "--weights", weights_rel,
        "--unet-batch-size", hyperparams["unet_batch_size"],
        "--det-threshold", hyperparams["det_threshold"],
        "--ilp-edge-weight", hyperparams["ilp_edge_weight"],
        "--ilp-appearance-weight", hyperparams["ilp_appearance_weight"],
        "--ilp-disappearance-weight", hyperparams["ilp_disappearance_weight"],
        "--ilp-division-weight", hyperparams["ilp_division_weight"],
        "--use-ilp",
    ]


def run_prediction(command: Sequence[str], cwd: Path, env: dict) -> dict:
    """Runs the predict command from `cwd` (tracking_repo). Streams nothing
    but captures stdout/stderr tails and the return code for the execute
    report. Does not raise on a nonzero return code - the caller decides
    whether that is fatal (the top-level fallback handles it).
    """
    proc = subprocess.run(list(command), cwd=str(cwd), env=env, capture_output=True, text=True)
    result = {
        "command": list(command), "cwd": str(cwd), "returncode": proc.returncode,
        "ok": proc.returncode == 0, "stdout_tail": proc.stdout[-8000:], "stderr_tail": proc.stderr[-8000:],
    }
    print(f"  [predict] rc={proc.returncode}")
    if proc.returncode != 0:
        print(f"  [predict] stderr tail:\n{proc.stderr[-2000:]}")
    return result

# --------------------------------------------------------------------------- #
# 6. GEFF -> submission conversion
# --------------------------------------------------------------------------- #
def _read_geff_array(geff_root: Path, rel_path: str):
    """Reads one GEFF array (e.g. "nodes/ids", "nodes/props/t/values") if
    present, else returns None. Uses the version-independent manual zarr
    reader first; falls back to the installed `zarr` package (guaranteed
    present on Kaggle from the artifact wheels) if a chunk codec the manual
    reader lacks was used to write the predictions.
    """
    array_dir = geff_root / Path(rel_path)
    if not (array_dir / "zarr.json").exists():
        return None
    try:
        return read_full_zarr_array(array_dir)
    except Exception:
        try:
            import zarr  # noqa: F401
            return zarr.open(store=str(geff_root), path=rel_path, mode="r")[:]
        except Exception as exc:
            print(f"  [geff] could not read {rel_path} under {geff_root}: {exc!r}")
            return None


def _geff_root_of(candidate: Path) -> Path | None:
    """Returns `candidate` if it is itself a GEFF store, else searches one
    level down for a nested GEFF store (some writers emit `<name>/tracks`).
    """
    if (candidate / "nodes" / "ids" / "zarr.json").exists():
        return candidate
    if candidate.is_dir():
        for child in candidate.iterdir():
            if child.is_dir() and (child / "nodes" / "ids" / "zarr.json").exists():
                return child
    return None


def read_geff_graph(geff_root: Path) -> dict:
    """Reads a GEFF prediction store into plain arrays: node ids + t/z/y/x,
    optional node `solution` mask (keep only selected nodes), edge id pairs,
    optional edge `solution` mask (keep only selected edges), and optional
    `edge_prob` / `edge_dist` used only for deterministic tie-breaking when
    enforcing degree constraints.
    """
    node_ids = _read_geff_array(geff_root, "nodes/ids")
    if node_ids is None:
        raise ValueError(f"{geff_root}: no nodes/ids array")
    node_ids = np.asarray(node_ids).reshape(-1)
    t = _read_geff_array(geff_root, "nodes/props/t/values")
    z = _read_geff_array(geff_root, "nodes/props/z/values")
    y = _read_geff_array(geff_root, "nodes/props/y/values")
    x = _read_geff_array(geff_root, "nodes/props/x/values")
    node_solution = _read_geff_array(geff_root, "nodes/props/solution/values")

    edge_ids = _read_geff_array(geff_root, "edges/ids")
    edge_solution = _read_geff_array(geff_root, "edges/props/solution/values")
    edge_prob = _read_geff_array(geff_root, "edges/props/edge_prob/values")
    edge_dist = _read_geff_array(geff_root, "edges/props/edge_dist/values")

    def _col(arr, n):
        return np.asarray(arr).reshape(-1) if arr is not None else np.zeros(n, dtype=float)

    n_nodes = len(node_ids)
    nodes = pd.DataFrame({
        "raw_id": node_ids.astype(np.int64),
        "t": _col(t, n_nodes), "z": _col(z, n_nodes), "y": _col(y, n_nodes), "x": _col(x, n_nodes),
    })
    if node_solution is not None:
        mask = np.asarray(node_solution).reshape(-1).astype(bool)
        nodes = nodes[mask].reset_index(drop=True)

    if edge_ids is None or np.asarray(edge_ids).size == 0:
        edges = pd.DataFrame(columns=["raw_source", "raw_target", "edge_prob", "edge_dist"])
    else:
        edge_ids = np.asarray(edge_ids)
        if edge_ids.ndim == 1:
            edge_ids = edge_ids.reshape(-1, 2)
        n_edges = edge_ids.shape[0]
        edges = pd.DataFrame({
            "raw_source": edge_ids[:, 0].astype(np.int64), "raw_target": edge_ids[:, 1].astype(np.int64),
            "edge_prob": _col(edge_prob, n_edges), "edge_dist": _col(edge_dist, n_edges),
        })
        if edge_solution is not None:
            emask = np.asarray(edge_solution).reshape(-1).astype(bool)
            edges = edges[emask].reset_index(drop=True)

    return {"nodes": nodes, "edges": edges,
            "has_edge_prob": edge_prob is not None, "has_edge_dist": edge_dist is not None}


def convert_geff_graph_to_rows(
    geff_graph: dict, dataset_name: str, next_id: int, next_node_id: int,
) -> tuple[list[dict], int, int, dict]:
    """Converts one GEFF graph into submission node/edge rows, threading
    global-consecutive `id` and globally-unique `node_id` counters across
    datasets. Remaps raw GEFF node ids to positive consecutive Kaggle
    node_ids (per dataset); keeps only edges whose BOTH endpoints survived
    node selection; requires target_t > source_t; drops duplicate directed
    edges; enforces in_degree <= 1 and out_degree <= 2, breaking ties by
    higher edge_prob then lower edge_dist then stable order.
    """
    nodes = geff_graph["nodes"].copy()
    edges = geff_graph["edges"].copy()

    # Remap raw -> consecutive Kaggle node_id (globally unique via next_node_id).
    raw_to_kaggle: dict[int, int] = {}
    node_rows = []
    node_t = {}
    kid = next_node_id
    for row in nodes.itertuples():
        raw_to_kaggle[int(row.raw_id)] = kid
        node_t[kid] = int(row.t)
        node_rows.append({"kaggle_node_id": kid, "t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x)})
        kid += 1
    next_node_id_out = kid

    stats = {
        "dataset": dataset_name, "n_geff_nodes": len(nodes), "n_geff_edges": len(edges),
        "n_edges_dangling_dropped": 0, "n_edges_time_order_dropped": 0,
        "n_edges_duplicate_dropped": 0, "n_edges_indegree_dropped": 0, "n_edges_outdegree_dropped": 0,
    }

    kept_edges = []
    for row in edges.itertuples():
        s_raw, t_raw = int(row.raw_source), int(row.raw_target)
        if s_raw not in raw_to_kaggle or t_raw not in raw_to_kaggle:
            stats["n_edges_dangling_dropped"] += 1
            continue
        s_kid, t_kid = raw_to_kaggle[s_raw], raw_to_kaggle[t_raw]
        if not (node_t[t_kid] > node_t[s_kid]):
            stats["n_edges_time_order_dropped"] += 1
            continue
        kept_edges.append({
            "source_id": s_kid, "target_id": t_kid,
            "edge_prob": float(row.edge_prob), "edge_dist": float(row.edge_dist),
        })

    edges_df = pd.DataFrame(kept_edges, columns=["source_id", "target_id", "edge_prob", "edge_dist"])

    if len(edges_df):
        before = len(edges_df)
        edges_df = edges_df.drop_duplicates(subset=["source_id", "target_id"], keep="first")
        stats["n_edges_duplicate_dropped"] = before - len(edges_df)

        # Rank edges best-first: higher edge_prob, then lower edge_dist, then
        # stable order - so degree pruning keeps the most-confident edges.
        edges_df = edges_df.assign(_order=np.arange(len(edges_df)))
        edges_df = edges_df.sort_values(
            ["edge_prob", "edge_dist", "_order"], ascending=[False, True, True]
        ).reset_index(drop=True)

        # in_degree <= 1: each target keeps only its single best incoming edge.
        before = len(edges_df)
        edges_df = edges_df.drop_duplicates(subset=["target_id"], keep="first")
        stats["n_edges_indegree_dropped"] = before - len(edges_df)

        # out_degree <= 2: each source keeps at most its 2 best outgoing edges.
        before = len(edges_df)
        edges_df = edges_df.groupby("source_id", sort=False, group_keys=False).head(2)
        stats["n_edges_outdegree_dropped"] = before - len(edges_df)

    rows = []
    cur_id = next_id
    for nr in node_rows:
        rows.append({
            "id": cur_id, "dataset": dataset_name, "row_type": "node", "node_id": nr["kaggle_node_id"],
            "t": nr["t"], "z": nr["z"], "y": nr["y"], "x": nr["x"], "source_id": -1, "target_id": -1,
        })
        cur_id += 1
    for er in edges_df.itertuples() if len(edges_df) else []:
        rows.append({
            "id": cur_id, "dataset": dataset_name, "row_type": "edge", "node_id": -1,
            "t": -1, "z": -1, "y": -1, "x": -1, "source_id": int(er.source_id), "target_id": int(er.target_id),
        })
        cur_id += 1

    stats["n_submission_nodes"] = len(node_rows)
    stats["n_submission_edges"] = int(len(edges_df))
    return rows, cur_id, next_node_id_out, stats


def find_prediction_geff_stores(predictions_dir: Path) -> list[tuple[str, Path]]:
    """Finds each dataset's GEFF prediction store under `predictions_dir`,
    returning (dataset_name, geff_root) pairs. The dataset name is the store
    directory's stem with any `.geff`/`.zarr` suffix stripped.
    """
    results = []
    if not predictions_dir.exists():
        return results
    for child in sorted(predictions_dir.iterdir()):
        if child.name.startswith("."):
            continue
        geff_root = _geff_root_of(child)
        if geff_root is not None:
            name = _strip_zarr(child.name)
            if name.endswith(".geff"):
                name = name[:-5]
            results.append((name, geff_root))
    return results


def convert_predictions_to_submission(predictions_dir: Path, expected_datasets: Sequence[str]) -> tuple[pd.DataFrame, dict]:
    """Converts every GEFF prediction store under `predictions_dir` into one
    combined submission dataframe with globally-consecutive `id` and
    globally-unique `node_id`. Records per-dataset conversion stats.
    """
    stores = find_prediction_geff_stores(predictions_dir)
    all_rows = []
    per_dataset = []
    next_id, next_node_id = 0, 0
    for dataset_name, geff_root in stores:
        graph = read_geff_graph(geff_root)
        rows, next_id, next_node_id, stats = convert_geff_graph_to_rows(graph, dataset_name, next_id, next_node_id)
        all_rows.extend(rows)
        per_dataset.append(stats)
        print(f"  [geff] {dataset_name}: {stats['n_submission_nodes']} node(s), {stats['n_submission_edges']} edge(s)")

    submission_df = pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)
    conversion = {
        "predictions_dir": str(predictions_dir),
        "n_stores_found": len(stores), "found_datasets": [n for n, _ in stores],
        "expected_datasets": list(expected_datasets), "per_dataset": per_dataset,
    }
    return submission_df, conversion

# --------------------------------------------------------------------------- #
# 7. Validation (hidden-safe) + fallback writer + diagnostics
# --------------------------------------------------------------------------- #
def _per_dataset_degree_extremes(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[int, int, int, int]:
    """Returns (max_in_degree, max_out_degree, n_dangling, n_duplicate)
    across all datasets, scoping node-id lookups per dataset (a foreign
    submission's node_id is only unique WITHIN a dataset).
    """
    max_in, max_out, dangling, duplicate = 0, 0, 0, 0
    for dataset, e in edges.groupby("dataset"):
        valid_ids = set(nodes.loc[nodes["dataset"] == dataset, "node_id"])
        good = e[e["source_id"].isin(valid_ids) & e["target_id"].isin(valid_ids)]
        dangling += len(e) - len(good)
        deduped = good.drop_duplicates(subset=["source_id", "target_id"])
        duplicate += len(good) - len(deduped)
        if len(deduped):
            max_out = max(max_out, int(deduped.groupby("source_id").size().max()))
            max_in = max(max_in, int(deduped.groupby("target_id").size().max()))
    return max_in, max_out, dangling, duplicate


def validate_reference_submission(df: pd.DataFrame, expected_datasets: Sequence[str]) -> tuple[bool, str, dict]:
    """Hidden-safe structural + topology validation. The expected dataset
    set comes from the caller (mounted sample_submission / test data), never
    a hardcoded public set. Allows out_degree up to 2 (divisions) but
    forbids in_degree > 1 (multi-parent), dangling edges, and duplicate
    directed edges. Returns (all_passed, report, stats).
    """
    checks = []

    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("columns match required schema", list(df.columns) == SUBMISSION_COLUMNS, f"got {list(df.columns)}")
    na_cols = df.columns[df.isna().any()].tolist() if len(df) else []
    check("no missing values", not na_cols, f"NaN columns: {na_cols}")
    check("id consecutive from 0", df["id"].tolist() == list(range(len(df))))

    present = set(df["dataset"].unique()) if len(df) else set()
    expected = set(expected_datasets)
    check("every expected dataset present", not (expected - present), f"missing: {sorted(expected - present)}")
    check("no foreign/hardcoded-public datasets", not (present - expected), f"foreign: {sorted(present - expected)}")
    check("row_type only node/edge", set(df["row_type"].unique()) <= {"node", "edge"} if len(df) else True)

    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    node_id_unique_per_dataset = all(g["node_id"].is_unique for _, g in nodes.groupby("dataset")) if len(nodes) else True
    check("node_id unique per dataset", node_id_unique_per_dataset)

    max_in, max_out, dangling, duplicate = _per_dataset_degree_extremes(nodes, edges)
    check("no dangling edges", dangling == 0, f"{dangling} dangling")
    check("no duplicate directed edges", duplicate == 0, f"{duplicate} duplicate")
    check("no in_degree > 1", max_in <= 1, f"max_in_degree={max_in}")
    check("no out_degree > 2", max_out <= 2, f"max_out_degree={max_out}")

    lines = ["Reference submission validation:"]
    all_passed = True
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_passed = False
        line = f"  [{mark}] {c['name']}"
        if c["detail"] and not c["passed"]:
            line += f"  -- {c['detail']}"
        lines.append(line)
    stats = {"n_node_rows": len(nodes), "n_edge_rows": len(edges), "max_in_degree": max_in,
             "max_out_degree": max_out, "dangling": dangling, "duplicate": duplicate,
             "datasets": sorted(present)}
    return all_passed, "\n".join(lines), stats


def compute_edge_distance_summary(df: pd.DataFrame) -> dict:
    """Physical (µm) edge-length percentiles over all valid edges, using
    the competition's per-axis voxel scale.
    """
    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    if len(edges) == 0 or len(nodes) == 0:
        return {}
    coord = {}
    for row in nodes.itertuples():
        coord[(row.dataset, int(row.node_id))] = (float(row.z), float(row.y), float(row.x))
    dists = []
    for row in edges.itertuples():
        s = coord.get((row.dataset, int(row.source_id)))
        t = coord.get((row.dataset, int(row.target_id)))
        if s is None or t is None:
            continue
        dz = (t[0] - s[0]) * VOXEL_SIZE_UM["z"]
        dy = (t[1] - s[1]) * VOXEL_SIZE_UM["y"]
        dx = (t[2] - s[2]) * VOXEL_SIZE_UM["x"]
        dists.append(float(np.sqrt(dz * dz + dy * dy + dx * dx)))
    if not dists:
        return {}
    s = pd.Series(dists)
    return {"mean": float(s.mean()), "median": float(s.median()), "p90": float(s.quantile(0.90)),
            "p95": float(s.quantile(0.95)), "p99": float(s.quantile(0.99)), "max": float(s.max())}


def write_fallback_submission(sample_submission_path: Path, out_path: Path) -> dict:
    """Writes a VALID (if low-scoring) submission.csv derived from the
    mounted sample_submission.csv: exact columns, no missing values (node
    rows get source/target = -1; edge rows get node/t/z/y/x = -1; any
    remaining NaN -> -1), id renumbered consecutively from 0. Used only
    when the learned pipeline failed and no submission.csv exists, so a
    hidden failure still produces a scorable file instead of an exception.
    """
    df = pd.read_csv(sample_submission_path)
    for col in SUBMISSION_COLUMNS:
        if col not in df.columns:
            df[col] = -1
    df = df[SUBMISSION_COLUMNS].copy()
    if "row_type" in df.columns and len(df):
        is_node = df["row_type"] == "node"
        is_edge = df["row_type"] == "edge"
        for col in ("source_id", "target_id"):
            df.loc[is_node, col] = df.loc[is_node, col].fillna(-1)
        for col in ("node_id", "t", "z", "y", "x"):
            df.loc[is_edge, col] = df.loc[is_edge, col].fillna(-1)
    df = df.fillna(-1)
    df["id"] = range(len(df))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return {"source": str(sample_submission_path), "out_path": str(out_path), "shape": list(df.shape)}


def write_reference_diagnostics(df: pd.DataFrame, valid: bool, stats: dict, out_dir: Path) -> None:
    """Writes the variant-counts, topology-validation, and edge-distance
    diagnostic CSVs for the reference submission.
    """
    n_nodes = int((df["row_type"] == "node").sum()) if len(df) else 0
    n_edges = int((df["row_type"] == "edge").sum()) if len(df) else 0
    pd.DataFrame([{"variant": "reference_learned_graph", "valid": valid, "total_rows": len(df),
                   "n_node_rows": n_nodes, "n_edge_rows": n_edges}]).to_csv(
        out_dir / "milestone16_reference_variant_counts.csv", index=False)
    pd.DataFrame([{"variant": "reference_learned_graph", "max_in_degree": stats.get("max_in_degree"),
                   "max_out_degree": stats.get("max_out_degree"), "dangling": stats.get("dangling"),
                   "duplicate": stats.get("duplicate")}]).to_csv(
        out_dir / "milestone16_reference_topology_validation.csv", index=False)
    edge_summary = compute_edge_distance_summary(df)
    pd.DataFrame([{"variant": "reference_learned_graph", **edge_summary}]).to_csv(
        out_dir / "milestone16_reference_edge_distance_stats.csv", index=False)

# --------------------------------------------------------------------------- #
# 8. Orchestration + top-level graceful fallback
# --------------------------------------------------------------------------- #
_RUN_LOG: list[str] = []


def log(msg: str) -> None:
    print(msg)
    _RUN_LOG.append(str(msg))


def _write_log(out_dir: Path) -> None:
    try:
        (out_dir / "milestone16_reference_runner.log").write_text("\n".join(_RUN_LOG) + "\n")
    except Exception as exc:
        print(f"[warn] could not write runner log: {exc!r}")


def run_pipeline(working_dir: str = KAGGLE_WORKING_DIR, competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    """The full learned-graph flow: resolve artifact -> materialize repo ->
    sitecustomize -> offline wheel install -> import check -> DYNAMIC splits
    -> GPU predict -> GEFF->submission conversion -> validation -> write
    submission.csv. Raises on any fatal step so the top-level fallback can
    guarantee a valid submission.csv still exists.
    """
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = Path(competition_dir)
    test_dir = comp / "test"
    sample_submission_path = comp / "sample_submission.csv"

    log("=== Milestone 16: reference learned-graph runner ===")

    artifact = resolve_artifact()
    (out_dir / "milestone16_artifact_selection.json").write_text(json.dumps(artifact, indent=2))
    if not artifact["resolved"]:
        raise RuntimeError("support-pack artifact not found at any candidate mount (see milestone16_artifact_selection.json)")
    artifact_dir = Path(artifact["artifact_dir"])
    log(f"Artifact resolved: {artifact_dir}")

    materialize = materialize_repo(artifact_dir, out_dir)
    log(f"Repo materialized to {materialize['repo_dst']} (weights: {materialize['weights_mode']})")
    write_sitecustomize()
    env = build_subprocess_env()

    dep_report = install_dependencies(artifact_dir / "wheels", env)
    (out_dir / "milestone16_dependency_report.json").write_text(json.dumps(dep_report, indent=2))
    if not dep_report["all_ok"]:
        raise RuntimeError("offline wheel install failed (see milestone16_dependency_report.json)")

    import_report = verify_imports(env)
    if not import_report["all_ok"]:
        raise RuntimeError(f"post-install import check failed: {import_report['modules']}")
    log(f"Imports OK: {list(import_report['modules'].keys())}")

    repo_dst = Path(materialize["repo_dst"])
    shipped_template = repo_dst / SPLITS_FILENAME
    splits_result = prepare_splits(
        test_dir=test_dir, sample_submission_path=sample_submission_path,
        template_path=shipped_template if shipped_template.exists() else None,
        out_splits_path=repo_dst / SPLITS_FILENAME, diagnostic_path=out_dir / "milestone16_splits.json",
    )

    command = build_predict_command()
    (out_dir / "milestone16_run_command.json").write_text(json.dumps({"command": command, "cwd": str(repo_dst)}, indent=2))
    log(f"Predict command: {' '.join(command)}")

    exec_result = run_prediction(command, cwd=repo_dst, env=env)
    (out_dir / "milestone16_execute_result.json").write_text(json.dumps(exec_result, indent=2))
    if not exec_result["ok"]:
        raise RuntimeError("predict script returned nonzero (see milestone16_execute_result.json)")

    predictions_dir = repo_dst / "predictions" / "unknown" / "unet_transformer" / "split_0"
    expected_datasets = discover_test_datasets(test_dir, sample_submission_path)["chosen"]
    submission_df, conversion = convert_predictions_to_submission(predictions_dir, expected_datasets)
    (out_dir / "milestone16_geff_conversion.json").write_text(json.dumps(conversion, indent=2, default=str))

    valid, report, stats = validate_reference_submission(submission_df, expected_datasets)
    log(report)
    if not valid:
        raise RuntimeError("converted reference submission FAILED validation (falling back)")

    submission_path = out_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    write_reference_diagnostics(submission_df, valid, stats, out_dir)

    report_obj = {
        "artifact_dir": str(artifact_dir), "expected_datasets": expected_datasets,
        "split0_test_names": splits_result["split0_test_names"], "submission_shape": list(submission_df.shape),
        "n_node_rows": stats["n_node_rows"], "n_edge_rows": stats["n_edge_rows"],
        "max_in_degree": stats["max_in_degree"], "max_out_degree": stats["max_out_degree"],
        "valid": valid, "final_source": "reference_learned_graph",
        "final_instruction": "submission.csv built and validated. Do not submit until user reviews.",
    }
    (out_dir / "milestone16_reference_submission_report.json").write_text(json.dumps(report_obj, indent=2, default=str))
    log(f"submission.csv written: shape={submission_df.shape}, source=reference_learned_graph")
    log("submission.csv built and validated. Do not submit until user reviews.")
    return {"submission_df": submission_df, "final_source": "reference_learned_graph", "valid": valid, "report": report_obj}


def run_with_fallback(working_dir: str = KAGGLE_WORKING_DIR, competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    """Runs the learned pipeline; on ANY failure, if no submission.csv was
    produced, writes a valid fallback from sample_submission.csv so a
    hidden rerun still yields a scorable file instead of a thrown
    exception. Re-raises ONLY if even the fallback cannot be written.
    """
    _RUN_LOG.clear()
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / "submission.csv"
    sample_submission_path = Path(competition_dir) / "sample_submission.csv"
    try:
        result = run_pipeline(working_dir, competition_dir)
        _write_log(out_dir)
        return {"status": "ok", **result}
    except Exception:
        tb = traceback.format_exc()
        log("[error] learned pipeline failed:\n" + tb)
        if submission_path.exists():
            log("submission.csv already exists; keeping it (not overwriting with fallback).")
            _write_log(out_dir)
            return {"status": "pipeline_failed_submission_exists"}
        try:
            fb = write_fallback_submission(sample_submission_path, submission_path)
            (out_dir / "milestone16_failure_fallback_report.json").write_text(
                json.dumps({"traceback": tb, "fallback": fb}, indent=2))
            log("FALLBACK sample submission written because learned graph pipeline failed.")
            _write_log(out_dir)
            return {"status": "fallback_written", "fallback": fb}
        except Exception:
            fb_tb = traceback.format_exc()
            try:
                (out_dir / "milestone16_failure_fallback_report.json").write_text(
                    json.dumps({"traceback": tb, "fallback_error": fb_tb}, indent=2))
            except Exception:
                pass
            log("[fatal] could not write fallback submission either:\n" + fb_tb)
            _write_log(out_dir)
            raise

# --------------------------------------------------------------------------- #
# 11. Artifact introspection (runtime discovery of the reference's actual
#     flags, weight splits, command, and post-processing scripts)
# --------------------------------------------------------------------------- #
import re

# argparse flags that look like recovery/relinking/division/postprocess/
# ensemble knobs - used to auto-detect whether Variant B is available.
GAP_RELINK_FLAG_KEYWORDS = ("gap", "relink", "recover", "division", "merge",
                            "postprocess", "post-process", "ensemble", "smooth", "close")


def list_weight_splits(artifact_dir: Path) -> list[int]:
    """Lists the split indices that actually ship an edge_predictor_best.pth
    (weights/unet_transformer/split_<i>/edge_predictor_best.pth).
    """
    base = artifact_dir / "weights" / "unet_transformer"
    splits: list[int] = []
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir() and d.name.startswith("split_") and (d / "edge_predictor_best.pth").exists():
                try:
                    splits.append(int(d.name.split("_", 1)[1]))
                except ValueError:
                    continue
    return sorted(splits)


def discover_predict_flags(predict_script_path: Path) -> list[str]:
    """Greps the predict script's source for every argparse flag it
    declares (`add_argument("--flag")`) - the definitive list of what the
    script will accept, without ever executing it.
    """
    if not predict_script_path.exists():
        return []
    text = predict_script_path.read_text(errors="replace")
    flags = set(re.findall(r"add_argument\(\s*['\"](--[A-Za-z0-9][A-Za-z0-9\-_]*)['\"]", text))
    return sorted(flags)


def discover_reference_commands(artifact_dir: Path, max_hits: int = 60) -> list[dict]:
    """Scans the shipped repo/notebooks/readmes for lines that invoke
    predict_unet_transformer.py - i.e. the original reference command(s),
    so M17 can reproduce them exactly.
    """
    hits: list[dict] = []
    exts = {".py", ".ipynb", ".md", ".sh", ".txt", ".json", ".cfg", ".yaml", ".yml"}
    try:
        for p in artifact_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            try:
                text = p.read_text(errors="replace")
            except Exception:
                continue
            if "predict_unet_transformer" not in text:
                continue
            for line in text.splitlines():
                if "predict_unet_transformer" in line:
                    hits.append({"file": str(p.relative_to(artifact_dir)), "line": line.strip()[:400]})
                    if len(hits) >= max_hits:
                        return hits
    except OSError:
        pass
    return hits


def list_postprocess_scripts(artifact_dir: Path) -> list[str]:
    scripts_dir = artifact_dir / "repo" / "scripts"
    if not scripts_dir.exists():
        return []
    return sorted(p.name for p in scripts_dir.glob("*.py"))


def read_manifest(artifact_dir: Path) -> dict:
    manifest_path = artifact_dir / "ARTIFACT_MANIFEST.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(errors="replace"))
    except Exception:
        return {}


def introspect_artifact(artifact_dir: Path, out_dir: Path) -> dict:
    """Discovers everything M17's variants need from the mounted artifact -
    accepted predict flags, available weight splits, the original reference
    command(s), post-processing scripts, and manifest metadata - and writes
    milestone17_artifact_introspection.json. This is the step that actually
    answers "what settings did the reference use / what can we tune", at
    runtime on Kaggle, without ever guessing.
    """
    predict_script = artifact_dir / "repo" / "scripts" / "predict_unet_transformer.py"
    predict_flags = discover_predict_flags(predict_script)
    gap_relink_flags = [f for f in predict_flags if any(kw in f.lower() for kw in GAP_RELINK_FLAG_KEYWORDS)]

    report = {
        "artifact_dir": str(artifact_dir),
        "weight_splits": list_weight_splits(artifact_dir),
        "predict_flags": predict_flags,
        "gap_or_relink_or_division_flags": gap_relink_flags,
        "postprocess_scripts": list_postprocess_scripts(artifact_dir),
        "reference_command_hits": discover_reference_commands(artifact_dir),
        "manifest_keys": sorted(read_manifest(artifact_dir).keys()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "milestone17_artifact_introspection.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"  [introspect] weight splits: {report['weight_splits']}")
    print(f"  [introspect] predict flags ({len(predict_flags)}): {predict_flags}")
    print(f"  [introspect] gap/relink/division flags: {gap_relink_flags}")
    print(f"  [introspect] postprocess scripts: {report['postprocess_scripts']}")
    print(f"  [introspect] reference command hits: {len(report['reference_command_hits'])}")
    return report

# --------------------------------------------------------------------------- #
# 12. Variant registry + FLAG-VALIDATED command builder
# --------------------------------------------------------------------------- #
# M16's known-good predict parameters (the exact settings that scored 0.874).
BASE_PREDICT_PARAMS = {
    "unet_batch_size": "4", "det_threshold": "0.99", "ilp_edge_weight": "-1.0",
    "ilp_appearance_weight": "0.1", "ilp_disappearance_weight": "0.1", "ilp_division_weight": "1.0",
}
FLAG_FOR_PARAM = {
    "unet_batch_size": "--unet-batch-size", "det_threshold": "--det-threshold",
    "ilp_edge_weight": "--ilp-edge-weight", "ilp_appearance_weight": "--ilp-appearance-weight",
    "ilp_disappearance_weight": "--ilp-disappearance-weight", "ilp_division_weight": "--ilp-division-weight",
}

# Each variant is a SMALL override on M16's known-good command. `split`
# defaults to 0; `use_gap_flags` enables Variant B's discovery-gated gap/
# relink flags; `extra_flags` are explicit additions validated against the
# discovered accepted-flags set.
M17_VARIANTS = {
    "A": {"label": "reference_exact", "param_overrides": {}, "split": 0, "use_gap_flags": False, "extra_flags": []},
    "B": {"label": "gap_recovery", "param_overrides": {}, "split": 0, "use_gap_flags": True, "extra_flags": []},
    # Variant C's single active change vs M16: det-threshold 0.99 -> 0.985, a
    # modest recall lever (more detections survive) with ILP weights held at
    # reference. A blind ILP-weight change is riskier than a small threshold
    # nudge; ILP-weight alternatives are documented for a follow-up sweep.
    "C": {"label": "tuned_det_threshold_0985", "param_overrides": {"det_threshold": "0.985"}, "split": 0,
          "use_gap_flags": False, "extra_flags": []},
}


def weights_rel_for_split(split: int) -> str:
    return f"weights/unet_transformer/split_{split}/edge_predictor_best.pth"


def select_weight_split(requested_split: int, available_splits: Sequence[int]) -> int:
    """Chooses the requested split if it actually ships, else the first
    available (or 0 when introspection could not enumerate any - e.g. an
    off-Kaggle dry run).
    """
    if not available_splits:
        return 0
    if requested_split in available_splits:
        return requested_split
    return available_splits[0]


def build_base_command(params: dict, split: int, splits_filename: str, data_dir: str) -> list[str]:
    cmd = [
        sys.executable, PREDICT_SCRIPT_REL, "--data-dir", data_dir, "--splits", splits_filename,
        "--split", str(split), "--weights", weights_rel_for_split(split),
    ]
    for key, flag in FLAG_FOR_PARAM.items():
        cmd += [flag, str(params[key])]
    cmd += ["--use-ilp"]
    return cmd


def extract_flag_usage_from_hits(hits: Sequence[dict], flag: str) -> list[str]:
    """Returns the EXACT tokens the reference command used for `flag`
    (`[flag]` for a bare toggle, `[flag, value]` for a valued flag), copied
    verbatim from a discovered reference command line - so Variant B only
    ever passes a gap/relink flag the way the reference itself proved it
    works, never a guessed value.
    """
    for h in hits:
        toks = str(h.get("line", "")).replace("\\", " ").split()
        if flag in toks:
            i = toks.index(flag)
            if i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                val = toks[i + 1].strip().strip(",").strip("'\"")
                if val:
                    return [flag, val]
            return [flag]
    return []


def build_variant_command(
    variant_name: str, introspection: dict,
    data_dir: str = f"{KAGGLE_COMPETITION_INPUT_DIR}/test", splits_filename: str = SPLITS_FILENAME,
) -> tuple[list[str], dict]:
    """Builds one variant's predict command as a small override on M16's
    known-good command, VALIDATING every added flag against the discovered
    accepted-flags set. An unsupported flag is dropped with a note rather
    than handed to predict (which would crash on an unrecognized argument),
    so a variant can never regress M16's stability - at worst it reduces to
    the reference-exact command.
    """
    accepted = set(introspection.get("predict_flags", []))
    available_splits = list(introspection.get("weight_splits", [])) or [0]
    hits = introspection.get("reference_command_hits", [])
    gap_flags = list(introspection.get("gap_or_relink_or_division_flags", []))

    vdef = M17_VARIANTS[variant_name]
    params = dict(BASE_PREDICT_PARAMS)
    params.update(vdef.get("param_overrides", {}))
    split = select_weight_split(vdef.get("split", 0), available_splits)
    cmd = build_base_command(params, split, splits_filename, data_dir)

    notes = {
        "variant": variant_name, "label": vdef["label"], "split_used": split,
        "available_splits": available_splits, "param_overrides": vdef.get("param_overrides", {}),
        "extra_flags_added": [], "dropped_flags": [], "gap_flags_discovered": gap_flags, "gap_notes": [],
    }

    if vdef.get("use_gap_flags"):
        # Only NEW gap flags matter here - a keyword like "division" also
        # matches the base --ilp-division-weight, which must never be
        # re-appended as a duplicate on top of the base command.
        base_flags_present = set(cmd)
        candidate_gap_flags = [f for f in gap_flags if f not in base_flags_present]
        if not candidate_gap_flags:
            notes["gap_notes"].append(
                "no NEW gap/relink/postprocess flag beyond the base command found in the predict "
                "script - variant reduces to reference-exact")
        else:
            for flag in candidate_gap_flags:
                if flag not in accepted:
                    notes["dropped_flags"].append(flag)
                    continue
                usage = extract_flag_usage_from_hits(hits, flag)
                if usage:
                    cmd += usage
                    notes["extra_flags_added"].append(usage)
                else:
                    notes["gap_notes"].append(
                        f"{flag} is declared by the predict script but its exact reference usage/value "
                        f"was not found in any discovered command - NOT guessing its argument (this flag "
                        f"is skipped; review milestone17_artifact_introspection.json and specify it)")

    for tok in vdef.get("extra_flags", []):
        flag = tok[0] if isinstance(tok, list) else tok
        if flag in accepted:
            cmd += (list(tok) if isinstance(tok, list) else [tok])
            notes["extra_flags_added"].append(tok)
        else:
            notes["dropped_flags"].append(flag)

    return cmd, notes

# --------------------------------------------------------------------------- #
# 21. Reused metric primitives (physical_distance_um, linefit, prune, relabel)
# --------------------------------------------------------------------------- #
# All gates are physical micrometres (VOXEL_SIZE_UM / _VOXEL_SCALE come from
# the reused M16 header). Every added edge spans a UNIT timepoint (t -> t+1)
# so the evaluator can credit it; a direct t -> t+2 edge is never produced.

# Safe-division gates (geometric second-child admission).
M30_DIV_PARENT_CHILD_MAX_UM = 4.7    # source -> new (second) child
M30_DIV_SISTER_MAX_UM = 6.85         # existing child <-> new child
M30_DIV_EXISTING_CHILD_MAX_UM = 7.45  # the original single edge must not already be over-stretched
M30_DIV_FRAME_CAP_FRAC = 0.0072      # per-timepoint cap on admitted divisions
M30_DIV_GLOBAL_CAP_FRAC = 0.004      # per-dataset cap on admitted divisions

# Single-frame gap recovery (end@t -> synthetic@t+1 -> start@t+2).
M30_GAP1_MAX_TOTAL_UM = 6.2
M30_GAP1_CAP_FRAC = 0.006
M30_GAP1_CAP_ABS = 300

# Two-frame gap recovery (end@t -> synth@t+1 -> synth@t+2 -> start@t+3).
M30_GAP2_MAX_STEP_UM = 4.4           # per-unit-step budget (total <= 3*step by construction)
M30_GAP2_MAX_TOTAL_UM = 10.2
M30_GAP2_VELOCITY_COS_MIN = -0.25    # incoming velocity vs gap velocity must not strongly reverse
M30_GAP2_VELOCITY_NORMDIFF_UM = 6.0  # |per-step gap velocity - incoming velocity| budget
M30_GAP2_CAP_FRAC = 0.0045
M30_GAP2_CAP_ABS = 180

# Line-fit smoothing (topology-preserving coordinate blend).
M30_LINEFIT_WINDOW = 2
M30_LINEFIT_WEIGHT = 0.72

# Isolated-node pruning (degree-0 predicted nodes = single-frame tracks).
M30_PRUNE_MAX_FRAC = 0.6             # safety valve: if MORE than this fraction is isolated, skip (likely a read problem, not over-prediction)


def physical_distance_um(pos_a, pos_b) -> float:
    """Euclidean distance in micrometres between two (z, y, x) voxel-index
    positions, applying the anisotropic VOXEL_SIZE_UM scale. `_VOXEL_SCALE`
    is [z, y, x] to match the coordinate order.
    """
    d = (np.asarray(pos_a, dtype=float) - np.asarray(pos_b, dtype=float)) * _VOXEL_SCALE
    return float(np.sqrt(float(np.dot(d, d))))


def build_base_graph(geff_graph: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduces the M16 converter's node remap + topology enforcement
    (raw -> consecutive local node_id starting at 0; drop dangling and
    backward-in-time and duplicate edges; in_degree<=1 keeping the best
    incoming; out_degree<=2 keeping the two best outgoing, ranked by higher
    edge_prob then lower edge_dist then stable order) but returns a plain
    (pred_nodes, pred_edges) pair - the SAME topology the M16 baseline
    submits. With no post-processing applied the downstream converter
    therefore reproduces the M16 baseline exactly.

    pred_nodes columns: node_id (int, >=0), t, z, y, x.
    pred_edges columns: source_id, target_id (both local node_id).
    """
    nodes = geff_graph["nodes"].copy()
    edges = geff_graph["edges"].copy()

    raw_to_local: dict[int, int] = {}
    node_rows = []
    node_t: dict[int, int] = {}
    for local_id, row in enumerate(nodes.itertuples()):
        raw_to_local[int(row.raw_id)] = local_id
        node_t[local_id] = int(row.t)
        node_rows.append({"node_id": local_id, "t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x)})
    pred_nodes = pd.DataFrame(node_rows, columns=["node_id", "t", "z", "y", "x"])

    kept = []
    for row in edges.itertuples():
        s_raw, t_raw = int(row.raw_source), int(row.raw_target)
        if s_raw not in raw_to_local or t_raw not in raw_to_local:
            continue
        s_kid, t_kid = raw_to_local[s_raw], raw_to_local[t_raw]
        if not (node_t[t_kid] > node_t[s_kid]):
            continue
        kept.append({"source_id": s_kid, "target_id": t_kid, "edge_prob": float(row.edge_prob), "edge_dist": float(row.edge_dist)})

    edges_df = pd.DataFrame(kept, columns=["source_id", "target_id", "edge_prob", "edge_dist"])
    if len(edges_df):
        edges_df = edges_df.drop_duplicates(subset=["source_id", "target_id"], keep="first")
        edges_df = edges_df.assign(_order=np.arange(len(edges_df)))
        edges_df = edges_df.sort_values(["edge_prob", "edge_dist", "_order"], ascending=[False, True, True]).reset_index(drop=True)
        edges_df = edges_df.drop_duplicates(subset=["target_id"], keep="first")
        edges_df = edges_df.groupby("source_id", sort=False, group_keys=False).head(2).reset_index(drop=True)

    pred_edges = edges_df[["source_id", "target_id"]].reset_index(drop=True) if len(edges_df) else pd.DataFrame(columns=["source_id", "target_id"])
    return pred_nodes, pred_edges


def _degree_maps(pred_edges: pd.DataFrame) -> tuple[dict, dict]:
    out_degree = pred_edges.groupby("source_id").size().to_dict() if len(pred_edges) else {}
    in_degree = pred_edges.groupby("target_id").size().to_dict() if len(pred_edges) else {}
    return out_degree, in_degree


def add_micro_safe_divisions(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    parent_child_max_um: float = M30_DIV_PARENT_CHILD_MAX_UM, sister_max_um: float = M30_DIV_SISTER_MAX_UM,
    existing_child_max_um: float = M30_DIV_EXISTING_CHILD_MAX_UM,
    frame_cap_frac: float = M30_DIV_FRAME_CAP_FRAC, global_cap_frac: float = M30_DIV_GLOBAL_CAP_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Admits a geometrically safe SECOND outgoing edge for a source that
    currently has EXACTLY one accepted child, turning it into a division.
    The new child must be an unmatched node (current in_degree==0) so no
    multi-parent is ever created; the source is never given a third child;
    the existing child must be exactly ONE timepoint ahead (so the sister
    edge is unit-timepoint and metric-valid) and its own edge must already
    be within `existing_child_max_um`; the new child must be within
    `parent_child_max_um` of the source and within `sister_max_um` of the
    existing child. Candidates are admitted closest-first under per-frame
    and per-dataset caps favouring the most confident (nearest) divisions.
    """
    diag = {"n_divisions_added": 0, "n_candidates_considered": 0}
    if len(pred_edges) == 0 or len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag

    pos = pred_nodes.set_index("node_id")[["z", "y", "x"]]
    t_of = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))
    out_degree, in_degree = _degree_maps(pred_edges)

    # sources with exactly one child whose existing edge is unit-timepoint
    # and short enough to be a credible parent.
    existing_child_of: dict[int, int] = {}
    for row in pred_edges.itertuples():
        s, t = int(row.source_id), int(row.target_id)
        if out_degree.get(s, 0) != 1:
            continue
        if t_of[t] != t_of[s] + 1:
            continue
        if physical_distance_um(pos.loc[s].to_numpy(float), pos.loc[t].to_numpy(float)) > existing_child_max_um:
            continue
        existing_child_of[s] = t

    if not existing_child_of:
        return pred_nodes.copy(), pred_edges.copy(), diag

    # orphan pool: unmatched nodes (in_degree==0) keyed by timepoint.
    orphans_by_t: dict[int, list[int]] = {}
    for nid in pred_nodes["node_id"]:
        if in_degree.get(nid, 0) == 0:
            orphans_by_t.setdefault(int(t_of[nid]), []).append(int(nid))

    candidates = []
    for s, child in existing_child_of.items():
        child_t = t_of[child]
        s_pos = pos.loc[s].to_numpy(float)
        child_pos = pos.loc[child].to_numpy(float)
        for cand in orphans_by_t.get(child_t, []):
            if cand == child or cand == s:
                continue
            c_pos = pos.loc[cand].to_numpy(float)
            d_parent = physical_distance_um(s_pos, c_pos)
            if d_parent > parent_child_max_um:
                continue
            d_sister = physical_distance_um(child_pos, c_pos)
            if d_sister > sister_max_um:
                continue
            candidates.append({"source_id": s, "new_child": cand, "t": child_t, "d_parent": d_parent})
    diag["n_candidates_considered"] = len(candidates)
    if not candidates:
        return pred_nodes.copy(), pred_edges.copy(), diag

    candidates.sort(key=lambda c: c["d_parent"])
    frame_totals: dict[int, int] = {}
    for t in t_of.values():
        frame_totals[int(t)] = frame_totals.get(int(t), 0) + 1

    # Caps limit division DENSITY but never forbid every admission outright:
    # the per-dataset global cap bounds the total (on real data int(frac*N)
    # dominates; the floor of 1 only matters on small/dry-run graphs), and
    # the per-frame cap is floored the same way so a lone confident division
    # in a small frame is still reachable while the global cap holds the line.
    global_cap = max(1, int(global_cap_frac * max(len(pred_nodes), 1)))
    frame_counts: dict[int, int] = {}
    new_edges = []
    used_children: set[int] = set()
    for c in candidates:
        if len(new_edges) >= global_cap:
            break
        s, cand, ft = c["source_id"], c["new_child"], c["t"]
        if cand in used_children:
            continue
        if in_degree.get(cand, 0) != 0:
            continue  # became matched earlier this pass (never multi-parent)
        if out_degree.get(s, 0) != 1:
            continue  # source already divided this pass (never a third child)
        frame_cap = max(1, int(frame_cap_frac * max(frame_totals.get(ft, 0), 1)))
        if frame_counts.get(ft, 0) >= frame_cap:
            continue
        new_edges.append({"source_id": s, "target_id": cand})
        out_degree[s] = out_degree.get(s, 0) + 1
        in_degree[cand] = in_degree.get(cand, 0) + 1
        frame_counts[ft] = frame_counts.get(ft, 0) + 1
        used_children.add(cand)

    result_edges = pd.concat([pred_edges, pd.DataFrame(new_edges, columns=["source_id", "target_id"])], ignore_index=True) if new_edges else pred_edges.copy()
    diag["n_divisions_added"] = len(new_edges)
    return pred_nodes.copy(), result_edges, diag


def _ends_and_starts(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame) -> tuple[list, dict, dict]:
    """Returns tracklet END nodes (out_degree==0, not at the last
    timepoint) and START nodes (in_degree==0, not at the first timepoint)
    indexed by timepoint, plus the node->predecessor map for velocity gates.
    """
    out_degree, in_degree = _degree_maps(pred_edges)
    t_of = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))
    max_t, min_t = int(pred_nodes["t"].max()), int(pred_nodes["t"].min())
    ends = [(int(nid), int(t_of[nid])) for nid in pred_nodes["node_id"] if out_degree.get(nid, 0) == 0 and t_of[nid] < max_t]
    starts_by_t: dict[int, list[int]] = {}
    for nid in pred_nodes["node_id"]:
        if in_degree.get(nid, 0) == 0 and t_of[nid] > min_t:
            starts_by_t.setdefault(int(t_of[nid]), []).append(int(nid))
    pred_of = dict(zip(pred_edges["target_id"], pred_edges["source_id"])) if len(pred_edges) else {}
    return ends, starts_by_t, pred_of


def close_single_frame_gaps(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    max_total_um: float = M30_GAP1_MAX_TOTAL_UM, cap_frac: float = M30_GAP1_CAP_FRAC, cap_abs: int = M30_GAP1_CAP_ABS,
    next_synth_id: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, int]:
    """Recovers a single missing frame: an END node at t and a START node
    at t+2 within `max_total_um` are bridged by ONE synthetic node at t+1
    (straight-line midpoint) plus two UNIT-timepoint edges end->synth and
    synth->start. A direct end->start (t -> t+2) edge is never created.
    Synthetic node_ids are negative so they can never collide with a real
    (>=0) id; they are relabelled positive later. Each end/start is used at
    most once; admitted closest-first under a cap.
    """
    diag = {"n_gap1_candidates": 0, "n_gap1_closed": 0}
    if len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag, next_synth_id
    ends, starts_by_t, _ = _ends_and_starts(pred_nodes, pred_edges)
    pos = pred_nodes.set_index("node_id")[["z", "y", "x"]]

    candidates = []
    for end_id, end_t in ends:
        e_pos = pos.loc[end_id].to_numpy(float)
        for start_id in starts_by_t.get(end_t + 2, []):
            s_pos = pos.loc[start_id].to_numpy(float)
            total = physical_distance_um(e_pos, s_pos)
            if total <= max_total_um:
                candidates.append({"end": end_id, "start": start_id, "mid_t": end_t + 1, "total": total})
    diag["n_gap1_candidates"] = len(candidates)
    if not candidates:
        return pred_nodes.copy(), pred_edges.copy(), diag, next_synth_id

    candidates.sort(key=lambda c: c["total"])
    cap = min(cap_abs, max(1, int(cap_frac * max(len(pred_nodes), 1))))
    new_nodes, new_edges = [], []
    used_ends, used_starts = set(), set()
    sid = next_synth_id
    for c in candidates:
        if len(new_nodes) >= cap:
            break
        if c["end"] in used_ends or c["start"] in used_starts:
            continue
        e_pos = pos.loc[c["end"]].to_numpy(float)
        s_pos = pos.loc[c["start"]].to_numpy(float)
        mid = (e_pos + s_pos) / 2.0
        new_nodes.append({"node_id": sid, "t": c["mid_t"], "z": float(mid[0]), "y": float(mid[1]), "x": float(mid[2])})
        new_edges.append({"source_id": c["end"], "target_id": sid})
        new_edges.append({"source_id": sid, "target_id": c["start"]})
        used_ends.add(c["end"])
        used_starts.add(c["start"])
        sid -= 1

    result_nodes = pd.concat([pred_nodes, pd.DataFrame(new_nodes, columns=["node_id", "t", "z", "y", "x"])], ignore_index=True) if new_nodes else pred_nodes.copy()
    result_edges = pd.concat([pred_edges, pd.DataFrame(new_edges, columns=["source_id", "target_id"])], ignore_index=True) if new_edges else pred_edges.copy()
    diag["n_gap1_closed"] = len(new_nodes)
    return result_nodes, result_edges, diag, sid


def recover_two_frame_gaps(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    max_step_um: float = M30_GAP2_MAX_STEP_UM, max_total_um: float = M30_GAP2_MAX_TOTAL_UM,
    velocity_cos_min: float = M30_GAP2_VELOCITY_COS_MIN, velocity_normdiff_um: float = M30_GAP2_VELOCITY_NORMDIFF_UM,
    cap_frac: float = M30_GAP2_CAP_FRAC, cap_abs: int = M30_GAP2_CAP_ABS, next_synth_id: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, int]:
    """Recovers two missing frames: an END node at t and a START node at
    t+3 are bridged by TWO synthetic nodes at t+1 and t+2 (linear
    interpolation) plus three UNIT-timepoint edges. Gated on per-step and
    total physical distance AND on velocity consistency - when the end node
    has a predecessor, the incoming velocity and the gap velocity must not
    strongly reverse (cosine >= `velocity_cos_min`) and their per-step
    magnitudes must agree within `velocity_normdiff_um`. No direct multi-
    frame edge is ever created. Synthetic ids are negative; admitted
    closest-first under a cap.
    """
    diag = {"n_gap2_candidates": 0, "n_gap2_recovered": 0}
    if len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag, next_synth_id
    ends, starts_by_t, pred_of = _ends_and_starts(pred_nodes, pred_edges)
    pos = pred_nodes.set_index("node_id")[["z", "y", "x"]]
    t_of = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))

    def _scaled(nid):
        return pos.loc[nid].to_numpy(float) * _VOXEL_SCALE

    candidates = []
    for end_id, end_t in ends:
        e_um = _scaled(end_id)
        # incoming velocity (per unit step) if the end has a predecessor.
        v_in = None
        if end_id in pred_of:
            p = pred_of[end_id]
            dt = int(t_of[end_id]) - int(t_of[p])
            if dt > 0:
                v_in = (e_um - _scaled(p)) / float(dt)
        for start_id in starts_by_t.get(end_t + 3, []):
            s_um = _scaled(start_id)
            total = float(np.linalg.norm(s_um - e_um))
            if total > max_total_um:
                continue
            v_gap = (s_um - e_um) / 3.0
            if float(np.linalg.norm(v_gap)) > max_step_um:
                continue
            if v_in is not None:
                nin, ngap = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_gap))
                if nin > 1e-9 and ngap > 1e-9:
                    cos = float(np.dot(v_in, v_gap) / (nin * ngap))
                    if cos < velocity_cos_min:
                        continue
                    if float(np.linalg.norm(v_gap - v_in)) > velocity_normdiff_um:
                        continue
            candidates.append({"end": end_id, "start": start_id, "t0": end_t, "total": total})
    diag["n_gap2_candidates"] = len(candidates)
    if not candidates:
        return pred_nodes.copy(), pred_edges.copy(), diag, next_synth_id

    candidates.sort(key=lambda c: c["total"])
    cap = min(cap_abs, max(1, int(cap_frac * max(len(pred_nodes), 1))))
    new_nodes, new_edges = [], []
    used_ends, used_starts = set(), set()
    sid = next_synth_id
    for c in candidates:
        if len(new_nodes) >= 2 * cap:  # each recovery adds two nodes
            break
        if c["end"] in used_ends or c["start"] in used_starts:
            continue
        e_pos = pos.loc[c["end"]].to_numpy(float)
        s_pos = pos.loc[c["start"]].to_numpy(float)
        m1 = e_pos + (s_pos - e_pos) * (1.0 / 3.0)
        m2 = e_pos + (s_pos - e_pos) * (2.0 / 3.0)
        id1, id2 = sid, sid - 1
        new_nodes.append({"node_id": id1, "t": c["t0"] + 1, "z": float(m1[0]), "y": float(m1[1]), "x": float(m1[2])})
        new_nodes.append({"node_id": id2, "t": c["t0"] + 2, "z": float(m2[0]), "y": float(m2[1]), "x": float(m2[2])})
        new_edges.append({"source_id": c["end"], "target_id": id1})
        new_edges.append({"source_id": id1, "target_id": id2})
        new_edges.append({"source_id": id2, "target_id": c["start"]})
        used_ends.add(c["end"])
        used_starts.add(c["start"])
        sid -= 2

    result_nodes = pd.concat([pred_nodes, pd.DataFrame(new_nodes, columns=["node_id", "t", "z", "y", "x"])], ignore_index=True) if new_nodes else pred_nodes.copy()
    result_edges = pd.concat([pred_edges, pd.DataFrame(new_edges, columns=["source_id", "target_id"])], ignore_index=True) if new_edges else pred_edges.copy()
    diag["n_gap2_recovered"] = len(new_nodes) // 2
    return result_nodes, result_edges, diag, sid


def linefit_smooth_nodes(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    window: int = M30_LINEFIT_WINDOW, weight: float = M30_LINEFIT_WEIGHT,
) -> tuple[pd.DataFrame, dict]:
    """Topology-preserving coordinate smoothing - never adds/removes a node
    or edge, only reassigns (z, y, x) for "linear interior" nodes (exactly
    one predecessor at t-1 AND exactly one successor at t+1). Division
    nodes, branch points, and chain endpoints are left untouched. Blends
    weight*original + (1-weight)*mean(up-to-`window` neighbours each side).
    """
    diag = {"n_nodes_smoothed": 0}
    if len(pred_nodes) == 0 or len(pred_edges) == 0:
        return pred_nodes.copy(), diag

    succ = dict(zip(pred_edges["source_id"], pred_edges["target_id"]))
    pred = dict(zip(pred_edges["target_id"], pred_edges["source_id"]))
    out_degree, in_degree = _degree_maps(pred_edges)
    node_lookup = pred_nodes.set_index("node_id")
    t_lookup = node_lookup["t"].to_dict()

    def is_linear_interior(nid) -> bool:
        if out_degree.get(nid, 0) != 1 or in_degree.get(nid, 0) != 1:
            return False
        s_id, p_id = succ.get(nid), pred.get(nid)
        if s_id is None or p_id is None:
            return False
        return t_lookup.get(s_id) == t_lookup[nid] + 1 and t_lookup.get(p_id) == t_lookup[nid] - 1

    def collect(nid, direction, steps):
        out = []
        cur = nid
        for _ in range(steps):
            nxt = succ.get(cur) if direction == "succ" else pred.get(cur)
            if nxt is None:
                break
            out.append(nxt)
            if not is_linear_interior(nxt):
                break
            cur = nxt
        return out

    smoothed: dict = {}
    for nid in pred_nodes["node_id"]:
        if not is_linear_interior(nid):
            continue
        neigh = collect(nid, "succ", window) + collect(nid, "pred", window)
        if not neigh:
            continue
        neigh_avg = node_lookup.loc[neigh, ["z", "y", "x"]].to_numpy(float).mean(axis=0)
        original = node_lookup.loc[nid, ["z", "y", "x"]].to_numpy(float)
        smoothed[nid] = weight * original + (1.0 - weight) * neigh_avg

    result = pred_nodes.copy()
    if smoothed:
        idx = result.set_index("node_id")
        for nid, p in smoothed.items():
            idx.loc[nid, ["z", "y", "x"]] = p
        result = idx.reset_index()[["node_id", "t", "z", "y", "x"]]
    diag["n_nodes_smoothed"] = len(smoothed)
    return result, diag


def prune_isolated_nodes(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, max_frac: float = M30_PRUNE_MAX_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Removes degree-0 nodes (no incoming AND no outgoing edge) - the
    single-frame tracks that inflate the predicted node count T_pred and
    are penalised by the metric with zero edge benefit. Removing them can
    never lower edge_jaccard (they carry no edges) or division_jaccard
    (out_degree 0 is not a division). A safety valve skips pruning entirely
    if MORE than `max_frac` of nodes are isolated, since that signals a
    graph-read problem rather than genuine over-prediction (better to keep
    the M16 baseline than gut it).
    """
    diag = {"n_isolated": 0, "n_pruned": 0, "prune_skipped_reason": None}
    if len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag
    matched = set(pred_edges["source_id"]) | set(pred_edges["target_id"]) if len(pred_edges) else set()
    isolated_mask = ~pred_nodes["node_id"].isin(matched)
    n_isolated = int(isolated_mask.sum())
    diag["n_isolated"] = n_isolated
    if n_isolated == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag
    if n_isolated > max_frac * len(pred_nodes):
        diag["prune_skipped_reason"] = f"{n_isolated}/{len(pred_nodes)} isolated exceeds max_frac={max_frac}; kept baseline"
        return pred_nodes.copy(), pred_edges.copy(), diag
    result_nodes = pred_nodes[~isolated_mask].reset_index(drop=True)
    diag["n_pruned"] = n_isolated
    return result_nodes, pred_edges.copy(), diag


def relabel_positive(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, start_id: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Remaps every node_id (real >=0 AND synthetic <0) to positive
    consecutive ids beginning at `start_id`, in node-row order, and rewrites
    edges through the same map (dropping any edge whose endpoint no longer
    exists, e.g. after isolated pruning). Guarantees no negative id ever
    reaches the submission. Returns (nodes, edges, next_start).
    """
    id_map: dict[int, int] = {}
    rows = []
    cur = start_id
    for row in pred_nodes.itertuples():
        id_map[int(row.node_id)] = cur
        rows.append({"node_id": cur, "t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x)})
        cur += 1
    nodes_out = pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x"])

    e_rows = []
    for row in pred_edges.itertuples() if len(pred_edges) else []:
        s, t = int(row.source_id), int(row.target_id)
        if s in id_map and t in id_map:
            e_rows.append({"source_id": id_map[s], "target_id": id_map[t]})
    edges_out = pd.DataFrame(e_rows, columns=["source_id", "target_id"])
    return nodes_out, edges_out, cur

# --------------------------------------------------------------------------- #
# 41. M30 raw CANDIDATE graph reader + richness diagnostic (v2 foundation)
# --------------------------------------------------------------------------- #
# CRITICAL: the reused read_geff_graph applies the edges/props/solution mask, so
# it returns the ILP SOLUTION, not the candidate graph. v2 must re-solve linking
# from the RAW candidate edges, so read_candidate_geff reads edges WITHOUT the
# solution mask and keeps edge_prob + edge_dist for every unit-timepoint
# candidate. build_base_graph is NEVER run before v2.
from dataclasses import dataclass, field, asdict

M30_FINAL_SOURCE = "metric_aware_global_relinker_v2"


def read_candidate_geff(geff_root: Path) -> dict:
    """Reads the RAW candidate graph: all detection nodes (node solution mask
    applied - those are the tracked cells) and ALL candidate edges with edge_prob
    / edge_dist, WITHOUT applying the edge solution mask. Reports how many edges
    the ILP solution kept so the diagnostic can tell a full candidate graph from
    an ILP-solution-only store."""
    node_ids = _read_geff_array(geff_root, "nodes/ids")
    if node_ids is None:
        raise ValueError(f"{geff_root}: no nodes/ids array")
    node_ids = np.asarray(node_ids).reshape(-1)
    t = _read_geff_array(geff_root, "nodes/props/t/values")
    z = _read_geff_array(geff_root, "nodes/props/z/values")
    y = _read_geff_array(geff_root, "nodes/props/y/values")
    x = _read_geff_array(geff_root, "nodes/props/x/values")
    node_solution = _read_geff_array(geff_root, "nodes/props/solution/values")

    edge_ids = _read_geff_array(geff_root, "edges/ids")
    edge_solution = _read_geff_array(geff_root, "edges/props/solution/values")
    edge_prob = _read_geff_array(geff_root, "edges/props/edge_prob/values")
    edge_dist = _read_geff_array(geff_root, "edges/props/edge_dist/values")

    def _col(arr, n, default=0.0):
        return np.asarray(arr).reshape(-1) if arr is not None else np.full(n, default, dtype=float)

    n_nodes = len(node_ids)
    nodes = pd.DataFrame({"raw_id": node_ids.astype(np.int64), "t": _col(t, n_nodes),
                          "z": _col(z, n_nodes), "y": _col(y, n_nodes), "x": _col(x, n_nodes)})
    if node_solution is not None:
        nodes = nodes[np.asarray(node_solution).reshape(-1).astype(bool)].reset_index(drop=True)

    has_edge_prob = edge_prob is not None
    has_edge_dist = edge_dist is not None
    if edge_ids is None or np.asarray(edge_ids).size == 0:
        edges = pd.DataFrame(columns=["raw_source", "raw_target", "edge_prob", "edge_dist", "in_solution"])
        n_edges_total = n_edges_solution = 0
    else:
        edge_ids = np.asarray(edge_ids)
        if edge_ids.ndim == 1:
            edge_ids = edge_ids.reshape(-1, 2)
        n_edges_total = edge_ids.shape[0]
        # edge_prob is left as NaN when ABSENT (never silently defaulted to 1.0).
        ep = np.asarray(edge_prob).reshape(-1).astype(float) if has_edge_prob else np.full(n_edges_total, np.nan)
        ed = np.asarray(edge_dist).reshape(-1).astype(float) if has_edge_dist else np.full(n_edges_total, np.nan)
        insol = np.asarray(edge_solution).reshape(-1).astype(bool) if edge_solution is not None else np.zeros(n_edges_total, dtype=bool)
        edges = pd.DataFrame({"raw_source": edge_ids[:, 0].astype(np.int64), "raw_target": edge_ids[:, 1].astype(np.int64),
                              "edge_prob": ep, "edge_dist": ed, "in_solution": insol})
        n_edges_solution = int(insol.sum())
    return {"nodes": nodes, "edges": edges, "has_edge_prob": has_edge_prob, "has_edge_dist": has_edge_dist,
            "n_edges_total": n_edges_total, "n_edges_solution": n_edges_solution}


def build_candidate_graph_v2(cand_geff: dict) -> tuple:
    """Local-id candidate graph. cand_nodes: node_id,t,z,y,x (consecutive from 0).
    cand_edges: source_id,target_id,edge_prob,edge_dist,in_solution for every
    UNIT-timepoint candidate whose both endpoints survived node selection. Also
    returns per-candidate diagnostics (unit vs non-unit counts). edge_prob is
    preserved verbatim - NEVER defaulted to 1.0."""
    nodes = cand_geff["nodes"].copy()
    edges = cand_geff["edges"].copy()
    raw_to_local, rows, node_t = {}, [], {}
    for lid, r in enumerate(nodes.itertuples()):
        raw_to_local[int(r.raw_id)] = lid
        node_t[lid] = int(r.t)
        rows.append({"node_id": lid, "t": int(r.t), "z": float(r.z), "y": float(r.y), "x": float(r.x)})
    cand_nodes = pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x"])
    unit, non_unit_count, dangling = [], 0, 0
    for r in (edges.itertuples() if len(edges) else []):
        s_raw, t_raw = int(r.raw_source), int(r.raw_target)
        if s_raw not in raw_to_local or t_raw not in raw_to_local:
            dangling += 1
            continue
        s, t = raw_to_local[s_raw], raw_to_local[t_raw]
        dt = node_t[t] - node_t[s]
        if dt == 1:
            unit.append({"source_id": s, "target_id": t, "edge_prob": float(r.edge_prob),
                         "edge_dist": float(r.edge_dist), "in_solution": bool(r.in_solution)})
        elif dt >= 1:
            non_unit_count += 1
        # dt<=0 (backward / same-frame) dropped.
    cand_edges = pd.DataFrame(unit, columns=["source_id", "target_id", "edge_prob", "edge_dist", "in_solution"])
    diag = {"n_unit_candidates": len(cand_edges), "n_non_unit_candidates": non_unit_count, "n_dangling": dangling}
    return cand_nodes, cand_edges, diag


def _quantiles(arr, qs):
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.quantile(a, q)) for q in qs}


def analyze_edge_prob(cand_edges: pd.DataFrame, has_edge_prob: bool) -> dict:
    """edge_prob availability + degeneracy profile. Degenerate when absent, all
    NaN, <=2 unique values, or essentially all 0/1."""
    out = {"edge_prob_available": bool(has_edge_prob), "edge_prob_non_degenerate": False,
           "edge_prob_dtype": None, "edge_prob_min": None, "edge_prob_unique_count": 0,
           "frac_edge_prob_0_or_1": None, "quantiles": {}}
    if not has_edge_prob or len(cand_edges) == 0:
        return out
    ep = cand_edges["edge_prob"].to_numpy(dtype=float)
    valid = ep[~np.isnan(ep)]
    out["edge_prob_dtype"] = str(cand_edges["edge_prob"].dtype)
    if len(valid) == 0:
        return out
    out["edge_prob_min"] = float(valid.min())
    uniq = np.unique(np.round(valid, 6))
    out["edge_prob_unique_count"] = int(len(uniq))
    frac01 = float(np.mean((np.isclose(valid, 0.0)) | (np.isclose(valid, 1.0))))
    out["frac_edge_prob_0_or_1"] = frac01
    out["quantiles"] = _quantiles(valid, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    out["edge_prob_non_degenerate"] = bool(len(uniq) > 2 and frac01 < 0.98 and float(valid.max()) > float(valid.min()))
    return out


def candidate_richness_stats(cand_nodes: pd.DataFrame, cand_edges: pd.DataFrame) -> dict:
    """Per-source / per-target candidate counts and the sparsity fractions the
    Stage-0 classifier needs."""
    t_of = dict(zip(cand_nodes["node_id"].astype(int), cand_nodes["t"].astype(int)))
    max_t = int(cand_nodes["t"].max()) if len(cand_nodes) else 0
    min_t = int(cand_nodes["t"].min()) if len(cand_nodes) else 0
    out_cnt = cand_edges.groupby("source_id").size().to_dict() if len(cand_edges) else {}
    in_cnt = cand_edges.groupby("target_id").size().to_dict() if len(cand_edges) else {}
    src_nodes = [int(n) for n in cand_nodes["node_id"] if t_of[int(n)] < max_t]
    tgt_nodes = [int(n) for n in cand_nodes["node_id"] if t_of[int(n)] > min_t]
    cps = np.array([out_cnt.get(n, 0) for n in src_nodes], dtype=float) if src_nodes else np.array([0.0])
    cpt = np.array([in_cnt.get(n, 0) for n in tgt_nodes], dtype=float) if tgt_nodes else np.array([0.0])
    n_src = max(len(src_nodes), 1)
    return {
        "n_sources": len(src_nodes), "n_targets": len(tgt_nodes),
        "candidates_per_source_mean": float(cps.mean()), "candidates_per_source_median": float(np.median(cps)),
        "candidates_per_source_p75": float(np.quantile(cps, 0.75)), "candidates_per_source_p90": float(np.quantile(cps, 0.90)),
        "candidates_per_source_p95": float(np.quantile(cps, 0.95)), "candidates_per_source_max": float(cps.max()),
        "candidates_per_target_mean": float(cpt.mean()), "candidates_per_target_median": float(np.median(cpt)),
        "candidates_per_target_p90": float(np.quantile(cpt, 0.90)), "candidates_per_target_max": float(cpt.max()),
        "pct_sources_0_candidates": float(np.mean(cps == 0)), "pct_sources_1_candidate": float(np.mean(cps == 1)),
        "pct_sources_ge2_candidates": float(np.mean(cps >= 2)), "pct_sources_ge3_candidates": float(np.mean(cps >= 3)),
    }


def classify_candidate_graph(prob_info: dict, rich: dict) -> str:
    """FULL_CANDIDATE_GRAPH / MODERATELY_SPARSE / ILP_SOLUTION_LIKE /
    EDGE_PROB_UNAVAILABLE per the Stage-0 thresholds."""
    if not prob_info["edge_prob_available"] or not prob_info["edge_prob_non_degenerate"]:
        return "EDGE_PROB_UNAVAILABLE"
    cps_mean = rich["candidates_per_source_mean"]
    pct_ge2 = rich["pct_sources_ge2_candidates"]
    if cps_mean >= 1.25 and pct_ge2 >= 0.15:
        return "FULL_CANDIDATE_GRAPH"
    if cps_mean <= 1.10 and pct_ge2 < 0.05:
        return "ILP_SOLUTION_LIKE"
    return "MODERATELY_SPARSE"


def recommend_submit_variant(candidate_class: str) -> str:
    return {
        "FULL_CANDIDATE_GRAPH": "M30_A (then B, E, D)",
        "MODERATELY_SPARSE": "M30_C (then D, E)",
        "ILP_SOLUTION_LIKE": "M30_C (then D, E)",
        "EDGE_PROB_UNAVAILABLE": "NONE - edge_prob unavailable/degenerate; DO_NOT_SUBMIT any v2 variant",
    }.get(candidate_class, "NONE")

# --------------------------------------------------------------------------- #
# 42. M30 metric-aware GLOBAL RELINKER v2 (re-solve linking from candidates)
# --------------------------------------------------------------------------- #
# v1 patched the ILP solution; v2 RE-SOLVES unit-timepoint linking from the raw
# candidate graph. Fused cost = -log(edge_prob) + w_dist*(dist_um/7.0) +
# w_motion*motion_deviation_um. Two independent Hungarian stages (primary +
# division), each with EXPLICIT per-source no-link dummy columns so a source may
# stay unlinked. Every produced edge is unit-timepoint; in<=1/out<=2 enforced.
M30_COST_DIST_SCALE = 7.0
M30_PROB_EPS = 1e-6


@dataclass
class LinkConfig:
    null_link_cost: float = 4.5
    w_dist: float = 1.0
    w_motion: float = 0.4
    max_link_um: float = 7.5
    min_link_prob: float = 0.05
    max_cost: float = 12.0
    div_enable: bool = True
    div_min_prob: float = 0.20
    div_child_max_um: float = 6.5
    div_sister_max_um: float = 9.0
    div_global_cap_frac: float = 0.020
    div_no_split_cost: float = 5.0
    gap_sizes: tuple = (1, 2)
    gap_max_step_um: float = 4.6
    gap_max_total_um: dict = field(default_factory=lambda: {1: 6.5, 2: 10.5, 3: 14.5})
    linefit_enable: bool = False
    linefit_window: int = 2
    linefit_weight: float = 0.75
    prune_enable: bool = True
    prune_max_frac: float = 0.6
    knn_enable: bool = False
    knn_k: int = 3
    knn_max_um: float = 4.5
    knn_default_prob: float = 0.20


def _m30_assign(cost: np.ndarray) -> list:
    n_r, n_c = cost.shape
    if n_r == 0 or n_c == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        order = sorted(((float(cost[i, j]), i, j) for i in range(n_r) for j in range(n_c)), key=lambda z: z[0])
        ur, uc, pairs = set(), set(), []
        for _, i, j in order:
            if i in ur or j in uc:
                continue
            ur.add(i)
            uc.add(j)
            pairs.append((i, j))
        return pairs


def _fused_cost(prob, dist_um, motion_dev_um, cfg: LinkConfig) -> float:
    p = max(float(prob), M30_PROB_EPS)
    return -float(np.log(p)) + cfg.w_dist * (float(dist_um) / M30_COST_DIST_SCALE) + cfg.w_motion * float(motion_dev_um)


def _source_velocities(cand_nodes, cand_edges):
    """Per-source velocity (um/step) from its best-probability incoming raw
    candidate, for the motion-consistency term. None when no predecessor."""
    pos = cand_nodes.set_index("node_id")[["z", "y", "x"]]
    best_in = {}
    for r in (cand_edges.itertuples() if len(cand_edges) else []):
        tgt = int(r.target_id)
        p = float(r.edge_prob) if not np.isnan(r.edge_prob) else 0.0
        if tgt not in best_in or p > best_in[tgt][1]:
            best_in[tgt] = (int(r.source_id), p)
    vel = {}
    for nid, (pred, _p) in best_in.items():
        if pred in pos.index and nid in pos.index:
            vel[nid] = (pos.loc[nid].to_numpy(float) - pos.loc[pred].to_numpy(float)) * _VOXEL_SCALE
    return vel, pos


def knn_augment_candidates(cand_nodes, cand_edges, cfg: LinkConfig) -> tuple:
    """Add up to top-k geometric candidates per source within knn_max_um that are
    NOT already raw candidates (no duplicates), with a conservative fixed
    knn_default_prob. Returns (augmented_edges, n_added)."""
    if not cfg.knn_enable:
        return cand_edges.assign(is_knn=False) if len(cand_edges) else pd.DataFrame(columns=list(cand_edges.columns) + ["is_knn"]), 0
    pos = cand_nodes.set_index("node_id")[["z", "y", "x"]]
    by_t = {}
    for r in cand_nodes.itertuples():
        by_t.setdefault(int(r.t), []).append(int(r.node_id))
    existing = set((int(r.source_id), int(r.target_id)) for r in cand_edges.itertuples()) if len(cand_edges) else set()
    t_of = dict(zip(cand_nodes["node_id"].astype(int), cand_nodes["t"].astype(int)))
    max_t = int(cand_nodes["t"].max()) if len(cand_nodes) else 0
    new = []
    for s in cand_nodes["node_id"]:
        s = int(s)
        if t_of[s] >= max_t:
            continue
        sp = pos.loc[s].to_numpy(float)
        cands = []
        for tg in by_t.get(t_of[s] + 1, []):
            if (s, tg) in existing:
                continue
            d = physical_distance_um(sp, pos.loc[tg].to_numpy(float))
            if d <= cfg.knn_max_um:
                cands.append((d, tg))
        cands.sort(key=lambda z: z[0])
        for d, tg in cands[: cfg.knn_k]:
            new.append({"source_id": s, "target_id": tg, "edge_prob": float(cfg.knn_default_prob),
                        "edge_dist": float(d), "in_solution": False, "is_knn": True})
    base = cand_edges.assign(is_knn=False) if len(cand_edges) else pd.DataFrame(columns=["source_id", "target_id", "edge_prob", "edge_dist", "in_solution", "is_knn"])
    aug = pd.concat([base, pd.DataFrame(new, columns=base.columns)], ignore_index=True) if new else base
    return aug, len(new)


def _primary_linking(cand_nodes, edges, vel, pos, cfg: LinkConfig):
    """Phase 1: one slot per source, each target <=1 parent, Hungarian over real
    candidate targets + per-source no-link dummies. A real link is taken only if
    it passes min_prob / max_link_um / max_cost AND beats null_link_cost."""
    t_of = dict(zip(cand_nodes["node_id"].astype(int), cand_nodes["t"].astype(int)))
    by_src_t = {}
    for r in (edges.itertuples() if len(edges) else []):
        by_src_t.setdefault(t_of[int(r.source_id)], []).append(r)
    selected = []          # dicts: source_id,target_id,edge_prob,edge_dist,fused,is_knn
    n_no_link = 0
    used_targets = set()
    for t, rows in by_src_t.items():
        srcs = sorted(set(int(r.source_id) for r in rows))
        tgts = sorted(set(int(r.target_id) for r in rows))
        s_idx = {s: i for i, s in enumerate(srcs)}
        t_idx = {tg: j for j, tg in enumerate(tgts)}
        n_s, n_t = len(srcs), len(tgts)
        BIG = 1e7
        cost = np.full((n_s, n_t + n_s), BIG, dtype=float)
        meta = {}
        for r in rows:
            s, tg = int(r.source_id), int(r.target_id)
            prob = float(r.edge_prob) if not np.isnan(r.edge_prob) else 0.0
            dist = float(r.edge_dist) if not np.isnan(r.edge_dist) else physical_distance_um(pos.loc[s].to_numpy(float), pos.loc[tg].to_numpy(float))
            if prob < cfg.min_link_prob or dist > cfg.max_link_um:
                continue
            mdev = 0.0
            if s in vel:
                pred_um = pos.loc[s].to_numpy(float) * _VOXEL_SCALE + vel[s]
                mdev = float(np.linalg.norm(pos.loc[tg].to_numpy(float) * _VOXEL_SCALE - pred_um))
            fc = _fused_cost(prob, dist, mdev, cfg)
            if fc > cfg.max_cost or fc >= cfg.null_link_cost:
                continue
            i, j = s_idx[s], t_idx[tg]
            if fc < cost[i, j]:
                cost[i, j] = fc
                meta[(i, j)] = {"source_id": s, "target_id": tg, "edge_prob": prob, "edge_dist": dist,
                                "fused": fc, "is_knn": bool(getattr(r, "is_knn", False))}
        for i in range(n_s):
            cost[i, n_t + i] = cfg.null_link_cost
        for i, j in _m30_assign(cost):
            if j >= n_t:
                n_no_link += 1
                continue
            if cost[i, j] >= BIG:
                n_no_link += 1
                continue
            m = meta.get((i, j))
            if m is None or m["target_id"] in used_targets:
                n_no_link += 1
                continue
            selected.append(m)
            used_targets.add(m["target_id"])
    return selected, n_no_link, used_targets


def _division_linking(cand_nodes, edges, primary, used_targets, pos, cfg: LinkConfig):
    """Phase 2: only sources that got a primary child (out_degree 1) may take a
    SECOND child among currently-UNASSIGNED targets. Division gates applied
    BEFORE assignment; independent Hungarian with per-parent no-division dummies
    (a rejected candidate never steals another source's target); global cap."""
    diag = {"division_candidates": 0, "divisions_selected": 0}
    if not cfg.div_enable or not primary:
        return [], diag
    t_of = dict(zip(cand_nodes["node_id"].astype(int), cand_nodes["t"].astype(int)))
    child_of = {m["source_id"]: m["target_id"] for m in primary}
    parents = sorted(child_of.keys())
    cand_by_src = {}
    for r in (edges.itertuples() if len(edges) else []):
        cand_by_src.setdefault(int(r.source_id), []).append(r)
    entries = []   # (parent, target, fused, prob, dist)
    avail_targets = set()
    for p in parents:
        c1 = child_of[p]
        c1_pos = pos.loc[c1].to_numpy(float)
        p_pos = pos.loc[p].to_numpy(float)
        for r in cand_by_src.get(p, []):
            tg = int(r.target_id)
            if tg == c1 or tg in used_targets:
                continue
            if t_of[tg] != t_of[c1]:
                continue
            prob = float(r.edge_prob) if not np.isnan(r.edge_prob) else 0.0
            if prob < cfg.div_min_prob:
                continue
            d_pc = physical_distance_um(p_pos, pos.loc[tg].to_numpy(float))
            if d_pc > cfg.div_child_max_um:
                continue
            if physical_distance_um(c1_pos, pos.loc[tg].to_numpy(float)) > cfg.div_sister_max_um:
                continue
            fc = _fused_cost(prob, d_pc, 0.0, cfg)
            entries.append({"parent": p, "target": tg, "fused": fc, "edge_prob": prob, "edge_dist": d_pc})
            avail_targets.add(tg)
    diag["division_candidates"] = len(entries)
    if not entries:
        return [], diag
    elig_parents = sorted(set(e["parent"] for e in entries))
    tgt_list = sorted(avail_targets)
    p_idx = {p: i for i, p in enumerate(elig_parents)}
    t_idx = {tg: j for j, tg in enumerate(tgt_list)}
    n_p, n_t = len(elig_parents), len(tgt_list)
    BIG = 1e7
    cost = np.full((n_p, n_t + n_p), BIG, dtype=float)
    cell = {}
    for e in entries:
        i, j = p_idx[e["parent"]], t_idx[e["target"]]
        if e["fused"] < cost[i, j]:
            cost[i, j] = e["fused"]
            cell[(i, j)] = e
    for i in range(n_p):
        cost[i, n_t + i] = cfg.div_no_split_cost
    picks = []
    for i, j in _m30_assign(cost):
        if j >= n_t or cost[i, j] >= BIG:
            continue
        picks.append(cell[(i, j)])
    # global cap: keep the best (lowest fused) divisions.
    global_cap = max(1, int(cfg.div_global_cap_frac * max(len(cand_nodes), 1)))
    picks.sort(key=lambda e: e["fused"])
    picks = picks[:global_cap]
    div_edges = [{"source_id": e["parent"], "target_id": e["target"], "edge_prob": e["edge_prob"],
                  "edge_dist": e["edge_dist"], "fused": e["fused"], "is_knn": False} for e in picks]
    diag["divisions_selected"] = len(div_edges)
    return div_edges, diag


def stitch_gaps_m30(nodes, edges, cfg: LinkConfig):
    """Post-link gap stitching for cfg.gap_sizes. END@t -> START@t+g+1 bridged by
    g synthetic nodes + g+1 UNIT edges, gated on per-step and total distance.
    Never a direct multi-frame edge."""
    diag = {f"gap{g}_closed": 0 for g in (1, 2, 3)}
    if len(nodes) == 0:
        return nodes.copy(), edges.copy(), diag, -1
    sid = -1
    new_nodes, new_edges = [], []
    used_e, used_s = set(), set()
    for g in cfg.gap_sizes:
        max_total = cfg.gap_max_total_um.get(g, 14.5) if isinstance(cfg.gap_max_total_um, dict) else cfg.gap_max_total_um
        cur_edges = edges if not new_edges else pd.concat([edges, pd.DataFrame([{"source_id": e["source_id"], "target_id": e["target_id"]} for e in new_edges])], ignore_index=True)
        out_d, in_d = (cur_edges.groupby("source_id").size().to_dict(), cur_edges.groupby("target_id").size().to_dict()) if len(cur_edges) else ({}, {})
        cur_nodes = nodes if not new_nodes else pd.concat([nodes, pd.DataFrame(new_nodes)], ignore_index=True)
        pos = cur_nodes.set_index("node_id")[["z", "y", "x"]]
        t_all = dict(zip(cur_nodes["node_id"].astype(int), cur_nodes["t"].astype(int)))
        max_t, min_t = int(cur_nodes["t"].max()), int(cur_nodes["t"].min())
        ends = [int(n) for n in cur_nodes["node_id"] if out_d.get(int(n), 0) == 0 and t_all[int(n)] <= max_t - (g + 1) and int(n) not in used_e]
        starts_by_t = {}
        for n in cur_nodes["node_id"]:
            n = int(n)
            if in_d.get(n, 0) == 0 and t_all[n] > min_t and n not in used_s:
                starts_by_t.setdefault(t_all[n], []).append(n)
        ends_by_t = {}
        for e in ends:
            ends_by_t.setdefault(t_all[e], []).append(e)
        for t, elist in ends_by_t.items():
            slist = [s for s in starts_by_t.get(t + g + 1, []) if s not in used_s]
            if not slist:
                continue
            BIG = 1e7
            cost = np.full((len(elist), len(slist)), BIG, dtype=float)
            for i, e in enumerate(elist):
                e_um = pos.loc[e].to_numpy(float) * _VOXEL_SCALE
                for j, s in enumerate(slist):
                    s_um = pos.loc[s].to_numpy(float) * _VOXEL_SCALE
                    total = float(np.linalg.norm(s_um - e_um))
                    if total > max_total or (total / (g + 1)) > cfg.gap_max_step_um:
                        continue
                    cost[i, j] = total
            for i, j in _m30_assign(cost):
                if cost[i, j] >= BIG:
                    continue
                e, s = elist[i], slist[j]
                if e in used_e or s in used_s:
                    continue
                e_pos = pos.loc[e].to_numpy(float)
                s_pos = pos.loc[s].to_numpy(float)
                prev = e
                for k in range(1, g + 1):
                    m = e_pos + (s_pos - e_pos) * (k / float(g + 1))
                    nid = sid
                    sid -= 1
                    new_nodes.append({"node_id": nid, "t": t + k, "z": float(m[0]), "y": float(m[1]), "x": float(m[2])})
                    new_edges.append({"source_id": prev, "target_id": nid})
                    prev = nid
                new_edges.append({"source_id": prev, "target_id": s})
                used_e.add(e)
                used_s.add(s)
                diag[f"gap{g}_closed"] += 1
    res_nodes = pd.concat([nodes, pd.DataFrame(new_nodes, columns=["node_id", "t", "z", "y", "x"])], ignore_index=True) if new_nodes else nodes.copy()
    res_edges = pd.concat([edges, pd.DataFrame([{"source_id": e["source_id"], "target_id": e["target_id"]} for e in new_edges], columns=["source_id", "target_id"])], ignore_index=True) if new_edges else edges.copy()
    return res_nodes, res_edges, diag, sid


def run_relinker_v2(cand_nodes, cand_edges, cfg: LinkConfig, candidate_class: str) -> tuple:
    """Full v2 on ONE dataset's candidate graph: kNN augment (optional) ->
    primary Hungarian (with no-link) -> division Hungarian (with no-division) ->
    stitch gaps -> linefit (optional) -> prune. Returns (nodes, edges2, diag)."""
    diag = {"raw_candidate_edges": int(len(cand_edges)), "candidate_class": candidate_class}
    aug_edges, n_knn = knn_augment_candidates(cand_nodes, cand_edges, cfg)
    diag["knn_candidates_added"] = int(n_knn)
    diag["augmented_candidate_edges"] = int(len(aug_edges))
    vel, pos = _source_velocities(cand_nodes, aug_edges)

    primary, n_no_link, used_targets = _primary_linking(cand_nodes, aug_edges, vel, pos, cfg)
    diag["primary_links_selected"] = len(primary)
    diag["primary_no_link_selected"] = int(n_no_link)
    div_edges, dd = _division_linking(cand_nodes, aug_edges, primary, used_targets, pos, cfg)
    diag.update(dd)

    sel = primary + div_edges
    diag["links_selected_from_raw"] = int(sum(1 for m in sel if not m.get("is_knn")))
    diag["links_selected_from_knn"] = int(sum(1 for m in sel if m.get("is_knn")))
    if sel:
        probs = np.array([m["edge_prob"] for m in sel], dtype=float)
        dists = np.array([m["edge_dist"] for m in sel], dtype=float)
        fused = np.array([m["fused"] for m in sel], dtype=float)
        diag["mean_selected_edge_prob"] = float(probs.mean())
        diag["selected_edge_prob_quantiles"] = _quantiles(probs, [0.05, 0.25, 0.5, 0.75, 0.95])
        diag["mean_selected_edge_distance_um"] = float(dists.mean())
        diag["mean_selected_fused_cost"] = float(fused.mean())
        diag["selected_fused_cost_quantiles"] = _quantiles(fused, [0.05, 0.25, 0.5, 0.75, 0.95])
    else:
        diag["mean_selected_edge_prob"] = None
        diag["selected_edge_prob_quantiles"] = {}
        diag["mean_selected_edge_distance_um"] = None
        diag["mean_selected_fused_cost"] = None
        diag["selected_fused_cost_quantiles"] = {}

    edges2 = pd.DataFrame([{"source_id": m["source_id"], "target_id": m["target_id"]} for m in sel], columns=["source_id", "target_id"])
    nodes = cand_nodes[["node_id", "t", "z", "y", "x"]].copy()
    nodes, edges2, gd, _sid = stitch_gaps_m30(nodes, edges2, cfg)
    diag.update(gd)
    diag["synthetic_nodes_added"] = gd["gap1_closed"] + 2 * gd["gap2_closed"] + 3 * gd["gap3_closed"]

    if cfg.linefit_enable:
        nodes, lf = linefit_smooth_nodes(nodes, edges2, window=cfg.linefit_window, weight=cfg.linefit_weight)
        diag["n_smoothed"] = lf.get("n_nodes_smoothed", 0)
    else:
        diag["n_smoothed"] = 0
    if cfg.prune_enable:
        nodes, edges2, pr = prune_isolated_nodes(nodes, edges2, max_frac=cfg.prune_max_frac)
        diag["isolated_nodes_pruned"] = pr.get("n_pruned", 0)
    else:
        diag["isolated_nodes_pruned"] = 0

    out_dF = edges2.groupby("source_id").size().to_dict() if len(edges2) else {}
    diag["divisions_total"] = int(sum(1 for v in out_dF.values() if v >= 2))
    diag["final_nodes"] = int(len(nodes))
    diag["final_edges"] = int(len(edges2))
    return nodes, edges2, diag


def convert_relinker_to_submission(predictions_dir: Path, expected_datasets, cfg: LinkConfig, candidate_class: str) -> tuple:
    """Per dataset: read RAW candidate GEFF -> build_candidate_graph_v2 ->
    run_relinker_v2 -> relabel positive-consecutive -> emit rows. Aggregates the
    v2 diagnostics + the Stage-0 richness/edge_prob profile (recomputed here so
    the submit report is self-consistent)."""
    stores = find_prediction_geff_stores(predictions_dir)
    all_rows, per_dataset = [], []
    next_id = next_node_id = 0
    keys = ["raw_candidate_edges", "knn_candidates_added", "augmented_candidate_edges",
            "primary_links_selected", "primary_no_link_selected", "division_candidates", "divisions_selected",
            "links_selected_from_raw", "links_selected_from_knn", "gap1_closed", "gap2_closed", "gap3_closed",
            "synthetic_nodes_added", "isolated_nodes_pruned", "n_smoothed", "divisions_total", "final_nodes", "final_edges"]
    agg = {k: 0 for k in keys}
    emptied = []
    all_probs, all_dists, all_fused = [], [], []
    prob_available_all, prob_nondegen_all = True, True
    rich_means, rich_ge2 = [], []
    for dataset_name, geff_root in stores:
        cg = read_candidate_geff(geff_root)
        cand_nodes, cand_edges, cdiag = build_candidate_graph_v2(cg)
        prob_info = analyze_edge_prob(cand_edges, cg["has_edge_prob"])
        rich = candidate_richness_stats(cand_nodes, cand_edges)
        prob_available_all = prob_available_all and prob_info["edge_prob_available"]
        prob_nondegen_all = prob_nondegen_all and prob_info["edge_prob_non_degenerate"]
        rich_means.append(rich["candidates_per_source_mean"]); rich_ge2.append(rich["pct_sources_ge2_candidates"])
        fn, fe, d = run_relinker_v2(cand_nodes, cand_edges, cfg, candidate_class)
        fn, fe, next_node_id = relabel_positive(fn, fe, next_node_id)
        if len(fn) == 0:
            emptied.append(dataset_name)
        for row in fn.itertuples():
            all_rows.append({"id": next_id, "dataset": dataset_name, "row_type": "node", "node_id": int(row.node_id),
                             "t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x), "source_id": -1, "target_id": -1})
            next_id += 1
        for row in (fe.itertuples() if len(fe) else []):
            all_rows.append({"id": next_id, "dataset": dataset_name, "row_type": "edge", "node_id": -1,
                             "t": -1, "z": -1, "y": -1, "x": -1, "source_id": int(row.source_id), "target_id": int(row.target_id)})
            next_id += 1
        for k in keys:
            agg[k] += int(d.get(k, 0))
        if d.get("mean_selected_edge_prob") is not None:
            all_probs.append((d["mean_selected_edge_prob"], d["primary_links_selected"] + d["divisions_selected"]))
            all_dists.append((d["mean_selected_edge_distance_um"], d["primary_links_selected"] + d["divisions_selected"]))
            all_fused.append((d["mean_selected_fused_cost"], d["primary_links_selected"] + d["divisions_selected"]))
        per_dataset.append({"dataset": dataset_name, "nodes": d["final_nodes"], "edges": d["final_edges"],
                            "candidate_diag": cdiag, "edge_prob": prob_info, "richness": rich, "v2": d})
        print(f"  [m30] {dataset_name}: raw_cand={d['raw_candidate_edges']} knn+={d['knn_candidates_added']} "
              f"primary={d['primary_links_selected']}(no_link={d['primary_no_link_selected']}) div={d['divisions_selected']} "
              f"-> {d['final_nodes']}/{d['final_edges']} (gap1={d['gap1_closed']} gap2={d['gap2_closed']} gap3={d['gap3_closed']})")

    def _wmean(pairs):
        num = sum(v * w for v, w in pairs); den = sum(w for _, w in pairs)
        return float(num / den) if den else None
    submission_df = pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)
    conversion = {
        "predictions_dir": str(predictions_dir), "n_stores_found": len(stores),
        "found_datasets": [n for n, _ in stores], "expected_datasets": list(expected_datasets),
        "candidate_class": candidate_class, "link_config": asdict(cfg),
        "edge_prob_available": bool(prob_available_all), "edge_prob_non_degenerate": bool(prob_nondegen_all),
        "candidates_per_source_mean": float(np.mean(rich_means)) if rich_means else 0.0,
        "candidates_per_source_p90": float(np.mean([p["richness"]["candidates_per_source_p90"] for p in per_dataset])) if per_dataset else 0.0,
        "pct_sources_ge2_candidates": float(np.mean(rich_ge2)) if rich_ge2 else 0.0,
        "mean_selected_edge_prob": _wmean(all_probs), "mean_selected_edge_distance_um": _wmean(all_dists),
        "mean_selected_fused_cost": _wmean(all_fused), "emptied_datasets": emptied, **agg, "per_dataset": per_dataset,
    }
    return submission_df, conversion

# --------------------------------------------------------------------------- #
# 43. M30 variant registry + 400ep artifact guard + candidate-mode resolution
# --------------------------------------------------------------------------- #
M30_400EP_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
M30_400EP_NAME_SUBSTR = "400ep"
M30_WEIGHTS_REL = "weights/unet_transformer/split_0/edge_predictor_best.pth"

# Universal submission sanity window (safety gate).
M30_MIN_NODE_ROWS, M30_MAX_NODE_ROWS = 110000, 165000
M30_MIN_EDGE_ROWS, M30_MAX_EDGE_ROWS = 100000, 155000
M30_MAX_SYNTHETIC = 12000
M30_MAX_DIVISIONS_TOTAL = 4000

_FULL = {"FULL_CANDIDATE_GRAPH"}
_SPARSE = {"ILP_SOLUTION_LIKE", "MODERATELY_SPARSE"}


def _linkcfg(**ov) -> LinkConfig:
    cfg = LinkConfig()
    for k, v in ov.items():
        setattr(cfg, k, v)
    return cfg


M30_VARIANTS = {
    # A: full-candidate raw relinker, balanced. Only for FULL_CANDIDATE_GRAPH.
    "A": {"label": "v2_full_candidates_balanced", "det_threshold": "0.99", "mode": "raw",
          "required_classes": _FULL,
          "config": _linkcfg(null_link_cost=4.5, w_dist=1.0, w_motion=0.4, max_link_um=7.5, min_link_prob=0.05,
                             max_cost=12.0, div_enable=True, div_min_prob=0.20, div_child_max_um=6.5,
                             div_sister_max_um=9.0, div_global_cap_frac=0.020, gap_sizes=(1, 2),
                             linefit_enable=False, prune_enable=True, knn_enable=False)},
    # B: A + gap3.
    "B": {"label": "v2_full_candidates_gap123", "det_threshold": "0.99", "mode": "raw",
          "required_classes": _FULL,
          "config": _linkcfg(null_link_cost=4.5, w_dist=1.0, w_motion=0.4, max_link_um=7.5, min_link_prob=0.05,
                             max_cost=12.0, div_enable=True, div_min_prob=0.20, div_child_max_um=6.5,
                             div_sister_max_um=9.0, div_global_cap_frac=0.020, gap_sizes=(1, 2, 3),
                             linefit_enable=False, prune_enable=True, knn_enable=False)},
    # C: sparse/ILP-like - raw + tight kNN augmentation. Primary sparse variant.
    "C": {"label": "v2_auto_sparse_knn_tight", "det_threshold": "0.99", "mode": "knn",
          "required_classes": _SPARSE,
          "config": _linkcfg(null_link_cost=4.2, w_dist=1.0, w_motion=0.4, max_link_um=7.0, min_link_prob=0.05,
                             max_cost=12.0, div_enable=True, div_min_prob=0.20, div_child_max_um=6.5,
                             div_sister_max_um=9.0, div_global_cap_frac=0.020, gap_sizes=(1, 2),
                             linefit_enable=False, prune_enable=True, knn_enable=True, knn_k=3,
                             knn_max_um=4.5, knn_default_prob=0.20)},
    # D: auto mode (raw if FULL, tight kNN if sparse) at det 0.95.
    "D": {"label": "v2_auto_det095", "det_threshold": "0.95", "mode": "auto", "required_classes": None,
          "config": _linkcfg(null_link_cost=4.2, w_dist=1.0, w_motion=0.4, max_link_um=7.0, min_link_prob=0.05,
                             max_cost=12.0, div_enable=True, div_min_prob=0.20, div_child_max_um=6.5,
                             div_sister_max_um=9.0, div_global_cap_frac=0.020, gap_sizes=(1, 2),
                             linefit_enable=False, prune_enable=True, knn_k=3, knn_max_um=4.5, knn_default_prob=0.20),
          "node_explosion_guard": True},
    # E: auto mode, relaxed divisions (tests the 0.1-weight division term).
    "E": {"label": "v2_division_relaxed", "det_threshold": "0.99", "mode": "auto", "required_classes": None,
          "config": _linkcfg(null_link_cost=4.5, w_dist=1.0, w_motion=0.4, max_link_um=7.5, min_link_prob=0.05,
                             max_cost=12.0, div_enable=True, div_min_prob=0.15, div_child_max_um=6.5,
                             div_sister_max_um=9.0, div_global_cap_frac=0.030, gap_sizes=(1, 2),
                             linefit_enable=False, prune_enable=True, knn_k=3, knn_max_um=4.5, knn_default_prob=0.20),
          "division_hard_guard": True},
}


def resolve_link_config(variant_name: str, candidate_class: str) -> LinkConfig:
    """Effective LinkConfig for the variant given the Stage-0 class. 'auto' turns
    kNN ON for sparse/ILP-like classes and OFF for a full candidate graph; 'knn'
    forces it ON; 'raw' forces it OFF."""
    vdef = M30_VARIANTS[variant_name]
    cfg = _linkcfg(**{k: getattr(vdef["config"], k) for k in vdef["config"].__dataclass_fields__})
    mode = vdef["mode"]
    if mode == "raw":
        cfg.knn_enable = False
    elif mode == "knn":
        cfg.knn_enable = True
    else:  # auto
        cfg.knn_enable = candidate_class in _SPARSE
    return cfg


def _sha256_file_m30(path) -> str | None:
    import hashlib
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _m30_read_manifest_name(root):
    mp = Path(root) / "ARTIFACT_MANIFEST.json"
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return None
    return m.get("artifact_name") or m.get("name") or m.get("artifact") or m.get("dataset")


def select_400ep_artifact_m30() -> dict:
    """Resolve the reference artifact and confirm it is the guarded 400ep pack
    (name contains '400ep' AND weight_sha256 matches). M30 still runs whatever
    resolve_artifact() returns, but reports whether the guard passed."""
    artifact = resolve_artifact()
    if not artifact.get("resolved"):
        return {"artifact_guard_passed": False, "resolved": False, "artifact_dir": None,
                "artifact_name": None, "weight_sha256": None, "expected_sha256": M30_400EP_SHA256}
    adir = Path(artifact["artifact_dir"])
    name = _m30_read_manifest_name(adir)
    sha = _sha256_file_m30(adir / M30_WEIGHTS_REL)
    blob = " ".join(str(x).lower() for x in [name, adir.name, str(adir)])
    passed = bool(sha == M30_400EP_SHA256 and M30_400EP_NAME_SUBSTR in blob)
    return {"artifact_guard_passed": passed, "resolved": True, "artifact_dir": str(adir),
            "artifact_name": name, "weight_sha256": sha, "expected_sha256": M30_400EP_SHA256,
            "name_has_400ep": (M30_400EP_NAME_SUBSTR in blob), "sha_matches": (sha == M30_400EP_SHA256)}


def build_m30_command(
    variant_name: str, introspection: dict,
    data_dir: str = f"{KAGGLE_COMPETITION_INPUT_DIR}/test", splits_filename: str = SPLITS_FILENAME,
) -> tuple:
    """Reference predict command with the variant's det-threshold (all other
    flags identical to M16/M19-C). Only D uses det 0.95."""
    vdef = M30_VARIANTS[variant_name]
    params = dict(BASE_PREDICT_PARAMS)
    params["det_threshold"] = vdef["det_threshold"]
    available_splits = list(introspection.get("weight_splits", [])) or [0]
    split = select_weight_split(0, available_splits)
    cmd = build_base_command(params, split, splits_filename, data_dir)
    notes = {"variant": variant_name, "label": vdef["label"], "det_threshold": params["det_threshold"],
             "split_used": split, "available_splits": available_splits, "mode": vdef["mode"],
             "predict_command_flags_identical_to_m16_except_det": True}
    return cmd, notes

# --------------------------------------------------------------------------- #
# 44. M30 Stage-0 candidate diagnostic + submit orchestration + v2 CV harness
# --------------------------------------------------------------------------- #
def _predict_on_artifact(variant_name, out_dir, competition_dir, det_threshold):
    """Resolve -> materialize -> deps -> splits -> reference predict at the given
    det-threshold. Returns (predictions_dir, expected_datasets, artifact_sel)."""
    comp = Path(competition_dir)
    test_dir = comp / "test"
    sample_submission_path = comp / "sample_submission.csv"
    artifact_sel = select_400ep_artifact_m30()
    if not artifact_sel["resolved"]:
        raise RuntimeError("support-pack artifact not found (see milestone30_artifact_selection.json)")
    artifact_dir = Path(artifact_sel["artifact_dir"])
    materialize = materialize_repo(artifact_dir, out_dir)
    write_sitecustomize()
    env = build_subprocess_env()
    introspection = introspect_artifact(artifact_dir, out_dir)
    dep_report = install_dependencies(artifact_dir / "wheels", env)
    if not dep_report["all_ok"]:
        raise RuntimeError("offline wheel install failed")
    if not verify_imports(env)["all_ok"]:
        raise RuntimeError("post-install import check failed")
    repo_dst = Path(materialize["repo_dst"])
    shipped = repo_dst / SPLITS_FILENAME
    prepare_splits(test_dir=test_dir, sample_submission_path=sample_submission_path,
                   template_path=shipped if shipped.exists() else None,
                   out_splits_path=repo_dst / SPLITS_FILENAME, diagnostic_path=out_dir / "milestone30_splits.json")
    cmd, notes = build_m30_command(variant_name, introspection)
    log("predict command: " + " ".join(cmd))
    exec_result = run_prediction(cmd, cwd=repo_dst, env=env)
    if not exec_result["ok"]:
        raise RuntimeError("predict script returned nonzero")
    predictions_dir = repo_dst / "predictions" / "unknown" / "unet_transformer" / f"split_{notes['split_used']}"
    if not predictions_dir.exists():
        predictions_dir = repo_dst / "predictions" / "unknown" / "unet_transformer" / "split_0"
    expected = discover_test_datasets(test_dir, sample_submission_path)["chosen"]
    return predictions_dir, expected, artifact_sel


def compute_stage0(predictions_dir: Path, expected_datasets) -> dict:
    """Per-dataset candidate richness + edge_prob profile + per-frame stats, and
    the aggregate candidate_graph_class. Shared by the diagnostic and the submit
    gate so the class is computed identically."""
    stores = find_prediction_geff_stores(predictions_dir)
    per_dataset, per_frame_rows = [], []
    prob_available_all = prob_nondegen_all = True if stores else False
    cps_means, ge2s = [], []
    tot_nodes = tot_unit = tot_nonunit = 0
    for dataset_name, geff_root in stores:
        cg = read_candidate_geff(geff_root)
        cand_nodes, cand_edges, cdiag = build_candidate_graph_v2(cg)
        prob_info = analyze_edge_prob(cand_edges, cg["has_edge_prob"])
        rich = candidate_richness_stats(cand_nodes, cand_edges)
        prob_available_all = prob_available_all and prob_info["edge_prob_available"]
        prob_nondegen_all = prob_nondegen_all and prob_info["edge_prob_non_degenerate"]
        cps_means.append(rich["candidates_per_source_mean"]); ge2s.append(rich["pct_sources_ge2_candidates"])
        tot_nodes += len(cand_nodes); tot_unit += cdiag["n_unit_candidates"]; tot_nonunit += cdiag["n_non_unit_candidates"]
        # per-frame candidate counts.
        t_of = dict(zip(cand_nodes["node_id"].astype(int), cand_nodes["t"].astype(int)))
        out_cnt = cand_edges.groupby("source_id").size().to_dict() if len(cand_edges) else {}
        by_t = {}
        for n in cand_nodes["node_id"]:
            by_t.setdefault(t_of[int(n)], []).append(int(n))
        for t, ns in sorted(by_t.items()):
            cs = np.array([out_cnt.get(n, 0) for n in ns], dtype=float)
            per_frame_rows.append({"dataset": dataset_name, "t": t, "n_nodes": len(ns),
                                   "cand_per_source_mean": float(cs.mean()) if len(cs) else 0.0,
                                   "pct_ge2": float(np.mean(cs >= 2)) if len(cs) else 0.0})
        per_dataset.append({"dataset": dataset_name, "raw_nodes": len(cand_nodes),
                            "n_unit_candidates": cdiag["n_unit_candidates"], "n_non_unit_candidates": cdiag["n_non_unit_candidates"],
                            "n_edges_total": cg["n_edges_total"], "n_edges_solution": cg["n_edges_solution"],
                            "edge_prob": prob_info, "richness": rich})
    agg_prob = {"edge_prob_available": bool(prob_available_all), "edge_prob_non_degenerate": bool(prob_nondegen_all)}
    agg_rich = {"candidates_per_source_mean": float(np.mean(cps_means)) if cps_means else 0.0,
                "pct_sources_ge2_candidates": float(np.mean(ge2s)) if ge2s else 0.0}
    candidate_class = classify_candidate_graph(agg_prob, agg_rich)
    return {"candidate_class": candidate_class, "n_datasets": len(stores),
            "dataset_names": [n for n, _ in stores], "raw_node_count": tot_nodes,
            "unit_candidate_edges": tot_unit, "non_unit_candidate_edges": tot_nonunit,
            "edge_prob_available": agg_prob["edge_prob_available"], "edge_prob_non_degenerate": agg_prob["edge_prob_non_degenerate"],
            "candidates_per_source_mean": agg_rich["candidates_per_source_mean"],
            "pct_sources_ge2_candidates": agg_rich["pct_sources_ge2_candidates"],
            "recommended_submit_variant": recommend_submit_variant(candidate_class),
            "per_dataset": per_dataset, "per_frame": per_frame_rows}


def run_m30_candidate_diagnostic(working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    """STAGE 0 (non-submit): run det=0.99 predict on the 400ep artifact and profile
    the raw candidate graph. Writes m30_candidate_graph_report.json + table.csv.
    Never writes a competition submission."""
    _RUN_LOG.clear()
    out_dir = Path(working_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report = {"kind": "candidate_graph_diagnostic", "submit": False}
    try:
        predictions_dir, expected, artifact_sel = _predict_on_artifact("A", out_dir, competition_dir, "0.99")
        report["artifact_name"] = artifact_sel["artifact_name"]
        report["weight_sha256"] = artifact_sel["weight_sha256"]
        report["artifact_guard_passed"] = artifact_sel["artifact_guard_passed"]
        s0 = compute_stage0(predictions_dir, expected)
        report.update(s0)
        report["status"] = "ok"
        pd.DataFrame(s0["per_frame"]).to_csv(out_dir / "m30_candidate_graph_table.csv", index=False)
    except Exception:
        report["status"] = "error"; report["traceback"] = traceback.format_exc()
    (out_dir / "m30_candidate_graph_report.json").write_text(json.dumps(report, indent=2, default=str))
    _write_log(out_dir)
    print("=== M30 STAGE-0 CANDIDATE GRAPH DIAGNOSTIC (non-submit) ===")
    if report.get("status") == "ok":
        print(f"  candidate_graph_class: {report['candidate_class']}")
        print(f"  edge_prob available/non-degenerate: {report['edge_prob_available']}/{report['edge_prob_non_degenerate']}")
        print(f"  candidates/source mean: {report['candidates_per_source_mean']:.3f}  pct sources >=2: {report['pct_sources_ge2_candidates']:.3f}")
        print(f"  unit candidates: {report['unit_candidate_edges']}  non-unit: {report['non_unit_candidate_edges']}")
        print(f"  recommended submit variant: {report['recommended_submit_variant']}")
    else:
        print("  diagnostic ERROR - see m30_candidate_graph_report.json")
    print("  No submission was created (Stage-0 diagnostic).")
    return report


def _m30_invariants(df, expected_datasets):
    nodes = df[df["row_type"] == "node"]; edges = df[df["row_type"] == "edge"]
    inv = {"n_node_rows": int(len(nodes)), "n_edge_rows": int(len(edges)), "max_in_degree": 0, "max_out_degree": 0,
           "direct_multiframe_edges": 0, "dangling_edges": 0, "id_consecutive": True, "no_nan": True,
           "datasets_present": sorted(df["dataset"].unique().tolist()), "missing_datasets": [], "emptied_datasets": []}
    ids = df["id"].to_numpy()
    inv["id_consecutive"] = bool(len(ids) == 0 or (ids.min() == 0 and ids.max() == len(ids) - 1 and len(set(ids.tolist())) == len(ids)))
    if len(nodes):
        inv["no_nan"] = bool(not nodes[["t", "z", "y", "x", "node_id"]].isna().any().any())
    for ds, g in df.groupby("dataset"):
        gn = g[g["row_type"] == "node"]; ge = g[g["row_type"] == "edge"]
        if len(gn) == 0:
            inv["emptied_datasets"].append(ds)
        t_of = dict(zip(gn["node_id"].astype(int), gn["t"].astype(int)))
        if len(ge):
            inv["max_in_degree"] = max(inv["max_in_degree"], int(ge.groupby("target_id").size().max()))
            inv["max_out_degree"] = max(inv["max_out_degree"], int(ge.groupby("source_id").size().max()))
            for r in ge.itertuples():
                s, t = int(r.source_id), int(r.target_id)
                if s not in t_of or t not in t_of:
                    inv["dangling_edges"] += 1
                elif t_of[t] - t_of[s] != 1:
                    inv["direct_multiframe_edges"] += 1
    inv["missing_datasets"] = [d for d in expected_datasets if d not in set(inv["datasets_present"])]
    return inv


def _m30_recommendation(variant_name, vdef, artifact_guard_passed, candidate_class, edge_prob_available,
                        edge_prob_non_degenerate, knn_enabled, valid, fallback_used, inv, synthetic, divisions_total) -> str:
    if not artifact_guard_passed:
        return "DO_NOT_SUBMIT_VALIDATION_FAILED"      # wrong/absent controlled artifact
    if not edge_prob_available or not edge_prob_non_degenerate or candidate_class == "EDGE_PROB_UNAVAILABLE":
        return "DO_NOT_SUBMIT_EDGE_PROB_UNAVAILABLE"
    req = vdef.get("required_classes")
    if req is not None and candidate_class not in req:
        return "DO_NOT_SUBMIT_WRONG_CANDIDATE_MODE"
    if candidate_class in _SPARSE and not knn_enabled:
        return "DO_NOT_SUBMIT_CANDIDATE_GRAPH_TOO_SPARSE"
    if not valid or fallback_used or not inv["no_nan"] or not inv["id_consecutive"]:
        return "DO_NOT_SUBMIT_VALIDATION_FAILED"
    if inv["dangling_edges"] > 0 or inv["direct_multiframe_edges"] > 0 or inv["max_in_degree"] > 1 or inv["max_out_degree"] > 2:
        return "DO_NOT_SUBMIT_VALIDATION_FAILED"
    if inv["missing_datasets"] or inv["emptied_datasets"]:
        return "DO_NOT_SUBMIT_MISSING_OR_EMPTY_DATASET"
    if vdef.get("node_explosion_guard") and (inv["n_node_rows"] > M30_MAX_NODE_ROWS or inv["n_edge_rows"] > M30_MAX_EDGE_ROWS):
        return "DO_NOT_SUBMIT_NODE_EXPLOSION"
    if divisions_total > M30_MAX_DIVISIONS_TOTAL:
        return "DO_NOT_SUBMIT_DIVISION_EXPLOSION"
    if not (M30_MIN_NODE_ROWS <= inv["n_node_rows"] <= M30_MAX_NODE_ROWS):
        return "DO_NOT_SUBMIT_UNSANE_NODE_COUNT"
    if not (M30_MIN_EDGE_ROWS <= inv["n_edge_rows"] <= M30_MAX_EDGE_ROWS):
        return "DO_NOT_SUBMIT_UNSANE_EDGE_COUNT"
    if synthetic > M30_MAX_SYNTHETIC:
        return "DO_NOT_SUBMIT_SYNTHETIC_EXPLOSION"
    return "OK_TO_SUBMIT_EXPERIMENTAL"


def _m30_report(variant_name, vdef, artifact_sel, det, candidate_class, cfg, conv, inv, valid, fallback_used,
                recommendation, instruction) -> dict:
    c = conv or {}; i = inv or {}
    return {
        "variant": variant_name, "label": vdef["label"],
        "artifact_name": artifact_sel.get("artifact_name"), "weight_sha256": artifact_sel.get("weight_sha256"),
        "artifact_guard_passed": artifact_sel.get("artifact_guard_passed"), "det_threshold": det,
        "candidate_graph_class": candidate_class,
        "edge_prob_available": c.get("edge_prob_available"), "edge_prob_non_degenerate": c.get("edge_prob_non_degenerate"),
        "raw_candidate_edges": c.get("raw_candidate_edges", 0), "augmented_candidate_edges": c.get("augmented_candidate_edges", 0),
        "knn_candidates_added": c.get("knn_candidates_added", 0),
        "candidates_per_source_mean": c.get("candidates_per_source_mean"), "candidates_per_source_p90": c.get("candidates_per_source_p90"),
        "pct_sources_ge2_candidates": c.get("pct_sources_ge2_candidates"),
        "null_link_cost": cfg.null_link_cost,
        "primary_links_selected": c.get("primary_links_selected", 0), "primary_no_link_selected": c.get("primary_no_link_selected", 0),
        "division_candidates": c.get("division_candidates", 0), "divisions_selected": c.get("divisions_selected", 0),
        "divisions_total": c.get("divisions_total", 0),
        "links_selected_from_raw": c.get("links_selected_from_raw", 0), "links_selected_from_knn": c.get("links_selected_from_knn", 0),
        "mean_selected_edge_prob": c.get("mean_selected_edge_prob"), "mean_selected_edge_distance_um": c.get("mean_selected_edge_distance_um"),
        "mean_selected_fused_cost": c.get("mean_selected_fused_cost"),
        "gap1_closed": c.get("gap1_closed", 0), "gap2_closed": c.get("gap2_closed", 0), "gap3_closed": c.get("gap3_closed", 0),
        "synthetic_nodes_added": c.get("synthetic_nodes_added", 0), "isolated_nodes_pruned": c.get("isolated_nodes_pruned", 0),
        "n_node_rows": i.get("n_node_rows", 0), "n_edge_rows": i.get("n_edge_rows", 0),
        "max_in_degree": i.get("max_in_degree", 0), "max_out_degree": i.get("max_out_degree", 0),
        "direct_multiframe_edges": i.get("direct_multiframe_edges", 0), "dangling_edges": i.get("dangling_edges", 0),
        "id_consecutive": i.get("id_consecutive"), "no_nan": i.get("no_nan"),
        "missing_datasets": i.get("missing_datasets", []), "emptied_datasets": i.get("emptied_datasets", []),
        "valid": valid, "fallback_used": fallback_used, "final_source": M30_FINAL_SOURCE,
        "recommendation": recommendation, "link_config": asdict(cfg), "mode": vdef["mode"],
        "final_instruction": instruction,
    }


def run_m30_variant_pipeline(variant_name, working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    out_dir = Path(working_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vdef = M30_VARIANTS[variant_name]
    log(f"=== Milestone 30 variant {variant_name} ({vdef['label']}) det={vdef['det_threshold']} mode={vdef['mode']} ===")
    predictions_dir, expected, artifact_sel = _predict_on_artifact(variant_name, out_dir, competition_dir, vdef["det_threshold"])
    (out_dir / "milestone30_artifact_selection.json").write_text(json.dumps(artifact_sel, indent=2, default=str))

    s0 = compute_stage0(predictions_dir, expected)
    (out_dir / "milestone30_candidate_graph_report.json").write_text(json.dumps(s0, indent=2, default=str))
    candidate_class = s0["candidate_class"]
    cfg = resolve_link_config(variant_name, candidate_class)
    log(f"candidate_class={candidate_class} knn_enabled={cfg.knn_enable} guard={artifact_sel['artifact_guard_passed']}")

    submission_df, conversion = convert_relinker_to_submission(predictions_dir, expected, cfg, candidate_class)
    (out_dir / "milestone30_v2_conversion.json").write_text(json.dumps(conversion, indent=2, default=str))

    valid, vreport, vstats = validate_reference_submission(submission_df, expected)
    log(vreport)
    inv = _m30_invariants(submission_df, expected)
    synthetic = int(conversion["synthetic_nodes_added"]); divisions_total = int(conversion["divisions_total"])
    submission_path = out_dir / "submission.csv"
    if valid:
        submission_df.to_csv(submission_path, index=False)
        submission_df.to_csv(out_dir / f"submission_m30_variant_{variant_name}.csv", index=False)
        write_reference_diagnostics(submission_df, valid, vstats, out_dir)

    recommendation = _m30_recommendation(variant_name, vdef, artifact_sel["artifact_guard_passed"], candidate_class,
                                         conversion["edge_prob_available"], conversion["edge_prob_non_degenerate"],
                                         cfg.knn_enable, valid, False, inv, synthetic, divisions_total)
    report_obj = _m30_report(variant_name, vdef, artifact_sel, vdef["det_threshold"], candidate_class, cfg, conversion, inv,
                             valid, False, recommendation,
                             instruction=("submission.csv built (M30 metric-aware global relinker v2, experimental). "
                                          + ("Review on-Kaggle gates then submit." if recommendation == "OK_TO_SUBMIT_EXPERIMENTAL"
                                             else f"{recommendation} - keep M19-C 0.880.")
                                          + ("" if artifact_sel["artifact_guard_passed"] else " NOTE: 400ep artifact guard FAILED (wrong/absent controlled artifact).")))
    (out_dir / "milestone30_reference_submission_report.json").write_text(json.dumps(report_obj, indent=2, default=str))
    (out_dir / "milestone30_failure_fallback_report.json").write_text(json.dumps(
        {"variant": variant_name, "fallback": {"used": False, "reason": "pipeline succeeded"}, "final_source": M30_FINAL_SOURCE}, indent=2))
    if not valid:
        raise RuntimeError("M30 submission FAILED validation (falling back)")

    print("\n=== Variant summary ===")
    print(f"  variant: {variant_name} ({vdef['label']})  det: {vdef['det_threshold']}  mode: {vdef['mode']}  class: {candidate_class}")
    print(f"  artifact: {artifact_sel['artifact_name']}  guard: {artifact_sel['artifact_guard_passed']}")
    print(f"  raw_cand={conversion['raw_candidate_edges']} knn+={conversion['knn_candidates_added']} "
          f"primary={conversion['primary_links_selected']}(no_link={conversion['primary_no_link_selected']}) "
          f"div={conversion['divisions_selected']}/{divisions_total}")
    print(f"  final: {inv['n_node_rows']}/{inv['n_edge_rows']}  gaps={conversion['gap1_closed']}/{conversion['gap2_closed']}/{conversion['gap3_closed']} synthetic={synthetic}")
    print(f"  topology: in<={inv['max_in_degree']} out<={inv['max_out_degree']} multiframe={inv['direct_multiframe_edges']} dangling={inv['dangling_edges']}")
    print(f"  RECOMMENDATION: {recommendation}")
    log("submission.csv built and validated. Do not submit until user reviews (M30 experimental).")
    return {"submission_df": submission_df, "variant": variant_name, "valid": valid, "recommendation": recommendation, "report": report_obj}


def run_m30_variant_with_fallback(variant_name, working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    _RUN_LOG.clear()
    out_dir = Path(working_dir); out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / "submission.csv"
    sample_submission_path = Path(competition_dir) / "sample_submission.csv"
    try:
        result = run_m30_variant_pipeline(variant_name, working_dir, competition_dir)
        _write_log(out_dir)
        return {"status": "ok", **result}
    except Exception:
        tb = traceback.format_exc(); log("[error] M30 pipeline failed:\n" + tb)
        if submission_path.exists():
            (out_dir / "milestone30_failure_fallback_report.json").write_text(json.dumps({"variant": variant_name, "traceback": tb, "fallback": {"used": False, "reason": "submission already written"}}, indent=2))
            _write_log(out_dir); return {"status": "pipeline_failed_submission_exists"}
        try:
            fb = write_fallback_submission(sample_submission_path, submission_path)
            (out_dir / "milestone30_failure_fallback_report.json").write_text(json.dumps({"variant": variant_name, "traceback": tb, "fallback": fb}, indent=2))
            print("\n=== Variant summary ===")
            print(f"  variant: {variant_name}  final_source: fallback_sample_submission  fallback_used: True")
            print("  RECOMMENDATION: DO_NOT_SUBMIT_VALIDATION_FAILED")
            _write_log(out_dir); return {"status": "fallback_written", "fallback": fb}
        except Exception:
            fb_tb = traceback.format_exc()
            try:
                (out_dir / "milestone30_failure_fallback_report.json").write_text(json.dumps({"variant": variant_name, "traceback": tb, "fallback_error": fb_tb}, indent=2))
            except Exception:
                pass
            _write_log(out_dir); raise


# --------------------------------------------------------------------------- #
# 45. M30 v2 LOCAL CV HARNESS (non-submit). Real v2 CV: discover train GT GEFF,
# run the v2 engine on the raw candidate graph, attempt the official repo metric,
# and report honestly. NEVER fabricates scores; NEVER writes a submission.
# --------------------------------------------------------------------------- #
def _m30_locate_official_metric_v2():
    """Try the repo's official metric (tracking_cellmot.metrics or an
    evaluate.py). Returns (callable|None, where|None). Never fabricates."""
    for modname, fnname in [("tracking_cellmot.metrics", "score"), ("tracking_cellmot.metrics", "evaluate"),
                            ("tracking_cellmot.metric", "score"), ("scripts.evaluate", "evaluate"),
                            ("evaluate", "evaluate"), ("evaluate", "score")]:
        try:
            mod = __import__(modname, fromlist=[fnname])
            fn = getattr(mod, fnname, None)
            if callable(fn):
                return fn, f"{modname}.{fnname}"
        except Exception:
            continue
    return None, None


def _m30_find_train_gt(competition_dir):
    comp = Path(competition_dir); found = []
    for root in [comp / "train", comp]:
        if not root.exists():
            continue
        for p in sorted(root.glob("*")):
            if not p.is_dir():
                continue
            gr = _geff_root_of(p)
            if gr is None:
                for child in list(p.glob("**/*.geff")) + list(p.glob("**/*graph*")):
                    gr = _geff_root_of(child) or (child if (child / "nodes").exists() else None)
                    if gr:
                        break
            if gr is not None:
                found.append({"dataset": p.name, "gt_geff": str(gr)})
    return found


def run_m30_v2_local_cv(working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    """Real v2 CV harness (non-submit). Discovers train GT GEFF, runs the v2
    engine (balanced + sparse-kNN where applicable), and attempts the official
    metric. Reports edge_jaccard/division_jaccard/total ONLY if obtained from the
    official metric; otherwise CV_NOT_WIRED with the exact missing piece. Writes
    m30_v2_local_cv_report.json. Never writes a competition submission."""
    _RUN_LOG.clear()
    out_dir = Path(working_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report = {"kind": "v2_local_cv_harness", "submit": False, "smoke_runs": [], "official_metric": None,
              "metric_wired": False, "missing": [], "cv_status": "CV_NOT_WIRED"}
    try:
        metric_fn, where = _m30_locate_official_metric_v2()
        report["official_metric"] = where; report["metric_wired"] = metric_fn is not None
        if metric_fn is None:
            report["missing"].append("official metric not importable (tried tracking_cellmot.metrics.{score,evaluate}, "
                                     "tracking_cellmot.metric.score, scripts.evaluate.evaluate, evaluate.{evaluate,score})")
        trains = _m30_find_train_gt(competition_dir)
        report["train_datasets_found"] = trains
        if not trains:
            report["missing"].append("no train dataset with a GT GEFF found under the mounted competition dir; local CV "
                                     "needs train GT (a test-only mount cannot be scored locally)")
        for tr in trains[:2]:
            entry = {"dataset": tr["dataset"], "ran": False, "error": None}
            try:
                cg = read_candidate_geff(Path(tr["gt_geff"]))
                cand_nodes, cand_edges, cdiag = build_candidate_graph_v2(cg)
                prob_info = analyze_edge_prob(cand_edges, cg["has_edge_prob"])
                rich = candidate_richness_stats(cand_nodes, cand_edges)
                klass = classify_candidate_graph(prob_info, rich)
                bal = resolve_link_config("A", klass)
                nb, eb, db = run_relinker_v2(cand_nodes, cand_edges, bal, klass)
                entry.update({"ran": True, "candidate_class": klass, "balanced_nodes": db["final_nodes"], "balanced_edges": db["final_edges"]})
                if klass in _SPARSE:
                    spc = resolve_link_config("C", klass)
                    ns, es, ds = run_relinker_v2(cand_nodes, cand_edges, spc, klass)
                    entry["sparse_knn_nodes"] = ds["final_nodes"]; entry["sparse_knn_edges"] = ds["final_edges"]
                if metric_fn is not None:
                    entry["note"] = ("engine ran; wire GT->prediction into the official metric to compute "
                                     "edge_jaccard/division_jaccard/total (left to the official harness - not fabricated)")
                else:
                    entry["note"] = "engine ran on the train candidate graph; NO official metric available -> NO CV score computed"
            except Exception as exc:
                entry["error"] = repr(exc)
            report["smoke_runs"].append(entry)
        if metric_fn is not None and trains:
            report["cv_status"] = "METRIC_IMPORTABLE_SCORING_CALL_TODO"
            report["missing"].append("GT->prediction adapter for the official metric signature is required to emit real "
                                     "edge_jaccard/division_jaccard/total; no score is fabricated here")
        report["status"] = "ok"
    except Exception:
        report["status"] = "error"; report["traceback"] = traceback.format_exc()
    (out_dir / "m30_v2_local_cv_report.json").write_text(json.dumps(report, indent=2, default=str))
    _write_log(out_dir)
    print("=== M30 v2 LOCAL CV HARNESS (non-submit) ===")
    print(f"  official metric wired: {report['metric_wired']} ({report['official_metric']})")
    print(f"  train GT datasets: {len(report.get('train_datasets_found', []))}   cv_status: {report['cv_status']}")
    for m in report["missing"]:
        print(f"   - MISSING: {m}")
    print("  No submission was created (CV harness).")
    return report

# --------------------------------------------------------------------------- #
# 46. M30 tests + top-level drivers (diagnostic / variant / v2 CV harness)
# --------------------------------------------------------------------------- #
def _mk30n(rows):
    return pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x"])


def _mk30e(rows):
    return pd.DataFrame(rows, columns=["source_id", "target_id", "edge_prob", "edge_dist", "in_solution"])


def _m30_test_raw_candidate_preservation():
    cg = {"nodes": pd.DataFrame({"raw_id": [10, 11, 12], "t": [0, 1, 2], "z": [0.0, 0, 0], "y": [0.0, 1, 2], "x": [0.0, 0, 0]}),
          "edges": pd.DataFrame({"raw_source": [10, 11, 10], "raw_target": [11, 12, 12],
                                 "edge_prob": [0.9, 0.8, 0.7], "edge_dist": [0.41, 0.41, 0.82], "in_solution": [True, True, False]}),
          "has_edge_prob": True, "has_edge_dist": True, "n_edges_total": 3, "n_edges_solution": 2}
    cand_nodes, cand_edges, diag = build_candidate_graph_v2(cg)
    assert diag["n_unit_candidates"] == 2 and diag["n_non_unit_candidates"] == 1, diag
    assert set(cand_edges["edge_prob"]) == {0.9, 0.8}, "unit-candidate edge_prob must be preserved verbatim"
    prob = analyze_edge_prob(cand_edges, True)
    assert prob["edge_prob_available"] is True


def _m30_test_missing_constant_edge_prob_guard():
    nodes = _mk30n([(0, 0, 0.0, 0.0, 0.0), (1, 1, 0.0, 1.0, 0.0)])
    # missing edge_prob (NaN) -> unavailable.
    miss = pd.DataFrame({"source_id": [0], "target_id": [1], "edge_prob": [np.nan], "edge_dist": [0.41], "in_solution": [False]})
    p_missing = analyze_edge_prob(miss, has_edge_prob=False)
    assert p_missing["edge_prob_available"] is False and p_missing["edge_prob_non_degenerate"] is False
    assert classify_candidate_graph(p_missing, candidate_richness_stats(nodes, miss)) == "EDGE_PROB_UNAVAILABLE"
    # constant edge_prob (all 0.5) -> degenerate.
    const = pd.DataFrame({"source_id": [0, 0], "target_id": [1, 1], "edge_prob": [0.5, 0.5], "edge_dist": [0.4, 0.4], "in_solution": [False, False]})
    p_const = analyze_edge_prob(const, has_edge_prob=True)
    assert p_const["edge_prob_available"] is True and p_const["edge_prob_non_degenerate"] is False, p_const


def _m30_test_no_link_and_good_link():
    # A: one POOR candidate (fused > null); B: one GOOD candidate (fused < null).
    nodes = _mk30n([(0, 0, 0.0, 0.0, 0.0), (2, 0, 0.0, 50.0, 0.0), (1, 1, 0.0, 1.0, 0.0), (3, 1, 0.0, 51.0, 0.0)])
    edges = pd.DataFrame({"source_id": [0, 2], "target_id": [1, 3], "edge_prob": [0.06, 0.95],
                          "edge_dist": [7.0, 0.5], "in_solution": [False, False]})
    cfg = _linkcfg(null_link_cost=3.5)
    pos = nodes.set_index("node_id")[["z", "y", "x"]]
    sel, n_no_link, used = _primary_linking(nodes, edges, {}, pos, cfg)
    linked = {(m["source_id"], m["target_id"]) for m in sel}
    assert (2, 3) in linked, "good real edge must beat the no-link dummy"
    assert (0, 1) not in linked and n_no_link >= 1, "poor real edge must lose to the no-link dummy"
    # primary keeps in<=1/out<=1.
    if sel:
        se = pd.DataFrame([(m["source_id"], m["target_id"]) for m in sel], columns=["s", "t"])
        assert int(se.groupby("t").size().max()) <= 1 and int(se.groupby("s").size().max()) <= 1


def _m30_test_division_independent_no_steal():
    # P1 and P2 each have a primary child; both want the SAME orphan O. Only one
    # may take it; the loser must NOT steal it (gets no-division).
    nodes = _mk30n([(0, 0, 0.0, 0.0, 0.0), (1, 1, 0.0, 0.5, 0.0),      # P1=0 -> child C1=1
                    (2, 0, 0.0, 0.2, 0.0), (3, 1, 0.0, 0.7, 0.0),      # P2=2 -> child C2=3
                    (9, 1, 0.0, 0.6, 0.0)])                            # orphan O=9 at t1
    edges = pd.DataFrame({"source_id": [0, 2], "target_id": [9, 9], "edge_prob": [0.9, 0.4],
                          "edge_dist": [0.2, 0.2], "in_solution": [False, False]})
    primary = [{"source_id": 0, "target_id": 1, "edge_prob": 0.9, "edge_dist": 0.2, "fused": 0.1, "is_knn": False},
               {"source_id": 2, "target_id": 3, "edge_prob": 0.9, "edge_dist": 0.2, "fused": 0.1, "is_knn": False}]
    pos = nodes.set_index("node_id")[["z", "y", "x"]]
    cfg = _linkcfg()
    divs, d = _division_linking(nodes, edges, primary, used_targets=set(), pos=pos, cfg=cfg)
    assert d["divisions_selected"] == 1, f"exactly one parent may adopt the single orphan, got {d}"
    tgts = [e["target_id"] for e in divs]
    assert tgts == [9] and len(set(tgts)) == 1, "orphan must not be double-assigned (no stealing)"


def _m30_test_division_gates_reject():
    # second child fails the sister-distance gate -> not even a candidate.
    nodes = _mk30n([(0, 0, 0.0, 0.0, 0.0), (1, 1, 0.0, 0.5, 0.0), (9, 1, 0.0, 40.0, 0.0)])
    edges = pd.DataFrame({"source_id": [0], "target_id": [9], "edge_prob": [0.9], "edge_dist": [0.3], "in_solution": [False]})
    primary = [{"source_id": 0, "target_id": 1, "edge_prob": 0.9, "edge_dist": 0.2, "fused": 0.1, "is_knn": False}]
    pos = nodes.set_index("node_id")[["z", "y", "x"]]
    divs, d = _division_linking(nodes, edges, primary, set(), pos, _linkcfg())
    assert d["division_candidates"] == 0 and d["divisions_selected"] == 0, d
    # low-prob second child rejected by div_min_prob.
    edges2 = pd.DataFrame({"source_id": [0], "target_id": [9], "edge_prob": [0.10], "edge_dist": [0.3], "in_solution": [False]})
    nodes2 = _mk30n([(0, 0, 0.0, 0.0, 0.0), (1, 1, 0.0, 0.5, 0.0), (9, 1, 0.0, 0.6, 0.0)])
    _divs, d2 = _division_linking(nodes2, edges2, primary, set(), nodes2.set_index("node_id")[["z", "y", "x"]], _linkcfg(div_min_prob=0.20))
    assert d2["divisions_selected"] == 0, "second child below div_min_prob must be rejected"


def _m30_test_knn_topk_no_dup():
    # source 0 at t0; targets 1,2,3,4 at t1 at increasing distance; raw candidate
    # 0->1 exists. kNN (k=3, radius covers 1,2,3 but not 4) must add 2,3 only (not
    # duplicate 1, not 4 beyond radius).
    nodes = _mk30n([(0, 0, 0.0, 0.0, 0.0), (1, 1, 0.0, 1.0, 0.0), (2, 1, 0.0, 2.0, 0.0),
                    (3, 1, 0.0, 3.0, 0.0), (4, 1, 0.0, 30.0, 0.0)])
    raw = _mk30e([(0, 1, 0.9, 0.41, False)])
    cfg = _linkcfg(knn_enable=True, knn_k=3, knn_max_um=1.5, knn_default_prob=0.20)   # 1.5um ~ 3.7 voxels
    aug, n_added = knn_augment_candidates(nodes, raw, cfg)
    added = aug[aug["is_knn"]]
    pairs = set(zip(added["source_id"], added["target_id"]))
    assert (0, 1) not in pairs, "kNN must not duplicate an existing raw candidate"
    assert (0, 4) not in pairs, "kNN must not add targets beyond knn_max_um"
    assert all(p == 0.20 for p in added["edge_prob"]), "kNN prob must be the conservative default"
    assert len(added) >= 1 and len(added) <= 3, f"kNN adds at most top-k, got {len(added)}"


def _m30_test_gaps_unit_only():
    nodes = _mk30n([(0, 0, 0.0, 0.0, 0.0), (1, 2, 0.0, 1.0, 0.0)])
    n2, e2, d, _ = stitch_gaps_m30(nodes, pd.DataFrame(columns=["source_id", "target_id"]), _linkcfg(gap_sizes=(1, 2)))
    assert d["gap1_closed"] == 1, d
    t_of = dict(zip(n2["node_id"], n2["t"]))
    for r in e2.itertuples():
        assert t_of[int(r.target_id)] - t_of[int(r.source_id)] == 1, "gap stitch produced a non-unit edge"


def _m30_test_end_to_end():
    rows, edges = [], []
    for tr in range(3):
        base = tr * 100
        for t in range(9):
            rows.append((base + t, t, 0.0, float(base + t), 0.0))
        edges += [(base + t, base + t + 1, 0.9, 0.41, False) for t in range(8)]
    cand_nodes = _mk30n(rows)
    cand_edges = _mk30e(edges)
    fn, fe, d = run_relinker_v2(cand_nodes, cand_edges, _linkcfg(), "FULL_CANDIDATE_GRAPH")
    fn, fe, _ = relabel_positive(fn, fe, 0)
    all_rows, nid = [], 0
    for r in fn.itertuples():
        all_rows.append({"id": nid, "dataset": "ds1", "row_type": "node", "node_id": int(r.node_id),
                         "t": int(r.t), "z": float(r.z), "y": float(r.y), "x": float(r.x), "source_id": -1, "target_id": -1}); nid += 1
    for r in fe.itertuples():
        all_rows.append({"id": nid, "dataset": "ds1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                         "source_id": int(r.source_id), "target_id": int(r.target_id)}); nid += 1
    df = pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)
    valid, rep, stats = validate_reference_submission(df, ["ds1"])
    assert valid, f"end-to-end v2 submission must be valid: {rep}"
    inv = _m30_invariants(df, ["ds1"])
    assert inv["direct_multiframe_edges"] == 0 and inv["dangling_edges"] == 0
    assert inv["max_in_degree"] <= 1 and inv["max_out_degree"] <= 2


def _m30_test_classification_and_mode_recommendation():
    full_prob = {"edge_prob_available": True, "edge_prob_non_degenerate": True}
    assert classify_candidate_graph(full_prob, {"candidates_per_source_mean": 1.4, "pct_sources_ge2_candidates": 0.25}) == "FULL_CANDIDATE_GRAPH"
    assert classify_candidate_graph(full_prob, {"candidates_per_source_mean": 1.05, "pct_sources_ge2_candidates": 0.02}) == "ILP_SOLUTION_LIKE"
    assert classify_candidate_graph(full_prob, {"candidates_per_source_mean": 1.18, "pct_sources_ge2_candidates": 0.10}) == "MODERATELY_SPARSE"
    assert classify_candidate_graph({"edge_prob_available": False, "edge_prob_non_degenerate": False}, {"candidates_per_source_mean": 2.0, "pct_sources_ge2_candidates": 0.5}) == "EDGE_PROB_UNAVAILABLE"
    ok_inv = {"n_node_rows": 140000, "n_edge_rows": 130000, "max_in_degree": 1, "max_out_degree": 2,
              "direct_multiframe_edges": 0, "dangling_edges": 0, "id_consecutive": True, "no_nan": True,
              "missing_datasets": [], "emptied_datasets": []}
    A = M30_VARIANTS["A"]; C = M30_VARIANTS["C"]
    # A on FULL (raw) -> OK; A on ILP_SOLUTION_LIKE -> wrong mode.
    assert _m30_recommendation("A", A, True, "FULL_CANDIDATE_GRAPH", True, True, False, True, False, ok_inv, 3000, 2000) == "OK_TO_SUBMIT_EXPERIMENTAL"
    assert _m30_recommendation("A", A, True, "ILP_SOLUTION_LIKE", True, True, False, True, False, ok_inv, 3000, 2000) == "DO_NOT_SUBMIT_WRONG_CANDIDATE_MODE"
    # C on ILP_SOLUTION_LIKE with knn enabled -> OK.
    assert _m30_recommendation("C", C, True, "ILP_SOLUTION_LIKE", True, True, True, True, False, ok_inv, 3000, 2000) == "OK_TO_SUBMIT_EXPERIMENTAL"
    # edge_prob unavailable dominates.
    assert _m30_recommendation("A", A, True, "EDGE_PROB_UNAVAILABLE", False, False, False, True, False, ok_inv, 3000, 2000) == "DO_NOT_SUBMIT_EDGE_PROB_UNAVAILABLE"
    # sparse graph with knn OFF -> too sparse.
    assert _m30_recommendation("D", M30_VARIANTS["D"], True, "MODERATELY_SPARSE", True, True, False, True, False, ok_inv, 3000, 2000) == "DO_NOT_SUBMIT_CANDIDATE_GRAPH_TOO_SPARSE"
    # explosion guards.
    assert _m30_recommendation("E", M30_VARIANTS["E"], True, "FULL_CANDIDATE_GRAPH", True, True, False, True, False, ok_inv, 3000, 5000) == "DO_NOT_SUBMIT_DIVISION_EXPLOSION"
    assert _m30_recommendation("D", M30_VARIANTS["D"], True, "FULL_CANDIDATE_GRAPH", True, True, False, True, False, {**ok_inv, "n_node_rows": 170000}, 3000, 2000) == "DO_NOT_SUBMIT_NODE_EXPLOSION"


def _m30_test_resolve_link_config():
    assert resolve_link_config("A", "FULL_CANDIDATE_GRAPH").knn_enable is False
    assert resolve_link_config("C", "ILP_SOLUTION_LIKE").knn_enable is True
    assert resolve_link_config("D", "FULL_CANDIDATE_GRAPH").knn_enable is False   # auto: raw when full
    assert resolve_link_config("D", "MODERATELY_SPARSE").knn_enable is True       # auto: knn when sparse
    for v, det in [("A", "0.99"), ("B", "0.99"), ("C", "0.99"), ("D", "0.95"), ("E", "0.99")]:
        cmd, notes = build_m30_command(v, {"weight_splits": [0]})
        assert cmd[cmd.index("--det-threshold") + 1] == det and cmd[cmd.index("--ilp-edge-weight") + 1] == "-1.0"


def run_milestone30_tests() -> None:
    """v2 global-relinker math on synthetic fixtures (no Kaggle/GPU): raw
    candidate preservation with edge_prob, missing/constant edge_prob guard,
    no-link dummy beats a poor edge / good edge beats the dummy, primary
    in<=1/out<=1, independent division assignment with no target stealing,
    division sister/probability gates, kNN top-k-within-radius no-duplicate,
    unit-timepoint gaps, end-to-end valid submission, candidate classification +
    mode recommendation + explosion guards, and the auto candidate-mode resolution."""
    _m30_test_raw_candidate_preservation()
    _m30_test_missing_constant_edge_prob_guard()
    _m30_test_no_link_and_good_link()
    _m30_test_division_independent_no_steal()
    _m30_test_division_gates_reject()
    _m30_test_knn_topk_no_dup()
    _m30_test_gaps_unit_only()
    _m30_test_end_to_end()
    _m30_test_classification_and_mode_recommendation()
    _m30_test_resolve_link_config()
    print("All milestone30_metric_aware_global_relinker_v2_runner tests passed (10/10).")


DEFAULT_M30_VARIANT = "A"


def run_milestone30_variant(variant_name=DEFAULT_M30_VARIANT, working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    print("=== Self-test: M30 metric-aware global relinker v2 (candidates, two-stage, guards) ===")
    run_milestone30_tests()
    if variant_name not in M30_VARIANTS:
        raise ValueError(f"unknown variant {variant_name!r}, expected one of {sorted(M30_VARIANTS)}")
    if not is_kaggle_env():
        print(f"[dry-run] /kaggle/input absent - self-tests only (variant {variant_name}). On Kaggle this runs the reference "
              "predict, profiles the raw candidate graph (Stage-0), gates the candidate mode, re-solves linking, and writes a guarded submission.")
        return {"status": "dry_run", "variant": variant_name}
    return run_m30_variant_with_fallback(variant_name, working_dir, competition_dir)


def run_milestone30_diagnostic(working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    print("=== Self-test: M30 v2 (diagnostic shares the candidate engine) ===")
    run_milestone30_tests()
    if not is_kaggle_env():
        print("[dry-run] /kaggle/input absent - self-tests only. On Kaggle this profiles the raw candidate graph and classifies it. Non-submit.")
        return {"status": "dry_run", "kind": "candidate_graph_diagnostic"}
    return run_m30_candidate_diagnostic(working_dir, competition_dir)


def run_milestone30_local_cv(working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    print("=== Self-test: M30 v2 (CV harness shares the candidate engine) ===")
    run_milestone30_tests()
    if not is_kaggle_env():
        print("[dry-run] /kaggle/input absent - self-tests only. On Kaggle this discovers train GT, runs v2, and honestly reports metric wiring. Never a submission.")
        return {"status": "dry_run", "kind": "v2_local_cv_harness"}
    return run_m30_v2_local_cv(working_dir, competition_dir)


if __name__ == "__main__":
    run_milestone30_variant(DEFAULT_M30_VARIANT)
