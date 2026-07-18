# M39/M40 offline bundle — Kaggle Dataset upload guide

This is the operator handbook for building the M39/M40 offline bundle and
uploading it as a Kaggle Dataset so the single-cell runner
(`M39_M40_DEPLOYMENT_KAGGLE_ONE_CELL.txt`) can run with **Internet Off**.

The single-cell runner never auto-submits to Kaggle. It writes
`/kaggle/working/submission.csv` only if all G1–G11 gates pass. Otherwise
it emits the byte-preserved **M19-C** fallback (public LB 0.880).

---

## Slot inventory the bundle MUST provide

| slot                | status in this repo | source                                                                                             |
|---------------------|---------------------|----------------------------------------------------------------------------------------------------|
| `vendor_metric`     | PRESENT             | `vendor/royerlab_cellmot/` — pinned to `royerlab/kaggle-cell-tracking-competition@7396b7e9`         |
| `m39_m40_package`   | PRESENT             | `m39_m40/` (scaffolding + `deployment/`)                                                            |
| `m19c_archive`      | PRESENT             | `archive/m19c/` (byte-preserved fallback + MANIFEST.sha256)                                         |
| `hoct_source`       | **NOT PRESENT**     | you must clone the official MIT-licensed HOCT repo and pin a commit                                 |
| `hoct_weights`      | **NOT PRESENT**     | the `general_v0` checkpoint that ships with HOCT                                                    |

Both HOCT slots are REQUIRED for the experimental path. Without them the
runner correctly refuses to proceed and falls back to M19-C.

---

## Step 1 — Vendor the HOCT source on a network-enabled machine

On a machine that has Internet access AND `git`:

```
git clone https://github.com/<OFFICIAL_HOCT_REPO>/hoct.git /tmp/hoct
cd /tmp/hoct
git checkout <pinned_commit_or_release_tag>

# Confirm the license is MIT and record it explicitly:
grep -i 'MIT' LICENSE  # must exist
```

Write a `PROVENANCE.json` next to the vendored package so the builder can
record where it came from:

```
cat > /tmp/hoct/PROVENANCE.json <<'JSON'
{
  "upstream_repository": "https://github.com/<OFFICIAL_HOCT_REPO>/hoct",
  "upstream_commit": "<pinned_commit_sha>",
  "upstream_release_tag": "<pinned_tag_or_null>",
  "license": "MIT",
  "license_path": "LICENSE",
  "vendored_at": "<YYYY-MM-DD>",
  "purpose": "Pinned OFFLINE snapshot of the official HOCT source so Kaggle can run with Internet OFF."
}
JSON
```

The builder records this file verbatim into
`BUNDLE_MANIFEST.json → provenance.hoct_provenance`.

---

## Step 2 — Vendor the `general_v0` checkpoint

Download the official `general_v0` weight file(s) into
`/tmp/hoct_weights/`. The builder copies the directory verbatim under
`weights/hoct/general_v0/` inside the bundle and records every file's
SHA256 in `BUNDLE_MANIFEST.json`.

---

## Step 3 — Build the offline bundle

From the repo root:

```
python -m m39_m40.deployment.bundle_builder \
    --comp-dir  competitions/biohub-cell-tracking-during-development \
    --bundle-root /tmp/m39_m40_bundle \
    --hoct-source  /tmp/hoct \
    --hoct-weights /tmp/hoct_weights \
    --marker-note "m39_m40 bundle for kaggle deployment"
```

Expected verdict on success:

```
"verdict": "COMPLETE"
```

If the builder reports `REFUSED_HOCT_MISSING`, one of the HOCT slots was
empty. Fix the arguments and re-run — the builder is deterministic.

---

## Step 4 — Verify the built bundle

```
python -c "
from pathlib import Path
from m39_m40.deployment.bundle_manifest import load_manifest, verify_manifest, BUNDLE_MANIFEST_FILENAME
root = Path('/tmp/m39_m40_bundle')
m = load_manifest(root, 'BUNDLE_MANIFEST.json')
print(verify_manifest(root, m)['verdict'])
"
```

Must print `VERIFIED`. Any other value means the operator ran a step out
of order (usually: forgot to re-run the builder after fixing a file). Do
not upload a bundle that is not `VERIFIED`.

---

## Step 5 — Upload as a Kaggle Dataset

Recommended CLI (assumes `~/.kaggle/kaggle.json` is configured):

```
cd /tmp/m39_m40_bundle
kaggle datasets init -p .
# edit dataset-metadata.json:
#   "id": "<your-username>/m39-m40-bundle"
#   "title": "M39/M40 offline bundle (tracking_cellmot + HOCT + M19-C archive)"
#   "licenses": [{"name": "other"}]      # bundle contains multi-source
kaggle datasets create -p . -r zip
```

Or via the web UI: “Create Dataset” → upload the whole
`/tmp/m39_m40_bundle/` directory. Set it to Private (or Public if you
have permission). Note the slug.

Attach the resulting dataset to a fresh Kaggle notebook alongside the
competition data and the 0902 reference bundle, then paste
`M39_M40_DEPLOYMENT_KAGGLE_ONE_CELL.txt` into a single cell.

---

## What the runner does not do

- It does not call `kaggle competitions submit` — you must click the
  Submit button in the Kaggle UI after the notebook run.
- It does not fabricate any score, runtime, or hash.
- It does not deterministically reconstruct `split_0` — it only accepts
  an OBSERVED split from the mounted reference bundle (dataset_splits.json
  or an explicit split file) or from checkpoint metadata.
- It does not overwrite `submission.csv` with the experimental output if
  the HOCT score is not ≥ the baseline on the observed CV — in that case
  M19-C's byte-preserved one-cell writes the submission instead.

---

## What to expect if HOCT is intentionally absent

The runner:

1. Detects `INCOMPLETE_HOCT` at G1 (bundle manifest).
2. Skips G2..G9.
3. Emits the M19-C fallback and writes the proven-0.880 submission.

This is the DESIGNED behaviour. It is safe to submit that fallback.
