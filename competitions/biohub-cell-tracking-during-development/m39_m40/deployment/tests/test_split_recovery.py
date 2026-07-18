"""Unit tests for split_recovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m39_m40 import PENDING_REAL_ENV
from m39_m40.deployment.split_recovery import (
    UPSTREAM_SPLITS_FILENAME,
    _parse_split_json,
    recover,
)


def _mk(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class ParseTests(unittest.TestCase):
    def test_dict_train_test(self):
        t, v = _parse_split_json({"train": ["a", "b"], "test": ["c"]})
        self.assertEqual(t, ["a", "b"])
        self.assertEqual(v, ["c"])

    def test_dict_train_validation(self):
        t, v = _parse_split_json({"train": ["a"], "validation": ["b"]})
        self.assertEqual(t, ["a"])
        self.assertEqual(v, ["b"])

    def test_list_wraps_first_split(self):
        t, v = _parse_split_json([{"train": ["a"], "test": ["b"]},
                                  {"train": ["c"], "test": ["d"]}])
        self.assertEqual((t, v), (["a"], ["b"]))

    def test_split_0_key(self):
        t, v = _parse_split_json({"split_0": {"train": ["a"], "test": ["b"]}})
        self.assertEqual((t, v), (["a"], ["b"]))

    def test_splits_list(self):
        t, v = _parse_split_json({"splits": [{"train": ["a"], "test": ["b"]}]})
        self.assertEqual((t, v), (["a"], ["b"]))


class RecoverTests(unittest.TestCase):
    def test_refused_when_no_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            r = recover(reference_bundle_root=Path(td))
            self.assertEqual(r.verdict, "REFUSED")
            self.assertEqual(r.split_source, PENDING_REAL_ENV)
            self.assertTrue(any("reconstruction" in x for x in r.reasons))

    def test_ok_observed_from_upstream_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk(root / "some" / "nested" / UPSTREAM_SPLITS_FILENAME,
                json.dumps({"train": ["a", "b", "c"], "test": ["d"]}))
            r = recover(reference_bundle_root=root)
            self.assertEqual(r.verdict, "OK_OBSERVED")
            self.assertEqual(r.split_source, "dataset_splits.json")
            self.assertEqual(r.declared_split_name, "split_0")
            self.assertEqual(r.train_datasets, ["a", "b", "c"])
            self.assertEqual(r.validation_datasets, ["d"])

    def test_refused_leakage_when_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk(root / UPSTREAM_SPLITS_FILENAME,
                json.dumps({"train": ["a", "b"], "test": ["b"]}))
            r = recover(reference_bundle_root=root)
            self.assertEqual(r.verdict, "REFUSED_LEAKAGE")
            self.assertEqual(r.overlap, ["b"])

    def test_explicit_split_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk(root / "reference" / "splits" / "split_0.json",
                json.dumps({"train": ["x"], "test": ["y"]}))
            r = recover(reference_bundle_root=root)
            self.assertEqual(r.verdict, "OK_OBSERVED")
            self.assertEqual(r.split_source, "explicit_split_file")

    def test_kaggle_test_splits_50ep(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk(root / "reference" / "kaggle_test_splits_50ep.json",
                json.dumps([{"train": ["m"], "test": ["n"]}]))
            r = recover(reference_bundle_root=root)
            self.assertEqual(r.verdict, "OK_OBSERVED")
            self.assertEqual(r.train_datasets, ["m"])
            self.assertEqual(r.validation_datasets, ["n"])

    def test_first_hit_wins_upstream_over_explicit(self):
        # If both exist, the upstream file takes precedence (cascade order).
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk(root / UPSTREAM_SPLITS_FILENAME,
                json.dumps({"train": ["u"], "test": ["v"]}))
            _mk(root / "reference" / "splits" / "split_0.json",
                json.dumps({"train": ["x"], "test": ["y"]}))
            r = recover(reference_bundle_root=root)
            self.assertEqual(r.split_source, "dataset_splits.json")
            self.assertEqual(r.train_datasets, ["u"])

    def test_missing_checkpoint_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake_ckpt = root / "does_not_exist.pth"
            r = recover(reference_bundle_root=root, checkpoint_paths=(fake_ckpt,))
            self.assertEqual(r.verdict, "REFUSED")
            self.assertTrue(any("checkpoint_missing" in x for x in r.reasons))


if __name__ == "__main__":
    unittest.main()
