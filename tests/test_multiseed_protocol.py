"""Tests for the Phase 20 multi-seed protocol enumeration and manifest.

These import the pure-Python combination/naming helpers from
scripts/run_transformer_multiseed.py directly (no subprocess, no
training, no network) to verify the planned 7 x 5 x 3 = 105 run matrix
is exactly right.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


multiseed_sweep = _load_module(
    "run_transformer_multiseed", REPO_ROOT / "scripts" / "run_transformer_multiseed.py"
)


class TestCombinationEnumeration(unittest.TestCase):
    def setUp(self):
        self.models = [
            "indobert_15g", "indobert_base_p1", "indobertweet",
            "indoroberta_15g", "mbert", "nusabert", "xlmr",
        ]
        self.folds = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]
        self.seeds = [42, 43, 44]

    def test_produces_exactly_105_combinations(self):
        combos = multiseed_sweep.all_combinations(self.models, self.folds, self.seeds)
        self.assertEqual(len(combos), 105)

    def test_combinations_span_exactly_7_models_5_folds_3_seeds(self):
        combos = multiseed_sweep.all_combinations(self.models, self.folds, self.seeds)
        models_seen = {m for m, _, _ in combos}
        folds_seen = {f for _, f, _ in combos}
        seeds_seen = {s for _, _, s in combos}
        self.assertEqual(models_seen, set(self.models))
        self.assertEqual(folds_seen, set(self.folds))
        self.assertEqual(seeds_seen, set(self.seeds))

    def test_no_duplicate_combinations(self):
        combos = multiseed_sweep.all_combinations(self.models, self.folds, self.seeds)
        self.assertEqual(len(combos), len(set(combos)))

    def test_run_name_is_unambiguous_per_fold_and_seed(self):
        name_a = multiseed_sweep.run_name("fold_2", 43)
        name_b = multiseed_sweep.run_name("fold_2", 44)
        name_c = multiseed_sweep.run_name("fold_3", 43)
        self.assertEqual(len({name_a, name_b, name_c}), 3)
        self.assertEqual(name_a, "fold_2__seed_43")


class TestManifestValidity(unittest.TestCase):
    def test_generated_manifest_reports_105_planned_runs(self):
        manifest_path = REPO_ROOT / "results" / "multiseed" / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("manifest not generated in this environment")
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertEqual(len(manifest["models"]), 7)
        self.assertEqual(len(manifest["folds"]), 5)
        self.assertEqual(len(manifest["seeds"]), 3)
        self.assertEqual(manifest["total_expected_runs"], 105)

    def test_manifest_schema_has_required_fields(self):
        manifest_path = REPO_ROOT / "results" / "multiseed" / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("manifest not generated in this environment")
        with open(manifest_path) as f:
            manifest = json.load(f)
        for field in (
            "experiment_id", "protocol_version", "models", "folds", "seeds",
            "total_expected_runs", "dataset_csv_sha256", "folds_json_sha256",
        ):
            self.assertIn(field, manifest)


class TestOldArtifactsNotSelected(unittest.TestCase):
    """The multi-seed run-state must never pick up Phase 13/15 single-seed
    artifacts as if they were multi-seed results — they live in entirely
    separate directories."""

    def test_is_complete_false_when_only_phase15_style_files_exist(self):
        with TemporaryDirectory() as tmp:
            import os

            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                # Simulate a Phase 13/15-style single-seed artifact
                # (results/metrics/<model>/<fold>.json) with NO multiseed
                # namespace present at all.
                os.makedirs("results/metrics/indobert_base_p1", exist_ok=True)
                with open("results/metrics/indobert_base_p1/fold_0.json", "w") as f:
                    json.dump({"macro_f1": 0.59}, f)
                os.makedirs("results/logs/indobert_base_p1", exist_ok=True)
                with open("results/logs/indobert_base_p1/fold_0.json", "w") as f:
                    json.dump({"status": "completed"}, f)

                self.assertFalse(
                    multiseed_sweep.is_complete("indobert_base_p1", "fold_0", 42)
                )
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
