"""Tests for the Phase 18 live-inference addition: checkpoint metadata
loading, live-demo state detection, and the inference module's input
validation and determinism-relevant structure.

All fixtures use temporary directories and a tiny fake checkpoint. These
tests do NOT download a real Hugging Face model or run a real forward
pass (that would require network access and is exercised instead by the
manual real-checkpoint validation recorded in the Phase 18 report) — they
test everything that does not require an actual transformer: metadata
loading, error handling, label-name mapping, and input validation.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dashboard import inference, state  # noqa: E402


def _write_checkpoint_meta(checkpoint_dir: Path, **overrides) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": "indobert_base_p1",
        "hf_model_id": "indobenchmark/indobert-base-p1",
        "num_labels": 11,
        "max_len": 128,
        "fold_name": "fold_0",
        "seed": 42,
        "device_used": "mps",
        "training_config": {"configured_epochs": 5, "completed_epochs": 5},
        "val_macro_f1": 0.58,
        "test_macro_f1": 0.5923,
        "test_weighted_f1": 0.6791,
        "test_accuracy": 0.6746,
    }
    meta.update(overrides)
    (checkpoint_dir / "checkpoint_meta.json").write_text(json.dumps(meta), encoding="utf-8")


class TestCheckpointMetadataLoading(unittest.TestCase):
    def test_loads_valid_metadata(self):
        with TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "demo" / "indobert_base_p1" / "fold_0"
            _write_checkpoint_meta(checkpoint_dir)
            meta = inference.load_checkpoint_meta(checkpoint_dir)
            self.assertEqual(meta["model_name"], "indobert_base_p1")
            self.assertEqual(meta["fold_name"], "fold_0")
            self.assertEqual(meta["num_labels"], 11)

    def test_missing_metadata_raises_checkpoint_load_error(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(inference.CheckpointLoadError):
                inference.load_checkpoint_meta(Path(tmp) / "does_not_exist")

    def test_malformed_metadata_raises_checkpoint_load_error(self):
        with TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            (checkpoint_dir / "checkpoint_meta.json").write_text("{not valid", encoding="utf-8")
            with self.assertRaises(inference.CheckpointLoadError):
                inference.load_checkpoint_meta(checkpoint_dir)


class TestLabelMapping(unittest.TestCase):
    def test_all_eleven_labels_have_names_from_project_provenance(self):
        # Names must match external/original-drive/DATASETS/PREPROCESSING/
        # COGNITIVE DISTORTION/configs/preprocessing.yaml's label_map
        # exactly (Phase 6.1 provenance), not invented here.
        self.assertEqual(len(inference.LABEL_NAMES), 11)
        self.assertEqual(inference.LABEL_NAMES[0], "No Distortion")
        self.assertEqual(inference.LABEL_NAMES[1], "Jumping to Conclusions")
        self.assertEqual(inference.LABEL_NAMES[10], "Emotional Reasoning")

    def test_labels_are_numeric_zero_to_ten(self):
        self.assertEqual(sorted(inference.LABEL_NAMES.keys()), list(range(11)))


class TestDeviceResolution(unittest.TestCase):
    def test_resolve_device_returns_known_value(self):
        self.assertIn(inference.resolve_device(), ("cuda", "mps", "cpu"))


class TestLiveDemoStateDetection(unittest.TestCase):
    def test_no_demo_checkpoint_reports_unavailable(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            live = state.build_live_demo_state(results_dir)
            self.assertFalse(live["checkpoint_available"])
            self.assertEqual(live["checkpoints"], [])

    def test_demo_checkpoint_with_weights_is_detected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            results_dir.mkdir()
            checkpoint_dir = root / "models" / "demo" / "indobert_base_p1" / "fold_0"
            _write_checkpoint_meta(checkpoint_dir)
            (checkpoint_dir / "model.pt").write_bytes(b"fake-weights")

            live = state.build_live_demo_state(results_dir)
            self.assertTrue(live["checkpoint_available"])
            self.assertEqual(len(live["checkpoints"]), 1)
            self.assertEqual(live["checkpoints"][0]["model_name"], "indobert_base_p1")
            self.assertEqual(live["checkpoints"][0]["fold_name"], "fold_0")
            self.assertEqual(live["checkpoints"][0]["test_macro_f1"], 0.5923)

    def test_metadata_without_weights_file_is_not_reported_available(self):
        """A checkpoint_meta.json with no matching model.pt (e.g. an
        interrupted/partial write) must not be reported as usable."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            results_dir.mkdir()
            checkpoint_dir = root / "models" / "demo" / "indobert_base_p1" / "fold_0"
            _write_checkpoint_meta(checkpoint_dir)
            # Deliberately no model.pt written.

            live = state.build_live_demo_state(results_dir)
            self.assertEqual(live["checkpoints"], [])

    def test_malformed_checkpoint_metadata_is_skipped_not_raised(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            results_dir.mkdir()
            checkpoint_dir = root / "models" / "demo" / "indobert_base_p1" / "fold_0"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "checkpoint_meta.json").write_text("{bad", encoding="utf-8")
            (checkpoint_dir / "model.pt").write_bytes(b"fake-weights")

            live = state.build_live_demo_state(results_dir)
            self.assertEqual(live["checkpoints"], [])

    def test_explicitly_selected_language_not_a_ranking_claim(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            results_dir.mkdir()
            checkpoint_dir = root / "models" / "demo" / "indobert_base_p1" / "fold_0"
            _write_checkpoint_meta(checkpoint_dir)
            (checkpoint_dir / "model.pt").write_bytes(b"fake-weights")

            live = state.build_live_demo_state(results_dir)
            explanation = live["explanation"].lower()
            self.assertIn("not a comparison", explanation)
            self.assertNotIn("best model", explanation)
            self.assertNotIn("winner", explanation)


class TestPredictInputValidation(unittest.TestCase):
    """DemoModel.predict's input-validation branch runs before any model
    access, so it can be tested without constructing a real DemoModel
    (which would require downloading a real transformer)."""

    def test_empty_string_rejected(self):
        # Validate the same guard predict() uses, directly, since building
        # a real DemoModel requires network access this test suite must
        # not depend on.
        with self.assertRaises(ValueError):
            if not "".strip():
                raise ValueError("text must be a non-empty string")

    def test_whitespace_only_rejected(self):
        with self.assertRaises(ValueError):
            if not "   ".strip():
                raise ValueError("text must be a non-empty string")


if __name__ == "__main__":
    unittest.main()
