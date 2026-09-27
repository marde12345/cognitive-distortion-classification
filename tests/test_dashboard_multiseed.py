"""Tests for the Phase 20 dashboard multi-seed state (src/dashboard/multiseed_loader.py
and its wiring into src/dashboard/state.py's build_dashboard_state).

All fixtures use temporary directories and synthetic JSON — no real
transformer checkpoints, no GPU/MPS training, no network access.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dashboard import multiseed_loader, state  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestNoResultsYet(unittest.TestCase):
    def test_reports_not_run_when_nothing_exists(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            ms = multiseed_loader.build_multiseed_state(results_dir)
            self.assertFalse(ms["manifest_available"])
            self.assertEqual(ms["overall_status"], "not_run")
            self.assertEqual(ms["completed_runs"], 0)
            self.assertEqual(ms["per_run"], [])

    def test_manifest_present_but_no_runs_still_not_run(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            _write_json(results_dir / "multiseed" / "manifest.json", {
                "experiment_id": "phase20_multiseed_transformer_sweep",
                "protocol_version": "1.0",
                "models": ["indobert_base_p1"],
                "folds": ["fold_0"],
                "seeds": [42, 43, 44],
                "total_expected_runs": 3,
            })
            ms = multiseed_loader.build_multiseed_state(results_dir)
            self.assertTrue(ms["manifest_available"])
            self.assertEqual(ms["overall_status"], "not_run")
            self.assertEqual(ms["total_expected_runs"], 3)


class TestPartialResults(unittest.TestCase):
    def test_partial_results_load_and_count_correctly(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            _write_json(results_dir / "multiseed" / "manifest.json", {
                "models": ["indobert_base_p1"], "folds": ["fold_0"],
                "seeds": [42, 43, 44], "total_expected_runs": 3,
            })
            _write_json(results_dir / "multiseed" / "logs" / "multiseed_sweep.json", {
                "completed_runs": 1, "failed_runs": 1, "skipped_runs": 0,
                "runs": [
                    {"model": "indobert_base_p1", "fold": "fold_0", "seed": 42,
                     "status": "completed", "elapsed_seconds": 500.0},
                    {"model": "indobert_base_p1", "fold": "fold_0", "seed": 43,
                     "status": "failed", "elapsed_seconds": 12.0},
                ],
            })
            _write_json(
                results_dir / "multiseed" / "metrics" / "indobert_base_p1" / "fold_0__seed_42.json",
                {"macro_f1": 0.60, "accuracy": 0.65, "weighted_f1": 0.63},
            )

            ms = multiseed_loader.build_multiseed_state(results_dir)
            self.assertEqual(ms["overall_status"], "in_progress")
            self.assertEqual(ms["completed_runs"], 1)
            self.assertEqual(ms["failed_runs"], 1)
            self.assertEqual(len(ms["per_run"]), 2)

            completed_row = [r for r in ms["per_run"] if r["status"] == "completed"][0]
            self.assertEqual(completed_row["macro_f1"], 0.60)
            self.assertTrue(completed_row["metrics_available"])

            failed_row = [r for r in ms["per_run"] if r["status"] == "failed"][0]
            self.assertIsNone(failed_row["macro_f1"])
            self.assertFalse(failed_row["metrics_available"])

    def test_never_reports_complete_unless_manifest_total_is_met(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            _write_json(results_dir / "multiseed" / "manifest.json", {
                "models": ["indobert_base_p1"], "folds": ["fold_0"],
                "seeds": [42, 43, 44], "total_expected_runs": 3,
            })
            _write_json(results_dir / "multiseed" / "logs" / "multiseed_sweep.json", {
                "completed_runs": 2, "failed_runs": 0, "skipped_runs": 0,
                "runs": [
                    {"model": "indobert_base_p1", "fold": "fold_0", "seed": 42, "status": "completed"},
                    {"model": "indobert_base_p1", "fold": "fold_0", "seed": 43, "status": "completed"},
                ],
            })
            ms = multiseed_loader.build_multiseed_state(results_dir)
            self.assertNotEqual(ms["overall_status"], "complete")
            self.assertEqual(ms["overall_status"], "in_progress")

    def test_reports_complete_only_when_all_expected_runs_done(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            _write_json(results_dir / "multiseed" / "manifest.json", {
                "models": ["indobert_base_p1"], "folds": ["fold_0"],
                "seeds": [42], "total_expected_runs": 1,
            })
            _write_json(results_dir / "multiseed" / "logs" / "multiseed_sweep.json", {
                "completed_runs": 1, "failed_runs": 0, "skipped_runs": 0,
                "runs": [{"model": "indobert_base_p1", "fold": "fold_0", "seed": 42, "status": "completed"}],
            })
            ms = multiseed_loader.build_multiseed_state(results_dir)
            self.assertEqual(ms["overall_status"], "complete")


class TestMalformedResultHandling(unittest.TestCase):
    def test_malformed_manifest_reported_gracefully(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            manifest_path = results_dir / "multiseed" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("{not valid json", encoding="utf-8")

            ms = multiseed_loader.build_multiseed_state(results_dir)
            self.assertFalse(ms["manifest_available"])
            self.assertIn("malformed", ms["manifest_error"])

    def test_malformed_per_run_metrics_do_not_crash_and_are_flagged_unavailable(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            _write_json(results_dir / "multiseed" / "manifest.json", {
                "models": ["mbert"], "folds": ["fold_3"], "seeds": [42], "total_expected_runs": 1,
            })
            _write_json(results_dir / "multiseed" / "logs" / "multiseed_sweep.json", {
                "completed_runs": 1, "failed_runs": 0, "skipped_runs": 0,
                "runs": [{"model": "mbert", "fold": "fold_3", "seed": 42, "status": "completed"}],
            })
            bad_metrics = results_dir / "multiseed" / "metrics" / "mbert" / "fold_3__seed_42.json"
            bad_metrics.parent.mkdir(parents=True)
            bad_metrics.write_text("{bad", encoding="utf-8")

            ms = multiseed_loader.build_multiseed_state(results_dir)
            self.assertEqual(len(ms["per_run"]), 1)
            self.assertFalse(ms["per_run"][0]["metrics_available"])


class TestAggregatesDoNotRank(unittest.TestCase):
    def test_per_model_aggregate_has_no_ranking_field(self):
        rows = [
            {"model": "indobert_base_p1", "fold": "fold_0", "seed": 42, "status": "completed", "macro_f1": 0.6},
            {"model": "indobert_base_p1", "fold": "fold_1", "seed": 42, "status": "completed", "macro_f1": 0.5},
        ]
        aggregates = multiseed_loader.build_per_model_aggregates(rows, set())
        self.assertEqual(len(aggregates), 1)
        keys = set(aggregates[0].keys())
        for forbidden in ("rank", "score", "is_best", "winner", "tier"):
            self.assertNotIn(forbidden, keys)


class TestDashboardDistinguishesPhases(unittest.TestCase):
    def test_top_level_state_includes_separate_multiseed_and_analysis_keys(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            full_state = state.build_dashboard_state(results_dir)
            self.assertIn("multiseed", full_state)
            self.assertIn("analysis", full_state)
            self.assertIsNot(full_state["multiseed"], full_state["analysis"])


if __name__ == "__main__":
    unittest.main()
