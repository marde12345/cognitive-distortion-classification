"""Tests for the dashboard data/state layer.

All fixtures are written to a temporary directory created per test, never
under the real `results/` tree. Nothing in this file writes to the real
repository's experiment artifacts.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dashboard import data_loader, state  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_manifest(**overrides) -> dict:
    manifest = {
        "created_at": "2026-09-25T14:39:29Z",
        "device": "mps",
        "models": ["model_a", "model_b"],
        "folds": ["fold_0", "fold_1"],
        "total_runs": 4,
        "completed_runs": 0,
        "failed_runs": 0,
        "skipped_runs": 0,
        "runs": [],
    }
    manifest.update(overrides)
    return manifest


class DataLoaderTests(unittest.TestCase):
    def test_missing_manifest(self):
        with TemporaryDirectory() as tmp:
            result = data_loader.load_manifest(Path(tmp))
            self.assertFalse(result.ok)
            self.assertEqual(result.error, "missing")

    def test_malformed_manifest_retains_error(self):
        with TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "logs" / "transformer_sweep.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text('{"models": ["a", ', encoding="utf-8")
            result = data_loader.load_manifest(Path(tmp))
            self.assertFalse(result.ok)
            self.assertTrue(result.error.startswith("malformed"))

    def test_missing_run_log_and_metrics(self):
        with TemporaryDirectory() as tmp:
            log_result = data_loader.load_run_log(Path(tmp), "model_a", "fold_0")
            metrics_result = data_loader.load_run_metrics(Path(tmp), "model_a", "fold_0")
            self.assertFalse(log_result.ok)
            self.assertFalse(metrics_result.ok)


class StateTests(unittest.TestCase):
    def test_no_manifest_no_previous_state(self):
        with TemporaryDirectory() as tmp:
            result = state.build_dashboard_state(Path(tmp))
            self.assertTrue(result["manifest_stale"])
            self.assertEqual(result["matrix"], [])

    def test_malformed_manifest_retains_previous_state(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            manifest = _base_manifest()
            _write_json(results_dir / "logs" / "transformer_sweep.json", manifest)
            good_state = state.build_dashboard_state(results_dir)
            self.assertFalse(good_state["manifest_stale"])

            (results_dir / "logs" / "transformer_sweep.json").write_text(
                '{"models": [', encoding="utf-8"
            )
            stale_state = state.build_dashboard_state(results_dir, previous_state=good_state)
            self.assertTrue(stale_state["manifest_stale"])
            self.assertEqual(stale_state["overview"], good_state["overview"])

    def test_empty_sweep_all_pending(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            _write_json(results_dir / "logs" / "transformer_sweep.json", _base_manifest())
            result = state.build_dashboard_state(results_dir)
            statuses = {c["status"] for row in result["matrix"] for c in row["cells"]}
            self.assertEqual(statuses, {"pending"})
            self.assertEqual(result["overview"]["pending_runs"], 4)

    def test_partially_completed_sweep(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            manifest = _base_manifest(
                completed_runs=1,
                runs=[
                    {
                        "model": "model_a",
                        "fold": "fold_0",
                        "status": "completed",
                        "elapsed_seconds": 100.0,
                        "exit_code": 0,
                    }
                ],
            )
            _write_json(results_dir / "logs" / "transformer_sweep.json", manifest)
            result = state.build_dashboard_state(results_dir)
            self.assertEqual(result["overview"]["completed_runs"], 1)
            self.assertEqual(result["overview"]["pending_runs"], 3)
            row = next(r for r in result["matrix"] if r["model"] == "model_a")
            self.assertEqual(row["cells"][0]["status"], "completed")
            self.assertEqual(row["cells"][1]["status"], "pending")

    def test_completed_run_reads_metrics(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            manifest = _base_manifest(
                completed_runs=1,
                runs=[
                    {
                        "model": "model_a",
                        "fold": "fold_0",
                        "status": "completed",
                        "elapsed_seconds": 100.0,
                        "exit_code": 0,
                    }
                ],
            )
            _write_json(results_dir / "logs" / "transformer_sweep.json", manifest)
            _write_json(
                results_dir / "metrics" / "model_a" / "fold_0.json",
                {"accuracy": 0.9, "macro_f1": 0.8, "weighted_f1": 0.85},
            )
            result = state.build_dashboard_state(results_dir)
            table = result["completed_runs"]
            self.assertEqual(len(table), 1)
            self.assertEqual(table[0]["accuracy"], 0.9)
            self.assertTrue(table[0]["metrics_available"])

    def test_completed_run_missing_metrics_file(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            manifest = _base_manifest(
                completed_runs=1,
                runs=[
                    {
                        "model": "model_a",
                        "fold": "fold_0",
                        "status": "completed",
                        "elapsed_seconds": 100.0,
                        "exit_code": 0,
                    }
                ],
            )
            _write_json(results_dir / "logs" / "transformer_sweep.json", manifest)
            result = state.build_dashboard_state(results_dir)
            table = result["completed_runs"]
            self.assertFalse(table[0]["metrics_available"])
            self.assertIsNone(table[0]["accuracy"])

    def test_failed_run(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            manifest = _base_manifest(
                failed_runs=1,
                runs=[
                    {
                        "model": "model_a",
                        "fold": "fold_0",
                        "status": "failed",
                        "elapsed_seconds": 5.0,
                        "exit_code": 1,
                    }
                ],
            )
            _write_json(results_dir / "logs" / "transformer_sweep.json", manifest)
            result = state.build_dashboard_state(results_dir)
            row = next(r for r in result["matrix"] if r["model"] == "model_a")
            self.assertEqual(row["cells"][0]["status"], "failed")
            self.assertEqual(result["completed_runs"], [])

    def test_running_run(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            manifest = _base_manifest(
                runs=[
                    {
                        "model": "model_a",
                        "fold": "fold_0",
                        "status": "running",
                        "elapsed_seconds": None,
                        "exit_code": None,
                    }
                ],
            )
            _write_json(results_dir / "logs" / "transformer_sweep.json", manifest)
            result = state.build_dashboard_state(results_dir)
            self.assertEqual(result["overview"]["running_runs"], 1)
            row = next(r for r in result["matrix"] if r["model"] == "model_a")
            self.assertEqual(row["cells"][0]["status"], "running")

    def test_pending_run_not_in_manifest_runs_list(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            _write_json(results_dir / "logs" / "transformer_sweep.json", _base_manifest())
            result = state.build_dashboard_state(results_dir)
            row = next(r for r in result["matrix"] if r["model"] == "model_b")
            self.assertEqual(row["cells"][1]["status"], "pending")

    def test_skipped_run(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            manifest = _base_manifest(
                skipped_runs=1,
                runs=[
                    {
                        "model": "model_a",
                        "fold": "fold_0",
                        "status": "skipped",
                        "start_time": None,
                        "end_time": None,
                        "elapsed_seconds": None,
                        "exit_code": None,
                    }
                ],
            )
            _write_json(results_dir / "logs" / "transformer_sweep.json", manifest)
            result = state.build_dashboard_state(results_dir)
            row = next(r for r in result["matrix"] if r["model"] == "model_a")
            self.assertEqual(row["cells"][0]["status"], "skipped")

    def test_missing_metric_file_for_model_detail(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            manifest = _base_manifest()
            _write_json(results_dir / "logs" / "transformer_sweep.json", manifest)
            _write_json(
                results_dir / "logs" / "model_a" / "fold_0.json",
                {"status": "completed", "elapsed_seconds": 10.0, "history": []},
            )
            detail = state.build_model_detail(results_dir, manifest, "model_a")
            fold0 = detail["folds"][0]
            self.assertTrue(fold0["log_available"])
            self.assertFalse(fold0["metrics_available"])
            self.assertIsNone(fold0["accuracy"])

    def test_missing_execution_log_for_model_detail(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            manifest = _base_manifest()
            _write_json(results_dir / "logs" / "transformer_sweep.json", manifest)
            _write_json(
                results_dir / "metrics" / "model_a" / "fold_0.json",
                {"accuracy": 0.5, "macro_f1": 0.4, "weighted_f1": 0.45},
            )
            detail = state.build_model_detail(results_dir, manifest, "model_a")
            fold0 = detail["folds"][0]
            self.assertFalse(fold0["log_available"])
            self.assertTrue(fold0["metrics_available"])
            self.assertEqual(fold0["accuracy"], 0.5)

    def test_all_runs_completed_cross_fold_stats(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            manifest = _base_manifest(
                models=["model_a"],
                folds=["fold_0", "fold_1"],
                total_runs=2,
                completed_runs=2,
                runs=[
                    {
                        "model": "model_a",
                        "fold": "fold_0",
                        "status": "completed",
                        "elapsed_seconds": 100.0,
                        "exit_code": 0,
                    },
                    {
                        "model": "model_a",
                        "fold": "fold_1",
                        "status": "completed",
                        "elapsed_seconds": 200.0,
                        "exit_code": 0,
                    },
                ],
            )
            _write_json(results_dir / "logs" / "transformer_sweep.json", manifest)
            _write_json(
                results_dir / "metrics" / "model_a" / "fold_0.json",
                {"accuracy": 0.8, "macro_f1": 0.7, "weighted_f1": 0.75},
            )
            _write_json(
                results_dir / "metrics" / "model_a" / "fold_1.json",
                {"accuracy": 0.6, "macro_f1": 0.5, "weighted_f1": 0.55},
            )
            result = state.build_dashboard_state(results_dir)
            self.assertEqual(result["overview"]["completed_runs"], 2)
            self.assertEqual(result["overview"]["pending_runs"], 0)

            detail = state.build_model_detail(results_dir, manifest, "model_a")
            stats = detail["cross_fold_stats"]
            self.assertAlmostEqual(stats["accuracy"]["mean"], 0.7)
            self.assertEqual(stats["n_folds_with_metrics"], 2)


if __name__ == "__main__":
    unittest.main()
