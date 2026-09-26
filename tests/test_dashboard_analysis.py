"""Tests for the Phase 17 dashboard additions: analysis-artifact loading,
live-demo checkpoint detection, and dashboard-server path/host handling.

All fixtures are written to a temporary directory created per test, never
under the real `results/` tree. Nothing in this file writes to the real
repository's experiment artifacts.
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dashboard import analysis_loader, state  # noqa: E402


def _analysis_dir(results_dir: Path) -> Path:
    d = results_dir / "analysis" / "transformer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class TestAggregateMetricsLoading(unittest.TestCase):
    def test_loads_valid_file(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            _write_json(_analysis_dir(results_dir) / "aggregate_metrics.json", {"ddof": 1, "models": {}})
            result = analysis_loader.load_aggregate_metrics(results_dir)
            self.assertTrue(result.ok)
            self.assertEqual(result.data["ddof"], 1)

    def test_missing_file_reported_not_raised(self):
        with TemporaryDirectory() as tmp:
            result = analysis_loader.load_aggregate_metrics(Path(tmp))
            self.assertFalse(result.ok)
            self.assertEqual(result.error, "missing")

    def test_malformed_json_reported_not_raised(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            path = _analysis_dir(results_dir) / "aggregate_metrics.json"
            path.write_text("{not valid json", encoding="utf-8")
            result = analysis_loader.load_aggregate_metrics(results_dir)
            self.assertFalse(result.ok)
            self.assertTrue(result.error.startswith("malformed"))


class TestCsvArtifactLoading(unittest.TestCase):
    def test_fold_metrics_csv_loads(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            _write_csv(
                _analysis_dir(results_dir) / "fold_metrics.csv",
                [{"model": "m", "fold": "fold_0", "macro_f1": "0.5"}],
            )
            result = analysis_loader.load_fold_metrics(results_dir)
            self.assertTrue(result.ok)
            self.assertEqual(result.data["rows"][0]["model"], "m")

    def test_per_class_metrics_csv_loads(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            _write_csv(
                _analysis_dir(results_dir) / "per_class_metrics.csv",
                [{"model": "m", "label": "0", "mean_f1": "0.7"}],
            )
            result = analysis_loader.load_per_class_metrics(results_dir)
            self.assertTrue(result.ok)
            self.assertEqual(result.data["rows"][0]["label"], "0")

    def test_missing_csv_reported_not_raised(self):
        with TemporaryDirectory() as tmp:
            result = analysis_loader.load_fold_metrics(Path(tmp))
            self.assertFalse(result.ok)
            self.assertEqual(result.error, "missing")

    def test_empty_csv_reported_as_malformed(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            path = _analysis_dir(results_dir) / "fold_metrics.csv"
            path.write_text("", encoding="utf-8")
            result = analysis_loader.load_fold_metrics(results_dir)
            self.assertFalse(result.ok)


class TestRemainingAnalysisArtifacts(unittest.TestCase):
    def test_anomaly_sensitivity_loads(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            _write_json(_analysis_dir(results_dir) / "anomaly_sensitivity.json", {"m/fold_1": {}})
            self.assertTrue(analysis_loader.load_anomaly_sensitivity(results_dir).ok)

    def test_runtime_analysis_loads(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            _write_json(_analysis_dir(results_dir) / "runtime_analysis.json", {"per_model": {}})
            self.assertTrue(analysis_loader.load_runtime_analysis(results_dir).ok)

    def test_statistical_tests_loads(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            _write_json(_analysis_dir(results_dir) / "statistical_tests.json", {"friedman_test": {}})
            self.assertTrue(analysis_loader.load_statistical_tests(results_dir).ok)


class TestPlotDiscoveryAndPathSafety(unittest.TestCase):
    def test_available_plots_only_lists_existing_known_files(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            plots_dir = _analysis_dir(results_dir) / "plots"
            plots_dir.mkdir()
            (plots_dir / "01_macro_f1_mean_std.png").write_bytes(b"fake-png")
            (plots_dir / "not_a_known_plot.png").write_bytes(b"fake-png")
            found = analysis_loader.available_plots(results_dir)
            self.assertEqual(found, ["01_macro_f1_mean_std.png"])

    def test_missing_plots_directory_yields_empty_list(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(analysis_loader.available_plots(Path(tmp)), [])

    def test_resolve_plot_path_accepts_known_existing_file(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            plots_dir = _analysis_dir(results_dir) / "plots"
            plots_dir.mkdir()
            (plots_dir / "01_macro_f1_mean_std.png").write_bytes(b"fake-png")
            resolved = analysis_loader.resolve_plot_path(results_dir, "01_macro_f1_mean_std.png")
            self.assertIsNotNone(resolved)
            self.assertTrue(resolved.is_file())

    def test_resolve_plot_path_rejects_unknown_name(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(analysis_loader.resolve_plot_path(Path(tmp), "evil.png"))

    def test_resolve_plot_path_rejects_traversal_attempt(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            # Even if an attacker-controlled name happens to collide with a
            # traversal string, it must fail the allowlist membership check.
            traversal = "../../../../../../etc/passwd"
            self.assertIsNone(analysis_loader.resolve_plot_path(results_dir, traversal))

    def test_resolve_plot_path_rejects_known_name_when_file_absent(self):
        with TemporaryDirectory() as tmp:
            # Name is in the allowlist but the file was never generated.
            self.assertIsNone(
                analysis_loader.resolve_plot_path(Path(tmp), "01_macro_f1_mean_std.png")
            )


class TestBuildAnalysisState(unittest.TestCase):
    def test_all_missing_reports_unavailable_without_raising(self):
        with TemporaryDirectory() as tmp:
            result = state.build_analysis_state(Path(tmp))
            self.assertFalse(result["aggregate_metrics_available"])
            self.assertFalse(result["fold_metrics_available"])
            self.assertFalse(result["per_class_metrics_available"])
            self.assertFalse(result["anomaly_sensitivity_available"])
            self.assertFalse(result["runtime_analysis_available"])
            self.assertFalse(result["statistical_tests_available"])
            self.assertEqual(result["available_plots"], [])
            self.assertFalse(result["plots_complete"])

    def test_model_order_is_fixed_experiment_order_not_a_ranking(self):
        with TemporaryDirectory() as tmp:
            result = state.build_analysis_state(Path(tmp))
            self.assertEqual(
                result["model_order"],
                ["indobert_15g", "indobert_base_p1", "indobertweet",
                 "indoroberta_15g", "mbert", "nusabert", "xlmr"],
            )
            self.assertIn("not ranked", result["model_order_note"].lower())

    def test_partial_artifacts_isolated_per_file(self):
        """One malformed file must not blank out the others."""
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            _write_json(_analysis_dir(results_dir) / "aggregate_metrics.json", {"models": {}})
            (_analysis_dir(results_dir) / "runtime_analysis.json").write_text("{bad", encoding="utf-8")
            result = state.build_analysis_state(results_dir)
            self.assertTrue(result["aggregate_metrics_available"])
            self.assertFalse(result["runtime_analysis_available"])


class TestLiveDemoState(unittest.TestCase):
    def test_no_checkpoint_reports_unavailable(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            live = state.build_live_demo_state(results_dir)
            self.assertFalse(live["checkpoint_available"])
            self.assertEqual(live["checkpoint_files"], [])
            self.assertIn("not available", live["message"].lower())

    def test_missing_models_dir_reports_unavailable_not_error(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            # models/ sibling directory intentionally does not exist.
            live = state.build_live_demo_state(results_dir)
            self.assertFalse(live["checkpoint_available"])

    def test_checkpoint_file_present_reports_available(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            results_dir.mkdir()
            models_dir = root / "models" / "some_model"
            models_dir.mkdir(parents=True)
            (models_dir / "pytorch_model.bin").write_bytes(b"fake-weights")
            live = state.build_live_demo_state(results_dir)
            self.assertTrue(live["checkpoint_available"])
            self.assertIn("some_model/pytorch_model.bin", live["checkpoint_files"])


class TestBuildDashboardStateIncludesAnalysisAndLiveDemo(unittest.TestCase):
    def test_present_even_when_manifest_missing(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            result = state.build_dashboard_state(results_dir)
            self.assertIn("analysis", result)
            self.assertIn("live_demo", result)
            self.assertTrue(result["manifest_stale"])

    def test_present_when_manifest_complete(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            _write_json(
                results_dir / "logs" / "transformer_sweep.json",
                {
                    "created_at": "2026-01-01T00:00:00Z", "device": "mps",
                    "models": ["m"], "folds": ["fold_0"],
                    "total_runs": 1, "completed_runs": 0, "failed_runs": 0,
                    "skipped_runs": 0, "runs": [],
                },
            )
            result = state.build_dashboard_state(results_dir)
            self.assertIn("analysis", result)
            self.assertIn("live_demo", result)
            self.assertFalse(result["manifest_stale"])


class TestDashboardServerHostHandling(unittest.TestCase):
    """CLI/host-binding behavior lives in scripts/run_dashboard.py, which is
    not itself a package module; import it directly by file path so these
    tests can exercise its argument parsing and LAN-IP helper without
    starting a real server or touching any real results/ directory."""

    def _load_module(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "scripts" / "run_dashboard.py"
        spec = importlib.util.spec_from_file_location("run_dashboard_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_default_host_is_localhost(self):
        """Parsing an empty argv (no --host given) must default to
        127.0.0.1, not 0.0.0.0 — the safe-by-default requirement."""
        module = self._load_module()
        parser = module.argparse.ArgumentParser(description=module.__doc__)
        parser.add_argument("--results-dir", type=Path, default=Path("results"))
        parser.add_argument("--port", type=int, default=8765)
        parser.add_argument("--host", type=str, default="127.0.0.1")
        parser.add_argument("--poll-interval", type=float, default=5.0)
        args = parser.parse_args([])
        self.assertEqual(args.host, "127.0.0.1")

    def test_explicit_host_overrides_default(self):
        module = self._load_module()
        parser = module.argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="127.0.0.1")
        args = parser.parse_args(["--host", "0.0.0.0"])
        self.assertEqual(args.host, "0.0.0.0")

    def test_local_lan_ip_returns_a_string_or_none(self):
        module = self._load_module()
        result = module.local_lan_ip()
        self.assertTrue(result is None or isinstance(result, str))

    def test_local_lan_ip_never_raises_on_socket_failure(self):
        module = self._load_module()
        import socket as real_socket

        class FailingSocket:
            def __init__(self, *a, **kw):
                raise OSError("no network")

        original = module.socket.socket
        module.socket.socket = FailingSocket
        try:
            self.assertIsNone(module.local_lan_ip())
        finally:
            module.socket.socket = original


if __name__ == "__main__":
    unittest.main()
