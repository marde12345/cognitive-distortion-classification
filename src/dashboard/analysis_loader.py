"""Read-only access to the Phase 16 statistical-analysis artifacts.

Every function here only reads files under
``results/analysis/transformer/``. None of them write, rename, delete, or
"repair" anything. Missing or malformed files are reported as soft
failures (never raised), exactly like ``data_loader.safe_read_json``, so
the dashboard can render an informative empty/error state per file
instead of crashing.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .data_loader import ReadResult, safe_read_json

ANALYSIS_SUBDIR = Path("analysis") / "transformer"

# Exact, known set of plot filenames the Phase 16 analysis script produces.
# Used as an allowlist so the dashboard's plot-serving route can never be
# used to read an arbitrary file (no path traversal): only a name in this
# set is ever opened, and only ever read from PLOTS_SUBDIR.
KNOWN_PLOTS = (
    "01_macro_f1_mean_std.png",
    "02_accuracy_mean_std.png",
    "03_weighted_f1_mean_std.png",
    "04_macro_f1_across_folds.png",
    "05_runtime_distribution.png",
    "06_per_class_f1_heatmap.png",
)


def _analysis_dir(results_dir: Path) -> Path:
    return results_dir / ANALYSIS_SUBDIR


def safe_read_csv(path: Path) -> ReadResult:
    """Read a CSV file into a list of dicts, never raising.

    Reuses ``ReadResult`` from data_loader for a consistent missing/
    malformed/ok vocabulary, even though the payload is a list, not a
    dict — callers use ``.ok``/``.error`` the same way either way.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ReadResult(None, "missing")
    except OSError as exc:
        return ReadResult(None, f"unreadable: {exc}")

    try:
        rows = list(csv.DictReader(raw.splitlines()))
    except csv.Error as exc:
        return ReadResult(None, f"malformed: {exc}")

    if not rows:
        return ReadResult(None, "malformed: no rows")

    # ReadResult.data is typed as dict|None; wrap the row list so the same
    # ok/error contract works for CSV as for JSON.
    return ReadResult({"rows": rows}, None)


def load_aggregate_metrics(results_dir: Path) -> ReadResult:
    return safe_read_json(_analysis_dir(results_dir) / "aggregate_metrics.json")


def load_fold_metrics(results_dir: Path) -> ReadResult:
    return safe_read_csv(_analysis_dir(results_dir) / "fold_metrics.csv")


def load_per_class_metrics(results_dir: Path) -> ReadResult:
    return safe_read_csv(_analysis_dir(results_dir) / "per_class_metrics.csv")


def load_anomaly_sensitivity(results_dir: Path) -> ReadResult:
    return safe_read_json(_analysis_dir(results_dir) / "anomaly_sensitivity.json")


def load_runtime_analysis(results_dir: Path) -> ReadResult:
    return safe_read_json(_analysis_dir(results_dir) / "runtime_analysis.json")


def load_statistical_tests(results_dir: Path) -> ReadResult:
    return safe_read_json(_analysis_dir(results_dir) / "statistical_tests.json")


def available_plots(results_dir: Path) -> list[str]:
    """Which of the known plot filenames actually exist on disk right now."""
    plots_dir = _analysis_dir(results_dir) / "plots"
    return [name for name in KNOWN_PLOTS if (plots_dir / name).is_file()]


def resolve_plot_path(results_dir: Path, name: str) -> Path | None:
    """Resolve a plot filename to a real path, or None if not allowed/found.

    ``name`` must be an exact match against ``KNOWN_PLOTS`` — this is the
    entire path-traversal defense for the plot-serving HTTP route: no
    user-supplied path component is ever joined onto a directory and
    opened without first passing this equality check against a fixed
    allowlist, so ``../../etc/passwd`` (or any other name) is rejected
    before any filesystem access is attempted.
    """
    if name not in KNOWN_PLOTS:
        return None
    path = _analysis_dir(results_dir) / "plots" / name
    return path if path.is_file() else None
