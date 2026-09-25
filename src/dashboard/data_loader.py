"""Read-only access to experiment sweep artifacts.

Every function here only reads files under a ``results/`` directory. None
of them write, rename, delete, or "repair" anything. Malformed or
temporarily-incomplete JSON (the sweep manifest is written incrementally
by a separate process) is reported as a soft failure, never raised past
this module, so a caller can keep displaying the last good state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReadResult:
    """Outcome of trying to read one JSON file.

    ``data`` is ``None`` when the file is missing or could not be parsed.
    ``error`` holds a short human-readable reason in that case, so callers
    (and tests) can distinguish "missing" from "malformed" without needing
    exception types to leak out of this module.
    """

    data: dict[str, Any] | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.data is not None


def safe_read_json(path: Path) -> ReadResult:
    """Read and parse a JSON file, never raising.

    Returns ``ReadResult(None, "missing")`` if the file does not exist,
    ``ReadResult(None, "malformed: ...")`` if it exists but is not valid
    JSON at the moment of reading (e.g. a concurrent writer left it
    truncated), and ``ReadResult(data, None)`` on success.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ReadResult(None, "missing")
    except OSError as exc:
        return ReadResult(None, f"unreadable: {exc}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ReadResult(None, f"malformed: {exc}")

    if not isinstance(data, dict):
        return ReadResult(None, "malformed: top-level JSON is not an object")

    return ReadResult(data, None)


def load_manifest(results_dir: Path) -> ReadResult:
    """Read ``results/logs/transformer_sweep.json``.

    This is the primary source of sweep-wide status. It is owned by the
    sweep runner and is never modified by the dashboard.
    """
    return safe_read_json(results_dir / "logs" / "transformer_sweep.json")


def load_run_log(results_dir: Path, model: str, fold: str) -> ReadResult:
    """Read ``results/logs/<model>/<fold>.json`` (per-run execution metadata)."""
    return safe_read_json(results_dir / "logs" / model / f"{fold}.json")


def load_run_metrics(results_dir: Path, model: str, fold: str) -> ReadResult:
    """Read ``results/metrics/<model>/<fold>.json`` (per-run evaluation metrics)."""
    return safe_read_json(results_dir / "metrics" / model / f"{fold}.json")


def prediction_csv_exists(results_dir: Path, model: str, fold: str) -> bool:
    """Whether a predictions CSV exists for a run.

    Informational only — per Phase 15 instructions, predictions must not be
    used to derive sweep status.
    """
    return (results_dir / "predictions" / model / f"{fold}.csv").exists()
