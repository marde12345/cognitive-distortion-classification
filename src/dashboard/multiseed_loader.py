"""Read-only access to Phase 20 multi-seed experiment artifacts.

Mirrors the isolation contract of data_loader.py: every function here only
reads files under results/multiseed/, never writes/repairs anything, and
turns a missing or malformed file into a soft failure (ReadResult) instead
of raising. As of Phase 20 no runs have been executed, so in the common
case this module reports "0 completed" against the manifest's planned
total -- it must never claim a run is complete unless both its metrics
file and its log file exist and the log's own status says "completed",
exactly like data_loader.is_complete-style checks elsewhere in this
project.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .data_loader import ReadResult, safe_read_json


def load_protocol_manifest(results_dir: Path) -> ReadResult:
    """Read results/multiseed/manifest.json (the static protocol descriptor)."""
    return safe_read_json(results_dir / "multiseed" / "manifest.json")


def load_run_state(results_dir: Path) -> ReadResult:
    """Read results/multiseed/logs/multiseed_sweep.json (the resumable,
    incrementally-updated run-state tracker written by
    scripts/run_transformer_multiseed.py)."""
    return safe_read_json(results_dir / "multiseed" / "logs" / "multiseed_sweep.json")


def load_live_state(results_dir: Path) -> ReadResult:
    """Read results/multiseed/live_state.json -- a lightweight, frequently
    refreshed progress summary (current_run, last_completed_run, counts)
    written by scripts/run_transformer_multiseed.py while training is in
    flight. Written atomically (tmp file + rename) by the writer, but this
    reader still treats a missing or malformed file as a soft failure like
    every other artifact in this module -- a dashboard poll landing between
    writes must never crash or show stale-as-if-live data."""
    return safe_read_json(results_dir / "multiseed" / "live_state.json")


def _run_name(fold: str, seed: int) -> str:
    return f"{fold}__seed_{seed}"


def load_run_metrics(results_dir: Path, model: str, fold: str, seed: int) -> ReadResult:
    """Read results/multiseed/metrics/<model>/<fold>__seed_<seed>.json."""
    return safe_read_json(
        results_dir / "multiseed" / "metrics" / model / f"{_run_name(fold, seed)}.json"
    )


def collect_per_run_results(
    results_dir: Path, run_state: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per (model, fold, seed) the run-state knows about, joined
    with that run's metrics file when the run is marked completed.

    A run whose log says "completed" but whose metrics file is missing or
    malformed is reported with metrics_available=False rather than being
    silently dropped or treated as if it had no result at all.
    """
    rows: list[dict[str, Any]] = []
    for run in run_state.get("runs", []) or []:
        model = run.get("model")
        fold = run.get("fold")
        seed = run.get("seed")
        status = run.get("status")
        row: dict[str, Any] = {
            "model": model,
            "fold": fold,
            "seed": seed,
            "status": status,
            "elapsed_seconds": run.get("elapsed_seconds"),
            "macro_f1": None,
            "accuracy": None,
            "weighted_f1": None,
            "metrics_available": False,
        }
        if status == "completed" and model is not None and fold is not None and seed is not None:
            metrics_result = load_run_metrics(results_dir, model, fold, seed)
            if metrics_result.ok:
                data = metrics_result.data or {}
                row["macro_f1"] = data.get("macro_f1")
                row["accuracy"] = data.get("accuracy")
                row["weighted_f1"] = data.get("weighted_f1")
                row["metrics_available"] = True
        rows.append(row)
    return rows


def _descriptive_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "mean": None, "std": None, "median": None,
            "min": None, "max": None, "range": None, "n": 0,
        }
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "n": len(values),
    }


def build_per_model_aggregates(
    per_run: list[dict[str, Any]], anomalous_runs: set[tuple[str, str, int]]
) -> list[dict[str, Any]]:
    """model -> descriptive stats over all its completed runs (all folds,
    all seeds pooled), plus completed/failed/anomalous counts. Anomalous
    runs are never excluded from the stats -- ``anomalous_runs`` is purely
    informational (a human-curated set, passed in by the caller; empty
    until a human identifies anomalies in this protocol's own results, the
    same way Phase 16's ANOMALOUS_RUNS was curated after inspection, not
    auto-detected)."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in per_run:
        by_model.setdefault(row["model"], []).append(row)

    out = []
    for model, rows in by_model.items():
        completed = [r for r in rows if r["status"] == "completed"]
        failed = [r for r in rows if r["status"] == "failed"]
        macro_f1s = [r["macro_f1"] for r in completed if isinstance(r["macro_f1"], (int, float))]
        n_anomalous = sum(
            1 for r in completed if (model, r["fold"], r["seed"]) in anomalous_runs
        )
        out.append({
            "model": model,
            "n_completed": len(completed),
            "n_failed": len(failed),
            "n_anomalous": n_anomalous,
            "macro_f1": _descriptive_stats(macro_f1s),
        })
    return out


def build_per_fold_aggregates(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(model, fold) -> descriptive stats aggregated across seeds."""
    by_key: dict[tuple[str, str], list[float]] = {}
    counts: dict[tuple[str, str], int] = {}
    for row in per_run:
        key = (row["model"], row["fold"])
        counts[key] = counts.get(key, 0) + (1 if row["status"] == "completed" else 0)
        if row["status"] == "completed" and isinstance(row["macro_f1"], (int, float)):
            by_key.setdefault(key, []).append(row["macro_f1"])

    out = []
    for (model, fold), values in sorted(by_key.items()):
        out.append({
            "model": model,
            "fold": fold,
            "n_completed": counts.get((model, fold), 0),
            "macro_f1": _descriptive_stats(values),
        })
    return out


def build_per_seed_aggregates(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(model, seed) -> descriptive stats aggregated across folds."""
    by_key: dict[tuple[str, int], list[float]] = {}
    counts: dict[tuple[str, int], int] = {}
    for row in per_run:
        key = (row["model"], row["seed"])
        counts[key] = counts.get(key, 0) + (1 if row["status"] == "completed" else 0)
        if row["status"] == "completed" and isinstance(row["macro_f1"], (int, float)):
            by_key.setdefault(key, []).append(row["macro_f1"])

    out = []
    for (model, seed), values in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append({
            "model": model,
            "seed": seed,
            "n_completed": counts.get((model, seed), 0),
            "macro_f1": _descriptive_stats(values),
        })
    return out


def build_multiseed_state(results_dir: Path) -> dict[str, Any]:
    """Top-level Phase 20 multi-seed state: planned protocol, run-state
    counters, per-run/per-model/per-fold/per-seed views.

    Never reports the experiment as complete unless the run-state's own
    ``completed_runs`` count equals the manifest's ``total_expected_runs``
    -- partial or absent data is always shown as such, not rounded up.
    """
    manifest_result = load_protocol_manifest(results_dir)
    run_state_result = load_run_state(results_dir)
    live_state_result = load_live_state(results_dir)

    manifest = manifest_result.data or {}
    run_state = run_state_result.data or {}
    live_state = live_state_result.data or {}

    planned_models = manifest.get("models", [])
    planned_folds = manifest.get("folds", [])
    planned_seeds = manifest.get("seeds", [])
    total_expected = manifest.get("total_expected_runs")

    per_run = collect_per_run_results(results_dir, run_state) if run_state_result.ok else []

    completed_runs = run_state.get("completed_runs", 0) if run_state_result.ok else 0
    failed_runs = run_state.get("failed_runs", 0) if run_state_result.ok else 0
    skipped_runs = run_state.get("skipped_runs", 0) if run_state_result.ok else 0

    current_run = live_state.get("current_run") if live_state_result.ok else None
    last_completed_run = live_state.get("last_completed_run") if live_state_result.ok else None

    if current_run is not None:
        # A run is actively in flight -- this always wins, even if the
        # (less frequently updated) run-state counts happen to already
        # read as "all expected runs accounted for" in some race window.
        overall_status = "in_progress"
    elif total_expected is not None and completed_runs == total_expected and total_expected > 0:
        overall_status = "complete"
    elif completed_runs == 0 and failed_runs == 0 and skipped_runs == 0:
        overall_status = "not_run"
    else:
        overall_status = "in_progress"

    progress_percent = (
        round(100.0 * completed_runs / total_expected, 1)
        if total_expected else 0.0
    )

    # Anomalous runs are never auto-detected here; Phase 20 has not run any
    # experiments yet, so this starts empty. A human curates this set (as
    # was done for Phase 16) once real results exist.
    anomalous_runs: set[tuple[str, str, int]] = set()

    return {
        "manifest_available": manifest_result.ok,
        "manifest_error": manifest_result.error,
        "protocol": {
            "experiment_id": manifest.get("experiment_id"),
            "protocol_version": manifest.get("protocol_version"),
            "models": planned_models,
            "folds": planned_folds,
            "seeds": planned_seeds,
            "total_expected_runs": total_expected,
        },
        "run_state_available": run_state_result.ok,
        "run_state_error": run_state_result.error,
        "live_state_available": live_state_result.ok,
        "live_state_error": live_state_result.error,
        "overall_status": overall_status,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "skipped_runs": skipped_runs,
        "total_expected_runs": total_expected,
        "progress_percent": progress_percent,
        "current_run": current_run,
        "last_completed_run": last_completed_run,
        "per_run": per_run,
        "per_model": build_per_model_aggregates(per_run, anomalous_runs),
        "per_fold": build_per_fold_aggregates(per_run),
        "per_seed": build_per_seed_aggregates(per_run),
        "anomalous_runs": [
            {"model": m, "fold": f, "seed": s} for (m, f, s) in sorted(anomalous_runs)
        ],
        "note": (
            "No automatic model selection is performed on multi-seed data. "
            "This section reports descriptive statistics only."
        ),
    }
