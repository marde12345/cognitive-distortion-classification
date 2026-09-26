"""Derive dashboard state from raw experiment artifacts.

This module has no knowledge of HTTP, HTML, or polling — it just turns
whatever `data_loader` was able to read into plain, JSON-serializable
dicts that the presentation layer renders. Keeping this separate means the
state-derivation logic (partial-JSON handling, status classification,
descriptive statistics) can be unit tested without spinning up a server.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from . import analysis_loader, data_loader

KNOWN_STATUSES = {"completed", "running", "failed", "pending", "skipped"}

# Experiment/configuration order (configs/models/*.yaml filename order).
# This is NOT a performance ranking; the dashboard must present models in
# this fixed order everywhere and say so explicitly.
MODEL_ORDER = (
    "indobert_15g",
    "indobert_base_p1",
    "indobertweet",
    "indoroberta_15g",
    "mbert",
    "nusabert",
    "xlmr",
)

ANOMALOUS_RUNS = (("indobertweet", "fold_1"), ("mbert", "fold_3"))


def _run_lookup(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for run in manifest.get("runs", []) or []:
        model = run.get("model")
        fold = run.get("fold")
        if model is not None and fold is not None:
            lookup[(model, fold)] = run
    return lookup


def build_matrix(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the model x fold status matrix.

    A (model, fold) pair not present in the manifest's ``runs`` list is
    classified as ``pending`` — the sweep has not reached it yet. Pairs
    that are present use the manifest's own ``status`` value verbatim, so
    any status the sweep runner writes (including ones not in
    ``KNOWN_STATUSES``) is surfaced rather than silently dropped.
    """
    models = manifest.get("models", []) or []
    folds = manifest.get("folds", []) or []
    lookup = _run_lookup(manifest)

    rows = []
    for model in models:
        cells = []
        for fold in folds:
            run = lookup.get((model, fold))
            if run is None:
                cells.append({"fold": fold, "status": "pending"})
            else:
                cells.append(
                    {
                        "fold": fold,
                        "status": run.get("status", "pending"),
                        "elapsed_seconds": run.get("elapsed_seconds"),
                        "exit_code": run.get("exit_code"),
                    }
                )
        rows.append({"model": model, "cells": cells})
    return rows


def build_overview(manifest: dict[str, Any]) -> dict[str, Any]:
    """Sweep-wide counters and measured/derived timing stats.

    ``completed_runs``/``failed_runs``/``skipped_runs``/``total_runs`` are
    taken directly from the manifest (measured, authoritative). Pending and
    running counts, and the average completed-run runtime, are derived here
    from the ``runs`` list.
    """
    models = manifest.get("models", []) or []
    folds = manifest.get("folds", []) or []
    total_runs = manifest.get("total_runs", len(models) * len(folds))
    completed_runs = manifest.get("completed_runs", 0)
    failed_runs = manifest.get("failed_runs", 0)
    skipped_runs = manifest.get("skipped_runs", 0)

    runs = manifest.get("runs", []) or []
    running_runs = sum(1 for r in runs if r.get("status") == "running")
    accounted = len(runs)
    pending_runs = max(total_runs - accounted, 0)

    completed_elapsed = [
        r["elapsed_seconds"]
        for r in runs
        if r.get("status") == "completed" and isinstance(r.get("elapsed_seconds"), (int, float))
    ]
    avg_runtime_seconds = (
        sum(completed_elapsed) / len(completed_elapsed) if completed_elapsed else None
    )
    total_elapsed_seconds = sum(completed_elapsed) if completed_elapsed else None

    return {
        "device": manifest.get("device"),
        "total_models": len(models),
        "total_folds": len(folds),
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "skipped_runs": skipped_runs,
        "running_runs": running_runs,
        "pending_runs": pending_runs,
        "avg_runtime_seconds": avg_runtime_seconds,
        "total_elapsed_seconds_measured": total_elapsed_seconds,
    }


def build_completed_runs_table(
    results_dir: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per completed run, in manifest/execution order (not sorted by metric)."""
    rows = []
    for run in manifest.get("runs", []) or []:
        if run.get("status") != "completed":
            continue
        model, fold = run.get("model"), run.get("fold")
        metrics_result = data_loader.load_run_metrics(results_dir, model, fold)
        metrics = metrics_result.data or {}
        rows.append(
            {
                "model": model,
                "fold": fold,
                "status": run.get("status"),
                "elapsed_seconds": run.get("elapsed_seconds"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "metrics_available": metrics_result.ok,
            }
        )
    return rows


def _mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std}


def build_model_detail(
    results_dir: Path, manifest: dict[str, Any], model: str
) -> dict[str, Any]:
    """Per-fold detail plus cross-fold descriptive statistics for one model.

    Used for the indobert_15g detail view, but works for any model present
    in the manifest so the dashboard keeps working as more models finish.
    """
    folds = manifest.get("folds", []) or []
    fold_rows = []
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    weighted_f1s: list[float] = []

    for fold in folds:
        log_result = data_loader.load_run_log(results_dir, model, fold)
        metrics_result = data_loader.load_run_metrics(results_dir, model, fold)
        log = log_result.data or {}
        metrics = metrics_result.data or {}

        row = {
            "fold": fold,
            "status": log.get("status"),
            "elapsed_seconds": log.get("elapsed_seconds"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "history": log.get("history") or metrics.get("history") or [],
            "log_available": log_result.ok,
            "metrics_available": metrics_result.ok,
        }
        fold_rows.append(row)

        if isinstance(row["accuracy"], (int, float)):
            accuracies.append(row["accuracy"])
        if isinstance(row["macro_f1"], (int, float)):
            macro_f1s.append(row["macro_f1"])
        if isinstance(row["weighted_f1"], (int, float)):
            weighted_f1s.append(row["weighted_f1"])

    return {
        "model": model,
        "folds": fold_rows,
        "cross_fold_stats": {
            "label": "Cross-fold descriptive statistics",
            "accuracy": _mean_std(accuracies),
            "macro_f1": _mean_std(macro_f1s),
            "weighted_f1": _mean_std(weighted_f1s),
            "n_folds": len(fold_rows),
            "n_folds_with_metrics": len(accuracies),
        },
    }


def build_run_detail(
    results_dir: Path, model: str, fold: str
) -> dict[str, Any]:
    """Full factual detail for a single run, merging log + metrics."""
    log_result = data_loader.load_run_log(results_dir, model, fold)
    metrics_result = data_loader.load_run_metrics(results_dir, model, fold)
    log = log_result.data or {}
    metrics = metrics_result.data or {}

    return {
        "model": model,
        "fold": fold,
        "status": log.get("status"),
        "device": log.get("device"),
        "configured_epochs": log.get("configured_epochs"),
        "completed_epochs": log.get("completed_epochs"),
        "batch_size": log.get("batch_size"),
        "max_seq_length": log.get("max_seq_length"),
        "elapsed_seconds": log.get("elapsed_seconds"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "history": log.get("history") or metrics.get("history") or [],
        "has_predictions": data_loader.prediction_csv_exists(results_dir, model, fold),
        "log_available": log_result.ok,
        "metrics_available": metrics_result.ok,
    }


def build_analysis_state(results_dir: Path) -> dict[str, Any]:
    """Assemble the Phase 16 statistical-analysis section, from disk, as-is.

    Every sub-artifact is read independently and failures are isolated:
    a missing or malformed file only blanks out its own section (with an
    explicit ``*_available: False`` / error string), it never prevents the
    rest of the analysis (or the sweep-status section) from rendering.
    Numbers are never recomputed or fabricated here — this function only
    reshapes what Phase 16 already wrote into a shape the UI can render,
    plus fixed plain-language copy that does not depend on the data.
    """
    aggregate = analysis_loader.load_aggregate_metrics(results_dir)
    fold_metrics = analysis_loader.load_fold_metrics(results_dir)
    per_class = analysis_loader.load_per_class_metrics(results_dir)
    sensitivity = analysis_loader.load_anomaly_sensitivity(results_dir)
    runtime = analysis_loader.load_runtime_analysis(results_dir)
    stats_tests = analysis_loader.load_statistical_tests(results_dir)
    plots = analysis_loader.available_plots(results_dir)

    return {
        "model_order": list(MODEL_ORDER),
        "model_order_note": (
            "Models are listed in experiment/configuration order, not "
            "ranked by performance."
        ),
        "aggregate_metrics": aggregate.data,
        "aggregate_metrics_available": aggregate.ok,
        "aggregate_metrics_error": aggregate.error,
        "fold_metrics": (fold_metrics.data or {}).get("rows") if fold_metrics.ok else None,
        "fold_metrics_available": fold_metrics.ok,
        "fold_metrics_error": fold_metrics.error,
        "per_class_metrics": (per_class.data or {}).get("rows") if per_class.ok else None,
        "per_class_metrics_available": per_class.ok,
        "per_class_metrics_error": per_class.error,
        "anomaly_sensitivity": sensitivity.data,
        "anomaly_sensitivity_available": sensitivity.ok,
        "anomaly_sensitivity_error": sensitivity.error,
        "anomalous_runs": [{"model": m, "fold": f} for m, f in ANOMALOUS_RUNS],
        "runtime_analysis": runtime.data,
        "runtime_analysis_available": runtime.ok,
        "runtime_analysis_error": runtime.error,
        "statistical_tests": stats_tests.data,
        "statistical_tests_available": stats_tests.ok,
        "statistical_tests_error": stats_tests.error,
        "available_plots": plots,
        "plots_complete": len(plots) == len(analysis_loader.KNOWN_PLOTS),
    }


def build_live_demo_state(results_dir: Path) -> dict[str, Any]:
    """Whether real, loadable model checkpoints exist.

    ``models_dir`` is a sibling of ``results_dir`` in the project layout
    (``<project>/models``, ``<project>/results``) — the dashboard is
    normally launched with ``--results-dir results`` from the project
    root, so ``results_dir.parent / "models"`` is the same ``models/``
    directory the training pipeline would ever write a checkpoint into.
    This function only checks for file existence; it never loads a model
    or performs inference.
    """
    models_dir = results_dir.parent / "models"
    checkpoint_files: list[str] = []
    if models_dir.is_dir():
        checkpoint_files = [str(p.relative_to(models_dir)) for p in models_dir.rglob("*") if p.is_file()]

    available = len(checkpoint_files) > 0
    return {
        "checkpoint_available": available,
        "checkpoint_files": checkpoint_files,
        "message": (
            "Live inference is available."
            if available
            else "Live inference is not available yet."
        ),
        "explanation": (
            "A model checkpoint was found and could be loaded for inference."
            if available
            else (
                "The experiments produced evaluation results and predictions, "
                "but the trained model weights were not persisted for later "
                "inference. This dashboard is prepared to support live "
                "inference once a checkpoint is added."
            )
        ),
    }


def build_dashboard_state(
    results_dir: Path, previous_state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Top-level entry point: read the manifest and assemble full state.

    If the manifest is currently unreadable (missing or malformed —
    e.g. the sweep process is mid-write), the previous valid state is
    returned unchanged with a ``manifest_stale`` flag set, instead of
    raising or presenting empty/broken data. If there is no previous state
    yet, an explicit "waiting for manifest" placeholder is returned.
    """
    manifest_result = data_loader.load_manifest(results_dir)

    # Independent of manifest freshness: these read their own files under
    # results/analysis/ and results/models/, and degrade on their own.
    analysis = build_analysis_state(results_dir)
    live_demo = build_live_demo_state(results_dir)

    if not manifest_result.ok:
        if previous_state is not None:
            stale = dict(previous_state)
            stale["manifest_stale"] = True
            stale["manifest_error"] = manifest_result.error
            stale["analysis"] = analysis
            stale["live_demo"] = live_demo
            return stale
        return {
            "manifest_stale": True,
            "manifest_error": manifest_result.error,
            "overview": None,
            "matrix": [],
            "completed_runs": [],
            "analysis": analysis,
            "live_demo": live_demo,
        }

    manifest = manifest_result.data
    state = {
        "manifest_stale": False,
        "manifest_error": None,
        "manifest_created_at": manifest.get("created_at"),
        "overview": build_overview(manifest),
        "matrix": build_matrix(manifest),
        "completed_runs": build_completed_runs_table(results_dir, manifest),
        "models": manifest.get("models", []) or [],
        "folds": manifest.get("folds", []) or [],
        "analysis": analysis,
        "live_demo": live_demo,
    }
    return state
