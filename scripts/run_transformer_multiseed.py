"""Resumable orchestration layer for the Phase 20 multi-seed transformer
experiment: 7 models x 5 folds x 3 seeds (42, 43, 44) = 105 runs.

This mirrors scripts/run_transformer_sweep.py's architecture exactly
(discover -> check completion -> invoke single-run subprocess -> record to
a resumable JSON manifest -> fail-fast on first failure), extended with an
explicit seed axis. It does NOT duplicate any training logic, and it never
touches the Phase 13/15 single-seed namespace
(results/metrics|logs|predictions/<model>/<fold>.*) -- every run here goes
through scripts/run_transformer_multiseed_single.py, which writes only
under results/multiseed/.

Usage:
    uv run python scripts/run_transformer_multiseed.py --dry-run
        Enumerate all 105 planned (model, fold, seed) combinations and
        which would be skipped as already-complete. Executes nothing,
        writes nothing.

    uv run python scripts/run_transformer_multiseed.py
        Run every not-yet-complete (model, fold, seed) combination, one at
        a time, stopping immediately on the first failure.

IMPORTANT: as of Phase 20, this script is execution-readiness infrastructure
only. Running it without --dry-run starts real training; Phase 20 itself
does not do this.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

import config_utils as cu
import loader

SEEDS = [42, 43, 44]
MULTISEED_ROOT = os.path.join("results", "multiseed")
RUN_STATE_PATH = os.path.join(MULTISEED_ROOT, "logs", "multiseed_sweep.json")
LIVE_STATE_PATH = os.path.join(MULTISEED_ROOT, "live_state.json")

# How often the on-disk live_state.json is refreshed while a run is in
# flight (cheap: a handful of small numbers), vs. how often a progress
# block is actually printed to the terminal (coarser, so a multi-hour
# run doesn't flood the terminal with a full block every couple seconds).
LIVE_STATE_WRITE_INTERVAL_SECONDS = 3
TERMINAL_PRINT_INTERVAL_SECONDS = 30


def discover_transformer_models():
    import yaml

    models = []
    for fname in sorted(os.listdir(cu.MODELS_CONFIG_DIR)):
        if not fname.endswith(".yaml"):
            continue
        path = os.path.join(cu.MODELS_CONFIG_DIR, fname)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        cfg = yaml.safe_load(raw) or {}
        if cfg.get("model_type") != "transformer":
            continue
        if "PLACEHOLDER" in raw:
            continue
        models.append(cfg["model_name"])
    return models


def discover_folds():
    paths = cu.get_paths()
    folds_data = loader.load_folds(paths, verbose=False)
    return sorted(folds_data["folds"].keys())


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def all_combinations(models, folds, seeds):
    """Every (model, fold, seed) triple, in a fixed, deterministic order."""
    return [(m, f, s) for m in models for f in folds for s in seeds]


def run_name(fold, seed):
    return f"{fold}__seed_{seed}"


def is_complete(model, fold, seed):
    name = run_name(fold, seed)
    metrics_path = os.path.join(MULTISEED_ROOT, "metrics", model, f"{name}.json")
    log_path = os.path.join(MULTISEED_ROOT, "logs", model, f"{name}.json")
    if not os.path.exists(metrics_path) or not os.path.exists(log_path):
        return False
    try:
        with open(log_path) as f:
            log = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return log.get("status") == "completed"


def load_run_state():
    if os.path.exists(RUN_STATE_PATH):
        with open(RUN_STATE_PATH) as f:
            return json.load(f)
    return None


def new_run_state(models, folds, seeds, device):
    return {
        "experiment_id": "phase20_multiseed_transformer_sweep",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "models": models,
        "folds": folds,
        "seeds": seeds,
        "total_runs": len(models) * len(folds) * len(seeds),
        "completed_runs": 0,
        "failed_runs": 0,
        "skipped_runs": 0,
        "runs": [],
    }


def save_run_state(state):
    os.makedirs(os.path.dirname(RUN_STATE_PATH), exist_ok=True)
    with open(RUN_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def load_live_state():
    if os.path.exists(LIVE_STATE_PATH):
        try:
            with open(LIVE_STATE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def new_live_state(experiment_id, total_expected_runs):
    return {
        "experiment_id": experiment_id,
        "total_expected_runs": total_expected_runs,
        "completed_runs": 0,
        "failed_runs": 0,
        "skipped_runs": 0,
        "current_run": None,
        "last_completed_run": None,
        "updated_at": None,
        "overall_status": "not_run",
    }


def _compute_overall_status(live_state):
    total = live_state.get("total_expected_runs")
    completed = live_state.get("completed_runs", 0)
    if live_state.get("current_run") is not None:
        return "in_progress"
    if total is not None and completed == total and total > 0:
        return "complete"
    if completed == 0 and live_state.get("failed_runs", 0) == 0 and live_state.get("skipped_runs", 0) == 0:
        return "not_run"
    return "in_progress"


def save_live_state(live_state):
    """Atomic write (tmp file + rename) so a concurrent dashboard read never
    sees a half-written JSON file."""
    live_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    live_state["overall_status"] = _compute_overall_status(live_state)
    os.makedirs(os.path.dirname(LIVE_STATE_PATH), exist_ok=True)
    tmp_path = LIVE_STATE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(live_state, f, indent=2)
    os.replace(tmp_path, LIVE_STATE_PATH)


def read_run_metrics(model, fold, seed):
    name = run_name(fold, seed)
    metrics_path = os.path.join(MULTISEED_ROOT, "metrics", model, f"{name}.json")
    try:
        with open(metrics_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return {
        "macro_f1": data.get("macro_f1"),
        "accuracy": data.get("accuracy"),
        "weighted_f1": data.get("weighted_f1"),
    }


def print_progress_block(index, total, model, fold, seed, elapsed, live_state, running=True):
    completed = live_state.get("completed_runs", 0)
    failed = live_state.get("failed_runs", 0)
    remaining = max(total - index + (0 if running else 1), 0)

    print()
    print(f"[{index}/{total}] {'RUNNING' if running else 'DONE'}")
    print(f"model={model}")
    print(f"fold={fold}")
    print(f"seed={seed}")
    if running:
        mins, secs = divmod(int(elapsed), 60)
        print(f"elapsed={mins:02d}:{secs:02d}")
    print()
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Remaining: {remaining}")

    last = live_state.get("last_completed_run")
    if last is not None:
        print()
        print("Last completed:")
        print(f"model={last['model']}")
        print(f"fold={last['fold']}")
        print(f"seed={last['seed']}")
        if last.get("macro_f1") is not None:
            print(f"macro_f1={last['macro_f1']:.4f}")
            print(f"accuracy={last['accuracy']:.4f}")
            print(f"weighted_f1={last['weighted_f1']:.4f}")
    sys.stdout.flush()


def run_one(model, fold, seed, index, total, live_state):
    """Run one (model, fold, seed) combination as a subprocess.

    Raw child stdout/stderr is redirected to a per-run log file under
    results/multiseed/logs/<model>/ (detailed record, never printed line
    by line to the parent terminal), while the parent terminal only gets
    a concise, periodically-refreshed progress block. results/multiseed/
    live_state.json is refreshed on the same cadence so the dashboard's
    polling always reflects genuinely current progress, not a stale
    snapshot from before this run started.
    """
    log_dir = os.path.join(MULTISEED_ROOT, "logs", model)
    os.makedirs(log_dir, exist_ok=True)
    raw_log_path = os.path.join(log_dir, f"{run_name(fold, seed)}.stdout.log")

    cmd = [
        "uv", "run", "python", "scripts/run_transformer_multiseed_single.py",
        "--model", model, "--fold", fold, "--seed", str(seed),
    ]

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    live_state["current_run"] = {
        "model": model, "fold": fold, "seed": seed,
        "started_at": started_at, "elapsed_seconds": 0,
    }
    save_live_state(live_state)
    print_progress_block(index, total, model, fold, seed, 0, live_state, running=True)

    t0 = time.time()
    last_file_write = t0
    last_terminal_print = t0
    with open(raw_log_path, "w") as raw_log:
        proc = subprocess.Popen(cmd, stdout=raw_log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(1)
            now = time.time()
            elapsed = now - t0
            if now - last_file_write >= LIVE_STATE_WRITE_INTERVAL_SECONDS:
                live_state["current_run"]["elapsed_seconds"] = round(elapsed, 1)
                save_live_state(live_state)
                last_file_write = now
            if now - last_terminal_print >= TERMINAL_PRINT_INTERVAL_SECONDS:
                print_progress_block(index, total, model, fold, seed, elapsed, live_state, running=True)
                last_terminal_print = now
        proc.wait()
    elapsed = time.time() - t0
    return proc.returncode, elapsed, raw_log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                         help="enumerate the planned matrix and skip/run decisions; execute nothing")
    args = parser.parse_args()

    models = discover_transformer_models()
    folds = discover_folds()
    device = resolve_device()
    combos = all_combinations(models, folds, SEEDS)

    print("=" * 60)
    print("MULTI-SEED TRANSFORMER EXPERIMENT (Phase 20)")
    print("=" * 60)
    print(f"Discovered models ({len(models)}): {models}")
    print(f"Discovered folds ({len(folds)}): {folds}")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Device: {device}")
    print(f"Total planned runs: {len(combos)}")
    print(f"Mode: {'DRY RUN (no execution)' if args.dry_run else 'EXECUTE'}")
    print("=" * 60)
    print()

    if args.dry_run:
        for i, (model, fold, seed) in enumerate(combos, start=1):
            status = "SKIP (already complete)" if is_complete(model, fold, seed) else "WOULD RUN"
            print(f"{i:3d}/{len(combos)}  {model} / {fold} / seed_{seed}  -> {status}")
        print()
        print(f"DRY RUN COMPLETE — {len(combos)} unique (model, fold, seed) combinations enumerated. "
              "No training was executed, no files were written.")
        return

    state = load_run_state()
    if state is None:
        state = new_run_state(models, folds, SEEDS, device)

    live_state = load_live_state()
    if live_state is None:
        live_state = new_live_state(state["experiment_id"], state["total_runs"])
    # Always resync counters from the authoritative run-state on startup
    # (e.g. after a restart), rather than trusting a possibly-stale
    # live_state.json left over from an interrupted process.
    live_state["completed_runs"] = state["completed_runs"]
    live_state["failed_runs"] = state["failed_runs"]
    live_state["skipped_runs"] = state["skipped_runs"]
    live_state["current_run"] = None
    save_live_state(live_state)

    total = len(combos)
    for i, (model, fold, seed) in enumerate(combos, start=1):
        already_done = is_complete(model, fold, seed)

        if already_done:
            state["skipped_runs"] += 1
            state["runs"].append({
                "model": model, "fold": fold, "seed": seed, "status": "skipped",
                "start_time": None, "end_time": None,
                "elapsed_seconds": None, "exit_code": None,
            })
            save_run_state(state)

            live_state["skipped_runs"] = state["skipped_runs"]
            metrics = read_run_metrics(model, fold, seed)
            if metrics is not None:
                live_state["last_completed_run"] = {
                    "model": model, "fold": fold, "seed": seed,
                    "completed_at": None, **metrics,
                }
            save_live_state(live_state)
            continue

        start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        exit_code, elapsed, raw_log_path = run_one(model, fold, seed, i, total, live_state)
        end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status = "completed" if exit_code == 0 else "failed"

        state["runs"].append({
            "model": model, "fold": fold, "seed": seed, "status": status,
            "start_time": start_time, "end_time": end_time,
            "elapsed_seconds": round(elapsed, 2), "exit_code": exit_code,
            "raw_log": raw_log_path,
        })
        if status == "completed":
            state["completed_runs"] += 1
        else:
            state["failed_runs"] += 1
        save_run_state(state)

        live_state["current_run"] = None
        live_state["completed_runs"] = state["completed_runs"]
        live_state["failed_runs"] = state["failed_runs"]
        if status == "completed":
            metrics = read_run_metrics(model, fold, seed) or {}
            live_state["last_completed_run"] = {
                "model": model, "fold": fold, "seed": seed,
                "completed_at": end_time, **metrics,
            }
        save_live_state(live_state)
        print_progress_block(i, total, model, fold, seed, elapsed, live_state, running=False)

        if status == "failed":
            print(f"MULTI-SEED SWEEP STOPPED: {model} / {fold} / seed_{seed} failed (exit code {exit_code}).")
            print(f"Run state: {RUN_STATE_PATH}")
            print(f"Raw log: {raw_log_path}")
            sys.exit(1)

    print("MULTI-SEED SWEEP COMPLETE.")
    print(f"Run state: {RUN_STATE_PATH}")


if __name__ == "__main__":
    main()
