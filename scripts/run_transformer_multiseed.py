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


def run_one(model, fold, seed):
    cmd = [
        "uv", "run", "python", "scripts/run_transformer_multiseed_single.py",
        "--model", model, "--fold", fold, "--seed", str(seed),
    ]
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
        sys.stdout.flush()
    proc.wait()
    elapsed = time.time() - t0
    print("-" * 60)
    return proc.returncode, elapsed


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

    for i, (model, fold, seed) in enumerate(combos, start=1):
        already_done = is_complete(model, fold, seed)
        print(f"Progress: {i}/{len(combos)}")
        print(f"Model: {model}  Fold: {fold}  Seed: {seed}")

        if already_done:
            print("Status: SKIP — already completed")
            print("=" * 60)
            print()
            state["skipped_runs"] += 1
            state["runs"].append({
                "model": model, "fold": fold, "seed": seed, "status": "skipped",
                "start_time": None, "end_time": None,
                "elapsed_seconds": None, "exit_code": None,
            })
            save_run_state(state)
            continue

        print("Status: RUNNING")
        print("-" * 60)
        start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        exit_code, elapsed = run_one(model, fold, seed)
        end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status = "completed" if exit_code == 0 else "failed"

        print(f"Status: {'SUCCESS' if exit_code == 0 else 'FAILED'}")
        print(f"Elapsed: {elapsed:.2f}s")
        print("=" * 60)
        print()

        state["runs"].append({
            "model": model, "fold": fold, "seed": seed, "status": status,
            "start_time": start_time, "end_time": end_time,
            "elapsed_seconds": round(elapsed, 2), "exit_code": exit_code,
        })
        if status == "completed":
            state["completed_runs"] += 1
        else:
            state["failed_runs"] += 1
        save_run_state(state)

        if status == "failed":
            print(f"MULTI-SEED SWEEP STOPPED: {model} / {fold} / seed_{seed} failed (exit code {exit_code}).")
            print(f"Run state: {RUN_STATE_PATH}")
            sys.exit(1)

    print("MULTI-SEED SWEEP COMPLETE.")
    print(f"Run state: {RUN_STATE_PATH}")


if __name__ == "__main__":
    main()
