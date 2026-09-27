"""Run a single transformer fine-tuning experiment for one model x one fold
x one explicit seed, for the Phase 20 multi-seed protocol.

This is a sibling of scripts/run_transformer.py, not a replacement for it.
scripts/run_transformer.py (Phase 12/13/15) is left completely untouched and
keeps writing to results/metrics/<model>/<fold>.json and
results/logs/<model>/<fold>.json exactly as before -- this script never
writes there.

This script writes to a separate namespace so a (model, fold, seed) triple
can never be confused with, or overwrite, a Phase 13/15 (model, fold) run:

    results/multiseed/metrics/<model>/<fold>__seed_<seed>.json
    results/multiseed/predictions/<model>/<fold>__seed_<seed>.csv
    results/multiseed/logs/<model>/<fold>__seed_<seed>.json

It reuses finetune.trainer.train_one_fold and
finetune.trainer.save_predictions_and_metrics unmodified -- the only
difference from run_transformer.py is (a) --seed is required and explicitly
passed to config_utils.set_seed instead of a hardcoded 42, and (b) the
`paths["results_root"]` passed to the trainer's save function points at
`results/multiseed` instead of `results`, so nothing under `results/metrics`,
`results/logs`, or `results/predictions` (the Phase 13/15 namespace) is ever
touched.

Usage:
    uv run python scripts/run_transformer_multiseed_single.py \\
        --model indobert_base_p1 --fold fold_0 --seed 43

--model, --fold, and --seed are all required; there is no hidden default
seed and no "run everything" mode in this script (that is
scripts/run_transformer_multiseed.py's job).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

import config_utils as cu
import loader
import finetune.trainer as tr

MULTISEED_ROOT = os.path.join("results", "multiseed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model_name from configs/models/*.yaml")
    parser.add_argument("--fold", required=True, help="fold name, e.g. fold_0")
    parser.add_argument("--seed", required=True, type=int, help="explicit random seed, e.g. 42/43/44")
    return parser.parse_args()


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_output_name(fold_name, seed):
    """Unambiguous identity for one (fold, seed) combination on disk."""
    return f"{fold_name}__seed_{seed}"


def write_execution_log(model_name, run_name, log_entry):
    log_dir = os.path.join(MULTISEED_ROOT, "logs", model_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.json")
    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)
    return log_path


def main():
    args = parse_args()
    model_name = args.model
    fold_name = args.fold
    seed = args.seed
    device = resolve_device()
    run_name = run_output_name(fold_name, seed)

    cu.set_seed(seed)
    paths = cu.get_paths()
    multiseed_paths = dict(paths)
    multiseed_paths["results_root"] = MULTISEED_ROOT

    cfg = cu.load_config()
    model_cfg = cu.load_config(model_name)
    tr_cfg = model_cfg["training"]

    print("=" * 60)
    print("Transformer Execution (multi-seed)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"HF Model: {model_cfg['hf_model_id']}")
    print(f"Fold: {fold_name}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Output namespace: {MULTISEED_ROOT}")
    print("=" * 60)
    print()

    print("Loading dataset...")
    folds_data = loader.load_folds(paths, verbose=False)
    df = loader.load_dataset(cfg, paths)
    print(f"Dataset loaded: {len(df)} rows")

    train_df, val_df, test_df = loader.get_fold_data(df, folds_data, fold_name, cfg)
    print(f"Fold split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print()

    log_entry = {
        "model_name": model_name,
        "hf_model_id": model_cfg["hf_model_id"],
        "fold": fold_name,
        "seed": seed,
        "device": device,
        "configured_epochs": tr_cfg["num_epochs"],
        "batch_size": tr_cfg["batch_size"],
        "max_seq_length": cfg["data"]["max_seq_length"],
        "status": "running",
        "start_time": None,
        "end_time": None,
        "elapsed_seconds": None,
        "completed_epochs": None,
        "history": None,
    }
    log_entry["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print("Starting training...")
    t0 = time.time()
    try:
        val_metrics, test_metrics, test_preds, test_probs, test_labels = tr.train_one_fold(
            model_name=model_name,
            hf_model_id=model_cfg["hf_model_id"],
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            cfg=cfg,
            paths=paths,
            num_labels=cfg["data"]["num_labels"],
            max_len=cfg["data"]["max_seq_length"],
            batch_size=tr_cfg["batch_size"],
            learning_rate=tr_cfg["learning_rate"],
            weight_decay=tr_cfg["weight_decay"],
            warmup_ratio=tr_cfg["warmup_ratio"],
            num_epochs=tr_cfg["num_epochs"],
            patience=cfg["training"]["early_stopping_patience"],
            fp16=cfg["training"]["fp16"],
            device=device,
            fold_name=run_name,
        )
    except Exception as e:
        elapsed = time.time() - t0
        log_entry["status"] = "failed"
        log_entry["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log_entry["elapsed_seconds"] = round(elapsed, 2)
        log_entry["error"] = f"{type(e).__name__}: {e}"
        write_execution_log(model_name, run_name, log_entry)
        raise

    elapsed = time.time() - t0
    print()
    print("Training completed")
    print(f"Elapsed: {elapsed:.2f}s")
    print()

    print("Saving predictions...")
    print("Saving metrics...")
    tr.save_predictions_and_metrics(
        model_name=model_name,
        fold_name=run_name,
        test_df=test_df,
        test_preds=test_preds,
        test_probs=test_probs,
        test_metrics=test_metrics,
        cfg=cfg,
        paths=multiseed_paths,
    )

    history = test_metrics.get("history", [])
    log_entry["status"] = "completed"
    log_entry["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log_entry["elapsed_seconds"] = round(elapsed, 2)
    log_entry["completed_epochs"] = len(history)
    log_entry["history"] = [
        {"epoch": h["epoch"], "train_loss": h["train_loss"], "val_macro_f1": h["val_macro_f1"]}
        for h in history
    ]
    log_path = write_execution_log(model_name, run_name, log_entry)
    print(f"Execution log: {log_path}")

    print()
    print(f"test macro_f1 = {test_metrics['macro_f1']:.4f}")
    print(f"val macro_f1  = {val_metrics['macro_f1']:.4f}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
