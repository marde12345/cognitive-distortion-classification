"""Train and persist ONE explicitly selected demo checkpoint.

This is a Phase 18 addition, separate from the Phase 13/15 sweep path.
It does not modify, rerun, or duplicate the sweep: it calls the same,
unmodified (except for one opt-in `checkpoint_path` parameter -- see
src/finetune/trainer.py) `train_one_fold` function used by the sweep, with
the same configuration a sweep run for the same model/fold would use, and
writes a persistent checkpoint that nothing else in the pipeline produces.

Refuses to overwrite an existing checkpoint directory -- if one is already
present, this script stops and reports the conflict rather than silently
retraining over it.

Usage:
    uv run python scripts/train_demo_checkpoint.py \
        --model indobert_base_p1 --fold fold_0

Defaults to --model indobert_base_p1 --fold fold_0, the explicitly
selected Phase 18 demo checkpoint (see docs/phases/PHASE_18_*.md for why).
This script does not select "the best model" -- it trains whatever
--model/--fold you give it.
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

DEMO_CHECKPOINT_ROOT = os.path.join("models", "demo")


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="indobert_base_p1")
    parser.add_argument("--fold", default="fold_0")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name, fold_name = args.model, args.fold
    checkpoint_dir = os.path.join(DEMO_CHECKPOINT_ROOT, model_name, fold_name)

    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
        print(f"STOP: checkpoint directory already exists and is not empty: {checkpoint_dir}")
        print("Refusing to overwrite an existing checkpoint. Remove it explicitly first if retraining is intended.")
        sys.exit(1)

    device = resolve_device()
    cu.set_seed(42)
    paths = cu.get_paths()
    cfg = cu.load_config()
    model_cfg = cu.load_config(model_name)
    tr_cfg = model_cfg["training"]

    print("=" * 60)
    print("Demo checkpoint training (Phase 18)")
    print("=" * 60)
    print(f"Model: {model_name}  (explicitly selected demo checkpoint -- not an automatic 'best model' choice)")
    print(f"HF Model: {model_cfg['hf_model_id']}")
    print(f"Fold: {fold_name}")
    print(f"Device: {device}")
    print(f"Checkpoint will be written to: {checkpoint_dir}")
    print("=" * 60)

    folds_data = loader.load_folds(paths, verbose=False)
    df = loader.load_dataset(cfg, paths)
    train_df, val_df, test_df = loader.get_fold_data(df, folds_data, fold_name, cfg)
    print(f"Dataset loaded: {len(df)} rows. Fold split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    t0 = time.time()
    val_metrics, test_metrics, test_preds, test_probs, test_labels = tr.train_one_fold(
        model_name=model_name,
        hf_model_id=model_cfg["hf_model_id"],
        train_df=train_df, val_df=val_df, test_df=test_df,
        cfg=cfg, paths=paths,
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
        fold_name=fold_name,
        checkpoint_path=checkpoint_dir,
    )
    elapsed = time.time() - t0

    print()
    print(f"Training completed in {elapsed:.2f}s")
    print(f"val macro_f1  = {val_metrics['macro_f1']:.4f}")
    print(f"test macro_f1 = {test_metrics['macro_f1']:.4f}")
    print(f"Checkpoint written: {checkpoint_dir}/model.pt, {checkpoint_dir}/checkpoint_meta.json")

    # Note: this intentionally does NOT call
    # tr.save_predictions_and_metrics() or write anywhere under results/ --
    # this script must not create or overwrite any Phase 13/15 experiment
    # artifact. The demo checkpoint's own evaluation numbers are recorded
    # only inside checkpoint_meta.json, alongside the weights.


if __name__ == "__main__":
    main()
