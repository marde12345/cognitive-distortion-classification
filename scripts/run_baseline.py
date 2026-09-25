"""Run a single baseline sanity experiment locally.

Minimal entry point mirroring the original notebook cells (14/17/20/22):
set seed, load config/paths, load dataset/folds, call the requested
baselines.sklearn_baseline.run_* function, print the summary.

Usage:
    uv run python scripts/run_baseline.py majority_class
    uv run python scripts/run_baseline.py tfidf_lr
    uv run python scripts/run_baseline.py tfidf_svm
    uv run python scripts/run_baseline.py svm_word2vec
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config_utils as cu
import loader
import baselines.sklearn_baseline as sb

RUNNERS = {
    "majority_class": sb.run_majority_class,
    "tfidf_lr": sb.run_tfidf_lr,
    "tfidf_svm": sb.run_tfidf_svm,
    "svm_word2vec": sb.run_svm_word2vec,
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in RUNNERS:
        print(f"Usage: uv run python scripts/run_baseline.py <{'|'.join(RUNNERS)}>")
        sys.exit(1)

    name = sys.argv[1]
    cu.set_seed(42)
    paths = cu.get_paths()
    cfg = cu.load_config()
    folds_data = loader.load_folds(paths, verbose=False)
    df = loader.load_dataset(cfg, paths)

    print(f"Dataset rows: {len(df)}")
    print(f"Folds: {len(folds_data['folds'])}")

    t0 = time.time()
    summary = RUNNERS[name](df, folds_data, cfg, paths)
    elapsed = time.time() - t0

    print(f"\n{name} summary:")
    print(f"  macro_f1_mean = {summary['macro_f1_mean']:.6f} +/- {summary['macro_f1_std']:.6f}")
    print(f"  weighted_f1_mean = {summary['weighted_f1_mean']:.6f} +/- {summary['weighted_f1_std']:.6f}")
    print(f"  accuracy_mean = {summary['accuracy_mean']:.6f} +/- {summary['accuracy_std']:.6f}")
    print(f"  n_folds = {summary['n_folds']}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
