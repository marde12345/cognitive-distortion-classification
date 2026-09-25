# NOTE:
# Original source for this module was not recoverable from Git history, the
# Git object database, the local filesystem, the notebook, or the Google
# Drive source tree (see docs/reconnaissance.md and the Phase 4A forensic
# report for the full investigation trail).
#
# This implementation is a forensic reconstruction based on:
# 1. Structural evidence recovered from the surviving orphaned CPython 3.13
#    bytecode artifact:
#    external/original-drive/MODELING/COGNITIVE DISTORTION/src/__pycache__/metrics.cpython-313.pyc
#    (exact signatures, imports, local variable names, keyword-argument
#    names, and output dictionary keys were recovered via marshal-level
#    code object inspection; opcode-level instruction order could not be
#    reliably read due to a 3.12/3.13 interpreter mismatch).
# 2. The verified caller contract in src/finetune/trainer.py and
#    src/baselines/sklearn_baseline.py.
#
# It should NOT be treated as the original source file. Behavior is
# believed to match the original for every field the real callers consume,
# but exact expression structure, whitespace, comments, and one default
# argument value (see average_metrics) could not be confirmed.

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)


def compute_metrics(y_true, y_pred, num_labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    labels_all = list(range(num_labels))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_all, zero_division=0,
    )

    per_class = {}
    for i, lbl in enumerate(labels_all):
        per_class[str(lbl)] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(acc),
        "per_class": per_class,
    }


def average_metrics(fold_metrics, num_labels):
    """Rata-rata metrik antar fold."""
    macro_f1s = [m["macro_f1"] for m in fold_metrics]
    weighted_f1s = [m["weighted_f1"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]

    per_class_avg = {}
    per_class_std = {}
    for lbl in range(num_labels):
        f1s = [m["per_class"][str(lbl)]["f1"] for m in fold_metrics]
        per_class_avg[str(lbl)] = float(np.mean(f1s))
        per_class_std[str(lbl)] = float(np.std(f1s))

    return {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "per_class_f1_mean": per_class_avg,
        "per_class_f1_std": per_class_std,
        "n_folds": len(fold_metrics),
    }
