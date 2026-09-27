"""Phase 19: build a read-only model-comparison package from existing
Phase 16 analysis artifacts.

This script does NOT retrain anything, does NOT touch the dataset, and
does NOT modify any existing artifact under results/analysis/transformer/.
It only reads results/analysis/transformer/aggregate_metrics.json and
results/analysis/transformer/anomaly_sensitivity.json (both produced by
Phase 16's scripts/analyze_transformer_sweep.py) and writes new files
under results/analysis/transformer/comparison/.

No ranking, scoring, or "best model" decision is computed or written by
this script. It reports descriptive statistics only.
"""

from __future__ import annotations

import json
import os

MODEL_ORDER = [
    "indobert_15g",
    "indobert_base_p1",
    "indobertweet",
    "indoroberta_15g",
    "mbert",
    "nusabert",
    "xlmr",
]

ANOMALOUS_FOLDS = {
    "indobertweet": ["fold_1"],
    "mbert": ["fold_3"],
}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS_DIR = os.path.join(ROOT, "results", "analysis", "transformer")
OUT_DIR = os.path.join(ANALYSIS_DIR, "comparison")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_rows(aggregate):
    rows = []
    for model in MODEL_ORDER:
        m = aggregate["models"][model]
        anomalous = ANOMALOUS_FOLDS.get(model, [])
        rows.append({
            "model": model,
            "macro_f1_mean": m["macro_f1"]["mean"],
            "macro_f1_std_ddof1": m["macro_f1"]["std"],
            "macro_f1_min": m["macro_f1"]["min"],
            "macro_f1_max": m["macro_f1"]["max"],
            "macro_f1_median": m["macro_f1"]["median"],
            "macro_f1_range": m["macro_f1"]["range"],
            "macro_f1_cv": m["macro_f1"]["cv"],
            "accuracy_mean": m["accuracy"]["mean"],
            "accuracy_std_ddof1": m["accuracy"]["std"],
            "weighted_f1_mean": m["weighted_f1"]["mean"],
            "weighted_f1_std_ddof1": m["weighted_f1"]["std"],
            "n_folds": m["macro_f1"]["n"],
            "n_anomalous_folds": len(anomalous),
            "anomalous_fold_names": ";".join(anomalous),
        })
    return rows


def write_csv(rows, path):
    fields = list(rows[0].keys())
    lines = [",".join(fields)]
    for r in rows:
        lines.append(",".join(str(r[f]) for f in fields))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    aggregate = load_json(os.path.join(ANALYSIS_DIR, "aggregate_metrics.json"))
    sensitivity = load_json(os.path.join(ANALYSIS_DIR, "anomaly_sensitivity.json"))

    rows = build_rows(aggregate)

    os.makedirs(OUT_DIR, exist_ok=True)

    write_csv(rows, os.path.join(OUT_DIR, "model_comparison.csv"))

    output = {
        "note": (
            "Official per-model aggregates are the full 5-fold statistics "
            "from Phase 16 (results/analysis/transformer/aggregate_metrics.json). "
            "This file does not replace or recompute them. No automatic "
            "model selection was performed."
        ),
        "source_artifacts": [
            "results/analysis/transformer/aggregate_metrics.json",
            "results/analysis/transformer/anomaly_sensitivity.json",
            "results/analysis/transformer/statistical_tests.json",
        ],
        "official_comparison_matrix": rows,
        "sensitivity_analysis": {
            model_fold: {
                "label": data["label"],
                "official_5fold_macro_f1_mean": data["official_5fold_macro_f1"]["mean"],
                "sensitivity_4fold_macro_f1_mean_excluding_anomaly": (
                    data["sensitivity_4fold_macro_f1_EXCLUDING_ANOMALY"]["mean"]
                ),
                "delta_macro_f1_mean_excl_minus_incl": data["delta_macro_f1_mean_excl_minus_incl"],
            }
            for model_fold, data in sensitivity.items()
        },
    }
    with open(os.path.join(OUT_DIR, "model_comparison.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {os.path.join(OUT_DIR, 'model_comparison.csv')}")
    print(f"Wrote {os.path.join(OUT_DIR, 'model_comparison.json')}")


if __name__ == "__main__":
    main()
