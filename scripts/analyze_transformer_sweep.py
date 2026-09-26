"""Read-only statistical analysis of the completed transformer sweep.

Reads only existing artifacts under results/{metrics,logs,predictions}/ and
external/original-drive/DATASETS/ (dataset, read-only, for fold sizes).
Writes analysis artifacts under results/analysis/transformer/ only.

Does not modify any existing result file, the dataset, the notebook, or
any source file. Does not train or retrain anything.
"""
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from scipy import stats

import config_utils as cu
import loader

MODELS = ["indobert_15g", "indobert_base_p1", "indobertweet",
          "indoroberta_15g", "mbert", "nusabert", "xlmr"]
FOLDS = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]
ANOMALOUS = {("indobertweet", "fold_1"), ("mbert", "fold_3")}

OUT_DIR = os.path.join("results", "analysis", "transformer")
PLOT_DIR = os.path.join(OUT_DIR, "plots")


def load_all():
    metrics, logs, pred_counts = {}, {}, {}
    for m in MODELS:
        metrics[m], logs[m], pred_counts[m] = {}, {}, {}
        for f in FOLDS:
            with open(f"results/metrics/{m}/{f}.json") as fh:
                metrics[m][f] = json.load(fh)
            with open(f"results/logs/{m}/{f}.json") as fh:
                logs[m][f] = json.load(fh)
            with open(f"results/predictions/{m}/{f}.csv") as fh:
                pred_counts[m][f] = sum(1 for _ in csv.DictReader(fh))
    return metrics, logs, pred_counts


def validate(metrics, logs, pred_counts, expected_test_sizes):
    problems = []
    for m in MODELS:
        for f in FOLDS:
            if logs[m][f].get("status") != "completed":
                problems.append(f"{m}/{f}: status != completed")
            for key in ("macro_f1", "weighted_f1", "accuracy", "per_class", "history"):
                if key not in metrics[m][f]:
                    problems.append(f"{m}/{f}: missing key {key}")
            if pred_counts[m][f] != expected_test_sizes[f]:
                problems.append(
                    f"{m}/{f}: prediction rows {pred_counts[m][f]} != expected {expected_test_sizes[f]}"
                )
    return problems


def describe(values, ddof=1):
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=ddof)) if len(arr) > 1 else 0.0,
        "std_ddof": ddof,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "cv": (float(np.std(arr, ddof=ddof) / np.mean(arr)) if len(arr) > 1 and np.mean(arr) != 0 else None),
        "n": len(arr),
    }


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    cu.set_seed(42)
    paths = cu.get_paths()
    cfg = cu.load_config()
    folds_data = loader.load_folds(paths, verbose=False)
    df = loader.load_dataset(cfg, paths)
    expected_test_sizes = {}
    for f in FOLDS:
        _, _, test_df = loader.get_fold_data(df, folds_data, f, cfg)
        expected_test_sizes[f] = len(test_df)

    metrics, logs, pred_counts = load_all()
    problems = validate(metrics, logs, pred_counts, expected_test_sizes)
    if problems:
        print("VALIDATION PROBLEMS FOUND:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print(f"Validation OK: all {len(MODELS)}x{len(FOLDS)} combinations structurally valid.")

    # ---- fold_metrics.csv ----
    fold_rows = []
    for m in MODELS:
        for f in FOLDS:
            d = metrics[m][f]
            fold_rows.append({
                "model": m, "fold": f,
                "macro_f1": d["macro_f1"], "weighted_f1": d["weighted_f1"],
                "accuracy": d["accuracy"],
                "anomalous": (m, f) in ANOMALOUS,
                "elapsed_seconds": logs[m][f]["elapsed_seconds"],
                "completed_epochs": logs[m][f]["completed_epochs"],
            })
    with open(os.path.join(OUT_DIR, "fold_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        w.writerows(fold_rows)

    # ---- aggregate_metrics.json (official = all 5 folds, ddof=1) ----
    aggregate = {"ddof": 1, "note": "sample standard deviation (ddof=1) used throughout", "models": {}}
    for m in MODELS:
        macro = [metrics[m][f]["macro_f1"] for f in FOLDS]
        acc = [metrics[m][f]["accuracy"] for f in FOLDS]
        wf1 = [metrics[m][f]["weighted_f1"] for f in FOLDS]
        aggregate["models"][m] = {
            "macro_f1": describe(macro),
            "accuracy": describe(acc),
            "weighted_f1": describe(wf1),
            "per_fold_macro_f1": dict(zip(FOLDS, macro)),
        }
    with open(os.path.join(OUT_DIR, "aggregate_metrics.json"), "w") as fh:
        json.dump(aggregate, fh, indent=2)

    # ---- anomaly_sensitivity.json ----
    sensitivity = {}
    for (m, anomalous_fold) in sorted(ANOMALOUS):
        included_macro = [metrics[m][f]["macro_f1"] for f in FOLDS]
        excluded_folds = [f for f in FOLDS if f != anomalous_fold]
        excluded_macro = [metrics[m][f]["macro_f1"] for f in excluded_folds]
        included_acc = [metrics[m][f]["accuracy"] for f in FOLDS]
        excluded_acc = [metrics[m][f]["accuracy"] for f in excluded_folds]
        included_wf1 = [metrics[m][f]["weighted_f1"] for f in FOLDS]
        excluded_wf1 = [metrics[m][f]["weighted_f1"] for f in excluded_folds]
        sensitivity[f"{m}/{anomalous_fold}"] = {
            "anomalous_run_metrics": metrics[m][anomalous_fold],
            "anomalous_run_history": logs[m][anomalous_fold]["history"],
            "other_folds_macro_f1": {f: metrics[m][f]["macro_f1"] for f in excluded_folds},
            "official_5fold_macro_f1": describe(included_macro),
            "sensitivity_4fold_macro_f1_EXCLUDING_ANOMALY": describe(excluded_macro),
            "delta_macro_f1_mean_excl_minus_incl": describe(excluded_macro)["mean"] - describe(included_macro)["mean"],
            "official_5fold_accuracy": describe(included_acc),
            "sensitivity_4fold_accuracy_EXCLUDING_ANOMALY": describe(excluded_acc),
            "official_5fold_weighted_f1": describe(included_wf1),
            "sensitivity_4fold_weighted_f1_EXCLUDING_ANOMALY": describe(excluded_wf1),
            "label": "Sensitivity analysis - NOT the official result. Official aggregate remains the full 5-fold result.",
        }
    with open(os.path.join(OUT_DIR, "anomaly_sensitivity.json"), "w") as fh:
        json.dump(sensitivity, fh, indent=2)

    # ---- per_class_metrics.csv ----
    num_labels = cfg["data"]["num_labels"]
    per_class_rows = []
    for m in MODELS:
        for lbl in range(num_labels):
            precisions, recalls, f1s, supports = [], [], [], []
            for f in FOLDS:
                pc = metrics[m][f]["per_class"][str(lbl)]
                precisions.append(pc["precision"]); recalls.append(pc["recall"])
                f1s.append(pc["f1"]); supports.append(pc["support"])
            per_class_rows.append({
                "model": m, "label": lbl,
                "mean_precision": float(np.mean(precisions)),
                "mean_recall": float(np.mean(recalls)),
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s, ddof=1)),
                "total_support_across_folds": sum(supports),
                "mean_support_per_fold": float(np.mean(supports)),
            })
    with open(os.path.join(OUT_DIR, "per_class_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_class_rows[0].keys()))
        w.writeheader()
        w.writerows(per_class_rows)

    # ---- statistical_tests.json ----
    macro_matrix = np.array([[metrics[m][f]["macro_f1"] for m in MODELS] for f in FOLDS])  # rows=folds, cols=models
    friedman_stat, friedman_p = stats.friedmanchisquare(*[macro_matrix[:, i] for i in range(len(MODELS))])

    pairwise = {}
    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            if i < j:
                d1 = np.array([metrics[m1][f]["macro_f1"] for f in FOLDS])
                d2 = np.array([metrics[m2][f]["macro_f1"] for f in FOLDS])
                pairwise[f"{m1}_vs_{m2}"] = {
                    "mean_diff": float(np.mean(d1) - np.mean(d2)),
                }

    ci95 = {}
    for m in MODELS:
        vals = np.array([metrics[m][f]["macro_f1"] for f in FOLDS])
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        n = len(vals)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        margin = tcrit * sd / (n ** 0.5)
        ci95[m] = {"mean": mean, "ci95_low": mean - margin, "ci95_high": mean + margin,
                   "n_folds": n, "note": "Wald-type t-interval on n=5 matched folds; illustrative only, not a rigorous population estimate"}

    statistical_tests = {
        "friedman_test": {
            "description": "Friedman test: non-parametric repeated-measures test across 7 models with fold as the matched/blocking factor (5 folds = 5 blocks). Tests whether the models' macro-F1 rank distributions differ across the same folds.",
            "statistic": float(friedman_stat),
            "df": len(MODELS) - 1,
            "p_value": float(friedman_p),
            "n_blocks_folds": len(FOLDS),
            "n_treatments_models": len(MODELS),
            "caveat": "Only 5 matched folds (blocks) is a very small sample for this test; treat any conclusion, significant or not, as weak evidence, not proof. Does NOT establish which specific pairs of models differ, only whether the 7 models' fold-rank distributions differ overall. Does not account for the two anomalous runs specially.",
        },
        "pairwise_mean_macro_f1_differences": pairwise,
        "confidence_intervals_95_macro_f1": ci95,
    }
    with open(os.path.join(OUT_DIR, "statistical_tests.json"), "w") as fh:
        json.dump(statistical_tests, fh, indent=2)

    # ---- runtime_analysis.json ----
    manifest = json.load(open("results/logs/transformer_sweep.json"))
    completed_runs = [r for r in manifest["runs"] if r["status"] == "completed"]
    runtime_by_model = {}
    for m in MODELS:
        times = [r["elapsed_seconds"] for r in completed_runs if r["model"] == m]
        runtime_by_model[m] = describe(times) if times else None
    all_times = [r["elapsed_seconds"] for r in completed_runs]
    fastest = min(completed_runs, key=lambda r: r["elapsed_seconds"])
    slowest = max(completed_runs, key=lambda r: r["elapsed_seconds"])
    runtime_analysis = {
        "per_model": runtime_by_model,
        "overall": describe(all_times),
        "total_sweep_wall_clock_seconds": None,  # filled from phase 15 report; not recomputed here
        "fastest_run": {"model": fastest["model"], "fold": fastest["fold"], "elapsed_seconds": fastest["elapsed_seconds"]},
        "slowest_run": {"model": slowest["model"], "fold": slowest["fold"], "elapsed_seconds": slowest["elapsed_seconds"]},
        "note": "Runtime observations only. No model-quality inference should be drawn from these numbers.",
    }
    with open(os.path.join(OUT_DIR, "runtime_analysis.json"), "w") as fh:
        json.dump(runtime_analysis, fh, indent=2)

    print("Analysis artifacts written to", OUT_DIR)
    print(f"Friedman chi2={friedman_stat:.4f}, df={len(MODELS)-1}, p={friedman_p:.4f}")


if __name__ == "__main__":
    main()
