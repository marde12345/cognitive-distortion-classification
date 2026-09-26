"""Generate simple, publication-friendly plots from the transformer sweep
analysis artifacts (results/analysis/transformer/*.json,*.csv).

Read-only with respect to experiment data; writes only PNG files under
results/analysis/transformer/plots/.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["indobert_15g", "indobert_base_p1", "indobertweet",
          "indoroberta_15g", "mbert", "nusabert", "xlmr"]
FOLDS = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]

OUT_DIR = os.path.join("results", "analysis", "transformer")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                      "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def load_fold_metrics():
    rows = []
    with open(os.path.join(OUT_DIR, "fold_metrics.csv")) as fh:
        for r in csv.DictReader(fh):
            r["macro_f1"] = float(r["macro_f1"])
            r["weighted_f1"] = float(r["weighted_f1"])
            r["accuracy"] = float(r["accuracy"])
            r["elapsed_seconds"] = float(r["elapsed_seconds"])
            r["anomalous"] = r["anomalous"] == "True"
            rows.append(r)
    return rows


def mean_std_bar(rows, metric, title, fname):
    fig, ax = plt.subplots(figsize=(7, 4))
    means, stds = [], []
    for m in MODELS:
        vals = [r[metric] for r in rows if r["model"] == m]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1))
    x = np.arange(len(MODELS))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_ylabel(title)
    ax.set_title(f"{title} — mean ± std across 5 folds (model order: config file order)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname))
    plt.close(fig)


def fold_lines(rows, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in MODELS:
        vals = [next(r["macro_f1"] for r in rows if r["model"] == m and r["fold"] == f) for f in FOLDS]
        anomalous_mask = [next(r["anomalous"] for r in rows if r["model"] == m and r["fold"] == f) for f in FOLDS]
        ax.plot(FOLDS, vals, marker="o", label=m, alpha=0.85)
        for xi, (v, a) in enumerate(zip(vals, anomalous_mask)):
            if a:
                ax.scatter([FOLDS[xi]], [v], color="red", zorder=5, s=60, marker="x")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 across folds per model (red X = flagged anomalous run)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname))
    plt.close(fig)


def runtime_box(rows, fname):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [[r["elapsed_seconds"] for r in rows if r["model"] == m] for m in MODELS]
    ax.boxplot(data, tick_labels=MODELS)
    ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_ylabel("Elapsed seconds")
    ax.set_title("Per-run runtime distribution by model (execution observation only)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname))
    plt.close(fig)


def per_class_heatmap(fname):
    data = {}
    with open(os.path.join(OUT_DIR, "per_class_metrics.csv")) as fh:
        for r in csv.DictReader(fh):
            data.setdefault(r["model"], {})[int(r["label"])] = float(r["mean_f1"])
    labels = sorted(next(iter(data.values())).keys())
    matrix = np.array([[data[m][lbl] for lbl in labels] for m in MODELS])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"L{l}" for l in labels])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title("Mean per-class F1 across 5 folds (per model)")
    for i in range(len(MODELS)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                     color="white" if matrix[i, j] < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="mean F1")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname))
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    rows = load_fold_metrics()
    mean_std_bar(rows, "macro_f1", "Macro-F1", "01_macro_f1_mean_std.png")
    mean_std_bar(rows, "accuracy", "Accuracy", "02_accuracy_mean_std.png")
    mean_std_bar(rows, "weighted_f1", "Weighted-F1", "03_weighted_f1_mean_std.png")
    fold_lines(rows, "04_macro_f1_across_folds.png")
    runtime_box(rows, "05_runtime_distribution.png")
    per_class_heatmap("06_per_class_f1_heatmap.png")
    print("Plots written to", PLOT_DIR)


if __name__ == "__main__":
    main()
