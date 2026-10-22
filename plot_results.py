"""Visualize the existing prototype CSV; no training or new metrics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    results = pd.read_csv(args.results_dir / "results.csv").set_index(["classifier", "m"])
    order = [("SVM", 1), ("SVM", 5), ("RF", 1), ("RF", 5)]
    results = results.loc[order]
    metrics = ["accuracy", "precision", "recall"]
    counts = pd.read_csv(args.results_dir / "bucket_counts.csv")
    test_counts = counts[counts["split"] == "test"].set_index("bucket")
    test_total = int(test_counts[["poisoned", "pristine"]].to_numpy().sum())
    poison_total = int(test_counts["poisoned"].sum())
    labels = [f"{family}\nm = {m}" for family, m in order]
    colors = ["#8bbdd9", "#17618c", "#edb77a", "#b95d17"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4))
    for ax, metric in zip(axes, metrics):
        values = results[f"overall_{metric}"].to_numpy()
        bars = ax.bar(np.arange(4), values, color=colors, width=0.68)
        ax.bar_label(bars, labels=[f"{v:.2%}" for v in values], padding=5, fontsize=10)
        ax.set_xticks(np.arange(4), labels)
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(metric.capitalize(), fontweight="bold")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.2)
    fig.suptitle("PathMNIST label-flip detection · overall test results", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.035,
             f"{test_total:,} test samples · {poison_total:,} poisoned · seed 42 · positive = poisoned / reject\n"
             "Precision: how often a rejection is correct. Recall: how much poison is detected.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=0.81, bottom=0.22, wspace=0.28)
    save_figure(fig, args.results_dir / "overall_metrics")

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.5), layout="constrained")
    for ax, metric in zip(axes, metrics):
        matrix = results[[f"class_{j}_{metric}" for j in range(9)]].to_numpy()
        heat = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_yticks(np.arange(4), [f"{family}  m={m}" for family, m in order])
        ax.set_xticks(np.arange(9), [f"Class {j}\n{int(test_counts.loc[j, 'poisoned'])} poisoned" for j in range(9)])
        ax.set_title(metric.capitalize(), loc="left", fontweight="bold")
        for row in range(4):
            for col in range(9):
                value = matrix[row, col]
                ax.text(col, row, f"{value:.1%}", ha="center", va="center",
                        color="white" if value > 0.65 else "#172b3a", fontsize=10)
        colorbar = fig.colorbar(heat, ax=ax, fraction=0.025, pad=0.02)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Per-class detection · buckets are routed by y_given\n"
                 "Same 0–100% scale in every panel · positive = poisoned / reject",
                 fontsize=15, fontweight="bold")
    save_figure(fig, args.results_dir / "per_class_metrics")


def save_figure(fig, stem):
    for extension in ("png", "pdf"):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
