import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def make_figure(out_path: str) -> None:
    # Left panel: vision tasks primary robustness metric (%)
    vision_labels = [
        "T01 CIFAR-10\nAcc@σ=0.10",
        "T02 PROTEINS\nAUC@σ=0.10",
        "T05 Pose\nPCK@0.05 (clean)",
        "T06 ReID\nAvg-Shift Rank-1",
        "T07 Chest-ray\nWorst-Shift Acc",
    ]
    b0 = [40.0, 73.8, 42.5, 43.0, 62.5]
    vat = [65.2, 66.8, 11.9, 65.8, 73.1]
    pmh = [80.4, 77.9, 39.7, 63.7, 82.5]

    # Right panel: TDI summary (lower is better)
    t04 = {"B0": 1.093, "VAT": 1.276, "PMH": 0.904}
    t08 = {"Baseline": 0.496, "PMH": 0.354}
    t09 = {"Pretrained": 1.230, "PMH": 0.936}

    colors = {
        "B0": "#ff8c1a",  # orange
        "VAT": "#d62728",  # red
        "PMH": "#1f77b4",  # blue
        "Baseline": "#b0b0b0",  # light gray
        "Pretrained": "#707070",  # dark gray
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.5), gridspec_kw={"wspace": 0.15})
    fig.suptitle("PMH Robustness Across All 9 Tasks — Replication Results", fontsize=24, y=0.98)

    # Left panel plotting
    x = np.arange(len(vision_labels))
    width = 0.24
    ax1.bar(x - width, b0, width=width, color=colors["B0"], edgecolor="white", label="B0")
    ax1.bar(x, vat, width=width, color=colors["VAT"], edgecolor="white", label="VAT")
    ax1.bar(x + width, pmh, width=width, color=colors["PMH"], edgecolor="white", label="PMH")

    for i, yv in enumerate(b0):
        ax1.text(i - width, yv + 1.0, f"{yv:.1f}%", ha="center", va="bottom", fontsize=10)
    for i, yv in enumerate(vat):
        ax1.text(i, yv + 1.0, f"{yv:.1f}%", ha="center", va="bottom", fontsize=10)
    for i, yv in enumerate(pmh):
        ax1.text(i + width, yv + 1.0, f"{yv:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax1.text(i + width, yv + 4.0, "*", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax1.set_title("Vision Tasks (T01–T07): Primary Robustness Metric", fontsize=17, pad=10)
    ax1.set_ylim(0, 102)
    ax1.set_ylabel("Metric value (%)", fontsize=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(vision_labels, rotation=28, ha="right", fontsize=12)
    ax1.grid(axis="y", alpha=0.35)
    ax1.legend(loc="upper center", ncol=3, fontsize=13, frameon=True)

    # Right panel plotting
    x2 = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
    vals = [t04["B0"], t04["VAT"], t04["PMH"], t08["Baseline"], t08["PMH"], t09["Pretrained"], t09["PMH"]]
    cols = [colors["B0"], colors["VAT"], colors["PMH"], colors["Baseline"], colors["PMH"], colors["Pretrained"], colors["PMH"]]
    ax2.bar(x2, vals, width=0.8, color=cols, edgecolor="white")

    for i, yv in enumerate(vals):
        ax2.text(x2[i], yv + 0.02, f"{yv:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold" if i in (1, 2, 6) else None)

    ax2.axhline(1.0, color="black", linewidth=1.5, alpha=0.7)
    ax2.text(2.6, 1.015, "Isometry", fontsize=12, fontweight="bold", color="black")
    ax2.text(1.95, 1.065, "PMH only method\nbelow isometry (1.0)", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))

    ax2.annotate("-28.6%", xy=(4, 0.354), xytext=(4.8, 0.54),
                 arrowprops=dict(arrowstyle="->", lw=2, color="black"),
                 fontsize=14, fontweight="bold", ha="center")

    ax2.set_title("T04, T08, T09: TDI (lower = better)", fontsize=17, pad=10)
    ax2.set_ylabel("TDI | lower = better", fontsize=15)
    ax2.set_ylim(0, 1.42)
    ax2.set_xticks([1, 3.5, 5.5])
    ax2.set_xticklabels(["T04 ViT\nTDI@0 ↓", "T08 BERT\nTDI@0 ↓", "T09 ImageNet\nViT TDI@0 ↓"], fontsize=13)
    ax2.grid(axis="y", alpha=0.35)

    # Correct, complete legend for right panel
    handles = [
        Patch(facecolor=colors["B0"], label="B0"),
        Patch(facecolor=colors["VAT"], label="VAT"),
        Patch(facecolor=colors["PMH"], label="PMH"),
        Patch(facecolor=colors["Baseline"], label="Baseline"),
        Patch(facecolor=colors["Pretrained"], label="Pretrained"),
    ]
    ax2.legend(handles=handles, loc="upper center", ncol=3, fontsize=12, frameon=True)

    fig.text(
        0.5,
        0.03,
        "All values from replication_seeded/artifacts/results/. T08 BERT TDI from semantic_tdi_results.json. "
        "T09 TDI from tdi_pretrained_baseline.json.",
        ha="center",
        fontsize=11,
    )
    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output = os.path.join(repo_root, "docs", "images", "cross_task_results.png")
    make_figure(output)
    print(f"Wrote figure to: {output}")
