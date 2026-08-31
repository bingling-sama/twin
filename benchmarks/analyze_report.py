#!/usr/bin/env python3
"""Generate visual analysis charts from the Twin benchmark report.

Usage:
    uv run python benchmarks/analyze_report.py
    uv run python benchmarks/analyze_report.py --save  # save PNGs to benchmarks/plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# ═══════════════════════════════════════════════════════════════════════════════
# Colour palette — MongoDB-inspired
# ═══════════════════════════════════════════════════════════════════════════════
C_GREEN = "#00AA5B"
C_GREEN_DARK = "#00684A"
C_TEAL = "#00B8A9"
C_BLUE = "#016BF8"
C_PURPLE = "#9C6ADE"
C_ORANGE = "#FF7722"
C_RED = "#DB3030"
C_INK = "#001E2B"
C_GRAY = "#5D6D74"
C_LIGHT = "#E8EDEB"
C_WHITE = "#FFFFFF"

FLAT_COLOUR = C_GRAY
IVF_COLOUR = C_BLUE
IVFPQ_COLOUR = C_GREEN
HNSW_COLOUR = C_PURPLE

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
    "figure.facecolor": C_WHITE,
    "axes.facecolor": C_WHITE,
    "axes.edgecolor": C_LIGHT,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.color": C_LIGHT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": C_INK,
    "axes.labelcolor": C_INK,
    "xtick.color": C_GRAY,
    "ytick.color": C_GRAY,
})


def save_or_show(fig, name: str, save: bool) -> None:
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOTS_DIR / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_WHITE)
        print(f"  Saved → {path}")
    else:
        fig.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Faiss Index Search Scaling
# ═══════════════════════════════════════════════════════════════════════════════
def plot_search_scaling(save: bool = False) -> None:
    sizes = [1_000, 10_000, 100_000]
    labels = ["1K", "10K", "100K"]

    data = {
        "IndexFlatL2":  [53.4,  427.1,  4900.0],
        "IndexIVFFlat": [None,  87.4,  None],
        "IndexHNSWFlat":[62.1,  160.7,   486.9],
        "IndexIVFPQ":   [66.2,   98.6,   232.0],
    }
    colours = {
        "IndexFlatL2":   FLAT_COLOUR,
        "IndexIVFFlat":  IVF_COLOUR,
        "IndexHNSWFlat": HNSW_COLOUR,
        "IndexIVFPQ":    IVFPQ_COLOUR,
    }
    markers = {
        "IndexFlatL2": "s",
        "IndexIVFFlat": "D",
        "IndexHNSWFlat": "o",
        "IndexIVFPQ": "P",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── (a) Line chart: latency vs index size ──
    for name, latencies in data.items():
        pts = [(s, lat) for s, lat in zip(sizes, latencies) if lat is not None]
        if len(pts) < 1:
            continue
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, color=colours[name], marker=markers[name],
                 markersize=10, linewidth=2.5, label=name, zorder=5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Index Size (vectors)")
    ax1.set_ylabel("Search Latency (μs / ms)")
    ax1.set_title("(a) Search Latency vs Index Size", fontweight="bold", loc="left")
    ax1.set_xticks(sizes)
    ax1.set_xticklabels(labels)
    ax1.legend(frameon=True, fancybox=False, edgecolor=C_LIGHT, fontsize=9)

    # Annotate IVFPQ at 100K
    ax1.annotate(
        "IVFPQ @100K: 232 μs\n(best large-scale)",
        xy=(100_000, 232), xytext=(25_000, 100),
        arrowprops=dict(arrowstyle="->", color=IVFPQ_COLOUR, lw=1.5),
        fontsize=9, color=IVFPQ_COLOUR, fontweight="bold",
    )

    # Annotate Flat at 100K
    ax1.annotate(
        "Flat @100K: 4.9 ms\n(linear scaling)",
        xy=(100_000, 4900), xytext=(30_000, 7000),
        arrowprops=dict(arrowstyle="->", color=FLAT_COLOUR, lw=1.5),
        fontsize=9, color=FLAT_COLOUR,
    )

    # ── (b) Bar chart: 10K comparison ──
    # Data from cross-index comparison (more precise single-query timing)
    comp_data = {
        "IndexFlatL2":   0.345,
        "IndexIVFFlat":  0.061,
        "IndexHNSWFlat": 0.138,
        "IndexIVFPQ":    0.070,
    }
    names = list(comp_data.keys())
    values = list(comp_data.values())
    bar_colours = [colours[n] for n in names]

    bars = ax2.bar(names, values, color=bar_colours, edgecolor="white", linewidth=1.2)
    ax2.set_ylabel("Search Latency (ms)")
    ax2.set_title("(b) Single-Query Latency at 10K Vectors", fontweight="bold", loc="left")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=9)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color=C_INK)

    # Relative speedup annotation
    fastest = min(values)
    for bar, val in zip(bars, values):
        ratio = val / fastest
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f"{ratio:.1f}×", ha="center", va="center", fontsize=8,
                 color="white", fontweight="bold")

    fig.suptitle("Faiss Index Search Performance", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_or_show(fig, "01_search_scaling", save)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Precision–Speed Trade-off (nprobe / efSearch sweeps)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_accuracy_speed_tradeoff(save: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    nprobe_values = [1, 4, 8, 16, 32]
    ivf_latency   = [36.6, 59.0, 82.2, 143.9, 342.2]
    ivfpq_latency = [43.6, 67.2, 96.6, 155.1, 274.8]

    efsearch_values = [16, 32, 64, 128, 256]
    hnsw_latency    = [64.6, 94.0, 164.6, 360.5, 761.2]

    ax.plot(nprobe_values, ivf_latency, color=IVF_COLOUR, marker="D",
            markersize=9, linewidth=2.5, label="IndexIVFFlat (nprobe)")
    ax.plot(nprobe_values, ivfpq_latency, color=IVFPQ_COLOUR, marker="P",
            markersize=9, linewidth=2.5, label="IndexIVFPQ (nprobe)")

    # Twin x-axis for HNSW efSearch
    ax2 = ax.twiny()
    ax2.plot(efsearch_values, hnsw_latency, color=HNSW_COLOUR, marker="o",
             markersize=9, linewidth=2.5, linestyle="--",
             label="IndexHNSWFlat (efSearch)")
    ax2.set_xlabel("HNSW efSearch", color=HNSW_COLOUR)
    ax2.tick_params(axis="x", colors=HNSW_COLOUR)

    ax.set_xlabel("IVF nprobe", color=IVF_COLOUR)
    ax.tick_params(axis="x", colors=IVF_COLOUR)
    ax.set_ylabel("Search Latency (μs)")
    ax.set_title("Precision–Speed Trade-off at 10K Vectors", fontweight="bold", loc="left")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=True,
              fancybox=False, edgecolor=C_LIGHT, fontsize=9, loc="upper left")

    # Annotate default operating points
    ax.annotate("IVFPQ default\nnprobe=8: 97 μs",
                xy=(8, 96.6), xytext=(10, 150),
                arrowprops=dict(arrowstyle="->", color=IVFPQ_COLOUR, lw=1.2),
                fontsize=8, color=IVFPQ_COLOUR, fontweight="bold")
    ax.annotate("IVF default\nnprobe=8: 82 μs",
                xy=(8, 82.2), xytext=(3, 50),
                arrowprops=dict(arrowstyle="->", color=IVF_COLOUR, lw=1.2),
                fontsize=8, color=IVF_COLOUR, fontweight="bold")
    ax.annotate("HNSW default\nefSearch=64: 165 μs",
                xy=(64, 164.6), xytext=(80, 200),
                arrowprops=dict(arrowstyle="->", color=HNSW_COLOUR, lw=1.2),
                fontsize=8, color=HNSW_COLOUR, fontweight="bold")

    fig.tight_layout()
    save_or_show(fig, "02_accuracy_speed_tradeoff", save)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Search Pipeline Stage Breakdown
# ═══════════════════════════════════════════════════════════════════════════════
def plot_pipeline_breakdown(save: bool = False) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    stages = ["CLIP Encode\n+ Faiss Search", "dHash\nFilter", "pHash\nFilter", "SSIM\nVerify"]
    times_ms = [3.722, 0.070, 0.002, 0.042]
    pcts = [96.6, 1.8, 0.1, 1.1]
    colours_stages = [C_BLUE, C_TEAL, C_GREEN, C_ORANGE]

    # ── (a) Horizontal bar — time ──
    y_pos = range(len(stages))
    ax1.barh(y_pos, times_ms, color=colours_stages, edgecolor="white", height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(stages, fontsize=10)
    ax1.set_xlabel("Latency (ms)")
    ax1.set_title("(a) Per-Stage Latency", fontweight="bold", loc="left")
    ax1.invert_yaxis()
    for i, (t, p) in enumerate(zip(times_ms, pcts)):
        ax1.text(t + 0.05, i, f"{t:.3f} ms ({p:.1f}%)",
                 va="center", fontsize=9, fontweight="bold", color=C_INK)

    # ── (b) Donut — CLIP dominates ──
    explode = (0.03, 0.03, 0.03, 0.03)
    wedges, texts, autotexts = ax2.pie(
        times_ms, labels=stages, autopct="%1.1f%%", explode=explode,
        colors=colours_stages, startangle=90, pctdistance=0.6,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(10)
    ax2.set_title("(b) Time Distribution", fontweight="bold", loc="left")

    # Center annotation
    ax2.text(0, 0, f"Total\n{sum(times_ms):.2f} ms", ha="center", va="center",
             fontsize=11, fontweight="bold", color=C_INK)

    fig.suptitle("Search Pipeline Stage Breakdown", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_or_show(fig, "03_pipeline_breakdown", save)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: IVFPQ Memory Advantage
# ═══════════════════════════════════════════════════════════════════════════════
def plot_memory_comparison(save: bool = False) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── (a) Actual serialised sizes ──
    types = ["IndexFlatL2", "IndexIVFPQ"]
    actual_mb = [20_480_045 / 1e6, 1_326_708 / 1e6]  # → MB
    theory_mb = [20_480_000 / 1e6, 640_000 / 1e6]

    x = np.arange(len(types))
    width = 0.3

    bars1 = ax1.bar(x - width / 2, actual_mb, width, color=[FLAT_COLOUR, IVFPQ_COLOUR],
                    edgecolor="white", label="Actual Serialised")
    bars2 = ax1.bar(x + width / 2, theory_mb, width, color=[FLAT_COLOUR, IVFPQ_COLOUR],
                    edgecolor="white", alpha=0.4, label="Theoretical Minimum")

    ax1.set_xticks(x)
    ax1.set_xticklabels(types, fontsize=10)
    ax1.set_ylabel("Size (MB)")
    ax1.set_title("(a) 10K Vectors: Serialised Index Size", fontweight="bold", loc="left")
    ax1.legend(frameon=True, fancybox=False, edgecolor=C_LIGHT, fontsize=9)

    for bar, val in zip(bars1, actual_mb):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f} MB", ha="center", fontweight="bold", fontsize=10)
    for bar, val in zip(bars2, theory_mb):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f} MB", ha="center", fontsize=8, color=C_GRAY)

    # ── (b) Projected scaling to 1M vectors ──
    scales = ["10K", "100K", "1M"]
    n_vectors = [10_000, 100_000, 1_000_000]
    flat_sizes = [n * 512 * 4 / 1e9 for n in n_vectors]  # GB
    pq_sizes = [n * 64 / 1e9 for n in n_vectors]          # GB (M=64, nbits=8)

    x2 = np.arange(len(scales))
    width2 = 0.3

    ax2.bar(x2 - width2 / 2, flat_sizes, width2, color=FLAT_COLOUR,
            edgecolor="white", label="IndexFlatL2 / IndexIVFFlat")
    ax2.bar(x2 + width2 / 2, pq_sizes, width2, color=IVFPQ_COLOUR,
            edgecolor="white", label="IndexIVFPQ (M=64, nbits=8)")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(scales, fontsize=10)
    ax2.set_ylabel("Size (GB)")
    ax2.set_title("(b) Projected Memory at Scale", fontweight="bold", loc="left")
    ax2.legend(frameon=True, fancybox=False, edgecolor=C_LIGHT, fontsize=9)

    for i, (flat, pq) in enumerate(zip(flat_sizes, pq_sizes)):
        ratio = flat / pq if pq > 0 else 0
        ax2.text(i, flat + 0.05, f"{flat:.2f} GB", ha="center", fontweight="bold", fontsize=9)
        ax2.text(i, pq + 0.02, f"{pq:.3f} GB", ha="center", fontweight="bold", fontsize=9)
        ax2.text(i, (flat + pq) / 2, f"{ratio:.0f}×", ha="center", va="center",
                 fontsize=18, fontweight="bold", color="white", alpha=0.8)

    fig.suptitle("IVFPQ Memory Advantage: Product Quantization Compression",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_or_show(fig, "04_memory_comparison", save)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Recall vs Latency — Pareto Frontier (10K vectors, Recall@50)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_recall_pareto(save: bool = False) -> None:
    """Pareto-optimal frontier: recall@50 vs per-query latency at 10K.

    Each point is a different (index_type, parameter) configuration.
    The ideal point is top-left: high recall, low latency.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── (a) Recall–Latency curves ──
    # IVF nprobe sweep
    ivf_recall = [0.0536, 0.1002, 0.1838, 0.3254, 0.5484, 0.8904]
    ivf_latency = [33.9, 43.0, 56.1, 77.5, 132.3, 357.9]
    ivf_probes = [1, 2, 4, 8, 16, 32]

    # IVFPQ nprobe sweep
    ivfpq_recall = [0.0516, 0.0918, 0.1458, 0.2136, 0.2846, 0.3488]
    ivfpq_latency = [44.8, 52.6, 70.4, 101.2, 165.4, 295.8]

    # HNSW efSearch sweep
    hnsw_recall = [0.2604, 0.4108, 0.5868, 0.8066, 0.9526]
    hnsw_latency = [93.4, 107.0, 227.7, 428.9, 776.5]
    hnsw_ef = [16, 32, 64, 128, 256]

    # Flat baseline
    flat_recall = 1.0
    flat_latency = 336.3

    # Plot curves
    ax1.plot(ivf_latency, ivf_recall, color=IVF_COLOUR, marker="D", markersize=8,
             linewidth=2.5, label="IndexIVFFlat (nprobe)")
    ax1.plot(ivfpq_latency, ivfpq_recall, color=IVFPQ_COLOUR, marker="P", markersize=8,
             linewidth=2.5, label="IndexIVFPQ (nprobe)")
    ax1.plot(hnsw_latency, hnsw_recall, color=HNSW_COLOUR, marker="o", markersize=8,
             linewidth=2.5, linestyle="--", label="IndexHNSWFlat (efSearch)")
    ax1.scatter([flat_latency], [flat_recall], color=FLAT_COLOUR, marker="s", s=120,
                zorder=10, label="IndexFlatL2 (exact)")

    # Annotate key points
    for lat, rec, probe in zip(ivf_latency, ivf_recall, ivf_probes):
        if probe in (1, 8, 32):
            ax1.annotate(f"nprobe={probe}", (lat, rec),
                         textcoords="offset points", xytext=(8, -8),
                         fontsize=7, color=IVF_COLOUR, alpha=0.8)
    for lat, rec, ef in zip(hnsw_latency, hnsw_recall, hnsw_ef):
        if ef in (16, 64, 256):
            ax1.annotate(f"ef={ef}", (lat, rec),
                         textcoords="offset points", xytext=(5, -12),
                         fontsize=7, color=HNSW_COLOUR, alpha=0.8)

    ax1.set_xlabel("Per-Query Latency (μs)")
    ax1.set_ylabel("Recall@50")
    ax1.set_title("(a) Recall–Latency Curves (10K Vectors)", fontweight="bold", loc="left")
    ax1.legend(frameon=True, fancybox=False, edgecolor=C_LIGHT, fontsize=8, loc="lower right")
    ax1.set_xlim(0, 850)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.9, color=C_GRAY, linestyle=":", alpha=0.3, linewidth=1)
    ax1.text(820, 0.905, "90% recall", fontsize=7, color=C_GRAY, ha="right")

    # Shade the Pareto-optimal region
    ax1.fill_between([0, 30], [0.05, 0.05], [0.1, 0.1], alpha=0.0)  # dummy for style
    ax1.annotate("← Better",
                 xy=(50, 0.95), fontsize=9, color=C_GREEN_DARK, fontweight="bold",
                 arrowprops=dict(arrowstyle="<-", color=C_GREEN_DARK, lw=1.5))
    ax1.annotate("Better →",
                 xy=(700, 0.08), fontsize=9, color=C_GREEN_DARK, fontweight="bold",
                 arrowprops=dict(arrowstyle="<-", color=C_GREEN_DARK, lw=1.5))

    # ── (b) Bar chart: recall at default settings ──
    default_names = ["IndexFlatL2\n(exact)", "IndexIVFFlat\n(nprobe=8)",
                     "IndexHNSWFlat\n(efSearch=64)", "IndexIVFPQ\n(nprobe=8)"]
    default_recall = [1.0, 0.3254, 0.5868, 0.2136]
    default_latency = [336.3, 77.5, 227.7, 101.2]
    default_colours = [FLAT_COLOUR, IVF_COLOUR, HNSW_COLOUR, IVFPQ_COLOUR]

    x_pos = np.arange(len(default_names))
    bars = ax2.bar(x_pos, default_recall, color=default_colours, edgecolor="white",
                   width=0.55)

    # Latency sub-label
    for i, (rec, lat) in enumerate(zip(default_recall, default_latency)):
        ax2.text(i, rec + 0.02, f"{rec:.3f}\n({lat:.0f} μs)",
                 ha="center", fontsize=9, fontweight="bold", color=C_INK)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(default_names, fontsize=8)
    ax2.set_ylabel("Recall@50")
    ax2.set_title("(b) Default Settings Comparison", fontweight="bold", loc="left")
    ax2.set_ylim(0, 1.2)

    # Highlight Flat bar
    bars[0].set_edgecolor(C_GREEN_DARK)
    bars[0].set_linewidth(2)

    fig.suptitle("Search Accuracy: Recall@50 vs Latency at 10K Vectors",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_or_show(fig, "05_recall_pareto", save)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Training Cost & Insertion Overhead
# ═══════════════════════════════════════════════════════════════════════════════
def plot_training_cost(save: bool = False) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── (a) Training time ──
    train_types = ["IndexIVFFlat", "IndexIVFPQ"]
    train_times = [0.2955, 6.339]  # seconds
    colours_train = [IVF_COLOUR, IVFPQ_COLOUR]

    bars = ax1.bar(train_types, train_times, color=colours_train, edgecolor="white",
                   width=0.4)
    ax1.set_ylabel("Training Time (seconds)")
    ax1.set_title("(a) Training Time: 50K Vectors", fontweight="bold", loc="left")

    for bar, t in zip(bars, train_times):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{t:.1f}s" if t > 1 else f"{t*1000:.0f}ms",
                 ha="center", fontweight="bold", fontsize=11)

    # Speedup annotation
    ratio = train_times[1] / train_times[0]
    ax1.annotate(f"IVFPQ training is\n{ratio:.0f}× slower\n(one-time cost)",
                xy=(1, train_times[1]), xytext=(1.5, train_times[1] * 0.6),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5),
                fontsize=9, color=C_RED, fontweight="bold")

    # ── (b) Add throughput (pure insert, Flat baseline) ──
    add_types = ["Flat", "IVFPQ\n(encode)"]
    add_times = [24.8, 255.4]  # μs for batch of 32, per-batch

    bars2 = ax2.bar(add_types, add_times, color=[FLAT_COLOUR, IVFPQ_COLOUR],
                    edgecolor="white", width=0.4)
    ax2.set_ylabel("Add 32 Vectors (μs)")
    ax2.set_title("(b) Insertion Overhead: 32-Vector Batch", fontweight="bold", loc="left")

    for bar, t in zip(bars2, add_times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f"{t:.0f} μs", ha="center", fontweight="bold", fontsize=11)

    ax2.annotate("PQ encoding adds\n~10× overhead per vector",
                xy=(1, 255.4), xytext=(0.5, 200),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5),
                fontsize=9, color=C_RED, fontweight="bold")

    fig.suptitle("Training Cost & Insertion Overhead", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_or_show(fig, "06_training_cost", save)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: CLIP Batch Encoding & Indexing Throughput
# ═══════════════════════════════════════════════════════════════════════════════
def plot_clip_and_indexing(save: bool = False) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── (a) CLIP batch encoding ──
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    encode_times = [3.6, 5.2, 10.2, 17.0, 32.5, 69.5, 142.4]  # ms

    ax1.plot(batch_sizes, encode_times, color=C_BLUE, marker="o", markersize=9,
             linewidth=2.5, zorder=5)

    # Ideal linear scaling line
    linear = [encode_times[0] * s for s in batch_sizes]
    ax1.plot(batch_sizes, linear, color=C_GRAY, linestyle="--", linewidth=1,
             alpha=0.5, label="Linear scaling (hypothetical)")

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Encoding Time (ms)")
    ax1.set_title("(a) CLIP Batch Encoding (ViT-B-32)", fontweight="bold", loc="left")
    ax1.legend(frameon=True, fancybox=False, edgecolor=C_LIGHT, fontsize=9)

    # Annotate efficiency
    efficiency = [linear[i] / encode_times[i] for i in range(len(batch_sizes))]
    for i, (bs, t, eff) in enumerate(zip(batch_sizes, encode_times, efficiency)):
        if bs >= 16:
            ax1.annotate(f"{eff:.1f}× GPU\nefficiency",
                        xy=(bs, t), xytext=(bs - 3, t + 20),
                        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2),
                        fontsize=8, color=C_GREEN, fontweight="bold")

    # ── (b) Full indexing pipeline breakdown ──
    # per-batch breakdown (batch_size=32)
    components = ["CLIP\nEncode", "dHash\n×32", "pHash\n×32", "Faiss\nAdd 32"]
    comp_times = [70.48, 0.9, 1.8, 0.2]  # ms (from CLAUDE.md)
    comp_colours = [C_BLUE, C_TEAL, C_GREEN, C_ORANGE]

    y_pos = range(len(components))
    ax2.barh(y_pos, comp_times, color=comp_colours, edgecolor="white", height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(components, fontsize=10)
    ax2.set_xlabel("Time (ms)")
    ax2.set_title("(b) Indexing Pipeline Breakdown (batch=32)", fontweight="bold", loc="left")
    ax2.invert_yaxis()

    for i, t in enumerate(comp_times):
        pct = t / sum(comp_times) * 100
        ax2.text(t + 1, i, f"{t:.1f} ms ({pct:.0f}%)",
                 va="center", fontsize=9, fontweight="bold", color=C_INK)

    fig.suptitle("CLIP Encoding & Indexing Throughput", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_or_show(fig, "07_clip_indexing", save)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8: Radar Chart — Index Type Decision Matrix
# ═══════════════════════════════════════════════════════════════════════════════
def plot_radar_decision_matrix(save: bool = False) -> None:
    """Radar chart comparing 4 index types across 5 dimensions.

    Dimensions (higher = better):
      - Search Speed:     inverse of latency at 10K
      - Memory Efficiency: inverse of bytes/vector
      - Scalability:      sub-linear scaling (100K vs 1K latency ratio)
      - Training Needed:  no training = better (HNSW wins)
      - Insert Speed:     inverse of add latency
    """
    dimensions = ["Search\nSpeed", "Recall\nAccuracy", "Memory\nEfficiency",
                  "Scalability\n(log N)", "No Training\nNeeded"]
    n_dim = len(dimensions)

    # Normalised scores (1-5 scale, higher = better) — informed by actual benchmark data
    # Recall@50 at default settings: Flat=1.0, IVF=0.33, IVFPQ=0.21, HNSW=0.59
    flat_scores  = [4, 5, 1, 1, 5]    # perfect recall, terrible memory, linear scaling
    ivf_scores   = [5, 2, 1, 3, 2]    # fastest search, moderate recall, needs training
    ivfpq_scores = [4, 1, 5, 5, 2]    # best memory+scale, worst recall at default nprobe
    hnsw_scores  = [3, 3, 1, 4, 5]    # good recall, no training, moderate speed

    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    for name, scores, colour, marker in [
        ("IndexFlatL2", flat_scores, FLAT_COLOUR, "s"),
        ("IndexIVFFlat", ivf_scores, IVF_COLOUR, "D"),
        ("IndexIVFPQ", ivfpq_scores, IVFPQ_COLOUR, "P"),
        ("IndexHNSWFlat", hnsw_scores, HNSW_COLOUR, "o"),
    ]:
        values = scores + scores[:1]
        ax.fill(angles, values, alpha=0.08, color=colour)
        ax.plot(angles, values, color=colour, marker=marker, markersize=8,
                linewidth=2.5, label=name)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7, color=C_GRAY)
    ax.set_title("Index Type Decision Matrix", fontweight="bold", pad=25, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              frameon=True, fancybox=False, edgecolor=C_LIGHT, fontsize=9)

    fig.tight_layout()
    save_or_show(fig, "08_radar_decision", save)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9: Summary Infographic
# ═══════════════════════════════════════════════════════════════════════════════
def plot_summary(save: bool = False) -> None:
    """Single-panel summary with key metrics as a styled table + highlights."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    title = "Twin Benchmark Analysis — Key Findings"
    ax.text(0.5, 0.93, title, transform=ax.transAxes, fontsize=16,
            fontweight="bold", ha="center", color=C_INK)

    # System info
    sys_info = (
        "Environment: AMD Ryzen 7 8845H (16 cores) · 14.9 GB RAM · NVIDIA GPU (599 MB VRAM used)\n"
        "Model: CLIP ViT-B-32 (512-dim) · Faiss Index: 1K–100K L2-normalised synthetic vectors"
    )
    ax.text(0.5, 0.87, sys_info, transform=ax.transAxes, fontsize=9, ha="center",
            color=C_GRAY, style="italic")

    # Key findings table
    findings = [
        ("Search Speed (10K)", "IVFFlat = fastest (61 μs)", "IVFPQ only 15% slower (70 μs)"),
        ("Scalability (100K)", "IVFPQ = best (232 μs)", "Flat = worst (4.9 ms, 21x slower)"),
        ("Memory (1M vectors)", "IVFPQ = 64 MB", "Flat/IVF/HNSW = 2 GB (32x larger)"),
        ("Training Cost (50K)", "IVFFlat = 0.3s", "IVFPQ = 6.3s (21x, one-time)"),
        ("Best For <10K", "IndexFlatL2", "Exact search, zero training, 53 us @1K"),
        ("Best For 10K-100K", "IndexIVFFlat / IndexHNSWFlat", "Fast, no PQ overhead"),
        (
            "Best For >100K",
            "IndexIVFPQ",
            "Only viable option memory-wise, tune nprobe for recall",
        ),
        (
            "Best Recall @ Speed",
            "IndexHNSWFlat (efSearch=128)",
            "80.7% recall at 429 us (1.3x slower than Flat, 4.8x faster than exact at scale)",
        ),
    ]

    col_labels = ["Metric", "Winner / Value", "Context"]
    table_data = [[f[0], f[1], f[2]] for f in findings]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=[0.18, 0.32, 0.40],
        bbox=[0.02, 0.02, 0.96, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style the table
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(C_LIGHT)
        cell.set_linewidth(0.5)
        if key[0] == 0:  # header
            cell.set_facecolor(C_INK)
            cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        elif key[1] == 0:  # metric column
            cell.set_text_props(fontweight="bold", color=C_INK)
        elif key[1] == 1:  # winner column
            cell.set_text_props(color=C_GREEN_DARK)
        else:
            cell.set_text_props(color=C_GRAY)

        # Row banding
        if key[0] > 0 and key[0] % 2 == 0:
            cell.set_facecolor("#F5F7F6")

    # Footer
    ax.text(0.5, -0.02, "Generated by benchmarks/analyze_report.py · Data: bench_results.json",
            transform=ax.transAxes, fontsize=7, ha="center", color=C_GRAY, style="italic")

    save_or_show(fig, "09_summary", save)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark analysis charts")
    parser.add_argument("--save", action="store_true", help="Save PNGs to benchmarks/plots/")
    args = parser.parse_args()

    print("Generating benchmark analysis charts...")
    print()

    plot_search_scaling(args.save)
    plot_accuracy_speed_tradeoff(args.save)
    plot_pipeline_breakdown(args.save)
    plot_memory_comparison(args.save)
    plot_recall_pareto(args.save)
    plot_training_cost(args.save)
    plot_clip_and_indexing(args.save)
    plot_radar_decision_matrix(args.save)
    plot_summary(args.save)

    print()
    print("Done. All charts generated.")
    if not args.save:
        print("Run with --save to save PNG files to benchmarks/plots/")


if __name__ == "__main__":
    main()
