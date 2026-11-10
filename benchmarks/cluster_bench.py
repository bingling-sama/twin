#!/usr/bin/env python3
"""Minecraft Texture Discovery & Clustering Benchmark Runner.

Evaluates throughput, cluster purity, recall, and stages telemetry for
seedless similar texture discovery via Greedy Leader Elimination.

Usage:
    python benchmarks/cluster_bench.py
    python benchmarks/cluster_bench.py --samples 300 --output benchmarks/cluster_report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.cluster.dataset import (  # noqa: E402
    build_duplicate_swarms_dataset,
    build_sparse_real_haystack,
    build_variant_clusters_dataset,
)
from benchmarks.cluster.evaluator import (  # noqa: E402
    ScenarioResult,
    evaluate_cluster_scenario,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cluster_bench")


def format_markdown_report(results: list[ScenarioResult]) -> str:
    """Generate comprehensive benchmark report formatted in Markdown."""
    lines: list[str] = [
        "# 📊 Minecraft Texture Seedless Discovery & Clustering Benchmark Report",
        "",
        "> Benchmark of `discover_similar_batches()` using **Greedy Leader Dynamic Elimination**",
        "> over native binary Faiss (`IndexBinaryFlat` SIMD pHash) and pixel-level `DiffMaskVerifier`.",  # noqa: E501
        "",
        "## 🏆 Executive Summary & Strategy Matrix",
        "",
        "| Scenario | Mode | Scale (Items / GT) | Latency | Throughput | Formed Batches | Purity | 100% Pure | GT Recall |",  # noqa: E501
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        scale_str = f"{r.total_items} / {r.gt_cluster_count}"
        lines.append(
            f"| **{r.scenario_name}** | `{r.mode}` | {scale_str} | "
            f"**{r.elapsed_ms:.1f}ms** | {r.probes_per_sec:,.0f} prob/s | "
            f"{r.discovered_batches} batches ({r.total_discovered_members} skins) | "
            f"**{r.avg_batch_purity:.1%}** | {r.perfect_batches_ratio:.1%} | "
            f"**{r.cluster_recall_rate:.1%}** |"
        )

    lines.append("")
    lines.append("## 🔬 Telemetry & Stage Latency Decomposition")
    lines.append("")
    lines.append(
        "| Scenario | Binary Recall (ms / %) | Parallel I/O (ms / %) | DiffMask Verify (ms / %) | Verified Pairs | Verify Rate |"  # noqa: E501
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        st = r.stages
        rec_ms = st.get("recall", {}).get("elapsed_ms", 0.0)
        io_ms = st.get("io_load", {}).get("elapsed_ms", 0.0)
        ver_ms = st.get("verify", {}).get("elapsed_ms", 0.0)
        tot_stage = max(rec_ms + io_ms + ver_ms, 0.001)

        lines.append(
            f"| `{r.scenario_name}` | {rec_ms:.1f}ms ({(rec_ms / tot_stage) * 100:.1f}%) | "
            f"{io_ms:.1f}ms ({(io_ms / tot_stage) * 100:.1f}%) | "
            f"{ver_ms:.1f}ms ({(ver_ms / tot_stage) * 100:.1f}%) | "
            f"{r.pairs_verified} | {r.pairs_per_sec:,.0f} pairs/s |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📝 Detailed Scenario Analysis")
    lines.append("")

    for r in results:
        lines.append(r.to_markdown_summary())

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Similarity Discovery Benchmarks")
    parser.add_argument("--samples", type=int, default=200, help="Background scale for haystack")
    parser.add_argument(
        "--output", type=str, default="benchmarks/cluster_report.md", help="Output report file"
    )
    parser.add_argument(
        "--no-zip", action="store_true", help="Force synthetic fallback instead of images.zip"
    )
    args = parser.parse_args()

    logger.info("Initializing Similarity Discovery Benchmark Suite...")
    results: list[ScenarioResult] = []

    # 1. Duplicate Swarms (Exact & Near-Lossless Duplicates)
    logger.info("--> Running Scenario 1: Exact Duplicate Swarms (Top-K Starvation Stress)")
    dup_items, dup_counts = build_duplicate_swarms_dataset(
        swarm_sizes=[30, 20, 15],
        distractor_count=40,
        seed=101,
    )
    r1 = evaluate_cluster_scenario(
        scenario_name="Exact Duplicate Swarms",
        items=dup_items,
        ground_truth_counts=dup_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=50,
        max_probes=200,
        seed_strategy="random",
    )
    results.append(r1)
    logger.info(
        "    Done: %d batches formed in %.2fms (Purity: %.1f%%)",
        r1.discovered_batches,
        r1.elapsed_ms,
        r1.avg_batch_purity * 100,
    )

    # 2. Large Scale Duplicate Penetration (50 identical skins)
    logger.info("--> Running Scenario 2: Large Duplicate Swarm (50 identical skins)")
    large_items, large_counts = build_duplicate_swarms_dataset(
        swarm_sizes=[50],
        distractor_count=20,
        seed=102,
    )
    r2 = evaluate_cluster_scenario(
        scenario_name="Massive 50-Skin Duplicate Swarm",
        items=large_items,
        ground_truth_counts=large_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=50,
        max_probes=200,
        seed_strategy="sequential",
    )
    results.append(r2)
    logger.info(
        "    Done: %d batches formed in %.2fms (Purity: %.1f%%)",
        r2.discovered_batches,
        r2.elapsed_ms,
        r2.avg_batch_purity * 100,
    )

    # 3. Multi-Cluster Derivative Variants (Recolors & Edits)
    logger.info("--> Running Scenario 3: Derivative Variant Clusters (Recolors / Edits)")
    var_items, var_counts = build_variant_clusters_dataset(
        num_clusters=5,
        items_per_cluster=6,
        distractor_count=40,
        seed=201,
    )
    r3 = evaluate_cluster_scenario(
        scenario_name="Derivative Variant Clusters",
        items=var_items,
        ground_truth_counts=var_counts,
        mode="variants",
        batch_size=5,
        max_batches=30,
        max_probes=200,
        seed_strategy="random",
    )
    results.append(r3)
    logger.info(
        "    Done: %d batches formed in %.2fms (Purity: %.1f%%)",
        r3.discovered_batches,
        r3.elapsed_ms,
        r3.avg_batch_purity * 100,
    )

    # 4. Sparse Real-World Haystack (Confirmed Duplicates)
    logger.info("--> Running Scenario 4: Sparse Real Haystack (Confirmed Duplicates)")
    real_items, real_counts = build_sparse_real_haystack(
        background_size=args.samples,
        num_duplicate_pairs=10,
        num_variant_clusters=5,
        use_zip=not args.no_zip,
        seed=301,
    )
    # Filter ground truth for confirmed duplicates only
    dup_gt_counts = {k: v for k, v in real_counts.items() if "dup" in k}
    r4 = evaluate_cluster_scenario(
        scenario_name="Sparse Real Haystack (Confirmed)",
        items=real_items,
        ground_truth_counts=dup_gt_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=30,
        max_probes=200,
        seed_strategy="random",
    )
    results.append(r4)
    logger.info(
        "    Done: %d batches formed in %.2fms (Purity: %.1f%%)",
        r4.discovered_batches,
        r4.elapsed_ms,
        r4.avg_batch_purity * 100,
    )

    # 5. Sparse Real-World Haystack (Variants Exploration)
    logger.info("--> Running Scenario 5: Sparse Real Haystack (Variants Exploration)")
    var_gt_counts = {k: v for k, v in real_counts.items() if "var" in k}
    r5 = evaluate_cluster_scenario(
        scenario_name="Sparse Real Haystack (Variants)",
        items=real_items,
        ground_truth_counts=var_gt_counts,
        mode="variants",
        batch_size=5,
        max_batches=30,
        max_probes=200,
        seed_strategy="random",
    )
    results.append(r5)
    logger.info(
        "    Done: %d batches formed in %.2fms (Purity: %.1f%%)",
        r5.discovered_batches,
        r5.elapsed_ms,
        r5.avg_batch_purity * 100,
    )

    # 6. Seed Strategy Comparison (Sequential Strategy)
    logger.info("--> Running Scenario 6: Seed Strategy Comparison (Sequential on Haystack)")
    r6 = evaluate_cluster_scenario(
        scenario_name="Sequential Seed Strategy",
        items=real_items,
        ground_truth_counts=dup_gt_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=30,
        max_probes=200,
        seed_strategy="sequential",
    )
    results.append(r6)
    logger.info(
        "    Done: %d batches formed in %.2fms (Purity: %.1f%%)",
        r6.discovered_batches,
        r6.elapsed_ms,
        r6.avg_batch_purity * 100,
    )

    # Generate and save report
    md_report = format_markdown_report(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_report, encoding="utf-8")
    logger.info("Benchmark report generated successfully at: %s", out_path)
    print("\n" + md_report)


if __name__ == "__main__":
    main()
