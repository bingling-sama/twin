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
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import PIL

# Ensure workspace root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import faiss  # noqa: E402

from benchmarks.cluster.dataset import (  # noqa: E402
    build_boundary_dataset,
    build_duplicate_swarms_dataset,
    build_sparse_real_haystack,
    build_transitivity_chain_dataset,
    build_variant_clusters_dataset,
)
from benchmarks.cluster.evaluator import (  # noqa: E402
    ScenarioResult,
    evaluate_cluster_scenario,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cluster_bench")


def get_environment_metadata() -> dict[str, str]:
    """Capture execution environment metadata for benchmark reproducibility."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        commit = "unknown"

    cpu_info = platform.processor() or platform.machine()
    return {
        "Commit SHA": commit,
        "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "OS Platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "CPU": cpu_info,
        "Faiss": getattr(faiss, "__version__", "unknown"),
        "Pillow": PIL.__version__,
        "PRNG Seed": "42 (Deterministic)",
    }


def format_markdown_report(
    results_with_stats: list[tuple[ScenarioResult, dict[str, float]]],
    env_meta: dict[str, str],
) -> str:
    """Generate comprehensive benchmark report formatted in Markdown."""
    lines: list[str] = [
        "# 📊 Minecraft Texture Seedless Discovery & Clustering Benchmark Report",
        "",
        "> Benchmark of `discover_similar_batches()` using **Greedy Leader Dynamic Elimination**",
        "> over native binary Faiss (`IndexBinaryFlat` SIMD pHash) and pixel-level `DiffMaskVerifier`.",  # noqa: E501
        "",
        "## 🖥️ Execution Environment Metadata",
        "",
        "| Parameter | Value |",
        "| :--- | :--- |",
    ]

    for k, v in env_meta.items():
        lines.append(f"| **{k}** | `{v}` |")

    lines.append("")
    lines.append("## 🏆 Executive Summary & Strategy Matrix")
    lines.append("")
    lines.append(
        "> **Note on Latencies & Rates**: "
        "`Latency` reflects Wall-clock Median (P95) over multiple statistical rounds. "
        "Throughput (`prob/s`) uses `end_to_end_ms` as the strict rate denominator."
    )
    lines.append("")
    lines.append(
        "| Scenario | Mode | Scale (Items/GT) | Wall Latency (Med/P95) | Throughput | Formed Batches | Purity | Coverage | Recall |"  # noqa: E501
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for r, stats in results_with_stats:
        scale_str = f"{r.total_items} / {r.gt_cluster_count}"
        med_lat = stats.get("median_ms", r.end_to_end_ms)
        p95_lat = stats.get("p95_ms", r.end_to_end_ms)
        lines.append(
            f"| **{r.scenario_name}** | `{r.mode}` | {scale_str} | "
            f"**{med_lat:.1f}ms** ({p95_lat:.1f}ms) | {r.probes_per_sec:,.0f} prob/s | "
            f"{r.discovered_batches} batches ({r.total_discovered_members} skins) | "
            f"**{r.avg_batch_purity:.1%}** | {r.member_coverage_rate:.1%} | "
            f"**{r.cluster_recall_rate:.1%}** |"
        )

    lines.append("")
    lines.append("## 🔬 Telemetry & Fine-Grained Latency Decomposition")
    lines.append("")
    lines.append(
        "> **Decomposition**: "
        "`service_query_ms` = `instrumented_stage_ms` (`recall` + `io` + `verify`) + `unattributed_ms` "  # noqa: E501
        "(dynamic elimination loop overhead & metadata retrieval)."
    )
    lines.append("")
    lines.append(
        "| Scenario | Service / Wall | Binary Recall | Parallel I/O | DiffMask Verify | Loop Overhead | Verified Pairs |"  # noqa: E501
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for r, _ in results_with_stats:
        st = r.stages
        rec_ms = float(st.get("recall", {}).get("elapsed_ms", 0.0))
        io_ms = float(st.get("io_load", {}).get("elapsed_ms", 0.0))
        ver_ms = float(st.get("verify", {}).get("elapsed_ms", 0.0))
        tot_stage = max(rec_ms + io_ms + ver_ms, 0.001)

        lines.append(
            f"| `{r.scenario_name}` | {r.service_query_ms:.1f}ms / {r.end_to_end_ms:.1f}ms | "
            f"{rec_ms:.1f}ms ({(rec_ms / tot_stage) * 100:.1f}%) | "
            f"{io_ms:.1f}ms ({(io_ms / tot_stage) * 100:.1f}%) | "
            f"{ver_ms:.1f}ms ({(ver_ms / tot_stage) * 100:.1f}%) | "
            f"{r.unattributed_ms:.1f}ms | "
            f"{r.pairs_verified} ({r.pairs_per_sec:,.0f} pairs/s) |"
        )

    lines.append("")
    lines.append("## ⚖️ Algorithmic Trade-offs & Variants Mode Honest Disclosure")
    lines.append("")
    lines.append(
        "> [!IMPORTANT]\n"
        "> **Variants Mode Trade-off Disclosure**:\n"
        "> In `mode='variants'`, search constraints are intentionally relaxed to Hamming $\\le 12$ and DiffMask $\\le 0.35$ "  # noqa: E501
        "> in order to discover derivative community recolors, minor clothing modifications, and edit series.\n"  # noqa: E501
        "> When exposed to hundreds of arbitrary community background distractors, cluster purity naturally settles at **~56% ~ 68%**.\n"  # noqa: E501
        "> This is a deliberate, mathematically sound engineering trade-off: **trading single-pass precision to maximize recall on loose artistic variants**.\n"  # noqa: E501
        "> Clustering suggestions generated under `variants` mode are intended for human review workflows and must never trigger unsupervised destructive deduplication."  # noqa: E501
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📝 Detailed Scenario Analysis")
    lines.append("")

    for r, _ in results_with_stats:
        lines.append(r.to_markdown_summary())

    return "\n".join(lines)


def run_scenario_with_stats(
    scenario_fn: Any,
    rounds: int = 3,
    **kwargs: Any,
) -> tuple[ScenarioResult, dict[str, float]]:
    """Run a scenario multiple rounds to compute robust median and p95 metrics."""
    results: list[ScenarioResult] = []
    latencies: list[float] = []

    for round_idx in range(rounds):
        res = scenario_fn(**kwargs)
        results.append(res)
        latencies.append(res.end_to_end_ms)

    # Sort results by end_to_end_ms to pick median representative
    sorted_idx = int(np.argsort(latencies)[len(latencies) // 2])
    representative = results[sorted_idx]

    stats = {
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }
    return representative, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Similarity Discovery Benchmarks")
    parser.add_argument("--samples", type=int, default=200, help="Background scale for haystack")
    parser.add_argument("--rounds", type=int, default=3, help="Statistical evaluation rounds")
    parser.add_argument(
        "--output", type=str, default="benchmarks/cluster_report.md", help="Output report file"
    )
    parser.add_argument(
        "--no-zip", action="store_true", help="Force synthetic fallback instead of images.zip"
    )
    args = parser.parse_args()

    logger.info("Initializing Similarity Discovery Benchmark Suite V2...")
    env_meta = get_environment_metadata()
    results_with_stats: list[tuple[ScenarioResult, dict[str, float]]] = []

    # 1. Duplicate Swarms (Exact & Near-Lossless Duplicates)
    logger.info("--> Running Scenario 1: Exact Duplicate Swarms (Top-K Starvation Stress)")
    dup_items, dup_counts = build_duplicate_swarms_dataset(
        swarm_sizes=[30, 20, 15],
        distractor_count=40,
        seed=101,
    )
    r1, s1 = run_scenario_with_stats(
        evaluate_cluster_scenario,
        rounds=args.rounds,
        scenario_name="Exact Duplicate Swarms",
        items=dup_items,
        ground_truth_counts=dup_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=50,
        max_probes=200,
        seed_strategy="random",
        random_seed=42,
    )
    results_with_stats.append((r1, s1))
    logger.info(
        "    Done: %d batches formed in %.2fms (Median: %.2fms, Purity: %.1f%%)",
        r1.discovered_batches,
        r1.end_to_end_ms,
        s1["median_ms"],
        r1.avg_batch_purity * 100,
    )

    # 2. Large Scale Duplicate Penetration (50 identical skins)
    logger.info("--> Running Scenario 2: Large Duplicate Swarm (50 identical skins)")
    large_items, large_counts = build_duplicate_swarms_dataset(
        swarm_sizes=[50],
        distractor_count=20,
        seed=102,
    )
    r2, s2 = run_scenario_with_stats(
        evaluate_cluster_scenario,
        rounds=args.rounds,
        scenario_name="Massive 50-Skin Duplicate Swarm",
        items=large_items,
        ground_truth_counts=large_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=50,
        max_probes=200,
        seed_strategy="sequential",
        random_seed=42,
    )
    results_with_stats.append((r2, s2))
    logger.info(
        "    Done: %d batches formed in %.2fms (Median: %.2fms, Purity: %.1f%%)",
        r2.discovered_batches,
        r2.end_to_end_ms,
        s2["median_ms"],
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
    r3, s3 = run_scenario_with_stats(
        evaluate_cluster_scenario,
        rounds=args.rounds,
        scenario_name="Derivative Variant Clusters",
        items=var_items,
        ground_truth_counts=var_counts,
        mode="variants",
        batch_size=5,
        max_batches=30,
        max_probes=200,
        seed_strategy="random",
        random_seed=42,
    )
    results_with_stats.append((r3, s3))
    logger.info(
        "    Done: %d batches formed in %.2fms (Median: %.2fms, Purity: %.1f%%)",
        r3.discovered_batches,
        r3.end_to_end_ms,
        s3["median_ms"],
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
    dup_gt_counts = {k: v for k, v in real_counts.items() if "dup" in k}
    r4, s4 = run_scenario_with_stats(
        evaluate_cluster_scenario,
        rounds=args.rounds,
        scenario_name="Sparse Real Haystack (Confirmed)",
        items=real_items,
        ground_truth_counts=dup_gt_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=30,
        max_probes=200,
        seed_strategy="random",
        random_seed=42,
    )
    results_with_stats.append((r4, s4))
    logger.info(
        "    Done: %d batches formed in %.2fms (Median: %.2fms, Purity: %.1f%%)",
        r4.discovered_batches,
        r4.end_to_end_ms,
        s4["median_ms"],
        r4.avg_batch_purity * 100,
    )

    # 5. Sparse Real-World Haystack (Variants Exploration)
    logger.info("--> Running Scenario 5: Sparse Real Haystack (Variants Exploration)")
    var_gt_counts = {k: v for k, v in real_counts.items() if "var" in k}
    r5, s5 = run_scenario_with_stats(
        evaluate_cluster_scenario,
        rounds=args.rounds,
        scenario_name="Sparse Real Haystack (Variants)",
        items=real_items,
        ground_truth_counts=var_gt_counts,
        mode="variants",
        batch_size=5,
        max_batches=30,
        max_probes=200,
        seed_strategy="random",
        random_seed=42,
    )
    results_with_stats.append((r5, s5))
    logger.info(
        "    Done: %d batches formed in %.2fms (Median: %.2fms, Purity: %.1f%%)",
        r5.discovered_batches,
        r5.end_to_end_ms,
        s5["median_ms"],
        r5.avg_batch_purity * 100,
    )

    # 6. Seed Strategy Comparison (Sequential on Haystack)
    logger.info("--> Running Scenario 6: Seed Strategy Comparison (Sequential on Haystack)")
    r6, s6 = run_scenario_with_stats(
        evaluate_cluster_scenario,
        rounds=args.rounds,
        scenario_name="Sequential Seed Strategy",
        items=real_items,
        ground_truth_counts=dup_gt_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=30,
        max_probes=200,
        seed_strategy="sequential",
        random_seed=42,
    )
    results_with_stats.append((r6, s6))
    logger.info(
        "    Done: %d batches formed in %.2fms (Median: %.2fms, Purity: %.1f%%)",
        r6.discovered_batches,
        r6.end_to_end_ms,
        s6["median_ms"],
        r6.avg_batch_purity * 100,
    )

    # 7. Threshold Boundary Verification
    logger.info("--> Running Scenario 7: Threshold Boundary Verification")
    b_items, b_counts = build_boundary_dataset(seed=401)
    r7, s7 = run_scenario_with_stats(
        evaluate_cluster_scenario,
        rounds=args.rounds,
        scenario_name="Threshold Boundary Gating",
        items=b_items,
        ground_truth_counts=b_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=5,
        seed_strategy="sequential",
        random_seed=42,
    )
    results_with_stats.append((r7, s7))
    logger.info(
        "    Done: %d batches formed in %.2fms (Purity: %.1f%%)",
        r7.discovered_batches,
        r7.end_to_end_ms,
        r7.avg_batch_purity * 100,
    )

    # 8. Transitivity Chain Elimination
    logger.info("--> Running Scenario 8: Transitivity Chain Elimination")
    t_items, t_counts = build_transitivity_chain_dataset(seed=501)
    r8, s8 = run_scenario_with_stats(
        evaluate_cluster_scenario,
        rounds=args.rounds,
        scenario_name="Anti-Transitivity Elimination",
        items=t_items,
        ground_truth_counts=t_counts,
        mode="variants",
        batch_size=5,
        max_batches=5,
        seed_strategy="sequential",
        random_seed=42,
    )
    results_with_stats.append((r8, s8))
    logger.info(
        "    Done: %d batches formed in %.2fms (Purity: %.1f%%)",
        r8.discovered_batches,
        r8.end_to_end_ms,
        r8.avg_batch_purity * 100,
    )

    # Generate and save report
    md_report = format_markdown_report(results_with_stats, env_meta)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_report, encoding="utf-8")
    logger.info("Benchmark report generated successfully at: %s", out_path)
    print("\n" + md_report)


if __name__ == "__main__":
    main()
