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
import hashlib
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import PIL

# Ensure workspace root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import faiss  # noqa: E402
import torch  # noqa: E402

from benchmarks.cluster.dataset import (  # noqa: E402
    build_adversarial_dataset,
    build_boundary_dataset,
    build_duplicate_swarms_dataset,
    build_sparse_real_haystack,
    build_transitivity_chain_dataset,
    build_variant_clusters_dataset,
    get_zip_fingerprint,
)
from benchmarks.cluster.evaluator import (  # noqa: E402
    ScenarioResult,
    evaluate_cluster_scenario,
)
from twin.core.config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cluster_bench")


def _compute_source_digest(paths: list[Path]) -> str:
    """Compute combined SHA-256 digest over core implementation and benchmark harness files."""
    digest = hashlib.sha256()
    for p in sorted(paths):
        digest.update(p.relative_to(_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        if p.is_file():
            digest.update(p.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def get_environment_metadata(
    rounds: int = 3,
    samples: int = 200,
    use_zip: bool = True,
) -> dict[str, str]:
    """Capture execution environment metadata for strict benchmark reproducibility."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="replace")
        dirty_tag = " (dirty)" if dirty_out.strip() else " (clean)"
    except Exception:
        commit = "unknown"
        dirty_tag = ""

    source_digest = _compute_source_digest(
        [
            _ROOT / "src" / "twin" / "services" / "cluster.py",
            _ROOT / "benchmarks" / "cluster" / "dataset.py",
            _ROOT / "benchmarks" / "cluster" / "evaluator.py",
            _ROOT / "benchmarks" / "cluster_bench.py",
        ]
    )

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info = f"{gpu_name} ({gpu_mem_gb:.1f} GB)"
    else:
        gpu_info = "CPU only (CUDA unavailable)"

    if use_zip:
        zip_fp = get_zip_fingerprint()
        if zip_fp.get("exists"):
            zip_str = (
                f"{zip_fp['size_bytes']:,} bytes | "
                f"{zip_fp['png_count']:,} PNGs | sha256={zip_fp['sha256']}"
            )
        else:
            zip_str = f"missing ({zip_fp['path']}, synthetic fallback used)"
    else:
        zip_str = "disabled (--no-zip)"

    cpu_info = platform.processor() or platform.machine()
    cpu_cores = os.cpu_count() or 1
    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    return {
        "Generated At": now_iso,
        "Commit SHA": f"{commit}{dirty_tag}",
        "Source Digest (SHA-256)": source_digest,
        "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "OS Platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "CPU": f"{cpu_info} ({cpu_cores} logical cores)",
        "GPU": gpu_info,
        "Torch / Faiss / Pillow": (
            f"{torch.__version__} / {getattr(faiss, '__version__', 'unknown')} / {PIL.__version__}"
        ),
        "Statistical Rounds": str(rounds),
        "Haystack Config": f"samples={samples}, use_zip={use_zip}",
        "images.zip Fingerprint": zip_str,
        "Confirmed Thresholds": (
            f"hamming <= {settings.discover_confirmed_hamming}, "
            f"diff_ratio <= {settings.discover_confirmed_diff_ratio}"
        ),
        "Variants Thresholds": (
            f"hamming <= {settings.discover_variants_hamming}, "
            f"diff_ratio <= {settings.discover_variants_diff_ratio}"
        ),
        "Recall Window": (
            f"min_k={settings.discover_min_recall_k}, "
            f"multiplier={settings.discover_recall_multiplier}"
        ),
    }


def format_markdown_report(
    results_with_stats: list[tuple[ScenarioResult, dict[str, Any]]],
    env_meta: dict[str, str],
) -> str:
    """Generate comprehensive benchmark report formatted in Markdown."""
    lines: list[str] = [
        "# 📊 Minecraft Texture Seedless Discovery & Clustering Benchmark Report",
        "",
        "> Benchmark of `discover_similar_batches()` using **Greedy Leader Dynamic Elimination**",
        "> over native binary Faiss (`IndexBinaryFlat` SIMD pHash) "
        "and pixel-level `DiffMaskVerifier`.",
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
        "`Latency` reports `Median (P95) [Min - Max]` wall-clock ms over `Statistical Rounds`. "
        "Throughput (`prob/s`) uses `end_to_end_ms` as the strict denominator. "
        "`Coverage` counts all recovered GT cluster members (seeds + peers)."
    )
    lines.append("")
    lines.append(
        "| Scenario | Mode | Scale (Items/GT) | Wall Latency Med/P95 [Min-Max] | Throughput | "
        "Formed Batches | Purity (Wrong) | Coverage | Recall |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for r, stats in results_with_stats:
        scale_str = f"{r.total_items} / {r.gt_cluster_count}"
        med_lat = float(stats.get("median_ms", r.end_to_end_ms))
        p95_lat = float(stats.get("p95_ms", r.end_to_end_ms))
        min_lat = float(stats.get("min_ms", r.end_to_end_ms))
        max_lat = float(stats.get("max_ms", r.end_to_end_ms))
        lines.append(
            f"| **{r.scenario_name}** | `{r.mode}` | {scale_str} | "
            f"**{med_lat:.1f}ms** ({p95_lat:.1f}ms) [{min_lat:.1f}-{max_lat:.1f}] | "
            f"{r.probes_per_sec:,.0f} prob/s | "
            f"{r.discovered_batches} batches ({r.total_discovered_members} skins) | "
            f"**{r.avg_batch_purity:.1%}** ({r.wrong_peer_count}) | "
            f"{r.member_coverage_rate:.1%} | **{r.cluster_recall_rate:.1%}** |"
        )

    lines.append("")
    lines.append("## 🔧 Scenario Reproducibility Matrix")
    lines.append("")
    lines.append(
        "| Scenario | Dataset Seed | PRNG Seed | Seed Strategy | batch_size | max_batches | "
        "max_probes |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r, stats in results_with_stats:
        lines.append(
            f"| `{r.scenario_name}` | `{stats.get('dataset_seed', '-')}` | "
            f"`{stats.get('random_seed', '-')}` | `{stats.get('seed_strategy', '-')}` | "
            f"{stats.get('batch_size', '-')} | {stats.get('max_batches', '-')} | "
            f"{stats.get('max_probes', '-')} |"
        )

    lines.append("")
    lines.append("## 🔬 Telemetry & Fine-Grained Latency Decomposition")
    lines.append("")
    lines.append(
        "> **Decomposition**: "
        "`end_to_end_ms` = `instrumented_stage_ms` (`recall` + `io` + `verify`) "
        "+ `unattributed_ms` (loop branching, metadata lookups, elimination, harness overhead); "
        "`service_query_ms` records internal `discover_similar_batches()` elapsed time."
    )
    lines.append("")
    lines.append(
        "| Scenario | Service / Wall | Binary Recall | Parallel I/O | DiffMask Verify | "
        "Unattributed (e2e) | Verified Pairs |"
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
        "> In `mode='variants'`, search constraints are relaxed to Hamming $\\le 12$ and "
        "DiffMask $\\le 0.35$ to discover derivative community recolors, minor clothing "
        "modifications, and edit series.\n"
        "> When exposed to arbitrary community skins (`Sparse Real Haystack (Variants)`), "
        "unlabeled background lookalikes may join discovered batches (see measured "
        "`Purity` and `Wrong` counts in the matrix above).\n"
        "> This reflects a deliberate recall-vs-precision engineering trade-off: clustering "
        "suggestions under `variants` mode are intended for human review workflows rather "
        "than unsupervised deletion."
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
    dataset_seed: int | str = "-",
    **kwargs: Any,
) -> tuple[ScenarioResult, dict[str, Any]]:
    """Run a scenario multiple rounds to compute robust median, p95, and min/max metrics."""
    results: list[ScenarioResult] = []
    latencies: list[float] = []

    for _round_idx in range(rounds):
        res = scenario_fn(**kwargs)
        results.append(res)
        latencies.append(res.end_to_end_ms)

    # Sort results by end_to_end_ms to pick median representative
    sorted_idx = int(np.argsort(latencies)[len(latencies) // 2])
    representative = results[sorted_idx]

    stats: dict[str, Any] = {
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "dataset_seed": dataset_seed,
        "random_seed": kwargs.get("random_seed", "-"),
        "seed_strategy": kwargs.get("seed_strategy", "-"),
        "batch_size": kwargs.get("batch_size", "-"),
        "max_batches": kwargs.get("max_batches", "-"),
        "max_probes": kwargs.get("max_probes", "-"),
    }
    return representative, stats


def _run_and_record(
    results_with_stats: list[tuple[ScenarioResult, dict[str, Any]]],
    step_label: str,
    rounds: int,
    dataset_seed: int | str,
    **kwargs: Any,
) -> None:
    """Execute one benchmark scenario, append stats, and log standardized summary."""
    logger.info("--> Running %s", step_label)
    res, stats = run_scenario_with_stats(
        evaluate_cluster_scenario,
        rounds=rounds,
        dataset_seed=dataset_seed,
        **kwargs,
    )
    results_with_stats.append((res, stats))
    logger.info(
        "    Done: %d batches formed in %.2fms (Med: %.2fms, P95: %.2fms, "
        "Purity: %.1f%%, Coverage: %.1f%%, Recall: %.1f%%)",
        res.discovered_batches,
        res.end_to_end_ms,
        stats["median_ms"],
        stats["p95_ms"],
        res.avg_batch_purity * 100,
        res.member_coverage_rate * 100,
        res.cluster_recall_rate * 100,
    )


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

    use_zip = not args.no_zip
    logger.info("Initializing Similarity Discovery Benchmark Suite...")
    env_meta = get_environment_metadata(rounds=args.rounds, samples=args.samples, use_zip=use_zip)
    results_with_stats: list[tuple[ScenarioResult, dict[str, Any]]] = []

    # 1. Duplicate Swarms (Exact & Near-Lossless Duplicates)
    dup_items, dup_counts = build_duplicate_swarms_dataset(
        swarm_sizes=[30, 20, 15], distractor_count=40, seed=101
    )
    _run_and_record(
        results_with_stats,
        "Scenario 1: Exact Duplicate Swarms (Top-K Starvation Stress)",
        rounds=args.rounds,
        dataset_seed=101,
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

    # 2. Large Scale Duplicate Penetration (50 identical skins)
    large_items, large_counts = build_duplicate_swarms_dataset(
        swarm_sizes=[50], distractor_count=20, seed=102
    )
    _run_and_record(
        results_with_stats,
        "Scenario 2: Massive 50-Skin Duplicate Swarm",
        rounds=args.rounds,
        dataset_seed=102,
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

    # 3. Multi-Cluster Derivative Variants (Recolors & Edits)
    var_items, var_counts = build_variant_clusters_dataset(
        num_clusters=5, items_per_cluster=6, distractor_count=40, seed=201
    )
    _run_and_record(
        results_with_stats,
        "Scenario 3: Derivative Variant Clusters (Recolors / Edits)",
        rounds=args.rounds,
        dataset_seed=201,
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

    # 4. Sparse Real-World Haystack (Confirmed & Variants & Sequential)
    real_items, real_counts = build_sparse_real_haystack(
        background_size=args.samples,
        num_duplicate_pairs=10,
        num_variant_clusters=5,
        use_zip=use_zip,
        seed=301,
    )
    dup_gt_counts = {k: v for k, v in real_counts.items() if "dup" in k}
    _run_and_record(
        results_with_stats,
        "Scenario 4: Sparse Real Haystack (Confirmed Duplicates)",
        rounds=args.rounds,
        dataset_seed=301,
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

    var_gt_counts = {k: v for k, v in real_counts.items() if "var" in k}
    _run_and_record(
        results_with_stats,
        "Scenario 5: Sparse Real Haystack (Variants Exploration)",
        rounds=args.rounds,
        dataset_seed=301,
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

    _run_and_record(
        results_with_stats,
        "Scenario 6: Seed Strategy Comparison (Sequential on Haystack)",
        rounds=args.rounds,
        dataset_seed=301,
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

    # 7 & 8. Threshold Boundary Verification (Confirmed & Variants)
    bc_items, bc_counts = build_boundary_dataset(seed=401, mode="confirmed")
    _run_and_record(
        results_with_stats,
        "Scenario 7: Threshold Boundary Gating (Confirmed)",
        rounds=args.rounds,
        dataset_seed=401,
        scenario_name="Threshold Boundary Gating (Confirmed)",
        items=bc_items,
        ground_truth_counts=bc_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=5,
        max_probes=200,
        seed_strategy="sequential",
        random_seed=42,
    )

    bv_items, bv_counts = build_boundary_dataset(seed=401, mode="variants")
    _run_and_record(
        results_with_stats,
        "Scenario 8: Threshold Boundary Gating (Variants)",
        rounds=args.rounds,
        dataset_seed=401,
        scenario_name="Threshold Boundary Gating (Variants)",
        items=bv_items,
        ground_truth_counts=bv_counts,
        mode="variants",
        batch_size=5,
        max_batches=5,
        max_probes=200,
        seed_strategy="sequential",
        random_seed=42,
    )

    # 9. Transitivity Chain Elimination
    t_items, t_counts = build_transitivity_chain_dataset(seed=501)
    _run_and_record(
        results_with_stats,
        "Scenario 9: Anti-Transitivity Elimination",
        rounds=args.rounds,
        dataset_seed=501,
        scenario_name="Anti-Transitivity Elimination",
        items=t_items,
        ground_truth_counts=t_counts,
        mode="variants",
        batch_size=5,
        max_batches=5,
        max_probes=200,
        seed_strategy="sequential",
        random_seed=42,
    )

    # 10. Adversarial Torture Suite
    adv_items, adv_counts = build_adversarial_dataset(seed=600)
    _run_and_record(
        results_with_stats,
        "Scenario 10: Adversarial Torture Suite (Confirmed)",
        rounds=args.rounds,
        dataset_seed=600,
        scenario_name="Adversarial Torture Suite",
        items=adv_items,
        ground_truth_counts=adv_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=10,
        max_probes=200,
        seed_strategy="random",
        random_seed=42,
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
