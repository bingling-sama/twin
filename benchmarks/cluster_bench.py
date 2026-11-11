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
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Initialize OpenMP thread controls before importing faiss/torch/numpy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import PIL  # noqa: E402

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
    assess_quality_risk,
    evaluate_cluster_scenario,
    summarize_stage_breakdown,
)
from benchmarks.cluster.meta import (  # noqa: E402
    compute_cluster_source_digest,
    get_cluster_dataset_fingerprint,
)
from twin.core.config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cluster_bench")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_SCENARIO_NAMES: tuple[str, ...] = (
    "Exact Duplicate Swarms",
    "Massive 50-Skin Duplicate Swarm",
    "Derivative Variant Clusters",
    "Sparse Real Haystack (Confirmed)",
    "Sparse Real Haystack (Variants)",
    "Sequential Seed Strategy",
    "Threshold Boundary Gating (Confirmed)",
    "Threshold Boundary Gating (Variants)",
    "Anti-Transitivity Elimination",
    "Adversarial Torture Suite",
)


def _git_head_and_clean(repo_root: Path = _ROOT) -> tuple[str, bool]:
    """Return (full_head_sha, is_clean) from git for report provenance checks."""
    commit = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        )
        .decode("ascii")
        .strip()
    )
    dirty_out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL
    ).decode("utf-8", errors="replace")
    return commit, not bool(dirty_out.strip())


def get_environment_metadata(
    rounds: int = 3,
    samples: int = 200,
    use_zip: bool = True,
) -> dict[str, str]:
    """Capture execution environment metadata for strict benchmark reproducibility."""
    try:
        commit, is_clean = _git_head_and_clean(_ROOT)
        dirty_tag = " (clean)" if is_clean else " (dirty)"
    except Exception:
        commit = "unknown"
        dirty_tag = ""

    source_digest = compute_cluster_source_digest()
    dataset_fingerprint = get_cluster_dataset_fingerprint()

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
        "Dataset Fingerprint (SHA-256)": dataset_fingerprint,
        "Python": platform.python_version(),
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
        "Throughput (`prob/s` and `pairs/s`) uses `end_to_end_ms` as the strict denominator. "
        "`Coverage` counts all recovered GT cluster members (seeds + peers). "
        "`Quality Risk` flags scenarios with wrong peers, impure batches, recall < 100%, "
        "or coverage < 100%."
    )
    lines.append("")
    lines.append(
        "| Scenario | Mode | Scale (Items/GT) | Wall Latency Med/P95 [Min-Max] | Throughput | "
        "Formed Batches | Purity / Impure Rate (Wrong) | Coverage | Recall | Quality Risk |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for r, stats in results_with_stats:
        scale_str = f"{r.total_items} / {r.gt_cluster_count}"
        med_lat = float(stats.get("median_ms", r.end_to_end_ms))
        p95_lat = float(stats.get("p95_ms", r.end_to_end_ms))
        min_lat = float(stats.get("min_ms", r.end_to_end_ms))
        max_lat = float(stats.get("max_ms", r.end_to_end_ms))
        risk_flags = assess_quality_risk(r)
        risk_badge = f"RISK: {', '.join(risk_flags)}" if risk_flags else "Clean"
        lines.append(
            f"| **{r.scenario_name}** | `{r.mode}` | {scale_str} | "
            f"**{med_lat:.1f}ms** ({p95_lat:.1f}ms) [{min_lat:.1f}-{max_lat:.1f}] | "
            f"{r.probes_per_sec:,.0f} prob/s | "
            f"{r.discovered_batches} batches ({r.total_discovered_members} skins) | "
            f"**{r.avg_batch_purity:.1%}** / {r.impure_batch_rate:.1%} ({r.wrong_peer_count}) | "
            f"{r.member_coverage_rate:.1%} | **{r.cluster_recall_rate:.1%}** | `{risk_badge}` |"
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
        "`service_query_ms` records internal `discover_similar_batches()` elapsed time; "
        "`pairs/s` uses `end_to_end_ms` as denominator."
    )
    lines.append("")
    lines.append(
        "| Scenario | Service / Stages / Wall | Binary Recall | Parallel I/O | "
        "DiffMask Verify | Unattributed (e2e) | Verified Pairs (e2e) |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for r, _ in results_with_stats:
        bd = summarize_stage_breakdown(r.stages)
        lines.append(
            f"| `{r.scenario_name}` | "
            f"{r.service_query_ms:.1f}ms / {r.instrumented_stage_ms:.1f}ms / "
            f"{r.end_to_end_ms:.1f}ms | "
            f"{bd['recall_ms']:.1f}ms ({bd['recall_pct']:.1f}%) | "
            f"{bd['io_ms']:.1f}ms ({bd['io_pct']:.1f}%) | "
            f"{bd['verify_ms']:.1f}ms ({bd['verify_pct']:.1f}%) | "
            f"{r.unattributed_ms:.1f}ms | "
            f"{r.pairs_verified} ({r.pairs_per_sec:,.0f} pairs/s, denominator: end_to_end_ms) |"
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


def _parse_metadata_table(report_text: str) -> dict[str, str]:
    """Parse **Key** | `Value` metadata rows from markdown report."""
    meta: dict[str, str] = {}
    row_re = re.compile(r"^\|\s*\*\*(.+?)\*\*\s*\|\s*`(.*)`\s*\|\s*$")
    for line in report_text.splitlines():
        match = row_re.match(line.strip())
        if match:
            meta[match.group(1).strip()] = match.group(2).strip()
    return meta


def verify_cluster_report(
    report_path: Path | str,
    *,
    repo_root: Path = _ROOT,
) -> list[str]:
    """Verify report completeness, clean Commit SHA == HEAD, and full SHA-256 digests."""
    path = Path(report_path)
    if not path.is_file():
        return [f"Report file not found: {path}"]

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return [f"Report file is empty: {path}"]

    errors: list[str] = []
    meta = _parse_metadata_table(text)
    required_meta_keys = (
        "Generated At",
        "Commit SHA",
        "Source Digest (SHA-256)",
        "Dataset Fingerprint (SHA-256)",
        "Python",
        "OS Platform",
        "CPU",
        "GPU",
        "Torch / Faiss / Pillow",
        "Statistical Rounds",
        "Haystack Config",
        "images.zip Fingerprint",
        "Confirmed Thresholds",
        "Variants Thresholds",
        "Recall Window",
    )
    for key in required_meta_keys:
        if not meta.get(key):
            errors.append(f"Missing required report metadata field: {key}")

    try:
        head_sha, repo_clean = _git_head_and_clean(repo_root)
    except Exception as exc:
        errors.append(f"Failed to query git HEAD/status for report verification: {exc}")
        head_sha, repo_clean = "", False

    if head_sha:
        expected_commit = f"{head_sha} (clean)"
        recorded_commit = meta.get("Commit SHA", "")
        if recorded_commit != expected_commit:
            errors.append(
                f"Report Commit SHA mismatch or dirty state: expected {expected_commit!r}, "
                f"got {recorded_commit!r}"
            )
        if not repo_clean:
            errors.append(
                "Working tree is dirty; commit or clean changes before certifying report."
            )

    expected_src = compute_cluster_source_digest()
    recorded_src = meta.get("Source Digest (SHA-256)", "")
    if not _SHA256_RE.match(recorded_src):
        errors.append(
            f"Report Source Digest must be full 64-char SHA-256 hex, got {recorded_src!r}"
        )
    elif recorded_src != expected_src:
        errors.append(
            f"Report Source Digest mismatch: expected {expected_src}, got {recorded_src}"
        )

    expected_ds = get_cluster_dataset_fingerprint()
    recorded_ds = meta.get("Dataset Fingerprint (SHA-256)", "")
    if not _SHA256_RE.match(recorded_ds):
        errors.append(
            f"Report Dataset Fingerprint must be full 64-char SHA-256 hex, got {recorded_ds!r}"
        )
    elif recorded_ds != expected_ds:
        errors.append(
            f"Report Dataset Fingerprint mismatch: expected {expected_ds}, got {recorded_ds}"
        )

    required_sections = (
        "## 🖥️ Execution Environment Metadata",
        "## 🏆 Executive Summary & Strategy Matrix",
        "## 🔧 Scenario Reproducibility Matrix",
        "## 🔬 Telemetry & Fine-Grained Latency Decomposition",
        "## 📝 Detailed Scenario Analysis",
    )
    for section in required_sections:
        if section not in text:
            errors.append(f"Missing required report section: {section}")

    for scenario_name in _EXPECTED_SCENARIO_NAMES:
        if f"| **{scenario_name}** |" not in text:
            errors.append(f"Missing Executive Summary row for scenario: {scenario_name}")
        if f"### Scenario: {scenario_name} " not in text:
            errors.append(f"Missing Detailed Scenario section for: {scenario_name}")

    if "denominator: end_to_end_ms" not in text or "instrumented stages:" not in text:
        errors.append("Missing explicit denominator or instrumented stage breakdown labels.")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Similarity Discovery Benchmarks")
    parser.add_argument("--samples", type=int, default=200, help="Background scale for haystack")
    parser.add_argument("--rounds", type=int, default=3, help="Statistical evaluation rounds")
    parser.add_argument(
        "--output", type=str, default="benchmarks/cluster_report.md", help="Output report file"
    )
    parser.add_argument(
        "--no-zip", action="store_true", help="Force synthetic fallback instead of images.zip"
    )
    parser.add_argument(
        "--verify-report",
        type=str,
        default=None,
        help="Verify an existing cluster_report.md against HEAD, clean status, and SHA-256 digests",
    )
    args = parser.parse_args(argv)

    if args.verify_report:
        errors = verify_cluster_report(args.verify_report)
        if errors:
            print("CLUSTER REPORT VERIFICATION FAILED:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        print(f"CLUSTER REPORT VERIFIED CLEAN: {args.verify_report}")
        return 0

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
