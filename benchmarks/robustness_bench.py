#!/usr/bin/env python3
"""Minecraft Texture Retrieval Robustness Benchmark Runner.

Evaluates and compares search pipelines against real Minecraft textures
under multiple controlled perturbations (grayscale, color tint, JPEG compression, etc.).

Usage:
    uv run python benchmarks/robustness_bench.py --samples 50
    uv run python benchmarks/robustness_bench.py --samples 100 --strategies current_diffmask ssim
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.robustness.dataset import (  # noqa: E402
    PerturbedQuery,
    build_evaluation_queries,
    sample_anchor_dataset,
)
from benchmarks.robustness.evaluator import (  # noqa: E402
    BenchmarkReport,
    evaluate_strategy,
    populate_benchmark_index,
)
from twin.models.clip_model import load as load_clip  # noqa: E402
from twin.services.embedding import compute_embedding  # noqa: E402
from twin.services.indexer import indexer  # noqa: E402
from twin.services.search import search  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("robustness_bench")


# ── Strategies definitions ──────────────────────────────────────────────────


def _strategy_current_diffmask(q: PerturbedQuery, top_k: int = 20) -> dict[str, Any]:
    """Optimized current pipeline using Flat SIMD pHash + DiffMask verification."""
    return search(
        q.image,
        top_k=top_k,
        datasource=q.datasource,
        verifier_type="diffmask",
    )


def _strategy_ssim_pipeline(q: PerturbedQuery, top_k: int = 20) -> dict[str, Any]:
    """Pipeline using legacy SSIM as verifier instead of DiffMask."""
    return search(
        q.image,
        top_k=top_k,
        datasource=q.datasource,
        verifier_type="ssim",
    )


def _strategy_pure_phash_binary(q: PerturbedQuery, top_k: int = 20) -> dict[str, Any]:
    """Pure perceptual hash binary recall without deep verifier."""
    return search(
        q.image,
        top_k=top_k,
        datasource=q.datasource,
        phash_threshold=20,
        ssim_threshold=0.0,
    )


def _strategy_pure_clip_dense(q: PerturbedQuery, top_k: int = 20) -> dict[str, Any]:
    """Dense CLIP vector search (cosine / L2 similarity only)."""
    vec = compute_embedding(q.image)
    distances, ids = indexer.search(vec, k=top_k, datasource=q.datasource)
    cands = []
    for d, idx in zip(distances, ids):
        meta = indexer.get_metadata(idx, datasource=q.datasource) or {}
        cands.append(
            {
                "id": idx,
                "filename": meta.get("filename", ""),
                "distance": float(d),
                "match_level": "none",
                "stages_passed": 1,
            }
        )
    return {
        "results": cands,
        "count": len(cands),
        "datasource": q.datasource,
        "stages": {"dense_clip": {"elapsed_ms": 0.0}},
    }


AVAILABLE_STRATEGIES = {
    "current_diffmask": _strategy_current_diffmask,
    "ssim_verifier": _strategy_ssim_pipeline,
    "pure_phash": _strategy_pure_phash_binary,
    "pure_clip": _strategy_pure_clip_dense,
}


# ── Comparison summary formatting ───────────────────────────────────────────


def print_comparison_table(reports: list[BenchmarkReport]) -> str:
    """Render cross-strategy comparison summary in markdown table."""
    headers = [
        "Strategy",
        "Recall@1",
        "Recall@5",
        "MRR",
        "Suspected",
        "Confirmed",
        "P50 Latency",
        "Mean Latency",
    ]
    rows = []
    for r in reports:
        rows.append(
            f"| **{r.strategy_name}** | **{r.recall_at_1:.2%}** | {r.recall_at_5:.2%} | "
            f"{r.mrr:.4f} | {r.suspected_rate:.2%} | {r.confirmed_rate:.2%} | "
            f"{r.latency_p50_ms:.2f}ms | {r.latency_mean_ms:.2f}ms |"
        )

    table_md = "\n".join(
        [
            "## 🏆 Strategy Comparison Summary",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |",
            *rows,
        ]
    )
    return table_md


# ── Main CLI ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Twin Texture Robustness Benchmark")
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of anchor textures to sample (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K search candidates (default: 20)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["current_diffmask", "ssim_verifier", "pure_phash", "pure_clip"],
        choices=list(AVAILABLE_STRATEGIES.keys()),
        help="Strategies to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output markdown/json report",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    logger.info("Initializing OpenCLIP and dependencies...")
    load_clip()

    logger.info("Sampling %d anchor textures (seed=%d)...", args.samples, args.seed)
    anchors = sample_anchor_dataset(sample_size=args.samples, seed=args.seed)
    logger.info("Sampled %d anchor textures successfully.", len(anchors))

    reports: list[BenchmarkReport] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info("Indexing anchor textures into in-memory Faiss index and temp storage...")
        populate_benchmark_index(anchors, tmp_dir=tmp_dir)
        logger.info("Index populated. Indexer count: %d", indexer.count)

        logger.info(
            "Generating evaluation queries across perturbation types (with identity baseline)..."
        )
        queries = build_evaluation_queries(anchors)
        logger.info("Generated %d evaluation query test cases.", len(queries))

        for s_name in args.strategies:
            logger.info("Evaluating strategy '%s' against %d queries...", s_name, len(queries))
            fn = AVAILABLE_STRATEGIES[s_name]
            rep = evaluate_strategy(
                strategy_name=s_name,
                search_fn=lambda q, fn=fn: fn(q, top_k=args.top_k),
                queries=queries,
            )
            reports.append(rep)
            logger.info(
                "[%s] Recall@1=%.2f%%, Recall@5=%.2f%%, "
                "Suspected=%.2f%%, Confirmed=%.2f%%, P50=%.2fms",
                s_name,
                rep.recall_at_1 * 100,
                rep.recall_at_5 * 100,
                rep.suspected_rate * 100,
                rep.confirmed_rate * 100,
                rep.latency_p50_ms,
            )

    # Print markdown summaries
    print("\n" + "=" * 80)
    summary_table = print_comparison_table(reports)
    print(summary_table)
    print("=" * 80 + "\n")

    for rep in reports:
        print(rep.to_markdown_table())
        print("\n" + "-" * 60 + "\n")

    total_time = time.perf_counter() - t_start
    logger.info("Benchmark finished in %.2fs.", total_time)

    # Optional file output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            payload = {
                "samples": args.samples,
                "queries_count": len(queries),
                "total_time_seconds": round(total_time, 2),
                "strategies": [
                    {
                        "name": r.strategy_name,
                        "recall_at_1": r.recall_at_1,
                        "recall_at_5": r.recall_at_5,
                        "mrr": r.mrr,
                        "confirmed_rate": r.confirmed_rate,
                        "suspected_rate": r.suspected_rate,
                        "detected_rate": r.detected_rate,
                        "latency_p50_ms": r.latency_p50_ms,
                        "latency_p95_ms": r.latency_p95_ms,
                        "latency_mean_ms": r.latency_mean_ms,
                        "by_perturbation": r.by_perturbation,
                    }
                    for r in reports
                ],
            }
            out_path.write_text(json.dumps(payload, indent=2))
        else:
            full_md = summary_table + "\n\n" + "\n\n".join(r.to_markdown_table() for r in reports)
            out_path.write_text(full_md, encoding="utf-8")
        logger.info("Report written to %s", out_path)


if __name__ == "__main__":
    main()
