"""Evaluation metrics engine for Minecraft texture robustness benchmarks.

Computes Recall@1, Recall@5, MRR, verification success rate, and latency distributions
across various perturbation categories and retrieval strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from benchmarks.robustness.dataset import AnchorSample, PerturbedQuery
from twin.services.embedding import compute_embedding
from twin.services.hasher import compute_ahash, compute_dhash, compute_phash
from twin.services.indexer import indexer


@dataclass
class QueryResultMetric:
    """Evaluation result for a single test query."""

    query_id: str
    target_filename: str
    perturbation_type: str
    rank: int | None  # 1-indexed rank, or None if not recalled
    hit_at_1: bool
    hit_at_5: bool
    reciprocal_rank: float
    confirmed: bool
    suspected: bool
    match_level: str
    elapsed_ms: float
    stages: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark metrics across queries."""

    strategy_name: str
    total_queries: int
    recall_at_1: float
    recall_at_5: float
    mrr: float
    confirmed_rate: float
    suspected_rate: float
    detected_rate: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    by_perturbation: dict[str, dict[str, float]]
    raw_results: list[QueryResultMetric] = field(default_factory=list)

    def to_markdown_table(self) -> str:
        """Format aggregated metrics into a human-readable markdown table."""
        lines = [
            f"### Strategy: {self.strategy_name}",
            f"- **Total Queries**: {self.total_queries}",
            f"- **Recall@1**: {self.recall_at_1:.2%}",
            f"- **Recall@5**: {self.recall_at_5:.2%}",
            f"- **MRR**: {self.mrr:.4f}",
            f"- **Detected (Confirmed + Suspected)**: {self.detected_rate:.2%}",
            f"- **Confirmed Rate**: {self.confirmed_rate:.2%}",
            f"- **Suspected Rate**: {self.suspected_rate:.2%}",
            f"- **Latency (P50 / P95 / Mean)**: "
            f"{self.latency_p50_ms:.2f}ms / {self.latency_p95_ms:.2f}ms / "
            f"{self.latency_mean_ms:.2f}ms",
            "",
            "| Perturbation | Count | Recall@1 | Recall@5 | Suspected | Confirmed | P50 (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for p_name, stats in sorted(self.by_perturbation.items()):
            lines.append(
                f"| `{p_name}` | {int(stats['count'])} | {stats['recall_at_1']:.1%} | "
                f"{stats['recall_at_5']:.1%} | {stats['suspected_rate']:.1%} | "
                f"{stats['confirmed_rate']:.1%} | {stats['latency_p50_ms']:.2f}ms |"
            )
        return "\n".join(lines)


def populate_benchmark_index(anchors: list[AnchorSample]) -> None:
    """Index anchor textures cleanly into the singleton indexer."""
    indexer.clear()
    for a in anchors:
        vec = compute_embedding(a.image)
        meta = {
            "filename": a.filename,
            "path": f"/virtual/{a.datasource}/{a.filename}",
            "datasource": a.datasource,
            "texture_type": a.texture_type.value,
            "ahash": compute_ahash(a.image),
            "dhash": compute_dhash(a.image),
            "phash": compute_phash(a.image),
        }
        indexer.add_item(vec, meta, datasource=a.datasource)


def evaluate_strategy(
    strategy_name: str,
    search_fn: Callable[[PerturbedQuery], dict[str, Any]],
    queries: list[PerturbedQuery],
) -> BenchmarkReport:
    """Evaluate a search strategy against a set of perturbed queries."""
    results: list[QueryResultMetric] = []

    for q in queries:
        t0 = time.perf_counter()
        resp = search_fn(q)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        cands = resp.get("results", [])
        rank: int | None = None
        target_item: dict[str, Any] | None = None

        for idx, c in enumerate(cands, start=1):
            if c.get("filename") == q.target_filename:
                rank = idx
                target_item = c
                break

        hit_1 = rank == 1
        hit_5 = rank is not None and rank <= 5
        rr = (1.0 / rank) if rank is not None else 0.0
        m_level = target_item.get("match_level", "none") if target_item else "none"
        confirmed = m_level == "confirmed"
        suspected = m_level == "suspected"

        results.append(
            QueryResultMetric(
                query_id=q.query_id,
                target_filename=q.target_filename,
                perturbation_type=q.perturbation_type,
                rank=rank,
                hit_at_1=hit_1,
                hit_at_5=hit_5,
                reciprocal_rank=rr,
                confirmed=confirmed,
                suspected=suspected,
                match_level=m_level,
                elapsed_ms=elapsed_ms,
                stages=resp.get("stages", {}),
            )
        )

    # Compute overall aggregates
    n = len(results)
    if n == 0:
        return BenchmarkReport(
            strategy_name=strategy_name,
            total_queries=0,
            recall_at_1=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            confirmed_rate=0.0,
            suspected_rate=0.0,
            detected_rate=0.0,
            latency_mean_ms=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            by_perturbation={},
        )

    r1 = sum(1 for r in results if r.hit_at_1) / n
    r5 = sum(1 for r in results if r.hit_at_5) / n
    mrr = sum(r.reciprocal_rank for r in results) / n
    conf_rate = sum(1 for r in results if r.confirmed) / n
    susp_rate = sum(1 for r in results if r.match_level == "suspected") / n
    det_rate = sum(1 for r in results if r.confirmed or r.match_level == "suspected") / n

    latencies = [r.elapsed_ms for r in results]
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))

    # Compute per-perturbation breakdown
    by_p: dict[str, list[QueryResultMetric]] = {}
    for r in results:
        by_p.setdefault(r.perturbation_type, []).append(r)

    p_breakdown: dict[str, dict[str, float]] = {}
    for p_name, sub_results in by_p.items():
        k = len(sub_results)
        sub_lats = [r.elapsed_ms for r in sub_results]
        p_breakdown[p_name] = {
            "count": float(k),
            "recall_at_1": sum(1 for r in sub_results if r.hit_at_1) / k,
            "recall_at_5": sum(1 for r in sub_results if r.hit_at_5) / k,
            "mrr": sum(r.reciprocal_rank for r in sub_results) / k,
            "confirmed_rate": sum(1 for r in sub_results if r.confirmed) / k,
            "suspected_rate": sum(1 for r in sub_results if r.match_level == "suspected") / k,
            "detected_rate": sum(
                1 for r in sub_results if r.confirmed or r.match_level == "suspected"
            ) / k,
            "latency_p50_ms": float(np.percentile(sub_lats, 50)),
            "latency_mean_ms": float(np.mean(sub_lats)),
        }

    return BenchmarkReport(
        strategy_name=strategy_name,
        total_queries=n,
        recall_at_1=r1,
        recall_at_5=r5,
        mrr=mrr,
        confirmed_rate=conf_rate,
        suspected_rate=susp_rate,
        detected_rate=det_rate,
        latency_mean_ms=mean_lat,
        latency_p50_ms=p50_lat,
        latency_p95_ms=p95_lat,
        by_perturbation=p_breakdown,
        raw_results=results,
    )
