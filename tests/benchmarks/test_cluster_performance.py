"""Performance-only benchmarks for seedless texture discovery (Gate 4).

Contains only @pytest.mark.smoke tests using pytest-benchmark.
All dataset build, disk writes, and Indexer population happen strictly OUTSIDE the timed
benchmark callable so timing isolates discover_similar_batches().
Functional correctness and boundary assertions live in test_cluster_correctness.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.cluster.dataset import (
    BenchmarkItem,
    create_base_skin,
    create_variant_skin,
)
from benchmarks.cluster.evaluator import populate_cluster_index
from twin.services.cluster import discover_similar_batches
from twin.services.indexer import Indexer


def _build_perf_dataset(
    num_duplicates: int = 30,
    num_variants: int = 15,
    num_noise: int = 25,
) -> list[BenchmarkItem]:
    """Generate deterministic dataset items for timing benchmarks."""
    items: list[BenchmarkItem] = []

    base_dup = create_base_skin(seed=42)
    for i in range(num_duplicates):
        items.append(
            BenchmarkItem(
                filename=f"dup_{i:04d}.png",
                image=base_dup.copy(),
                cluster_id="cluster_dup_1",
            )
        )

    base_var = create_base_skin(seed=100)
    for i in range(num_variants):
        items.append(
            BenchmarkItem(
                filename=f"var_{i:04d}.png",
                image=create_variant_skin(base_var, variant_idx=i + 1),
                cluster_id="cluster_var_1",
            )
        )

    for i in range(num_noise):
        items.append(
            BenchmarkItem(
                filename=f"noise_{i:04d}.png",
                image=create_base_skin(seed=1000 + i * 17),
                cluster_id=None,
            )
        )

    return items


def _benchmark_discovery(
    benchmark: Any,
    tmp_path: Path,
    items: list[BenchmarkItem],
    **discover_kwargs: Any,
) -> dict[str, Any]:
    """Build sandbox index outside timing and measure discover_similar_batches()."""
    sandbox = Indexer()
    populate_cluster_index(
        items, tmp_dir=tmp_path / "images", datasource="skin", indexer_instance=sandbox
    )
    return benchmark(
        discover_similar_batches,
        datasource="skin",
        indexer_instance=sandbox,
        **discover_kwargs,
    )


@pytest.mark.smoke
def test_bench_cluster_confirmed_duplicates(benchmark: Any, tmp_path: Path) -> None:
    """Measure confirmed duplicate clustering throughput and latency."""
    items = _build_perf_dataset(num_duplicates=25, num_variants=0, num_noise=20)
    _benchmark_discovery(
        benchmark,
        tmp_path,
        items,
        max_batches=10,
        batch_size=5,
        mode="confirmed",
        max_probes=100,
        seed_strategy="random",
        random_seed=42,
    )


@pytest.mark.smoke
def test_bench_cluster_adaptive_recall_scaling(benchmark: Any, tmp_path: Path) -> None:
    """Measure adaptive recall scaling latency on a 50-duplicate swarm."""
    items = _build_perf_dataset(num_duplicates=50, num_variants=0, num_noise=10)
    _benchmark_discovery(
        benchmark,
        tmp_path,
        items,
        max_batches=20,
        batch_size=5,
        mode="confirmed",
        max_probes=200,
        seed_strategy="sequential",
        random_seed=42,
    )


@pytest.mark.smoke
def test_bench_cluster_variants_mode(benchmark: Any, tmp_path: Path) -> None:
    """Measure variants mode discovery latency on recolored derivative textures."""
    items = _build_perf_dataset(num_duplicates=10, num_variants=20, num_noise=20)
    _benchmark_discovery(
        benchmark,
        tmp_path,
        items,
        max_batches=10,
        batch_size=5,
        mode="variants",
        max_probes=100,
        seed_strategy="random",
        random_seed=42,
    )

