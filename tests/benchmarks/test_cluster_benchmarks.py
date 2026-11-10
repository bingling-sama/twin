"""Benchmarks and correctness tests for seedless texture discovery via Greedy Leader Elimination.

Four-tiered verification:
  1. Performance Benchmarks: Integrates with pytest-benchmark (--benchmark-only)
  2. Memory & Disk Sentinel Isolation: Verifies global indexer singleton & host disk are untouched
  3. Deterministic Reproducibility: Verifies identical batches with fixed PRNG seed
  4. Mathematical Contracts & Boundaries: Verifies strict threshold rejection & anti-transitivity
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmarks.cluster.dataset import (
    BenchmarkItem,
    build_boundary_dataset,
    build_transitivity_chain_dataset,
    create_base_skin,
    create_variant_skin,
)
from benchmarks.cluster.evaluator import (
    evaluate_cluster_scenario,
    isolated_index_environment,
    populate_cluster_index,
)
from twin.core.config import settings
from twin.services.cluster import discover_similar_batches
from twin.services.indexer import Indexer, indexer

try:
    import pytest_benchmark  # noqa: F401
except ImportError:

    @pytest.fixture
    def benchmark():
        """Fallback runner when pytest-benchmark is not installed."""

        def _runner(func, *args, **kwargs):
            return func(*args, **kwargs)

        return _runner


def _create_synthetic_skin(seed: int, recolor_torso: bool = False) -> Image.Image:
    """Create a 64x64 standard RGBA skin for benchmark testing."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((64, 64, 4), dtype=np.uint8)

    r = int(rng.randint(50, 200))
    g = int(rng.randint(50, 200))
    b = int(rng.randint(50, 200))

    # Base head and torso
    arr[8:16, 8:24] = [r, g, b, 255]
    if not recolor_torso:
        arr[20:32, 20:36] = [(r + 40) % 255, (g + 40) % 255, (b + 40) % 255, 255]
    else:
        arr[20:32, 20:36] = [(r + 140) % 255, (g + 140) % 255, (b + 140) % 255, 255]
    arr[32:48, 0:64] = [r // 2, g // 2, b // 2, 255]
    arr[0:48, 0:64, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


def _build_cluster_dataset(
    num_duplicates: int = 30,
    num_variants: int = 15,
    num_noise: int = 25,
) -> list[BenchmarkItem]:
    """Generate in-memory benchmark items for test population."""
    items: list[BenchmarkItem] = []

    # 1. Duplicates
    base_dup = _create_synthetic_skin(seed=42)
    for i in range(num_duplicates):
        items.append(
            BenchmarkItem(
                filename=f"dup_{i:04d}.png",
                image=base_dup.copy(),
                cluster_id="cluster_dup_1",
            )
        )

    # 2. Variants
    base_var = _create_synthetic_skin(seed=100)
    for i in range(num_variants):
        var_img = create_variant_skin(base_var, variant_idx=i + 1)
        items.append(
            BenchmarkItem(
                filename=f"var_{i:04d}.png",
                image=var_img,
                cluster_id="cluster_var_1",
            )
        )

    # 3. Noise
    for i in range(num_noise):
        noise_img = _create_synthetic_skin(seed=1000 + i * 17)
        items.append(
            BenchmarkItem(
                filename=f"noise_{i:04d}.png",
                image=noise_img,
                cluster_id=None,
            )
        )

    return items


@pytest.mark.smoke
def test_bench_cluster_confirmed_duplicates(benchmark, tmp_path: Path) -> None:
    """Benchmark confirmed mode: pure duplicates discovery with zero diff."""
    items = _build_cluster_dataset(num_duplicates=25, num_variants=0, num_noise=20)
    img_dir = tmp_path / "images"
    idx_dir = tmp_path / "index"

    sandbox = Indexer()
    with isolated_index_environment(index_dir=idx_dir):
        populate_cluster_index(
            items, tmp_dir=img_dir, datasource="skin", indexer_instance=sandbox
        )

        res = benchmark(
            discover_similar_batches,
            datasource="skin",
            max_batches=10,
            batch_size=5,
            mode="confirmed",
            max_probes=100,
            seed_strategy="random",
            indexer_instance=sandbox,
            random_seed=42,
        )

        batches = res["batches"]
        assert len(batches) >= 4
        for b in batches:
            assert b["mode"] == "confirmed"
            assert b["max_diff_ratio"] <= 0.05
            for m in b["members"]:
                assert m["diff_ratio"] == 0.0


@pytest.mark.smoke
def test_bench_cluster_adaptive_recall_scaling(benchmark, tmp_path: Path) -> None:
    """Benchmark adaptive recall expansion on a large swarm of 50 identical duplicates."""
    items = _build_cluster_dataset(num_duplicates=50, num_variants=0, num_noise=10)
    img_dir = tmp_path / "images"
    idx_dir = tmp_path / "index"

    sandbox = Indexer()
    with isolated_index_environment(index_dir=idx_dir):
        populate_cluster_index(
            items, tmp_dir=img_dir, datasource="skin", indexer_instance=sandbox
        )

        res = benchmark(
            discover_similar_batches,
            datasource="skin",
            max_batches=20,
            batch_size=5,
            mode="confirmed",
            max_probes=200,
            seed_strategy="sequential",
            indexer_instance=sandbox,
            random_seed=42,
        )

        batches = res["batches"]
        assert len(batches) == 10
        total_items_in_batches = sum(b["size"] for b in batches)
        assert total_items_in_batches == 50

        stages = res["stages"]
        assert stages["recall"]["probes_evaluated"] >= 10
        assert stages["verify"]["batches_formed"] == 10


@pytest.mark.smoke
def test_bench_cluster_variants_mode(benchmark, tmp_path: Path) -> None:
    """Benchmark variants mode on recolored derivative textures."""
    items = _build_cluster_dataset(num_duplicates=10, num_variants=20, num_noise=20)
    img_dir = tmp_path / "images"
    idx_dir = tmp_path / "index"

    sandbox = Indexer()
    with isolated_index_environment(index_dir=idx_dir):
        populate_cluster_index(
            items, tmp_dir=img_dir, datasource="skin", indexer_instance=sandbox
        )

        res = benchmark(
            discover_similar_batches,
            datasource="skin",
            max_batches=10,
            batch_size=5,
            mode="variants",
            max_probes=100,
            seed_strategy="random",
            indexer_instance=sandbox,
            random_seed=42,
        )

        batches = res["batches"]
        assert len(batches) >= 2
        for b in batches:
            assert b["mode"] == "variants"
            assert b["max_diff_ratio"] <= 0.35


def test_cluster_memory_and_disk_isolation_sentinel(tmp_path: Path) -> None:
    """Regression test: verify benchmark execution preserves host disk and memory singleton."""
    host_dir = tmp_path / "host_data"
    host_dir.mkdir(parents=True)
    sentinel_file = host_dir / "sentinel.txt"
    sentinel_file.write_text("CRITICAL_HOST_INDEX_DATA", encoding="utf-8")

    sentinel_faiss = host_dir / "skin" / "index.flat.faiss"
    sentinel_faiss.parent.mkdir(parents=True)
    sentinel_faiss.write_text("FAISS_BINARY_CONTENT", encoding="utf-8")

    old_path = settings.index_path

    # 1. Pre-populate global indexer singleton with 10 dummy items in memory
    indexer.clear()
    dummy_vec = np.zeros(512, dtype=np.float32)
    for i in range(10):
        indexer.add_item(
            dummy_vec,
            {"filename": f"host_singleton_{i}.png", "path": f"/host/{i}.png"},
            datasource="skin",
        )
    assert indexer.get_count("skin") == 10

    # Point settings.index_path to host_dir containing critical data
    settings.index_path = str(host_dir)

    try:
        items = _build_cluster_dataset(num_duplicates=10, num_variants=0, num_noise=5)
        # Execute benchmark scenario (must use decoupled sandbox indexer)
        res = evaluate_cluster_scenario(
            scenario_name="Sentinel Host Protection Test",
            items=items,
            ground_truth_counts={"cluster_dup_1": 10},
            mode="confirmed",
            max_batches=5,
            batch_size=5,
        )
        assert res.discovered_batches == 2

        # Assert disk sentinel files were NEVER unlinked or modified
        assert sentinel_file.is_file(), "Sentinel file was deleted by benchmark!"
        assert sentinel_file.read_text(encoding="utf-8") == "CRITICAL_HOST_INDEX_DATA"
        assert sentinel_faiss.is_file(), "Host faiss index was deleted by benchmark!"
        assert sentinel_faiss.read_text(encoding="utf-8") == "FAISS_BINARY_CONTENT"

        # Assert global in-memory singleton was completely unpolluted!
        assert indexer.get_count("skin") == 10, (
            f"Global indexer was mutated! Count is {indexer.get_count('skin')}, expected 10."
        )
        first_meta = indexer.get_metadata(0, datasource="skin")
        assert first_meta is not None and first_meta["filename"] == "host_singleton_0.png"
    finally:
        settings.index_path = old_path
        indexer.clear()


def test_cluster_deterministic_reproducibility() -> None:
    """Verify identical batches and metrics across runs with the same PRNG seed."""
    items = _build_cluster_dataset(num_duplicates=15, num_variants=10, num_noise=15)
    gt_counts = {"cluster_dup_1": 15, "cluster_var_1": 10}

    # Run 1
    res1 = evaluate_cluster_scenario(
        scenario_name="Deterministic Run 1",
        items=items,
        ground_truth_counts=gt_counts,
        mode="confirmed",
        random_seed=42,
        seed_strategy="random",
    )

    # Run 2
    res2 = evaluate_cluster_scenario(
        scenario_name="Deterministic Run 2",
        items=items,
        ground_truth_counts=gt_counts,
        mode="confirmed",
        random_seed=42,
        seed_strategy="random",
    )

    assert res1.discovered_batches == res2.discovered_batches
    assert res1.avg_batch_purity == res2.avg_batch_purity
    assert res1.cluster_recall_rate == res2.cluster_recall_rate
    assert res1.member_coverage_rate == res2.member_coverage_rate
    assert res1.max_observed_diff_ratio == res2.max_observed_diff_ratio


def test_cluster_boundary_rejection() -> None:
    """Verify strict rejection when diff_ratio exceeds threshold."""
    items, gt_counts = build_boundary_dataset(seed=401)

    # In confirmed mode (max_diff = 0.05):
    # - conf_member_near (diff ~ 0.045) should be grouped with conf_base
    # - conf_distractor_beyond (diff ~ 0.065) must NOT be grouped
    res = evaluate_cluster_scenario(
        scenario_name="Boundary Rejection Test",
        items=items,
        ground_truth_counts=gt_counts,
        mode="confirmed",
        batch_size=5,
        max_batches=5,
        seed_strategy="sequential",
    )

    assert res.discovered_batches == 1
    assert res.avg_batch_purity == 1.0
    assert res.max_observed_diff_ratio <= 0.05


def test_cluster_transitivity_chain_elimination() -> None:
    """Verify Greedy Leader Dynamic Elimination prevents non-transitive cluster collapse."""
    items, gt_counts = build_transitivity_chain_dataset(seed=501)

    # A ~ B (0.18), B ~ C (0.20), A !~ C (0.48)
    # Under variants mode (threshold = 0.35):
    # Cluster AB should form, B eliminated, C must NOT be in cluster AB
    res = evaluate_cluster_scenario(
        scenario_name="Transitivity Chain Test",
        items=items,
        ground_truth_counts=gt_counts,
        mode="variants",
        batch_size=5,
        max_batches=5,
        seed_strategy="sequential",
    )

    assert res.discovered_batches == 1
    assert res.avg_batch_purity == 1.0
    assert res.max_observed_diff_ratio <= 0.35


def test_evaluator_recall_bounded_in_zero_to_one() -> None:
    """Regression test: verify Recall numerator is strictly filtered to target GT set."""
    items = _build_cluster_dataset(num_duplicates=10, num_variants=10, num_noise=10)

    res = evaluate_cluster_scenario(
        scenario_name="Mixed Dataset Bounded Recall Test",
        items=items,
        ground_truth_counts={"cluster_var_1": 10},
        mode="variants",
        max_batches=10,
        batch_size=5,
        random_seed=42,
    )
    assert 0.0 <= res.cluster_recall_rate <= 1.0
    assert 0.0 <= res.member_coverage_rate <= 1.0
    assert res.cluster_recall_rate == 1.0


def test_evaluator_group_recovery_requires_peer() -> None:
    """Regression test: verify a cluster is NOT recovered if batch members are only distractors."""
    phantom_base = create_base_skin(seed=999)
    items: list[BenchmarkItem] = [
        BenchmarkItem(
            filename="phantom_seed.png",
            image=phantom_base,
            cluster_id="phantom_cluster",
        )
    ]
    for i in range(20):
        items.append(
            BenchmarkItem(
                filename=f"distractor_{i:03d}.png",
                image=create_base_skin(seed=10000 + i * 37),
                cluster_id=None,
            )
        )

    res = evaluate_cluster_scenario(
        scenario_name="No-Peer Phantom Test",
        items=items,
        ground_truth_counts={"phantom_cluster": 1},
        mode="confirmed",
        max_batches=5,
        batch_size=5,
        random_seed=42,
    )
    assert res.cluster_recall_rate == 0.0
